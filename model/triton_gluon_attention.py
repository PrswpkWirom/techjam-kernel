"""Blackwell Gluon attention core used by :mod:`triton_fused_attention`.

Each program owns one query tile and one batch/head pair. QK, masked softmax,
and P@V are kept in registers/shared MMA storage; no score or probability
matrix is written to global memory.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.tools.triton_to_gluon_translater.translator_helpers import default_blocked_layout
from triton.language.extra import libdevice


_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)
_SUPPORTED_SEQUENCE_LENGTHS = (32, 64, 128)
_SUPPORTED_WARPS = (1, 2, 4, 8)
_FP32_FUSED_HEAD_DIMS = (64,)

# These values are deliberately compile-time kernel modes.  Keeping the mode
# out of the element-wise dataflow lets Gluon emit one specialization per
# model dtype while sharing the QK/mask/softmax/PV implementation.
FP16_MODE = gl.constexpr(0)
BF16_MODE = gl.constexpr(1)
FP32_TF32_MODE = gl.constexpr(2)


@gluon.jit
def _mma(
    a, b, M: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
    INPUT_PRECISION: gl.constexpr,
):
    # This is the same operand packing rule used by Triton's own
    # Triton-to-Gluon translator.  FP16/BF16 consume two 16-bit values per
    # 32-bit register lane; FP32 consumes one.
    a_bitwidth: gl.constexpr = a.type.element_ty.primitive_bitwidth
    b_bitwidth: gl.constexpr = b.type.element_ty.primitive_bitwidth
    min_bitwidth: gl.constexpr = min(a_bitwidth, b_bitwidth)
    k_width: gl.constexpr = max(32 // min_bitwidth, 1)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[gl.num_warps(), 1], instr_shape=[16, 8]
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(
        parent=mma_layout, operand_index=0, k_width=k_width
    )
    b_layout: gl.constexpr = gl.DotOperandLayout(
        parent=mma_layout, operand_index=1, k_width=k_width
    )
    a = gl.convert_layout(a, a_layout)
    b = gl.convert_layout(b, b_layout)
    acc = gl.zeros([M, N], gl.float32, layout=mma_layout)
    return mma_v2(a, b, acc, input_precision=INPUT_PRECISION)


@gluon.jit
def _add_rn(a, b):
    return libdevice.add_rn(a, b)


@gluon.jit
def _kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, mask_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_mb, stride_ms,
    num_heads: gl.constexpr, sequence_length: gl.constexpr,
    scale: gl.constexpr, HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    CAUSAL: gl.constexpr, HAS_MASK: gl.constexpr,
    DTYPE_MODE: gl.constexpr, INPUT_PRECISION: gl.constexpr,
):
    qtile = gl.program_id(0)
    bh = gl.program_id(1)
    batch = bh // num_heads
    head = bh - batch * num_heads
    parent_q: gl.constexpr = default_blocked_layout([BLOCK_M, HEAD_DIM], gl.num_warps())
    parent_k: gl.constexpr = default_blocked_layout([BLOCK_N, HEAD_DIM], gl.num_warps())
    parent_score: gl.constexpr = default_blocked_layout([BLOCK_M, BLOCK_N], gl.num_warps())
    parent_1d_q: gl.constexpr = gl.SliceLayout(1, parent_q)
    parent_1d_k: gl.constexpr = gl.SliceLayout(1, parent_k)
    parent_1d_score_m: gl.constexpr = gl.SliceLayout(1, parent_score)
    parent_1d_score_n: gl.constexpr = gl.SliceLayout(0, parent_score)
    m = gl.arange(0, BLOCK_M, layout=parent_1d_q) + qtile * BLOCK_M
    d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, parent_q))
    n = gl.arange(0, BLOCK_N, layout=parent_1d_k)
    score_m = gl.arange(0, BLOCK_M, layout=parent_1d_score_m) + qtile * BLOCK_M
    score_n = gl.arange(0, BLOCK_N, layout=parent_1d_score_n)
    q_valid = m < sequence_length
    k_valid = n < sequence_length
    q_valid_score = gl.convert_layout(q_valid, parent_1d_score_m)
    k_valid_score = gl.convert_layout(k_valid, parent_1d_score_n)
    q = gl.load(
        q_ptr + batch * stride_qb + head * stride_qh
        + m[:, None] * stride_qs + d[None, :],
        mask=q_valid[:, None], other=0.0,
    )
    k = gl.load(
        k_ptr + batch * stride_kb + head * stride_kh
        + n[:, None] * stride_ks + d[None, :],
        mask=k_valid[:, None], other=0.0,
    )
    v = gl.load(
        v_ptr + batch * stride_vb + head * stride_vh
        + n[:, None] * stride_vs + d[None, :],
        mask=k_valid[:, None], other=0.0,
    )
    scores = _mma(
        q, k.trans(), BLOCK_M, HEAD_DIM, BLOCK_N,
        INPUT_PRECISION=INPUT_PRECISION,
    )
    scores = gl.convert_layout(scores, parent_score)
    if DTYPE_MODE == FP16_MODE:
        scores = scores.to(gl.float16)
        scores = (scores.to(gl.float32) * scale).to(gl.float16).to(gl.float32)
    elif DTYPE_MODE == BF16_MODE:
        scores = scores.to(gl.bfloat16)
        scores = (scores.to(gl.float32) * scale).to(gl.bfloat16).to(gl.float32)
    else:
        # FP32 QK is intentionally not routed through a narrower dtype.  The
        # MMA helper receives input_precision="tf32" for this specialization.
        scores = scores * scale
    included = q_valid_score[:, None] & k_valid_score[None, :]
    if CAUSAL:
        included = included & (score_n[None, :] <= score_m[:, None])
    if HAS_MASK:
        key_is_valid = gl.load(
            mask_ptr + batch * stride_mb + n * stride_ms,
            mask=k_valid, other=0,
        )
        key_is_valid = gl.convert_layout(key_is_valid, parent_1d_score_n)
        included = included & key_is_valid[None, :]
    scores = gl.where(included, scores, -float("inf"))
    row_max = gl.max(scores, axis=1)
    origin = gl.where(row_max == -float("inf"), 0.0, row_max)
    numerator = libdevice.exp(scores - origin[:, None])
    denominator = gl.reduce(numerator, axis=1, combine_fn=_add_rn)
    denominator = gl.convert_layout(denominator, parent_1d_score_m)
    denominator = gl.where(denominator > 0.0, denominator, 1.0)
    probabilities = libdevice.div_rn(numerator, denominator[:, None])
    if DTYPE_MODE == FP16_MODE:
        probabilities = probabilities.to(gl.float16)
    elif DTYPE_MODE == BF16_MODE:
        probabilities = probabilities.to(gl.bfloat16)
    probabilities = gl.convert_layout(probabilities, parent_score)
    values = gl.convert_layout(v, parent_k)
    accumulator = _mma(
        probabilities, values, BLOCK_M, BLOCK_N, HEAD_DIM,
        INPUT_PRECISION=INPUT_PRECISION,
    )
    accumulator = gl.convert_layout(accumulator, parent_q)
    if DTYPE_MODE == FP16_MODE:
        out = accumulator.to(gl.float16)
    elif DTYPE_MODE == BF16_MODE:
        out = accumulator.to(gl.bfloat16)
    else:
        out = accumulator
    gl.store(
        out_ptr + batch * stride_ob + head * stride_oh
        + m[:, None] * stride_os + d[None, :],
        out, mask=q_valid[:, None],
    )


def triton_gluon_full_attention(
    q, k, v, mask=None, causal=False, scale=None,
    block_m=128, block_n=None, num_warps=8, output_bshd=False,
):
    """Blackwell full-row attention using Gluon's register MMA path.

    This is an internal accelerator entry point, but it still owns its
    dispatch boundary: malformed tensors raise a useful error, while valid
    inputs outside the compiled Gluon contract use the exact PyTorch
    operation-order fallback.  That keeps callers correct if the benchmark
    adds a dtype, shape, or device that this kernel does not specialize.
    """
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, H, S, D], got {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have the same [B, H, S, D] shape")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    batch, num_heads, sequence_length, head_dim = q.shape
    if sequence_length <= 0 or head_dim <= 0:
        raise ValueError("sequence length and head dimension must be positive")
    if mask is not None:
        if mask.shape != (batch, sequence_length):
            raise ValueError(
                "mask must have shape "
                f"{(batch, sequence_length)}, got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        if mask.device != q.device:
            raise ValueError("mask and q must be on the same device")
    if block_m <= 0 or block_n is not None and block_n <= 0:
        raise ValueError("block sizes must be positive")
    if block_n is None:
        block_n = triton_next_power_of_2(q.shape[2])
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    def reference() -> torch.Tensor:
        # BaselineSelfAttention materializes contiguous [B,H,S,D] views before
        # its QK/PV matmuls.  Keep this copy strictly on the fallback path; the
        # fused kernel consumes the original strides directly.
        q_ref = q.contiguous()
        k_ref = k.contiguous()
        v_ref = v.contiguous()
        scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
        if causal:
            causal_mask = torch.ones(
                (sequence_length, sequence_length),
                device=q.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        result = torch.matmul(probabilities, v_ref)
        return result.transpose(1, 2).contiguous() if output_bshd else result

    # Gluon's mma_v2 lowering is intentionally restricted to the validated
    # Blackwell tile envelope. Partial sequence lengths use the reference to
    # avoid changing the baseline's reduction/rounding behavior.
    supported = (
        q.device.type == "cuda"
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (q.dtype != torch.float32 or head_dim in _FP32_FUSED_HEAD_DIMS)
        and head_dim in _SUPPORTED_HEAD_DIMS
        and sequence_length in _SUPPORTED_SEQUENCE_LENGTHS
        and block_m in (16, 32, 64, 128)
        and block_n in (32, 64, 128)
        and block_n >= sequence_length
        and num_warps in _SUPPORTED_WARPS
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and torch.cuda.get_device_capability(q.device)[0] >= 12
        and not (
            torch.is_grad_enabled()
            and (q.requires_grad or k.requires_grad or v.requires_grad)
        )
    )
    if not supported:
        return reference()

    if output_bshd:
        # The adapter consumes [B,S,H,D]. Writing this layout directly avoids
        # three input head-repacking copies and the post-attention transpose.
        out = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1], q.shape[3]),
            device=q.device,
            dtype=q.dtype,
        )
        output_strides = (out.stride(0), out.stride(2), out.stride(1))
    else:
        out = torch.empty_like(q)
        output_strides = (out.stride(0), out.stride(1), out.stride(2))
    if mask is None:
        mask = q
        smb = sms = 0
        hm = False
    else:
        smb, sms = mask.stride(0), mask.stride(1)
        hm = True
    if q.dtype == torch.float16:
        dtype_mode = FP16_MODE
        input_precision = None
    elif q.dtype == torch.bfloat16:
        dtype_mode = BF16_MODE
        input_precision = None
    else:
        dtype_mode = FP32_TF32_MODE
        input_precision = "tf32"
    _kernel[((q.shape[2] + block_m - 1) // block_m, q.shape[0] * q.shape[1])](
        q, k, v, out, mask,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output_strides[0], output_strides[1], output_strides[2],
        smb, sms,
        num_heads=q.shape[1], sequence_length=q.shape[2], scale=scale,
        HEAD_DIM=q.shape[-1], BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal, HAS_MASK=hm,
        DTYPE_MODE=dtype_mode, INPUT_PRECISION=input_precision,
        num_warps=num_warps,
    )
    return out


def triton_next_power_of_2(value):
    return 1 << (value - 1).bit_length()
