"""Fused inference-only self-attention for the Transformer benchmark.

The fast path computes tiled QK^T, masking, online softmax, and PV in one
Triton kernel. It writes only the final [B, H, S, D] context tensor; full score
and probability tensors are never materialized. Unsupported inputs use the
value-equivalent PyTorch path so this module remains safe outside benchmark
shapes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


@triton.jit
def _fused_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_token_mask_ptr,
    output_ptr,
    stride_q_batch,
    stride_q_head,
    stride_q_sequence,
    stride_k_batch,
    stride_k_head,
    stride_k_sequence,
    stride_v_batch,
    stride_v_head,
    stride_v_sequence,
    stride_output_batch,
    stride_output_head,
    stride_output_sequence,
    stride_mask_batch,
    num_heads: tl.constexpr,
    sequence_length: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_VALID_TOKEN_MASK: tl.constexpr,
) -> None:
    """Compute one query tile for one batch element and attention head."""
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dimension_offsets = tl.arange(0, HEAD_DIM)
    query_in_bounds = query_offsets < sequence_length

    q_offsets = (
        batch * stride_q_batch
        + head * stride_q_head
        + query_offsets[:, None] * stride_q_sequence
        + dimension_offsets[None, :]
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=query_in_bounds[:, None],
        other=0.0,
    )

    # Pass 1 computes FP32 online-softmax statistics without storing scores.
    # Pass 2 recomputes each score tile and accumulates normalized P @ V. The
    # second pass preserves the baseline's probability-rounding point while
    # still removing both [S, S] global-memory intermediates.
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    key_loop_end = sequence_length
    if CAUSAL:
        key_loop_end = tl.minimum(
            (query_tile + 1) * BLOCK_M, sequence_length
        )

    for key_start in tl.range(
        0, key_loop_end, BLOCK_N, num_stages=NUM_STAGES
    ):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_in_bounds = key_offsets < sequence_length

        k_offsets = (
            batch * stride_k_batch
            + head * stride_k_head
            + key_offsets[:, None] * stride_k_sequence
            + dimension_offsets[None, :]
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=key_in_bounds[:, None],
            other=0.0,
        )

        # q and k retain their model dtype so tl.dot can use Tensor Cores. Its
        # accumulator is FP32; scaling and online-softmax updates remain FP32.
        scores = tl.dot(q, tl.trans(k)).to(q.dtype)
        scores = (scores * scale).to(q.dtype).to(tl.float32)
        included = query_in_bounds[:, None] & key_in_bounds[None, :]

        if CAUSAL:
            included &= key_offsets[None, :] <= query_offsets[:, None]

        if HAS_VALID_TOKEN_MASK:
            key_is_valid = tl.load(
                valid_token_mask_ptr
                + batch * stride_mask_batch
                + key_offsets,
                mask=key_in_bounds,
                other=0,
            ).to(tl.int1)
            included &= key_is_valid[None, :]

        scores = tl.where(included, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)

        # A completely masked row has max=-inf. Use zero as a harmless exponent
        # origin until a valid key appears, avoiding inf-inf and NaNs.
        exponent_origin = tl.where(
            new_max == -float("inf"), 0.0, new_max
        )
        rescale = tl.where(
            running_max == -float("inf"),
            0.0,
            libdevice.exp(running_max - exponent_origin),
        )
        probabilities = tl.where(
            included,
            libdevice.exp(scores - exponent_origin[:, None]),
            0.0,
        )

        if BLOCK_N == 128:
            halves = tl.reshape(
                probabilities, (BLOCK_M, 2, 64)
            ).permute(0, 2, 1)
            first_half, second_half = tl.split(halves)
            first_quarters = tl.reshape(
                first_half, (BLOCK_M, 2, 32)
            ).permute(0, 2, 1)
            second_quarters = tl.reshape(
                second_half, (BLOCK_M, 2, 32)
            ).permute(0, 2, 1)
            part_0, part_1 = tl.split(first_quarters)
            part_2, part_3 = tl.split(second_quarters)
            lane_sum = ((part_0 + part_1) + part_2) + part_3
            block_sum = tl.sum(lane_sum, axis=1)
        else:
            block_sum = tl.sum(probabilities, axis=1)
        running_sum = running_sum * rescale + block_sum
        running_max = new_max

    denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
    exponent_origin = tl.where(
        running_max == -float("inf"), 0.0, running_max
    )
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for key_start in tl.range(
        0, key_loop_end, BLOCK_N, num_stages=NUM_STAGES
    ):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_in_bounds = key_offsets < sequence_length
        k_offsets = (
            batch * stride_k_batch
            + head * stride_k_head
            + key_offsets[:, None] * stride_k_sequence
            + dimension_offsets[None, :]
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=key_in_bounds[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)).to(q.dtype)
        scores = (scores * scale).to(q.dtype).to(tl.float32)
        included = query_in_bounds[:, None] & key_in_bounds[None, :]
        if CAUSAL:
            included &= key_offsets[None, :] <= query_offsets[:, None]
        if HAS_VALID_TOKEN_MASK:
            key_is_valid = tl.load(
                valid_token_mask_ptr
                + batch * stride_mask_batch
                + key_offsets,
                mask=key_in_bounds,
                other=0,
            ).to(tl.int1)
            included &= key_is_valid[None, :]

        probabilities = tl.where(
            included,
            libdevice.div_rn(
                libdevice.exp(scores - exponent_origin[:, None]),
                denominator[:, None],
            ),
            0.0,
        )
        v_offsets = (
            batch * stride_v_batch
            + head * stride_v_head
            + key_offsets[:, None] * stride_v_sequence
            + dimension_offsets[None, :]
        )
        v = tl.load(
            v_ptr + v_offsets,
            mask=key_in_bounds[:, None],
            other=0.0,
        )
        accumulator = tl.dot(probabilities.to(q.dtype), v, accumulator)

    output = accumulator
    output_offsets = (
        batch * stride_output_batch
        + head * stride_output_head
        + query_offsets[:, None] * stride_output_sequence
        + dimension_offsets[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=query_in_bounds[:, None],
    )


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    scale: float,
) -> torch.Tensor:
    """Value-equivalent fallback matching the organizer's attention order."""
    sequence_length = q.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.ones(
            (sequence_length, sequence_length),
            device=q.device,
            dtype=torch.bool,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    if valid_token_mask is not None:
        scores = scores.masked_fill(
            ~valid_token_mask[:, None, None, :], float("-inf")
        )
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probabilities, v)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
) -> tuple[int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, H, S, D], got {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "q, k, and v must have the same [B, H, S, D] shape; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")

    batch, num_heads, sequence_length, head_dim = q.shape
    if sequence_length <= 0 or head_dim <= 0:
        raise ValueError("sequence length and head dimension must be positive")
    if valid_token_mask is not None:
        if valid_token_mask.shape != (batch, sequence_length):
            raise ValueError(
                "valid_token_mask must have shape "
                f"{(batch, sequence_length)}, got {tuple(valid_token_mask.shape)}"
            )
        if valid_token_mask.dtype != torch.bool:
            raise TypeError("valid_token_mask must have dtype torch.bool")
        if valid_token_mask.device != q.device:
            raise ValueError("valid_token_mask and q must be on the same device")
    return batch, num_heads, sequence_length, head_dim


def _can_use_triton_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    head_dim: int,
) -> bool:
    if q.device.type != "cuda":
        return False
    # The bf16 kernel is close but does not pass the organizer's per-element OR
    # tolerance for every tested partial tile, so correctness takes the fallback.
    if q.dtype != torch.float16:
        return False
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    if valid_token_mask is not None and valid_token_mask.stride(-1) != 1:
        return False
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return False
    return True


def _launch_configuration(
    sequence_length: int, head_dim: int, causal: bool
) -> tuple[int, int, int, int]:
    """Return a deterministic shape-specialized (M, N, warps, stages) tuple."""
    if sequence_length <= 64:
        return 32, 32, 4, 2
    if sequence_length <= 128 and head_dim <= 64:
        if causal:
            return 64, 64, 4, 3
        return 64, 128, 4, 3
    if head_dim <= 64:
        return 64, 64, 4, 3
    return 32, 32, 4, 2


def triton_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Return scaled dot-product self-attention context without [S, S] tensors."""
    batch, num_heads, sequence_length, head_dim = _validate_inputs(
        q, k, v, valid_token_mask
    )
    if scale is None:
        scale = head_dim**-0.5
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    if not _can_use_triton_fast_path(q, k, v, valid_token_mask, head_dim):
        return _reference_attention(q, k, v, valid_token_mask, causal, scale)

    output = torch.empty_like(q)
    block_m, block_n, num_warps, num_stages = _launch_configuration(
        sequence_length, head_dim, causal
    )
    # The no-mask specialization does not dereference this pointer. Reusing q
    # avoids an otherwise unnecessary allocation in the timed forward path.
    mask_pointer = valid_token_mask if valid_token_mask is not None else q
    grid = (triton.cdiv(sequence_length, block_m), batch * num_heads)
    _fused_attention_forward_kernel[grid](
        q,
        k,
        v,
        mask_pointer,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        valid_token_mask.stride(0) if valid_token_mask is not None else 0,
        num_heads=num_heads,
        sequence_length=sequence_length,
        scale=float(scale),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        CAUSAL=causal,
        HAS_VALID_TOKEN_MASK=valid_token_mask is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


class TritonFusedSelfAttention(nn.Module):
    """Baseline-compatible projections backed by fused Triton attention."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        # Names intentionally match BaselineSelfAttention for strict=True copies.
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = x.shape
        return (
            x.view(batch, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, sequence_length, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        context = triton_fused_attention(
            q,
            k,
            v,
            valid_token_mask=valid_token_mask,
            causal=causal,
            scale=self.scale,
        )
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, sequence_length, self.d_model)
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output
