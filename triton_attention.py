"""Triton implementation of the benchmark's self-attention softmax stage.

The public seam is :func:`triton_attention_softmax` and
:class:`TritonSelfAttention`. The latter intentionally exposes the same learned
projection names as the benchmark's baseline attention module so models can
copy weights with ``strict=True``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _attention_softmax_kernel(
    scores_ptr,
    valid_token_mask_ptr,
    probabilities_ptr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_VALID_TOKEN_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Compute one complete [key] softmax row per Triton program."""
    row = tl.program_id(0)
    key_position = tl.arange(0, BLOCK_SIZE)
    in_sequence = key_position < seq_len

    # scores is contiguous [B, H, S, S], flattened to [B * H * S, S].
    # Keeping scale, both masks, and softmax in this program means the score row
    # is loaded and the probability row is stored only once in global memory.
    score = tl.load(
        scores_ptr + row * seq_len + key_position,
        mask=in_sequence,
        other=-float("inf"),
    )
    # The baseline scales in the model dtype, then promotes to fp32 for
    # softmax. Preserve that rounding point before the fp32 reductions.
    #score = score*scale
    score = (score.to(tl.float32) * scale).to(scores_ptr.dtype.element_ty)
    score = score.to(tl.float32)

    # Flattened rows vary query first, then head, then batch.
    query_position = row % seq_len
    if CAUSAL:
        score = tl.where(key_position <= query_position, score, -float("inf"))

    if HAS_VALID_TOKEN_MASK:
        batch = row // (num_heads * seq_len)
        key_is_valid = tl.load(
            valid_token_mask_ptr + batch * seq_len + key_position,
            mask=in_sequence,
            other=0,
        )
        score = tl.where(key_is_valid, score, -float("inf"))

    # FP32 reductions match the baseline's stable fp32 softmax even when the
    # model uses fp16 or bf16. The store casts back to scores' element type.
    row_max = tl.max(score, axis=0)
    # libdevice.exp is more accurate than Triton's fast approximate exponential.
    # That precision matters because small errors compound through model layers.
    #numerator = tl.exp(score - row_max)
    numerator = libdevice.exp(score - row_max)
    if BLOCK_SIZE == 128:
        # PyTorch's CUDA persistent softmax gives each warp lane four values
        # spaced 32 columns apart, sums those serially, then reduces the warp.
        # Matching that association prevents one-ULP fp16 differences from
        # being amplified by later fp16 GEMMs in the full Transformer.
        first_half, second_half = tl.split(
            tl.trans(tl.reshape(numerator, (2, 64)))
        )
        part_0, part_1 = tl.split(
            tl.trans(tl.reshape(first_half, (2, 32)))
        )
        part_2, part_3 = tl.split(
            tl.trans(tl.reshape(second_half, (2, 32)))
        )
        lane_sum = ((part_0 + part_1) + part_2) + part_3
        denominator = tl.sum(lane_sum, axis=0)
    else:
        denominator = tl.sum(numerator, axis=0)
    probability = libdevice.div_rn(numerator, denominator)
    #probability = numerator / denominator
    tl.store(
        probabilities_ptr + row * seq_len + key_position,
        probability,
        mask=in_sequence,
    )


def triton_attention_softmax(
    scores: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Fuse attention scaling, key masking, and row-wise softmax on CUDA."""
    if scores.ndim != 4:
        raise ValueError(f"scores must have shape [B, H, S, S], got {scores.shape}")

    batch, num_heads, query_len, key_len = scores.shape
    if query_len != key_len:
        raise ValueError("self-attention scores must be square in the last two dims")
    if scores.device.type != "cuda":
        raise ValueError("triton_attention_softmax requires CUDA scores")
    if scores.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported scores dtype: {scores.dtype}")
    if not scores.is_contiguous():
        raise ValueError("scores must be contiguous")
    if valid_token_mask is not None:
        if valid_token_mask.shape != (batch, key_len):
            raise ValueError(
                "valid_token_mask must have shape "
                f"{(batch, key_len)}, got {tuple(valid_token_mask.shape)}"
            )
        if valid_token_mask.dtype != torch.bool:
            raise TypeError("valid_token_mask must have dtype torch.bool")
        if valid_token_mask.device != scores.device:
            raise ValueError("valid_token_mask and scores must be on the same device")
        if not valid_token_mask.is_contiguous():
            raise ValueError("valid_token_mask must be contiguous")

    probabilities = torch.empty_like(scores)
    block_size = triton.next_power_of_2(key_len)
    if block_size > 65536:
        raise ValueError(f"sequence length {key_len} is too large for this row kernel")

    # The mask pointer is unused in the no-mask specialization, but Triton still
    # requires a valid launch argument. Reusing the output avoids an allocation.
    mask_ptr = valid_token_mask if valid_token_mask is not None else probabilities
    # One warp owns a 128-wide row in the default benchmark. Besides avoiding
    # cross-warp reduction overhead, this mirrors PyTorch's persistent-softmax
    # reduction shape closely enough to avoid fp16 branch-point drift.
    if block_size <= 128:
        num_warps = 1
    elif block_size <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    row_count = batch * num_heads * query_len
    _attention_softmax_kernel[(row_count,)](
        scores,
        mask_ptr,
        probabilities,
        seq_len=key_len,
        num_heads=num_heads,
        scale=scale,
        CAUSAL=causal,
        HAS_VALID_TOKEN_MASK=valid_token_mask is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return probabilities


class TritonSelfAttention(nn.Module):
    """Baseline-compatible attention with a fused Triton softmax stage."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1))
        probabilities = triton_attention_softmax(
            scores,
            valid_token_mask,
            scale=self.scale,
            causal=causal,
        )
        context = torch.matmul(probabilities, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output
