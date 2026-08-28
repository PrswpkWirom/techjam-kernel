"""Triton attention kernels for the Transformer benchmark.

The Blackwell FP16 path uses a one-program Gluon kernel: QK, masked softmax,
and P@V all stay on chip, with Gluon's MMA lowering preserving the baseline
FP16 GEMM rounding. Other dtypes/devices/shapes use the value-equivalent
PyTorch fallback.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

from triton_gluon_attention import triton_gluon_full_attention


_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
    stride_mask_sequence,
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
    key_lane_offsets = tl.arange(0, BLOCK_N)
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

    # Keep all online-softmax state and the P@V accumulator in FP32. Q/K/V stay
    # in their model dtype so both tl.dot operations can use Tensor Cores.
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # For causal attention, no query in this program can see a key beyond the
    # end of its own query tile. The per-element causal mask below still handles
    # the diagonal tile exactly.
    key_loop_end = sequence_length
    if CAUSAL:
        key_loop_end = tl.minimum((query_tile + 1) * BLOCK_M, sequence_length)

    for key_start in tl.range(
        0, key_loop_end, BLOCK_N, num_stages=NUM_STAGES
    ):
        key_offsets = key_start + key_lane_offsets
        key_in_bounds = key_offsets < sequence_length

        k_offsets = (
            batch * stride_k_batch
            + head * stride_k_head
            + key_offsets[:, None] * stride_k_sequence
            + dimension_offsets[None, :]
        )
        v_offsets = (
            batch * stride_v_batch
            + head * stride_v_head
            + key_offsets[:, None] * stride_v_sequence
            + dimension_offsets[None, :]
        )
        k = tl.load(
            k_ptr + k_offsets,
            mask=key_in_bounds[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_offsets,
            mask=key_in_bounds[:, None],
            other=0.0,
        )

        # tl.dot accumulates QK^T in FP32. The organizer baseline materializes
        # QK^T in the model dtype and applies scale there before fp32 softmax, so
        # preserve those two rounding points for fp16/bf16 numerical agreement.
        scores = tl.dot(q, tl.trans(k)).to(q.dtype)
        scores = (scores.to(tl.float32) * scale).to(q.dtype).to(tl.float32)

        included = query_in_bounds[:, None] & key_in_bounds[None, :]
        if CAUSAL:
            included &= key_offsets[None, :] <= query_offsets[:, None]

        if HAS_VALID_TOKEN_MASK:
            key_is_valid = tl.load(
                valid_token_mask_ptr
                + batch * stride_mask_batch
                + key_offsets * stride_mask_sequence,
                mask=key_in_bounds,
                other=0,
            ).to(tl.int1)
            # Match the baseline exactly: valid_token_mask masks keys here.
            # Invalid query rows are zeroed only after out_proj in the adapter.
            included &= key_is_valid[None, :]

        scores = tl.where(included, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)

        # Rows can be fully masked (or be query-tail lanes). Avoid -inf - -inf
        # while keeping their recurrence state at zero until a valid key exists.
        exponent_origin = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.where(
            running_max == -float("inf"),
            0.0,
            libdevice.exp(running_max - exponent_origin),
        )
        p = tl.where(
            included,
            libdevice.exp(scores - exponent_origin[:, None]),
            0.0,
        )

        tile_sum = tl.sum(p, axis=1)
        new_sum = running_sum * alpha + tile_sum

        # Keep ACC as the normalized context for all keys processed so far.
        # This is algebraically the same online softmax recurrence, but it makes
        # the Tensor-Core input p/new_sum resemble the baseline's final softmax
        # probabilities before the model-dtype cast. That materially reduces
        # fp16/bf16 rounding drift across multiple Transformer layers.
        denominator = tl.where(new_sum > 0.0, new_sum, 1.0)
        previous_weight = tl.where(
            new_sum > 0.0,
            running_sum * alpha / denominator,
            0.0,
        )
        normalized_p = p / denominator[:, None]
        accumulator = accumulator * previous_weight[:, None]
        accumulator = tl.dot(normalized_p.to(v.dtype), v, accumulator)

        running_sum = new_sum
        running_max = new_max

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


@triton.jit
def _pv_accumulate_tile(
    probabilities,
    v_ptr,
    batch,
    head,
    key_start,
    sequence_length,
    stride_v_batch,
    stride_v_head,
    stride_v_sequence,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    accumulator,
):
    key_offsets = key_start + tl.arange(0, BLOCK_K)
    key_in_bounds = key_offsets < sequence_length
    dimension_offsets = tl.arange(0, HEAD_DIM)
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
    return tl.dot(probabilities.to(tl.float16), v, accumulator)


@triton.jit
def _pv_full_tile(
    probabilities,
    values,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Keep the single-tile P·V MMA lowering independent of mask dataflow."""
    accumulator = tl.zeros((HEAD_DIM, probabilities.shape[0]), tl.float32)
    result = tl.dot(
        tl.trans(values), tl.trans(probabilities.to(tl.float16)), accumulator
    )
    return tl.trans(result).to(tl.float32)


# Legacy Triton-language full-row experiment. The production adapter below
# dispatches to the Blackwell Gluon core instead; this remains available for
# isolated regression tests and future cross-architecture work.
@triton.jit
def _fused_full_attention_forward_kernel(
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
    stride_mask_sequence,
    num_heads: tl.constexpr,
    sequence_length: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PV_BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_VALID_TOKEN_MASK: tl.constexpr,
) -> None:
    """One-program full-row attention with no global score/probability tensor."""
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    dimension_offsets = tl.arange(0, HEAD_DIM)
    query_in_bounds = query_offsets < sequence_length
    key_in_bounds = key_offsets < sequence_length

    q_offsets = (
        batch * stride_q_batch
        + head * stride_q_head
        + query_offsets[:, None] * stride_q_sequence
        + dimension_offsets[None, :]
    )
    k_offsets = (
        batch * stride_k_batch
        + head * stride_k_head
        + key_offsets[:, None] * stride_k_sequence
        + dimension_offsets[None, :]
    )
    v_offsets = (
        batch * stride_v_batch
        + head * stride_v_head
        + key_offsets[:, None] * stride_v_sequence
        + dimension_offsets[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=query_in_bounds[:, None], other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=key_in_bounds[:, None], other=0.0)
    if PV_BLOCK_K == BLOCK_N:
        v = tl.load(v_ptr + v_offsets, mask=key_in_bounds[:, None], other=0.0)

    # Preserve the baseline's QK and scale rounding boundaries before fp32
    # softmax. All score/probability state remains in registers or shared/TMEM.
    scores = tl.dot(q, tl.trans(k)).to(q.dtype)
    scores = (scores.to(tl.float32) * scale).to(q.dtype).to(tl.float32)
    # Apply masks in the same independent score-select order as the exact
    # baseline-compatible softmax kernel. Keeping the predicates separate is
    # important on the MMA-produced layout, including an all-true mask.
    included = key_in_bounds[None, :]
    scores = tl.where(included, scores, -float("inf"))
    if CAUSAL:
        scores = tl.where(
            key_offsets[None, :] <= query_offsets[:, None],
            scores,
            -float("inf"),
        )
    key_is_valid = tl.full((BLOCK_N,), 1, tl.int1)
    if HAS_VALID_TOKEN_MASK:
        key_is_valid = tl.load(
            valid_token_mask_ptr
            + batch * stride_mask_batch
            + key_offsets * stride_mask_sequence,
            mask=key_in_bounds,
            other=0,
        ).to(tl.int1)
        scores = tl.where(key_is_valid[None, :], scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    exponent_origin = tl.where(row_max == -float("inf"), 0.0, row_max)
    # Masked scores are -inf, so exp naturally produces zero. Keeping this as
    # a direct row expression matches the standalone baseline-compatible
    # softmax layout for both all-true and partially padded masks.
    numerator = libdevice.exp(scores - exponent_origin[:, None])

    if BLOCK_N == 128:
        halves = tl.reshape(numerator, (BLOCK_M, 2, 64)).permute(0, 2, 1)
        first_half, second_half = tl.split(halves)
        first_quarters = tl.reshape(
            first_half, (BLOCK_M, 2, 32)
        ).permute(0, 2, 1)
        second_quarters = tl.reshape(
            second_half, (BLOCK_M, 2, 32)
        ).permute(0, 2, 1)
        part_0, part_1 = tl.split(first_quarters)
        part_2, part_3 = tl.split(second_quarters)
        lane_sum = libdevice.add_rn(part_0, part_1)
        lane_sum = libdevice.add_rn(lane_sum, part_2)
        lane_sum = libdevice.add_rn(lane_sum, part_3)
    elif BLOCK_N == 64:
        pairs = tl.reshape(numerator, (BLOCK_M, 2, 32)).permute(0, 2, 1)
        part_0, part_1 = tl.split(pairs)
        lane_sum = libdevice.add_rn(part_0, part_1)
    elif BLOCK_N == 32:
        lane_sum = numerator

    if BLOCK_N >= 32:
        reduction_16 = tl.reshape(
            lane_sum, (BLOCK_M, 2, 16)
        ).permute(0, 2, 1)
        reduction_16_left, reduction_16_right = tl.split(reduction_16)
        reduction_16 = libdevice.add_rn(
            reduction_16_left, reduction_16_right
        )
        reduction_8 = tl.reshape(
            reduction_16, (BLOCK_M, 2, 8)
        ).permute(0, 2, 1)
        reduction_8_left, reduction_8_right = tl.split(reduction_8)
        reduction_8 = libdevice.add_rn(reduction_8_left, reduction_8_right)
        reduction_4 = tl.reshape(
            reduction_8, (BLOCK_M, 2, 4)
        ).permute(0, 2, 1)
        reduction_4_left, reduction_4_right = tl.split(reduction_4)
        reduction_4 = libdevice.add_rn(reduction_4_left, reduction_4_right)
        reduction_2 = tl.reshape(
            reduction_4, (BLOCK_M, 2, 2)
        ).permute(0, 2, 1)
        reduction_2_left, reduction_2_right = tl.split(reduction_2)
        reduction_2 = libdevice.add_rn(reduction_2_left, reduction_2_right)
        reduction_1 = tl.reshape(
            reduction_2, (BLOCK_M, 2, 1)
        ).permute(0, 2, 1)
        reduction_1_left, reduction_1_right = tl.split(reduction_1)
        denominator = tl.reshape(
            libdevice.add_rn(reduction_1_left, reduction_1_right),
            (BLOCK_M,),
        )
    else:
        denominator = tl.sum(numerator, axis=1)

    denominator = tl.where(denominator > 0.0, denominator, 1.0)
    probabilities = libdevice.div_rn(numerator, denominator[:, None])

    # This is the only reduction-order-sensitive operation. Keep it in the
    # same program while accumulating in fp32, then store the model dtype.
    if PV_BLOCK_K == BLOCK_N:
        pv_probabilities = tl.reshape(probabilities, (BLOCK_M, BLOCK_N))
        pv_values = tl.reshape(v, (BLOCK_N, HEAD_DIM))
        accumulator = _pv_full_tile(
            pv_probabilities, pv_values, HEAD_DIM=HEAD_DIM, BLOCK_N=BLOCK_N
        )
    elif PV_BLOCK_K == 64:
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        probability_tiles = tl.reshape(probabilities, (BLOCK_M, 2, 64)).permute(
            0, 2, 1
        )
        probability_0, probability_1 = tl.split(probability_tiles)
        accumulator = _pv_accumulate_tile(
            probability_0,
            v_ptr,
            batch,
            head,
            0,
            sequence_length,
            stride_v_batch,
            stride_v_head,
            stride_v_sequence,
            HEAD_DIM=HEAD_DIM,
            BLOCK_K=64,
            accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_1,
            v_ptr,
            batch,
            head,
            64,
            sequence_length,
            stride_v_batch,
            stride_v_head,
            stride_v_sequence,
            HEAD_DIM=HEAD_DIM,
            BLOCK_K=64,
            accumulator=accumulator,
        )
    elif PV_BLOCK_K == 32:
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        groups = tl.reshape(probabilities, (BLOCK_M, 4, 32)).permute(0, 2, 1)
        groups_0, groups_1 = tl.split(groups)
        probability_0, probability_1 = tl.split(groups_0)
        probability_2, probability_3 = tl.split(groups_1)
        accumulator = _pv_accumulate_tile(
            probability_0, v_ptr, batch, head, 0, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=32, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_1, v_ptr, batch, head, 32, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=32, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_2, v_ptr, batch, head, 64, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=32, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_3, v_ptr, batch, head, 96, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=32, accumulator=accumulator,
        )
    elif PV_BLOCK_K == 16:
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        groups = tl.reshape(probabilities, (BLOCK_M, 8, 16)).permute(0, 2, 1)
        groups_0, groups_1 = tl.split(groups)
        groups_00, groups_01 = tl.split(groups_0)
        groups_10, groups_11 = tl.split(groups_1)
        probability_0, probability_1 = tl.split(groups_00)
        probability_2, probability_3 = tl.split(groups_01)
        probability_4, probability_5 = tl.split(groups_10)
        probability_6, probability_7 = tl.split(groups_11)
        accumulator = _pv_accumulate_tile(
            probability_0, v_ptr, batch, head, 0, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_1, v_ptr, batch, head, 16, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_2, v_ptr, batch, head, 32, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_3, v_ptr, batch, head, 48, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_4, v_ptr, batch, head, 64, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_5, v_ptr, batch, head, 80, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_6, v_ptr, batch, head, 96, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
        accumulator = _pv_accumulate_tile(
            probability_7, v_ptr, batch, head, 112, sequence_length,
            stride_v_batch, stride_v_head, stride_v_sequence,
            HEAD_DIM=HEAD_DIM, BLOCK_K=16, accumulator=accumulator,
        )
    else:
        accumulator = tl.dot(probabilities.to(q.dtype), v)
    output_value = accumulator
    output_offsets = (
        batch * stride_output_batch
        + head * stride_output_head
        + query_offsets[:, None] * stride_output_sequence
        + dimension_offsets[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        output_value,
        mask=query_in_bounds[:, None],
    )


@triton.jit
def _fused_qk_softmax_forward_kernel(
    q_ptr,
    k_ptr,
    valid_token_mask_ptr,
    probabilities_ptr,
    stride_q_batch,
    stride_q_head,
    stride_q_sequence,
    stride_k_batch,
    stride_k_head,
    stride_k_sequence,
    stride_probabilities_batch,
    stride_probabilities_head,
    stride_probabilities_query,
    stride_mask_batch,
    stride_mask_sequence,
    num_heads: tl.constexpr,
    sequence_length: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_VALID_TOKEN_MASK: tl.constexpr,
) -> None:
    """Fuse QK, scaling, masking, and a complete-row softmax.

    The probability matrix is intentionally materialized so the subsequent PV
    operation can use the same native matmul reduction as the organizer
    baseline. This isolates the only boundary that remained non-bit-exact in
    the fully fused kernel while still eliminating the score matrix and its
    separate softmax launch.
    """
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    dimension_offsets = tl.arange(0, HEAD_DIM)
    query_in_bounds = query_offsets < sequence_length
    key_in_bounds = key_offsets < sequence_length

    q_offsets = (
        batch * stride_q_batch
        + head * stride_q_head
        + query_offsets[:, None] * stride_q_sequence
        + dimension_offsets[None, :]
    )
    k_offsets = (
        batch * stride_k_batch
        + head * stride_k_head
        + key_offsets[:, None] * stride_k_sequence
        + dimension_offsets[None, :]
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=query_in_bounds[:, None],
        other=0.0,
    )
    k = tl.load(
        k_ptr + k_offsets,
        mask=key_in_bounds[:, None],
        other=0.0,
    )

    # Match the baseline's two model-dtype rounding points: native QK writes
    # fp16/bf16 scores, and multiplication by scale also returns that dtype
    # before the stable fp32 softmax.
    scores = tl.dot(q, tl.trans(k)).to(q.dtype)
    scores = (scores.to(tl.float32) * scale).to(q.dtype).to(tl.float32)

    included = query_in_bounds[:, None] & key_in_bounds[None, :]
    if CAUSAL:
        included &= key_offsets[None, :] <= query_offsets[:, None]
    if HAS_VALID_TOKEN_MASK:
        key_is_valid = tl.load(
            valid_token_mask_ptr
            + batch * stride_mask_batch
            + key_offsets * stride_mask_sequence,
            mask=key_in_bounds,
            other=0,
        ).to(tl.int1)
        included &= key_is_valid[None, :]

    scores = tl.where(included, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    exponent_origin = tl.where(row_max == -float("inf"), 0.0, row_max)
    numerator = tl.where(
        included,
        libdevice.exp(scores - exponent_origin[:, None]),
        0.0,
    )

    if BLOCK_N == 128:
        # Mirror PyTorch's persistent-softmax lane association: each lane first
        # sums values spaced 32 columns apart, then the lanes follow a fixed
        # binary reduction tree below.
        halves = tl.reshape(numerator, (BLOCK_M, 2, 64)).permute(0, 2, 1)
        first_half, second_half = tl.split(halves)
        first_quarters = tl.reshape(
            first_half, (BLOCK_M, 2, 32)
        ).permute(0, 2, 1)
        second_quarters = tl.reshape(
            second_half, (BLOCK_M, 2, 32)
        ).permute(0, 2, 1)
        part_0, part_1 = tl.split(first_quarters)
        part_2, part_3 = tl.split(second_quarters)
        lane_sum = libdevice.add_rn(part_0, part_1)
        lane_sum = libdevice.add_rn(lane_sum, part_2)
        lane_sum = libdevice.add_rn(lane_sum, part_3)
    elif BLOCK_N == 64:
        pairs = tl.reshape(numerator, (BLOCK_M, 2, 32)).permute(0, 2, 1)
        part_0, part_1 = tl.split(pairs)
        lane_sum = libdevice.add_rn(part_0, part_1)
    elif BLOCK_N == 32:
        lane_sum = numerator

    if BLOCK_N >= 32:
        reduction_16 = tl.reshape(
            lane_sum, (BLOCK_M, 2, 16)
        ).permute(0, 2, 1)
        reduction_16_left, reduction_16_right = tl.split(reduction_16)
        reduction_16 = libdevice.add_rn(
            reduction_16_left, reduction_16_right
        )
        reduction_8 = tl.reshape(
            reduction_16, (BLOCK_M, 2, 8)
        ).permute(0, 2, 1)
        reduction_8_left, reduction_8_right = tl.split(reduction_8)
        reduction_8 = libdevice.add_rn(reduction_8_left, reduction_8_right)
        reduction_4 = tl.reshape(
            reduction_8, (BLOCK_M, 2, 4)
        ).permute(0, 2, 1)
        reduction_4_left, reduction_4_right = tl.split(reduction_4)
        reduction_4 = libdevice.add_rn(reduction_4_left, reduction_4_right)
        reduction_2 = tl.reshape(
            reduction_4, (BLOCK_M, 2, 2)
        ).permute(0, 2, 1)
        reduction_2_left, reduction_2_right = tl.split(reduction_2)
        reduction_2 = libdevice.add_rn(reduction_2_left, reduction_2_right)
        reduction_1 = tl.reshape(
            reduction_2, (BLOCK_M, 2, 1)
        ).permute(0, 2, 1)
        reduction_1_left, reduction_1_right = tl.split(reduction_1)
        denominator = tl.reshape(
            libdevice.add_rn(reduction_1_left, reduction_1_right),
            (BLOCK_M,),
        )
    else:
        denominator = tl.sum(numerator, axis=1)

    denominator = tl.where(denominator > 0.0, denominator, 1.0)
    probabilities = tl.where(
        included,
        libdevice.div_rn(numerator, denominator[:, None]),
        0.0,
    )
    probability_offsets = (
        batch * stride_probabilities_batch
        + head * stride_probabilities_head
        + query_offsets[:, None] * stride_probabilities_query
        + key_offsets[None, :]
    )
    tl.store(
        probabilities_ptr + probability_offsets,
        probabilities,
        mask=query_in_bounds[:, None] & key_in_bounds[None, :],
    )


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
    # Validation here is deliberately limited to the common tensor contract.
    # Unsupported dtypes/head dimensions are dispatched to the PyTorch
    # reference below rather than rejected before the fallback can run.
    if not q.dtype.is_floating_point:
        raise TypeError(f"q/k/v must use a floating-point dtype, got {q.dtype}")

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


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    scale: float,
) -> torch.Tensor:
    """Value-equivalent fallback matching the organizer's operation order."""
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


def _launch_configuration(
    sequence_length: int, head_dim: int, causal: bool
) -> tuple[int, int, int, int]:
    """Conservative launch choices; benchmark-specific tuning comes after correctness."""
    if sequence_length <= 64:
        return 32, 32, 4, 2
    if head_dim <= 64:
        # The organizer default is S=128, D_head=64. 64x64 keeps both tl.dot
        # operands Tensor-Core friendly without the register pressure of 128x64.
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
    """Compute self-attention in one fused Triton kernel without SxS intermediates."""
    batch, num_heads, sequence_length, head_dim = _validate_inputs(
        q, k, v, valid_token_mask
    )
    if scale is None:
        scale = head_dim**-0.5
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    # The experimental bf16/float32 dot paths do not yet meet the organizer's
    # strict per-element gate across random partial tiles. Keep the optimized
    # kernel focused on its validated fp16 contract and preserve correctness
    # for the other advertised dtypes.
    if (
        q.device.type != "cuda"
        or q.dtype != torch.float16
        or head_dim not in _SUPPORTED_HEAD_DIMS
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
        or (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    ):
        return _reference_attention(q, k, v, valid_token_mask, causal, scale)

    output = torch.empty_like(q)
    block_m, block_n, num_warps, num_stages = _launch_configuration(
        sequence_length, head_dim, causal
    )

    # The no-mask specialization never dereferences this pointer. Reuse q rather
    # than allocate any placeholder tensor in the timed path.
    mask_pointer = valid_token_mask if valid_token_mask is not None else q
    mask_stride_batch = valid_token_mask.stride(0) if valid_token_mask is not None else 0
    mask_stride_sequence = (
        valid_token_mask.stride(1) if valid_token_mask is not None else 0
    )

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
        mask_stride_batch,
        mask_stride_sequence,
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


def triton_fused_full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
    output_bshd: bool = False,
) -> torch.Tensor:
    """Run the true full-row Blackwell-fused kernel for supported shapes."""
    batch, num_heads, sequence_length, head_dim = _validate_inputs(
        q, k, v, valid_token_mask
    )
    if scale is None:
        scale = head_dim**-0.5
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    block_n = triton.next_power_of_2(sequence_length)
    if (
        q.device.type != "cuda"
        or q.dtype != torch.float16
        or sequence_length > 128
        or sequence_length not in (32, 64, 128)
        or head_dim not in _SUPPORTED_HEAD_DIMS
        or block_n not in (32, 64, 128)
        or torch.cuda.get_device_capability(q.device)[0] < 12
        or (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    ):
        output = _reference_attention(q, k, v, valid_token_mask, causal, scale)
        if output_bshd:
            return output.transpose(1, 2).contiguous()
        return output

    if sequence_length <= 32:
        block_m = 32
        num_warps = 4
    elif sequence_length <= 64:
        block_m = 64
        num_warps = 4
    else:
        block_m = 64
        # Two 64-row programs per head expose more independent work than a
        # single 128-row program and reduce register pressure on Blackwell.
        num_warps = 4
    return triton_gluon_full_attention(
        q,
        k,
        v,
        mask=valid_token_mask,
        causal=causal,
        scale=float(scale),
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        output_bshd=output_bshd,
    )


def triton_fused_qk_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Historical QK/softmax experiment with a materialized probability tile.

    This API is retained for isolated comparisons. It is not used by
    :class:`TritonFusedSelfAttention`; the production path uses the Gluon
    full-fusion kernel and never allocates a global ``[B,H,S,S]`` tile.
    """
    batch, num_heads, sequence_length, head_dim = _validate_inputs(
        q, k, v, valid_token_mask
    )
    if scale is None:
        scale = head_dim**-0.5
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    if (
        q.device.type != "cuda"
        or q.dtype not in _SUPPORTED_DTYPES
        or head_dim not in _SUPPORTED_HEAD_DIMS
        or sequence_length > 128
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
        or (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
    ):
        return _reference_attention(q, k, v, valid_token_mask, causal, scale)

    probabilities = torch.empty(
        (batch, num_heads, sequence_length, sequence_length),
        device=q.device,
        dtype=q.dtype,
    )
    # On Blackwell, 32 query rows avoid the register-pressure cliff of the
    # 64-row program while keeping enough programs resident to hide PV traffic.
    block_m = 32
    block_n = triton.next_power_of_2(sequence_length)
    num_warps = 4 if block_n >= 64 else 2
    mask_pointer = valid_token_mask if valid_token_mask is not None else q
    mask_stride_batch = valid_token_mask.stride(0) if valid_token_mask is not None else 0
    mask_stride_sequence = (
        valid_token_mask.stride(1) if valid_token_mask is not None else 0
    )
    grid = (triton.cdiv(sequence_length, block_m), batch * num_heads)
    _fused_qk_softmax_forward_kernel[grid](
        q,
        k,
        mask_pointer,
        probabilities,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        probabilities.stride(0),
        probabilities.stride(1),
        probabilities.stride(2),
        mask_stride_batch,
        mask_stride_sequence,
        num_heads=num_heads,
        sequence_length=sequence_length,
        scale=float(scale),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CAUSAL=causal,
        HAS_VALID_TOKEN_MASK=valid_token_mask is not None,
        num_warps=num_warps,
        num_stages=3,
    )
    return torch.matmul(probabilities, v)


class TritonFusedSelfAttention(nn.Module):
    """Baseline-compatible adapter using Blackwell full-row attention."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        # Keep the baseline's exact learned parameter names for strict=True
        # state-dict copying.
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = x.shape
        return (
            x.view(batch, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
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

        needs_autograd = torch.is_grad_enabled() and (
            x.requires_grad
            or any(parameter.requires_grad for parameter in self.parameters())
        )
        can_use_fused_full_attention = (
            x.device.type == "cuda"
            and x.dtype == torch.float16
            and sequence_length <= 128
            and sequence_length in (32, 64, 128)
            and self.head_dim in _SUPPORTED_HEAD_DIMS
            and not needs_autograd
        )
        if can_use_fused_full_attention:
            context = triton_fused_full_attention(
                q,
                k,
                v,
                valid_token_mask=valid_token_mask,
                causal=causal,
                scale=self.scale,
                output_bshd=True,
            )
            context = context.view(batch, sequence_length, self.d_model)
        else:
            context = _reference_attention(
                q,
                k,
                v,
                valid_token_mask,
                causal,
                self.scale,
            )
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(batch, sequence_length, self.d_model)
            )
        output = self.out_proj(context)

        # Match BaselineSelfAttention: padding masks keys inside attention, then
        # invalid query outputs are zeroed after the output projection.
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output
