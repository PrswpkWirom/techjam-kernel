from __future__ import annotations

import math
import unittest

import torch

from triton_fused_attention import TritonFusedSelfAttention, triton_fused_attention
from triton_softmax import TritonSelfAttention, triton_attention_softmax


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    causal: bool,
) -> torch.Tensor:
    seq_len = q.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if causal:
        causal_mask = torch.ones(
            (seq_len, seq_len), device=q.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    if valid_token_mask is not None:
        scores = scores.masked_fill(
            ~valid_token_mask[:, None, None, :], float("-inf")
        )
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probabilities, v)


def _triton_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    causal: bool,
) -> torch.Tensor:
    """Run the materialized QK -> Triton softmax -> PV comparison path."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    probabilities = triton_attention_softmax(
        scores,
        valid_token_mask=valid_token_mask,
        causal=causal,
        scale=q.shape[-1] ** -0.5,
    )
    return torch.matmul(probabilities, v)


def _assert_official_tolerance(
    candidate: torch.Tensor, reference: torch.Tensor
) -> None:
    candidate_float = candidate.float()
    reference_float = reference.float()
    absolute_error = (candidate_float - reference_float).abs()
    passed = (absolute_error <= 0.002) | (
        absolute_error <= 0.02 * reference_float.abs()
    )
    if not bool(passed.all()):
        failed = int((~passed).sum().item())
        raise AssertionError(
            f"{failed}/{reference.numel()} elements failed; "
            f"max_abs={absolute_error.max().item():.6g}"
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class TritonFusedAttentionTests(unittest.TestCase):
    def _assert_attention_implementations(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        causal: bool,
    ) -> None:
        expected = _reference_attention(q, k, v, valid_token_mask, causal)
        softmax_actual = _triton_softmax_attention(
            q, k, v, valid_token_mask, causal
        )
        fused_actual = triton_fused_attention(
            q,
            k,
            v,
            valid_token_mask=valid_token_mask,
            causal=causal,
            scale=1.0 / math.sqrt(q.shape[-1]),
        )

        _assert_official_tolerance(softmax_actual, expected)
        _assert_official_tolerance(fused_actual, expected)

    def test_noncausal_attention_without_padding(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(1234)
        q = torch.randn(
            (2, 4, 128, 64),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        self._assert_attention_implementations(
            q, k, v, valid_token_mask=None, causal=False
        )

    def test_causal_attention_with_padding_and_partial_tiles(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(5678)
        q = torch.randn(
            (3, 2, 97, 64),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        positions = torch.arange(97, device="cuda")
        valid_token_mask = positions[None, :] < torch.tensor(
            [97, 71, 19], device="cuda"
        )[:, None]

        self._assert_attention_implementations(
            q, k, v, valid_token_mask=valid_token_mask, causal=True
        )

    def test_single_token_causal_attention(self) -> None:
        q = torch.randn((1, 1, 1, 32), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        valid_token_mask = torch.ones((1, 1), device="cuda", dtype=torch.bool)

        self._assert_attention_implementations(
            q, k, v, valid_token_mask=valid_token_mask, causal=True
        )

    def test_causal_attention_without_padding_mask(self) -> None:
        q = torch.randn((2, 2, 65, 64), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        self._assert_attention_implementations(
            q, k, v, valid_token_mask=None, causal=True
        )

    def test_bfloat16_attention_with_partial_tiles(self) -> None:
        q = torch.randn((2, 3, 33, 32), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        valid_token_mask = torch.arange(33, device="cuda")[None, :] < torch.tensor(
            [33, 11], device="cuda"
        )[:, None]

        self._assert_attention_implementations(
            q, k, v, valid_token_mask=valid_token_mask, causal=False
        )

    def test_modules_load_baseline_weights_and_zero_padded_queries(self) -> None:
        from torch_transformer_benchmark import BaselineSelfAttention

        torch.manual_seed(9012)
        baseline = BaselineSelfAttention(d_model=128, num_heads=2)
        softmax_attention = TritonSelfAttention(d_model=128, num_heads=2)
        fused = TritonFusedSelfAttention(d_model=128, num_heads=2)
        state_dict = baseline.state_dict()
        softmax_attention.load_state_dict(state_dict, strict=True)
        fused.load_state_dict(state_dict, strict=True)
        baseline = baseline.cuda().half().eval()
        softmax_attention = softmax_attention.cuda().half().eval()
        fused = fused.cuda().half().eval()
        x = torch.randn((2, 33, 128), device="cuda", dtype=torch.float16)
        valid_token_mask = torch.arange(33, device="cuda")[None, :] < torch.tensor(
            [33, 17], device="cuda"
        )[:, None]

        with torch.inference_mode():
            expected = baseline(x, valid_token_mask, causal=False)
            softmax_actual = softmax_attention(
                x, valid_token_mask, causal=False
            )
            fused_actual = fused(x, valid_token_mask, causal=False)

        _assert_official_tolerance(softmax_actual, expected)
        _assert_official_tolerance(fused_actual, expected)
        self.assertTrue(bool((softmax_actual[1, 17:] == 0).all()))
        self.assertTrue(bool((fused_actual[1, 17:] == 0).all()))


if __name__ == "__main__":
    unittest.main()
