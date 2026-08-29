"""Compatibility exports for the renamed Triton softmax implementation."""

from .triton_softmax import TritonSelfAttention, triton_attention_softmax

__all__ = ["TritonSelfAttention", "triton_attention_softmax"]
