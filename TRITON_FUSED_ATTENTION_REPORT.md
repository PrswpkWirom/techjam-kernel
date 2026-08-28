# Fused Triton attention experiment

Date: 2026-08-28

GPU: NVIDIA GeForce RTX 5070 Ti

Software: PyTorch 2.13.0+cu130, Triton 3.7.1

## Scope and tools

The task was to replace the materialized `QK^T -> softmax -> PV` attention core
with a custom FlashAttention-style Triton implementation while preserving the
benchmark interface, causal masking, `[B, S]` valid-token masks, and strict
state-dict copying. Codex used the local implementation, TDD, and review
workflows and consulted Triton's official fused-attention tutorial for current
forward-kernel patterns. No task work was delegated before the required final
review.

Reference: <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>

## Architecture

The old core launches a QK GEMM, writes `[B,H,S,S]` scores, launches a Triton
softmax that writes probabilities of the same size, then launches a PV GEMM.

The new opt-in core uses this flow:

```text
Q tile ─┐
K tiles ├─> tiled QK^T + mask + online FP32 (max, sum)
mask ───┘                       │
                               ├─> recompute normalized P tiles
V tiles ───────────────────────┘             │
                                             └─> FP32 P@V accumulator -> context
```

It never writes a complete score or probability matrix. A two-pass tiled form
was retained because it was measurably more accurate than one-pass rescaled PV:
pass 1 maintains online maximum and normalization statistics; pass 2 recomputes
score tiles, rounds fully normalized probabilities to fp16, and accumulates PV.

For the default `B=8, H=8, S=128, D=64` non-causal case, the launch grid is
`(ceil_div(S, 64), B*H) = (2, 64)`. One program owns 64 query rows for one
batch/head pair. The chosen configuration is `BLOCK_M=64`, `BLOCK_N=128`, four
warps, and three stages. It tied the fastest measured alternative while using
fewer warps than the `128x128` candidate. Causal `S=128` dispatch uses `64x64`;
its loop ends at the query tile boundary, so completely future K/V blocks are
not loaded.

Q/K/V load in fp16. QK uses Tensor Cores, then rounds at the baseline's fp16
score/scale boundary. Online maxima, sums, exponentials, division, and the PV
accumulator use fp32. Fully normalized P is cast to fp16 for Tensor-Core PV, and
the final context store casts to fp16. Bfloat16 and other unsupported inputs use
the exact PyTorch fallback because the experimental bf16 fast path missed the
official tolerance.

Causal masking compares key and query offsets in registers. The `[B,S]` mask
loads one boolean per key tile and broadcasts it on chip; padded query outputs
are zeroed after `out_proj`, matching the baseline module.

## Correctness

Direct fused-core tests pass the official per-element OR rule for non-causal,
causal, padded, unpadded, `S=1`, `S=33`, `S=97`, and `S=128` cases. The final
default attention-core comparison had max absolute error `0.000244141` and zero
failed elements. Its max relative error was `0.25`; those near-zero elements
still pass through the rule's absolute-error branch. Module weights load with
`strict=True`, and padded outputs are zero.

The custom fused class is intentionally not the default selector. Across the
six-layer default Transformer, its one-ULP attention differences compounded
into 1-2 failed near-zero output elements in representative three-trial runs.
PyTorch SDPA similarly failed 4-8 elements in four of five trials. Since
correctness is a hard gate, `UserOptimizedTransformer.attention_class` remains
`TritonSelfAttention`; changing it manually to `TritonFusedSelfAttention`
enables the experiment.

## Default attention-core benchmark

Fixed tensors, fp16, `B=8 H=8 S=128 D=64`; 100 ms warmup and 1000 ms sampling
through `triton.testing.do_bench`:

| Path | Median (ms) | Speedup vs original | Max abs error | Max relative error | Failed | Peak incremental allocation |
|---|---:|---:|---:|---:|---:|---:|
| QK + Triton softmax + PV | 0.026624 | 1.000x | 0 | 0 | 0 | 5,242,880 B |
| PyTorch SDPA | 0.024896 | 1.069x | 0.001953125 | not recorded | 0 | 1,050,624 B |
| Custom fused Triton | 0.014336 | 1.857x | 0.000244141 | 0.25 | 0 | 1,048,576 B |

These are attention-core results, not an official candidate speedup claim. The
correctness-gated default selector's full-model repeated run was:

```text
.venv/bin/python torch_transformer_benchmark.py --device cuda --dtype float16 \
  --warmup 20 --repeats 100 --benchmark-rounds 3
baseline median:  1.7903 ms
optimized median: 1.6192 ms
official speedup: 1.106x
correctness: PASS, max_abs=0, max_rel=0, failed=0/2,621,440
```

## Experiments rejected

- One-pass online PV: failed seven full-model elements in the first trial.
- Accurate exponential and fp16 score rounding alone: still failed 13 elements
  over three trials.
- Two-pass `64x64`: improved to two failures over three trials but did not clear
  the gate.
- Native-fp16 `tl.dot` output: regressed to 23 failures over five trials.
- Bfloat16 Triton fast path: failed 3/6,336 direct elements; now falls back.
- Configurations from `32x32` through `128x128` were measured. `64x128/4 warps`
  and `128x128/8 warps` tied at 0.014336 ms; the former was retained.

## Remaining bottlenecks and next work

The two-pass kernel doubles QK compute to preserve probability rounding. The
main unresolved issue is bit-level agreement with cuBLAS's PV reduction order
across six layers, not QK, softmax, direct attention tolerance, or memory
traffic. A diagnostic materialization found bit-identical Triton QK scores and
softmax probabilities; only the on-chip Triton PV dot produced the remaining
one-ULP differences. Useful next experiments are Blackwell warp-specialized
shared-memory layouts and a systematic comparison of PV accumulator ordering
against the exact cuBLAS algorithm. Any such variant must pass the full
Transformer gate before becoming the default selector.
