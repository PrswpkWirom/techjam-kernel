# TechJam Transformer Optimization Report

## Historical correctness-safe fused-attention adapter

Date: 2026-08-28

Environment:

- GPU: NVIDIA GeForce RTX 5070 Ti
- PyTorch: 2.13.0+cu130
- dtype and primary shape: FP16, `[B=256, S=128, D=512]`, 8 heads,
  6 layers, non-causal, no padding

AI-assisted workflow:

- Codex diagnosis isolated the first numerical divergence with uniform-score,
  one-hot-value, and layer-by-layer differential probes.
- The implementation used the `implement`, `tdd`, and `codebase-design`
  workflows. A failing six-layer regression was captured before the fix.
- Independent standards and specification review tasks checked the final diff.
- The user selected correctness over retaining an unscoreable fully fused path.

The experimental `triton_fused_attention()` kernel changes FP16 QK/softmax
reduction association. An isolated attention layer remained within tolerance,
but the differences accumulated through residual and FFN operations. With seed
1234, the original six-layer candidate failed 16 of 16,777,216 output elements
with maximum absolute error 0.0078125.

The compatibility class `TritonFusedSelfAttention` now uses native QK, the
custom exact Triton softmax, and native PV. The raw fully fused function remains
available for experiments but is not the official model path. Unsupported
devices, layouts, dtypes, and autograd execution use the value-equivalent
PyTorch fallback.

Validation command:

```bash
python torch_transformer_benchmark.py --batch-size 256 \
  --benchmark-rounds 16 --dtype float16
```

The integrity checker passed before and after the run. Accuracy was bit-exact
for all five trials: 0 failures in 83,886,080 elements. Median latency was
39.2537 ms for the baseline and 30.5978 ms for the corrected adapter, a valid
1.283x speedup.

An identical-input, alternating-order ablation used 20 warmups, 100 repeats,
and 3 rounds. The corrected adapter measured 30.4365 ms; raw fusion measured
28.0865 ms but failed 10 of 16,777,216 elements. The raw result is therefore
recorded only as a failed experiment and is not claimed as a valid speedup.

Automated verification:

```bash
python -m unittest -v test.test_triton_fused_attention
python -m py_compile model/triton_fused_attention.py model/triton_softmax.py \
  test/test_triton_fused_attention.py torch_transformer_benchmark.py
python tools/check_benchmark_integrity.py
```

Human verification: the user should rerun the validation command on the final
submitted commit and repeat it for every organizer-announced shape.

## Historical fused QK-softmax precision recovery

Date: 2026-08-29

The one-pass `64x64` online attention path reached a promising invalid speedup
of 1.405x, but failed 71 of 16,777,216 elements for the pinned six-layer seed.
Layer-by-layer differential testing showed zero official failures in the first
two blocks, followed by 1, 2, 35, and 126 failures after blocks three through
six. The first attention output had no official failures but was already
non-bit-exact, so residual and FFN operations amplified one-ULP differences.

The retained implementation fuses QK, model-dtype score scaling, masking, and
the complete-row fp32 softmax in one Triton kernel. It materializes only the
fp16 probability matrix and delegates PV to native `torch.matmul`, preserving
the organizer baseline's PV reduction order. A diagnostic proved Triton QK was
bit-exact; the remaining mismatch came from `tl.sum` reducing an MMA-produced
layout differently from PyTorch's persistent softmax. An explicit round-to-
nearest `32 -> 16 -> 8 -> 4 -> 2 -> 1` denominator tree made probabilities and
the complete six-layer output bit-exact. `BLOCK_M=32` and four warps was the
fastest low-register-pressure exact configuration in the local sweep.

Rejected experiments:

- changing the original online kernel from `BLOCK_N=64` to 128 reduced the
  final failures from 71 to 16 but did not pass;
- fused QK-softmax with native PV but generic `tl.sum` reduced failures to 10;
- storing and reloading scores inside the same kernel regressed to 14 failures;
- `libdevice.add_rn` for only the four per-lane numerator additions still left
  10 failures; the entire fixed reduction tree was required.

Validation commands:

```bash
python3 tools/check_benchmark_integrity.py
.venv/bin/python -m unittest -v test.test_triton_fused_attention
.venv/bin/python torch_transformer_benchmark.py --batch-size 256 \
  --benchmark-rounds 16 --dtype float16
```

Final RTX 5070 Ti result with PyTorch 2.13.0+cu130: all five accuracy trials
were bit-exact (`0/83,886,080` failures, `max_abs=0`). Baseline median latency
was 39.2704 ms and optimized median latency was 29.1389 ms, for a valid 1.348x
speedup. The complete nine-test attention suite passed, including causal and
padded partial tiles, bf16 correctness fallback, and CPU autograd fallback.

Additional five-trial FP16 six-layer checks were bit-exact for `S=33` with
padding, `S=64`, and causal padded `S=97`. The first `S=64` probe exposed one
failure while using generic `tl.sum`; extending the explicit reduction tree to
64- and 32-wide rows eliminated it. The one-repeat smoke timings for these edge
shapes are correctness diagnostics rather than final performance claims.

## Blackwell true full-fusion implementation

Date: 2026-08-29

The final adapter now uses `model/triton_gluon_attention.py` on FP16 Blackwell
(`compute capability 12.x`) for exact power-of-two sequence lengths 32, 64,
and 128. Each CTA owns a query tile and batch/head pair. Gluon `mma_v2`
performs both QK and P@V, while the score mask, `libdevice.exp`, and a custom
`libdevice.add_rn` reduction match the baseline's FP16 rounding on the tested
shapes. The kernel writes only the final context; it never allocates or writes
a global `[B,H,S,S]` probability tensor. Partial sequence lengths and
unsupported devices/dtypes use the value-equivalent reference fallback.

The planned `tcgen05`/TMEM route was also probed, but this Triton build fails
LLVM lowering of `tcgen05.wait` on the target environment. The implementation
therefore uses Gluon's documented `mma_v2` path as the safe Blackwell fallback;
the dispatch boundary keeps that hardware-specific choice isolated.

The adapter also keeps Q/K/V in their transposed views and asks the fused core
to write `[B,S,H,D]` directly. This removes the three head-repacking copies and
the post-attention transpose from the timed FP16 path.

Validation and benchmark command:

```bash
python3 tools/check_benchmark_integrity.py
.venv/bin/python torch_transformer_benchmark.py --batch-size 256 \
  --benchmark-rounds 16 --dtype float16 --benchmark-on-failure
```

On the NVIDIA GeForce RTX 5070 Ti with PyTorch 2.13.0+cu130, all five
accuracy trials were bit-exact (`0/83,886,080` failures, `max_abs=0`). The
latest run measured 39.7002 ms baseline versus 25.8242 ms optimized median
latency, a valid 1.537x speedup.

An identical second 16-round run also passed all five trials and measured
41.0487 ms baseline versus 27.0089 ms optimized (1.520x median speedup).

Shape/mask checks also passed with the official tolerance: FP16 `S=64` padded,
FP16 causal padded `S=97`, FP16 padded `S=33`, FP16 causal `S=32`, and BF16
fallback. The partial-length cases intentionally dispatch to the correctness
fallback; their smoke timings are not used for the headline speed claim.

## Published shape-matrix test tooling

Date: 2026-08-29

Added `tools/benchmark_shape_matrix.py`, `tools/run_benchmark_matrix.py`,
`tools/benchmark_log_parser.py`, and `tools/visualize_benchmark_matrix.py` to
execute the 14 Appendix 3.7 configurations through the unchanged official
evaluator. The runner stores each command, raw evaluator log, parsed result,
manifest, CSV, and JSON summary under `--output-dir`; plots include only cases
that pass the official accuracy gate. `--resume` checks the benchmark hash,
evaluator flags, shape list, and preflight settings before reusing a result.

The runner preflights the `B=32, S=100000, D=1024, H=16` case because the
baseline's dense `[B,H,S,S]` score tensor requires at least 10.24 TB in FP16;
`--force-unsafe-shapes` explicitly bypasses that safety check. A one-trial GPU
smoke run of case 1 recorded the existing optimized-model mismatch (1 failed
element, maximum absolute error `0.0078125`) and correctly skipped timing.
Exact smoke command: `.venv/bin/python tools/run_benchmark_matrix.py --case 1
--device cuda --dtype float16 --accuracy-trials 1 --warmup 1 --repeats 1
--benchmark-rounds 1 --output-dir /tmp/techjam-matrix-gpu-smoke`. Environment:
NVIDIA GeForce RTX 5070 Ti, PyTorch 2.13.0+cu130. Raw output is preserved at
`/tmp/techjam-matrix-gpu-smoke/cases/case-01/raw.log`.

## Repository organization

Date: 2026-08-29

The organizer-owned `torch_transformer_benchmark.py` remains at the repository
root. Kernel implementations now live in `model/`, tests in `test/`, benchmark
orchestration in `tools/`, generated outputs in `results/`, and documentation
in `docs/`. Only the editable `UserOptimizedTransformer` imports changed in the
official harness; the integrity checker continued to pass after the move.

Structural validation used the full 19-test suite, Python compilation, and a
real case-2 GPU smoke run through the relocated runner:

```bash
python3 tools/check_benchmark_integrity.py
.venv/bin/python -m unittest discover -v
.venv/bin/python tools/run_benchmark_matrix.py --case 2 --device cuda \
  --dtype float16 --accuracy-trials 1 --warmup 1 --repeats 1 \
  --benchmark-rounds 1 --output-dir /tmp/techjam-reorg-gpu-smoke
python3 tools/check_benchmark_integrity.py
```

The smoke run passed all 16,384 output elements on the NVIDIA GeForce RTX 5070
Ti with PyTorch 2.13.0+cu130. Its one-repeat timing is only a wiring diagnostic,
not a performance claim.
