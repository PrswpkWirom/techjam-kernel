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

## Case 13 FP16 D_head=32 long-sequence path

Date: 2026-08-30

Environment: NVIDIA GeForce RTX 5070 Ti (sm120), PyTorch 2.13.0+cu130,
Triton 3.7.1, FP16, causal `B=64,S=1024,D=128,H=4,L=4,FFN=128`.

The case previously fell through `TritonFusedSelfAttention.forward()` because
the long FP16 predicate required `self.head_dim == 64`. The existing tiled
kernel's `HEAD_DIM`, strides, causal/tail masks, and direct BSHD output were
already generic for 32; changing only that predicate was not numerically safe.
Forced dispatch reached the one-pass Triton kernel but failed 15 elements over
five full-model trials (`max_abs=0.0078125`). All requested tile/warp choices
had the same residual amplification pattern.

The D_head=32 path now uses two bounded Triton passes for rows after the first
stack block. The statistics pass stores only per-row FP32 max/inverse-sum
values; the output pass recomputes scores, normalizes once, rounds
probabilities at the model-dtype boundary, and performs tiled P@V. This avoids
the one-pass recurrence's per-tile FP16 renormalization. The first D_head=32
stack block uses an exact bounded native tile loop (16 query rows at a time,
batched across all 64 samples) because every Triton-only configuration still
showed one or more amplified failures. It does not call `_reference_attention()`;
the remaining three blocks enter the Triton path.

The adapter records the stack layer index so this numerical safety mode is
explicit and deterministic. D_head=32 adapter dispatch is intentionally limited
to the validated `S=1024` case; other D_head=32 lengths retain their prior
fallback. Unsupported devices, dtypes, layouts, autograd, and existing
D_head=64/BF16/FP32 cases retain their prior dispatch behavior.

The required case-13 sweep used the real attention grid and 40 CUDA-event
samples after warm-up:

| BLOCK_M/N | warps | median ms | stats regs/spills/shared | output regs/spills/shared |
|---|---:|---:|---:|---:|
| 32/64 | 4 | 0.842640 | 80/0/10,752 B | 98/0/18,432 B |
| 32/64 | 8 | 1.478528 | 54/0/10,752 B | 121/0/18,432 B |
| 64/64 | 4 | **0.590592** | 92/0/12,800 B | 121/0/20,480 B |
| 64/64 | 8 | 0.897968 | 79/0/13,312 B | 117/0/20,480 B |
| 64/128 | 4 | 0.801312 | 190/0/21,504 B | 180/0/36,992 B |
| 64/128 | 8 | 1.096000 | 144/0/21,504 B | 107/0/36,864 B |
| 128/64 | 4 | 0.658288 | 158/0/17,408 B | 168/2/24,576 B |
| 128/64 | 8 | 0.654048 | 95/0/17,408 B | 117/0/24,576 B |

Nearby `32/32` and `128/128` trials were slower; `128/128` reached 255
registers and spilled in several output variants. The selected launch is
`BLOCK_M=64`, `BLOCK_N=64`, four warps, three stages.

Correctness used the official rule (`abs <= 0.002 OR relative <= 0.02`). The
unpadded and `padding_ratio=0.25` five-trial case-13 runs both had zero failed
elements. The final official command reported:

```text
baseline median  = 69.7260 ms
optimized median = 14.2180 ms
median speedup   = 4.904x
```

The pre-change paired run was 81.5325 ms baseline versus 81.7267 ms optimized
(0.998x), so the candidate improved from the prior fallback by approximately
5.75x on this GPU, while the paired post-change comparison above is the valid
headline speedup. The full 33-test suite and benchmark-integrity check passed.

A CUDA profiler run after the change identified repeated native masked/
pointwise kernels as the largest remaining category (~2.571 ms), followed by
projection/FFN GEMMs and LayerNorm. These are the next optimization targets;
attention itself is no longer the dominant case-13 cost.

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

## Exact long-sequence path

Date: 2026-08-30

The existing tiled online-softmax Triton kernel is now the production attention
path for CUDA FP16 causal inputs with `D_head=64` and `S>=257`. It keeps Q/K/V
in transposed views, maintains FP32 online-softmax state, scans only through
the causal query tile, and can write the final context directly as `[B,S,H,D]`.
Its normalized running context is algebraically equivalent to the textbook
`acc/l` recurrence while preserving the baseline-compatible FP16 probability
rounding that passed the multi-layer tolerance checks.
The established Gluon path for `S=32/64/128` remains unchanged; unsupported
long dtypes, head dimensions, non-causal inputs, and autograd retain the
reference fallback.

The editable `UserOptimizedTransformer` seam recognizes the announced
`B=32,S=100000,D=1024,H=16,FFN=1024,L=2,causal=True` FP16 inference case and
runs the complete inherited Transformer one sample at a time. A single final
`[B,S,D]` output is preallocated and filled with slice copies. No attention,
score, probability, or causal tensor proportional to `S^2` is created.

Validation commands:

```bash
python3 tools/check_benchmark_integrity.py
.venv/bin/python -m unittest -v test.test_triton_fused_attention
.venv/bin/python tools/smoke_long_sequence.py --seed 1234
```

On the NVIDIA GeForce RTX 5070 Ti (15.47 GiB, PyTorch 2.13.0+cu130), the
optimized-only 100k smoke passed with output shape `(32, 100000, 1024)`,
`torch.float16`, all finite values, runtime `25972.164 ms`, peak allocated
memory `14127.2 MiB`, and peak reserved memory `14142.0 MiB`. A final rerun
measured `26464.723 ms` with the same memory peaks and output properties.

The protected official evaluator remains preflight-blocked for case 14 because
its baseline necessarily allocates dense attention. Therefore this result is a
memory/finite-output smoke test, not an official baseline comparison or
speedup claim. The long path also passed the competition tolerance at
`S=257,1024,2048,4096` for the two-layer D=1024 model with zero failed
elements; maximum absolute errors were `0.00390625`, `0.00390625`,
`0.00488281`, and `0.00488281`, respectively.

### BF16 long-sequence extension

Date: 2026-08-30

The exact case-14 whole-model microbatch dispatcher and optimized-only smoke
tool now also accept BF16. Directly changing the FP16 Triton online-softmax
kernel to BF16 was rejected for the manageable probe lengths: isolated
attention was close, but two-layer D=1024 validation failed the official
per-element rule. The exact S=100000 BF16 case therefore uses a dedicated
one-pass Triton kernel with the same on-chip online-softmax structure as FP16,
`BLOCK_M=32`, `BLOCK_N=32`, four warps, and two stages. It writes `[B,S,H,D]`
directly and creates no `[B,H,S,S]`, `[S,S]`, or global score/probability
storage. The four manageable reference lengths use a bounded native-softmax
fallback so their correctness checks remain exact.

BF16 invocation:

```bash
.venv/bin/python tools/smoke_long_sequence.py --dtype bfloat16
```

Two-layer D=1024 validation at S=257/1024/2048/4096 passed the official OR
tolerance with zero failing elements through the native-softmax fallback. On
the NVIDIA GeForce RTX 5070 Ti (PyTorch 2.13.0+cu130), the full fused BF16
smoke passed with output shape `(32,100000,1024)`, dtype `torch.bfloat16`, and
all finite values. Runtime was `29510.145 ms`, peak allocated memory was
`14127.2 MiB`, and peak reserved memory was `14142.0 MiB`. A warmed rerun
measured `27384.668 ms` for BF16 versus `22953.848 ms` for FP16 (1.19x).
This is a finite-output smoke for the
100k fused kernel; a dense baseline comparison at S=100000 is infeasible.

## BF16 and FP32 Gluon full-fusion extension

Date: 2026-08-29

The Gluon MMA adapter is now dtype-specialized behind the same full-row
QK/softmax/PV kernel. `_mma` derives operand `k_width` from the input primitive
bit widths (`2` for FP16/BF16 and `1` for FP32) and forwards a compile-time MMA
precision. FP16 and BF16 round QK and scaled scores at the model-dtype
boundaries, keep softmax state in FP32, round probabilities immediately before
PV, and store the model dtype. FP32 keeps all state in FP32 and uses
`input_precision="tf32"` for both MMA operations.

Reference-only execution now makes Q/K/V contiguous, matching
`BaselineSelfAttention`; the fused path still consumes the original
transposed strides and writes `[B,S,H,D]` directly.

On the NVIDIA GeForce RTX 5070 Ti with PyTorch 2.13.0+cu130, the headline
`B=256,S=128,D=512,H=8,L=6` case passed all 83,886,080 elements for each dtype:

| dtype | baseline median | optimized median | speedup | max abs | failures |
|---|---:|---:|---:|---:|---:|
| FP16 | 39.1447 ms | 25.4351 ms | 1.539x | 0 | 0 |
| BF16 | 39.1312 ms | 25.5070 ms | 1.534x | 0 | 0 |
| FP32/TF32 | 67.8260 ms | 53.0044 ms | 1.280x | 0.00118494 | 0 |

FP16 and BF16 remained fused for the current exact tile envelope: sequence
lengths 32/64/128 and head dimensions 16/32/64/128. Partial/long sequences and
head dimensions 8/256 use the contiguous reference fallback. Plain TF32 was
correct for the common head-dimension-64 path. D_head=32 and D_head=128 each
occasionally exceeded the strict 0.002 absolute threshold by one element after
the residual stack, so those FP32 shapes are correctness-gated to fallback.

The requested FP32 alternatives were tested and rejected on this Triton build:
`tf32x3` and `ieee` report unsupported MMA versions on Blackwell `mma_v2`.
An explicit Gluon denominator-tree port also cannot compile with the FP32
distributed layout's SplitOp/reshape constraints. The existing Gluon reduction
is retained for passing modes.

Validation performed:

```bash
python3 tools/check_benchmark_integrity.py
.venv/bin/python -m unittest -v test.test_triton_fused_attention
.venv/bin/python -m py_compile model/triton_gluon_attention.py \
  model/triton_fused_attention.py test/test_triton_fused_attention.py \
  torch_transformer_benchmark.py
```

All announced cases 1–13 passed three-trial accuracy smoke runs for FP16, BF16,
and FP32. Core cases 1/9/12 passed 20-trial checks for FP16/BF16; FP32 case 10
(D_head=64) passed 20 trials with its performance run. Case 14 remains
preflight-blocked because the protected dense baseline would require tens of
terabytes for its `[B,H,S,S]` tensor.

## FP32 D=32 short-sequence hybrid coverage

Date: 2026-08-30

The first FP32 D=32 milestone enables the Gluon full-row kernel only for
`S=128, head_dim=32` (the existing FP32 D=64 shapes remain unchanged). The
four-layer model keeps layer 0 on the exact contiguous PyTorch-compatible
attention path and uses Gluon for layers 1–3. This is the smallest change that
preserves correctness while retaining most of the fusion; all tested Gluon
launch configurations produced the same numerical result, so changing
`BLOCK_M` or warp count did not repair the fully fused stack.

The mismatch was localized with a deterministic case-1 probe (`seed=1253`):

| stage/probe | result |
|---|---:|
| fully fused four-layer stack | 1 failed element, `max_abs=0.00210861` |
| exact one-product QK, scale + softmax | `max_abs=7.6e-6` |
| real D=32 QK probability probe | `max_abs=4.88e-4` |
| uniform-probability P@V probe | `max_abs=8.51e-5` |
| varied-probability P@V contribution | up to `9.77e-4` |

The dominant error begins in the TF32 QK MMA accumulation/reduction order;
P@V MMA reduction adds a smaller independent error. Neither error usually
breaks a single attention call, but residual accumulation across four blocks
can cross the official gate in the final block. The scale and FP32 softmax
are not the primary source.

Validation on the NVIDIA GeForce RTX 5070 Ti (`torch 2.13.0+cu130`, CUDA
13.0, Triton 3.7.1) used the protected evaluator with `atol=0.002`,
`rtol=0.02`, TF32 enabled, `matmul-precision=high`, 100 accuracy trials,
20 warmups, 100 repeats, and 10 alternating benchmark rounds. The hybrid
path selected one exact call and three Gluon calls per forward, passed all
`104,857,600` checked elements with zero failures, and produced:

| run | baseline median | optimized median | speedup |
|---|---:|---:|---:|
| pre-change exact fallback | 1.4339 ms | 1.4371 ms | 0.998x |
| first | 1.4377 ms | 0.9614 ms | 1.495x |
| repeat | 1.4302 ms | 0.9516 ms | 1.503x |

The exact command was:

```bash
.venv/bin/python torch_transformer_benchmark.py \
  --batch-size 64 --seq-len 128 --d-model 128 --heads 4 \
  --ffn-dim 128 --layers 4 --causal --device cuda --dtype float32 \
  --padding-ratio 0.0 --input-scale 1.0 --accuracy-trials 100 \
  --rtol 0.02 --atol 0.002 --seed 1234 --warmup 20 --repeats 100 \
  --benchmark-rounds 10 --matmul-precision high --allow-tf32
```

### Case 6 policy selection

The published `B=10000,S=128,D_model=128,H=4,L=4` case exposed two failures
with the original `EFFF` placement over 20 trials. Before changing the kernel,
all requested one-exact placements and all six two-exact placements were
tested with the protected evaluator. The one-exact results were:

| policy | failed elements over 20 trials |
|---|---:|
| EFFF | 2 |
| FEFF | 10 |
| FFEF | 12 |
| FFFE | 17 |

`EEFF`, `EFEF`, and `EFFE` were the only two-exact policies with zero failures;
`FEEF`, `FEFE`, and `FFEE` failed 3, 4, and 5 elements respectively. The
policies use the same two exact and two Gluon launches, so their short timing
sweep was effectively tied. `EEFF` was selected for its larger numerical
margin and marginally lowest median.

The model-level dispatcher now selects `EFFF` for cases 1–5 and `EEFF` for
case 6. Unknown short FP32 D=32 configurations remain exact. The attention
adapter's public interface, parameter layout, causal/mask semantics, and
fused kernel are unchanged; the validated launch remains `BLOCK_M=32`,
`BLOCK_N=128`, four warps.

On the NVIDIA GeForce RTX 5070 Ti (`torch 2.13.0+cu130`, CUDA 13.0, Triton
3.7.1), case 6 passed 100 accuracy trials with zero failures across
`16,384,000,000` elements (`max_abs=0.00198889`). The protected timing used
`atol=0.002`, `rtol=0.02`, TF32 enabled, `matmul-precision=high`, no padding,
20 warmups, 100 repeats, and 10 alternating rounds:

| run | median latency | p90 latency | speedup |
|---|---:|---:|---:|
| baseline | 426.6848 ms | 427.4103 ms | — |
| optimized EEFF | 307.0767 ms | 307.7347 ms | 1.390x |

The exact benchmark command was:

```bash
.venv/bin/python torch_transformer_benchmark.py \
  --batch-size 10000 --seq-len 128 --d-model 128 --heads 4 \
  --ffn-dim 128 --layers 4 --causal --device cuda --dtype float32 \
  --padding-ratio 0.0 --input-scale 1.0 --accuracy-trials 20 \
  --rtol 0.02 --atol 0.002 --seed 1234 --warmup 20 --repeats 100 \
  --benchmark-rounds 10 --matmul-precision high --allow-tf32
```

A five-trial `padding_ratio=0.25` run also passed with zero failures. A
three-trial non-causal check selected the exact fallback and was bit-exact.
The numerical failure source remains TF32 QK accumulation/reduction order,
with a smaller independent P@V contribution; no launch tuning was needed.

Case 13 is the next milestone: extend the FP32 tiled attention core only
after this case-6 policy is retained by the full review and commit gates.

## FP32 completion: published cases 7–14

Date: 2026-08-30

Environment: NVIDIA GeForce RTX 5070 Ti (sm120, 16,603,101,504 bytes),
PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0.  Codex used the approved
implementation/TDD workflow to add dispatch tests before each kernel path,
then recorded rejected configurations rather than routing them silently.

### Dispatch and numerical policy

Unsupported FP32 attention now enters a baseline-operation-order branch before
adapter Q/K/V views are created.  This branch matches the organizer's
projection order, contiguous split-head layout, masking, FP32 softmax, P@V,
and output projection.  It fixes the previous nested-fallback regression and
remains the deliberate path for cases 8 and 9.

A representative otherwise-unsupported FP32 configuration
`B=64,S=128,D=48,H=4,FFN=48,L=4,causal` was also timed through the unchanged
official harness: 20 unpadded accuracy seeds were bit-exact and the median was
`1.0039 ms` baseline versus `1.0156 ms` optimized (`0.988x`). Its 20-seed
25%-padding check was likewise bit-exact. This is the intended approximately
1.0x no-regression result for shapes that have no retained custom kernel.

The selected FP32 specializations are deliberately separate:

- Case 12 extends the full-row Gluon whitelist to `(S=32,D_head=32)`. `FFFF`
  failed one of 20 trials; `EFFF` passed 100 unpadded and 20 padded trials.
  The selected launch is `BLOCK_M=32`, `BLOCK_N=32`, eight warps.
- Case 9 structurally compiled at `(128,128)`, but `FFFF` failed accuracy and
  the passing `FFEF` candidate measured 0.858x.  Its whitelist was removed
  and the final policy is exact `EEEE`.
- Cases 7 and 11 use a dedicated D_head=8 Gluon kernel. Q/K and V are padded
  to 16 lanes for TF32 MMA, while only eight output lanes are stored. It keeps
  FP32 online-softmax state and supports both 64- and 128-key tiles. Case 7
  selects `EFFF`, `64x64`, two warps; case 11 selects `FFFF`, `64x128`, four
  warps.
- Case 13 uses a dedicated FP32 tiled FlashAttention kernel: TF32 QK/PV MMA,
  FP32 scores, `m/l/acc` online-softmax state, direct BSHD output, causal and
  padding masks, and no global score/probability tensor. Its selected launch
  is `64x32`, four warps, three stages, and it passes as `FFFF`.
- Case 8's D=256 experiment accumulates QK over four D=64 TF32 chunks,
  normalizes once, then computes four D=64 PV chunks. It was correct in some
  short sweeps but failed 100-seed residual-stack certification even after a
  hybrid policy and offered no stable speed margin. The final policy is exact
  `EEEE`; the strict experimental kernel is not dispatched.
- Case 14 reuses the tiled core at D_head=64 with `64x32`, four warps, two
  stages. A bounded FP32 PyTorch oracle processes small query blocks and was
  checked against dense attention at manageable lengths and against beginning,
  middle, and mask-boundary rows at S=100000. The model-level B=1 probe
  passes two-layer dense-reference validation at S=1024 and full 100k finite/
  padding smoke tests.

The FP32 100k B=32 output contract cannot be executed on this 16 GB GPU:
input plus output alone require 24.41 GiB before model working state. The
smoke tool therefore preflights the full FP32 shape and requires a >=32 GiB
GPU. The protected baseline also cannot be timed at this shape because its
single FP32 score tensor requires 20.48 TB. No official case-14 speedup is
claimed.

### Final validation

The protected evaluator passed before and after every matrix run. Full tests:

```bash
.venv/bin/python -m unittest discover -v
.venv/bin/python -m py_compile torch_transformer_benchmark.py \
  model/triton_fused_attention.py model/triton_gluon_attention.py \
  tools/smoke_long_sequence.py test/test_triton_fused_attention.py
python3 tools/check_benchmark_integrity.py
```

The final first run covered cases 1–13; the independent second timing run
covered changed cases 7–13 plus the restored case-10 policy. Both used 20
accuracy trials, 20 warmups, 100 repeats, and 10 alternating rounds:

```bash
.venv/bin/python tools/run_benchmark_matrix.py \
  --case 1,2,3,4,5,6,7,8,9,10,11,12,13 --device cuda --dtype float32 \
  --accuracy-trials 20 --warmup 20 --repeats 100 --benchmark-rounds 10
.venv/bin/python tools/run_benchmark_matrix.py \
  --case 7,8,9,10,11,12,13 --device cuda --dtype float32 \
  --accuracy-trials 20 --warmup 20 --repeats 100 --benchmark-rounds 10
.venv/bin/python tools/smoke_long_sequence.py --dtype float32 \
  --batch-size 1 --padding-ratio 0.0 --warmup 1 --repeats 3
```

Cases 7–13 also passed 100 unpadded deterministic seeds and 20 seeds with
`padding_ratio=0.25`; rejected exact paths were included in the padded matrix.
The B=1 case-14 smoke had samples `1279.576, 1277.771, 1281.343 ms`, median
`1279.576 ms`, peak allocated `3595.9 MiB`, peak reserved `3622.0 MiB`, finite
FP32 output, and zero padded-query outputs. A separate 25%-padding 100k run
also passed.

| case | shape (B,S,D,H) | FP32 dispatch / policy | kernel family | correctness | baseline median | optimized median | speedup | limitation |
|---:|---|---|---|---|---:|---:|---:|---|
| 1 | 64,128,128,4 | EFFF | Gluon full-row D32 | PASS | 1.5267 ms | 1.0191 ms | 1.498x | — |
| 2 | 1,128,128,4 | EFFF | Gluon full-row D32 | PASS | 1.1080 ms | 1.0391 ms | 1.066x | — |
| 3 | 4,128,128,4 | EFFF | Gluon full-row D32 | PASS | 1.0175 ms | 0.9621 ms | 1.058x | — |
| 4 | 16,128,128,4 | EFFF | Gluon full-row D32 | PASS | 1.0790 ms | 1.0170 ms | 1.061x | — |
| 5 | 128,128,128,4 | EFFF | Gluon full-row D32 | PASS | 3.0199 ms | 2.0498 ms | 1.473x | — |
| 6 | 10000,128,128,4 | EEFF | Gluon full-row D32 | PASS | 408.9762 ms | 294.5715 ms | 1.388x | — |
| 7 | 64,128,32,4 | EFFF | padded-TF32 Gluon D8 | PASS | 0.9686 ms | 0.9577 ms | 1.011x | narrow margin, retained after two runs |
| 8 | 64,128,1024,4 | EEEE | early exact | PASS | 16.7618 ms | 16.7836 ms | 0.999x | D256 custom candidate rejected |
| 9 | 64,128,128,1 | EEEE | early exact | PASS | 0.8946 ms | 0.8946 ms | 1.000x | D128 Gluon candidate rejected |
| 10 | 64,128,128,2 | FFFF | Gluon full-row D64 | PASS | 1.1833 ms | 0.8221 ms | 1.439x | — |
| 11 | 64,128,128,16 | FFFF | padded-TF32 Gluon D8 | PASS | 6.5712 ms | 1.0407 ms | 6.314x | — |
| 12 | 64,32,128,4 | EFFF | Gluon full-row D32 | PASS | 1.0363 ms | 0.9749 ms | 1.063x | — |
| 13 | 64,1024,128,4 | FFFF | tiled FP32 TF32 MMA | PASS | 99.3595 ms | 8.3001 ms | 11.971x | — |
| 14 | 32,100000,1024,16 | FF (B=1 validated) | tiled FP32 TF32 MMA | bounded-oracle PASS | N/A | N/A (B=32) | N/A | requires >=32 GiB; dense baseline infeasible |
