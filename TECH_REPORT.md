# TechJam Transformer Optimization Report

## Correctness-safe fused-attention adapter

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
python -m unittest -v test_triton_fused_attention.py
python -m py_compile triton_fused_attention.py triton_softmax.py \
  test_triton_fused_attention.py torch_transformer_benchmark.py
python tools/check_benchmark_integrity.py
```

Human verification: the user should rerun the validation command on the final
submitted commit and repeat it for every organizer-announced shape.
