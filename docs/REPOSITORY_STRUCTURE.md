# Repository structure

```text
techjam-kernel/
├── torch_transformer_benchmark.py  # Organizer evaluator; remains at root
├── AGENTS.md                       # Repository-wide engineering rules
├── model/                          # Optimized model adapters and GPU kernels
│   ├── attention_dispatch.py       # Pure central attention/stack planner
│   ├── triton_fused_attention.py   # Fused/blocked attention executors
│   ├── triton_gluon_attention.py   # Gluon full-row executor
│   └── triton_softmax.py            # Softmax attention adapter
├── test/                           # CPU/GPU regression tests
│   └── test_attention_dispatch.py   # Torch-free planner/CLI parity tests
├── tools/                          # Integrity, matrix-runner, and plot tooling
│   └── inspect_attention_dispatch.py # Non-timed plan introspection CLI
├── results/                        # Generated logs, summaries, and plots
└── docs/                           # Technical reports and project notes
```

Run the unchanged evaluator directly from the repository root:

```bash
.venv/bin/python torch_transformer_benchmark.py --device cuda --dtype float16
```

Run the published shape matrix and keep its artifacts under `results/`:

```bash
.venv/bin/python tools/run_benchmark_matrix.py \
  --device cuda --dtype float16 \
  --output-dir results/official-fp16
```

Generate plots from a completed run:

```bash
.venv/bin/python tools/visualize_benchmark_matrix.py \
  results/official-fp16/summary.json \
  --output-dir results/official-fp16/plots
```

Run all tests and the mandatory integrity check:

```bash
.venv/bin/python -m unittest discover -v
python3 tools/check_benchmark_integrity.py
```
