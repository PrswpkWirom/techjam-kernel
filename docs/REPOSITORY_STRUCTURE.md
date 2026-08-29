# Repository structure

```text
techjam-kernel/
├── torch_transformer_benchmark.py  # Organizer evaluator; remains at root
├── AGENTS.md                       # Repository-wide engineering rules
├── model/                          # Optimized model adapters and GPU kernels
├── test/                           # CPU/GPU regression tests
├── tools/                          # Integrity, matrix-runner, and plot tooling
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
