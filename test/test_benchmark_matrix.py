from __future__ import annotations

import unittest
from pathlib import Path

from tools.benchmark_log_parser import evaluation_status, parse_benchmark_log
from tools.benchmark_shape_matrix import (
    ANNOUNCED_CASES,
    format_bytes,
    parse_case_ids,
    select_cases,
)
from tools.run_benchmark_matrix import (
    build_command,
    dtype_bytes,
    manifest_signature,
    matrix_exit_code,
    preflight_reason,
    text_output,
)
from tools.visualize_benchmark_matrix import valid_results


PASS_LOG = """
=== Configuration ===
TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)
device=cuda, dtype=torch.float16, torch=2.13.0+cu130
gpu=NVIDIA Test GPU
summary: PASS | max_abs=0 | max_rel=0 | failed=0/4194304
baseline : median=1.2500 ms | mean=1.3000 ms | p90=1.5000 ms | min=1.1000 ms | throughput=6553.60 token/s
optimized: median=0.6250 ms | mean=0.7000 ms | p90=0.8000 ms | min=0.6000 ms | throughput=13107.20 token/s
speedup  : 2.000x based on median latency
"""


class BenchmarkMatrixTests(unittest.TestCase):
    def test_published_matrix_has_exact_cases_and_order(self) -> None:
        self.assertEqual(len(ANNOUNCED_CASES), 14)
        self.assertEqual(ANNOUNCED_CASES[0].as_dict(), {
            "case_id": 1, "batch_size": 64, "d_model": 128, "heads": 4,
            "seq_len": 128, "layers": 4, "causal": True, "ffn_dim": 128,
        })
        self.assertEqual(ANNOUNCED_CASES[-1].seq_len, 100000)
        self.assertEqual(ANNOUNCED_CASES[-1].layers, 2)

    def test_case_selection_and_cli_shape_flags(self) -> None:
        selected = select_cases((8, 1))
        self.assertEqual([case.case_id for case in selected], [8, 1])
        self.assertIn("--causal", selected[1].benchmark_args())
        with self.assertRaises(ValueError):
            select_cases((99,))

    def test_case_id_parser_and_size_format(self) -> None:
        self.assertEqual(parse_case_ids("1, 4,14"), (1, 4, 14))
        with self.assertRaises(ValueError):
            parse_case_ids("1,nope")
        self.assertEqual(format_bytes(1024**4), "1.00 TiB")

    def test_parser_extracts_official_summaries(self) -> None:
        parsed = parse_benchmark_log(PASS_LOG)
        self.assertEqual(parsed.config["batch_size"], 64)
        self.assertEqual(parsed.accuracy_status, "PASS")
        self.assertEqual(parsed.baseline.median_ms, 1.25)
        self.assertEqual(parsed.optimized.median_ms, 0.625)
        self.assertEqual(parsed.speedup, 2.0)
        self.assertEqual(evaluation_status(0, parsed), "PASS")

    def test_parser_never_promotes_accuracy_failure_to_speedup(self) -> None:
        failed_log = PASS_LOG.replace("summary: PASS", "summary: FAIL").replace(
            "failed=0/", "failed=1/"
        )
        parsed = parse_benchmark_log(failed_log)
        self.assertEqual(evaluation_status(2, parsed), "ACCURACY_FAIL")
        empty = parse_benchmark_log("")
        self.assertEqual(evaluation_status(2, empty), "ERROR")

    def test_post_integrity_failure_overrides_case_results(self) -> None:
        passing = {"status": "PASS"}
        self.assertEqual(matrix_exit_code([passing], {"passed": True}), 0)
        self.assertEqual(matrix_exit_code([passing], {"passed": False}), 3)
        self.assertEqual(text_output(b"partial output"), "partial output")
        self.assertEqual(text_output(None), "")

    def test_runner_builds_shape_and_shared_flags_without_reimplementing_math(self) -> None:
        command = build_command(
            ANNOUNCED_CASES[0],
            benchmark_path=Path("torch_transformer_benchmark.py"),
            shared_args=("--device", "cuda", "--dtype", "float16"),
            python_executable="python3",
        )
        self.assertEqual(command[:3], ["python3", "torch_transformer_benchmark.py", "--batch-size"])
        self.assertIn("--causal", command)
        self.assertEqual(command[-4:], ["--device", "cuda", "--dtype", "float16"])

    def test_preflight_detects_extreme_dense_attention_but_can_be_forced(self) -> None:
        extreme = ANNOUNCED_CASES[-1]
        reason = preflight_reason(extreme, "float16", 64 * 1024**3, False)
        self.assertIsNotNone(reason)
        self.assertIsNone(preflight_reason(extreme, "float16", 64 * 1024**3, True))
        self.assertEqual(dtype_bytes("bfloat16"), 2)

    def test_visualizer_filters_nonpassing_results(self) -> None:
        parsed = parse_benchmark_log(PASS_LOG).as_dict()
        passing = {"status": "PASS", "parsed": parsed, "case": {"case_id": 1}}
        failed = {"status": "ACCURACY_FAIL", "parsed": parsed, "case": {"case_id": 2}}
        valid_summary = {"results": [passing, failed], "integrity_after": {"passed": True}}
        self.assertEqual(valid_results(valid_summary), [passing])
        cpu = {"status": "PASS", "parsed": {**parsed, "device": "cpu"}, "case": {"case_id": 3}}
        self.assertEqual(valid_results({**valid_summary, "results": [cpu]}), [])
        self.assertEqual(valid_results({"results": [passing], "integrity_after": {"passed": False}}), [])

    def test_resume_signature_covers_evaluator_and_shape_inputs(self) -> None:
        base = {
            "benchmark_sha256": "abc",
            "shared_evaluator_args": ["--dtype", "float16"],
            "cases": [{"case_id": 1}],
            "preflight": {"dense_attention_limit_bytes": 1},
            "runner": {"timeout_seconds": 10},
            "python": "/venv/bin/python",
            "git_commit": "deadbeef",
        }
        self.assertNotEqual(
            manifest_signature(base),
            manifest_signature({**base, "runner": {"timeout_seconds": 20}}),
        )
        changed = {**base, "shared_evaluator_args": ["--dtype", "float32"]}
        self.assertNotEqual(manifest_signature(base), manifest_signature(changed))


if __name__ == "__main__":
    unittest.main()
