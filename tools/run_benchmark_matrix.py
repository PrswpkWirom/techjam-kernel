#!/usr/bin/env python3
"""Run the official Transformer benchmark over the published shape matrix.

The official benchmark is intentionally executed as a subprocess. This keeps
its protected evaluator, timing placement, and correctness gate authoritative;
this script adds orchestration, persistence, and reporting only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .benchmark_log_parser import evaluation_status, parse_benchmark_log
    from .benchmark_shape_matrix import (
        ANNOUNCED_CASES,
        ShapeCase,
        format_bytes,
        parse_case_ids,
        select_cases,
    )
else:
    from benchmark_log_parser import evaluation_status, parse_benchmark_log
    from benchmark_shape_matrix import (
        ANNOUNCED_CASES,
        ShapeCase,
        format_bytes,
        parse_case_ids,
        select_cases,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "torch_transformer_benchmark.py"
INTEGRITY_CHECKER = REPO_ROOT / "tools" / "check_benchmark_integrity.py"
DEFAULT_DENSE_ATTENTION_LIMIT_BYTES = 64 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def build_shared_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Build evaluator flags that are shared by every shape case."""

    values: list[str] = [
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--padding-ratio",
        str(args.padding_ratio),
        "--input-scale",
        str(args.input_scale),
        "--accuracy-trials",
        str(args.accuracy_trials),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--benchmark-rounds",
        str(args.benchmark_rounds),
        "--compile-mode",
        args.compile_mode,
        "--matmul-precision",
        args.matmul_precision,
    ]
    if args.benchmark_on_failure:
        values.append("--benchmark-on-failure")
    if args.compile_baseline:
        values.append("--compile-baseline")
    if args.compile_user:
        values.append("--compile-user")
    if args.non_strict_weight_copy:
        values.append("--non-strict-weight-copy")
    values.append("--allow-tf32" if args.allow_tf32 else "--no-allow-tf32")
    return tuple(values)


def build_command(
    case: ShapeCase,
    benchmark_path: Path = BENCHMARK_PATH,
    shared_args: Sequence[str] = (),
    python_executable: str = sys.executable,
) -> list[str]:
    """Build an exact subprocess command for one published case."""

    return [python_executable, str(benchmark_path), *case.benchmark_args(), *shared_args]


def dtype_bytes(dtype: str) -> int:
    if dtype == "float32":
        return 4
    if dtype in {"float16", "bfloat16"}:
        return 2
    raise ValueError(f"unsupported dtype: {dtype}")


def preflight_reason(
    case: ShapeCase,
    dtype: str,
    limit_bytes: int,
    force_unsafe_shapes: bool,
) -> str | None:
    """Return a safety warning for clearly impossible dense-score cases."""

    required = case.dense_attention_bytes(dtype_bytes(dtype))
    if force_unsafe_shapes or required <= limit_bytes:
        return None
    return (
        "official baseline allocates a dense [B,H,S,S] attention tensor; "
        f"one tensor is estimated at {format_bytes(required)}, above the "
        f"configured preflight limit of {format_bytes(limit_bytes)}. "
        "Use --force-unsafe-shapes to attempt it explicitly."
    )


def run_integrity_check(output_path: Path) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        [sys.executable, str(INTEGRITY_CHECKER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return {
        "return_code": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": time.time() - started,
        "log": str(output_path),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must match before resuming saved case results."""

    return {
        "benchmark_sha256": manifest.get("benchmark_sha256"),
        "shared_evaluator_args": manifest.get("shared_evaluator_args"),
        "cases": manifest.get("cases"),
        "preflight": manifest.get("preflight"),
        "runner": {
            "timeout_seconds": (manifest.get("runner") or {}).get("timeout_seconds"),
            "preflight_only": (manifest.get("runner") or {}).get("preflight_only"),
        },
        "python": manifest.get("python"),
        "git_commit": manifest.get("git_commit"),
    }


def matrix_exit_code(results: Sequence[dict[str, Any]], integrity_after: dict[str, Any]) -> int:
    """Return a conservative process code for a completed matrix run."""

    if not integrity_after.get("passed", False):
        return 3
    return 0 if all(result["status"] == "PASS" for result in results) else 2


def text_output(value: str | bytes | None) -> str:
    """Normalize subprocess output across Python's timeout implementations."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_case(
    case: ShapeCase,
    case_dir: Path,
    command: Sequence[str],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "command.txt").write_text(
        subprocess.list2cmdline(list(command)) + "\n", encoding="utf-8"
    )
    started = time.time()
    timed_out = False
    try:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            process.kill()
            output, _ = process.communicate()
            output = text_output(exc.output) + text_output(output)
        return_code = process.returncode
    except OSError as exc:
        output = f"failed to start benchmark: {exc}\n"
        return_code = 127

    (case_dir / "raw.log").write_text(output, encoding="utf-8")
    parsed = parse_benchmark_log(output)
    status = "TIMEOUT" if timed_out else evaluation_status(return_code, parsed)
    result: dict[str, Any] = {
        "case": case.as_dict(),
        "command": list(command),
        "status": status,
        "return_code": return_code,
        "duration_seconds": time.time() - started,
        "parsed": parsed.as_dict(),
    }
    _write_json(case_dir / "result.json", result)
    return result


def preflight_result(case: ShapeCase, case_dir: Path, reason: str | None) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    status = "PREFLIGHT_BLOCKED" if reason else "PREFLIGHT_PASS"
    message = reason or "preflight passed; execution not requested"
    result = {
        "case": case.as_dict(),
        "command": None,
        "status": status,
        "return_code": None,
        "duration_seconds": 0.0,
        "reason": message,
        "parsed": None,
    }
    (case_dir / "raw.log").write_text(f"{status}: {message}\n", encoding="utf-8")
    _write_json(case_dir / "result.json", result)
    return result


def write_summary(
    output_dir: Path,
    results: Sequence[dict[str, Any]],
    integrity_after: dict[str, Any],
) -> None:
    summary = {
        "total_cases": len(results),
        "integrity_after": integrity_after,
        "status_counts": {
            status: sum(result["status"] == status for result in results)
            for status in sorted({result["status"] for result in results})
        },
        "results": list(results),
    }
    _write_json(output_dir / "summary.json", summary)

    fields = [
        "case_id",
        "batch_size",
        "d_model",
        "heads",
        "seq_len",
        "layers",
        "ffn_dim",
        "causal",
        "status",
        "accuracy_status",
        "accuracy_failed",
        "accuracy_total",
        "baseline_median_ms",
        "optimized_median_ms",
        "speedup",
        "duration_seconds",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            case = result["case"]
            parsed = result.get("parsed") or {}
            baseline = parsed.get("baseline") or {}
            optimized = parsed.get("optimized") or {}
            writer.writerow({
                "case_id": case["case_id"],
                "batch_size": case["batch_size"],
                "d_model": case["d_model"],
                "heads": case["heads"],
                "seq_len": case["seq_len"],
                "layers": case["layers"],
                "ffn_dim": case["ffn_dim"],
                "causal": case["causal"],
                "status": result["status"],
                "accuracy_status": parsed.get("accuracy_status"),
                "accuracy_failed": parsed.get("accuracy_failed"),
                "accuracy_total": parsed.get("accuracy_total"),
                "baseline_median_ms": baseline.get("median_ms"),
                "optimized_median_ms": optimized.get("median_ms"),
                "speedup": parsed.get("speedup") if result["status"] == "PASS" else None,
                "duration_seconds": result["duration_seconds"],
            })


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case", help="comma-separated one-based case IDs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--force-unsafe-shapes",
        action="store_true",
        help="attempt preflight-blocked dense-attention cases (may OOM the host)",
    )
    parser.add_argument("--dense-attention-limit-gib", type=float, default=64.0)
    parser.add_argument("--timeout", type=float, default=0.0, help="per-case timeout in seconds; 0 disables it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    case_ids = parse_case_ids(args.case) if args.case else None
    cases = select_cases(case_ids)
    if args.dense_attention_limit_gib <= 0:
        raise SystemExit("--dense-attention-limit-gib must be positive")
    if args.timeout < 0:
        raise SystemExit("--timeout must be non-negative")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "results" / f"run-{stamp}"
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume and not args.dry_run:
        raise SystemExit(f"output directory is not empty; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(exist_ok=True)

    shared_args = build_shared_args(args)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(BENCHMARK_PATH),
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "python": sys.executable,
        "shared_evaluator_args": list(shared_args),
        "cases": [case.as_dict() for case in cases],
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "preflight": {
            "dense_attention_limit_bytes": int(args.dense_attention_limit_gib * 1024**3),
            "force_unsafe_shapes": args.force_unsafe_shapes,
        },
        "runner": {
            "timeout_seconds": args.timeout,
            "resume": args.resume,
            "preflight_only": args.preflight_only,
        },
    }
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("--resume requires an existing manifest.json in --output-dir")
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read existing resume manifest: {exc}") from exc
        if manifest_signature(existing_manifest) != manifest_signature(manifest):
            raise SystemExit(
                "--resume arguments do not match the existing output manifest; "
                "choose a new --output-dir or use identical evaluator arguments"
            )
    _write_json(output_dir / "manifest.json", manifest)

    before = run_integrity_check(output_dir / "integrity-before.log")
    manifest["integrity_before"] = before
    _write_json(output_dir / "manifest.json", manifest)
    if not before["passed"]:
        print("integrity check failed before matrix execution", file=sys.stderr)
        return 3

    commands = {case.case_id: build_command(case, shared_args=shared_args) for case in cases}
    if args.dry_run:
        for case in cases:
            print(f"case {case.case_id:02d}: {subprocess.list2cmdline(commands[case.case_id])}")
        after = run_integrity_check(output_dir / "integrity-after.log")
        manifest["integrity_after"] = after
        _write_json(output_dir / "manifest.json", manifest)
        return 0 if after["passed"] else 3

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            case_dir = cases_dir / f"case-{case.case_id:02d}"
            saved_result = case_dir / "result.json"
            if args.resume and saved_result.exists():
                results.append(json.loads(saved_result.read_text(encoding="utf-8")))
                print(f"case {case.case_id:02d}: RESUMED ({results[-1]['status']})")
                continue

            reason = preflight_reason(
                case,
                args.dtype,
                int(args.dense_attention_limit_gib * 1024**3),
                args.force_unsafe_shapes,
            )
            if reason and not args.preflight_only:
                result = preflight_result(case, case_dir, reason)
            elif args.preflight_only:
                result = preflight_result(case, case_dir, reason)
            else:
                print(f"case {case.case_id:02d}: running")
                result = run_case(case, case_dir, commands[case.case_id], args.timeout or None)
            results.append(result)
            print(f"case {case.case_id:02d}: {result['status']}")
    finally:
        after = run_integrity_check(output_dir / "integrity-after.log")
        manifest["integrity_after"] = after
        _write_json(output_dir / "manifest.json", manifest)

    write_summary(output_dir, results, after)
    return matrix_exit_code(results, after)


if __name__ == "__main__":
    raise SystemExit(main())
