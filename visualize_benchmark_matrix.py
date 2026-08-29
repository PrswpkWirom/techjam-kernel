#!/usr/bin/env python3
"""Create conservative plots from a completed matrix run.

Only cases with status ``PASS`` are included in performance plots. Accuracy
failures and process/preflight failures remain visible in the summary JSON but
cannot be presented as valid speedups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise ValueError(f"invalid matrix summary: {path}")
    return value


def valid_results(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return CUDA results eligible for official performance claims."""

    if not (summary.get("integrity_after") or {}).get("passed", False):
        return []

    rows: list[dict[str, Any]] = []
    for result in summary["results"]:
        if result.get("status") != "PASS":
            continue
        parsed = result.get("parsed") or {}
        if not str(parsed.get("device", "")).startswith("cuda"):
            continue
        if parsed.get("speedup") is None:
            continue
        if not parsed.get("baseline") or not parsed.get("optimized"):
            continue
        rows.append(result)
    return rows


def _make_plots(summary: dict[str, Any], output_dir: Path) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("plotting requires matplotlib; install it or use summary.json/csv") from exc

    rows = valid_results(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        (output_dir / "no-valid-speedups.txt").write_text(
            "No PASS cases with complete timing summaries were available.\n",
            encoding="utf-8",
        )
        return 0

    labels = [f"#{row['case']['case_id']}" for row in rows]
    baseline = [row["parsed"]["baseline"]["median_ms"] for row in rows]
    optimized = [row["parsed"]["optimized"]["median_ms"] for row in rows]
    speedups = [row["parsed"]["speedup"] for row in rows]

    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))
    axis.bar(labels, speedups, color="#2f6f9f")
    axis.axhline(1.0, color="#555", linewidth=0.8)
    axis.set(
        title="Official benchmark speedup (PASS cases only)",
        ylabel="Baseline median / optimized median",
        xlabel="Published case",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "speedup-by-case.png", dpi=160)
    figure.savefig(output_dir / "speedup-by-case.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))
    positions = list(range(len(rows)))
    width = 0.38
    axis.bar([position - width / 2 for position in positions], baseline, width, label="Baseline", color="#999")
    axis.bar([position + width / 2 for position in positions], optimized, width, label="Optimized", color="#2f6f9f")
    axis.set_xticks(positions, labels)
    axis.set_yscale("log")
    axis.set(
        title="Official median latency (PASS cases only)",
        ylabel="Milliseconds (log scale)",
        xlabel="Published case",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "latency-comparison.png", dpi=160)
    figure.savefig(output_dir / "latency-comparison.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))
    baseline_throughput = [row["parsed"]["baseline"]["throughput_tokens_per_second"] for row in rows]
    optimized_throughput = [row["parsed"]["optimized"]["throughput_tokens_per_second"] for row in rows]
    axis.bar(
        [position - width / 2 for position in positions],
        baseline_throughput,
        width,
        label="Baseline",
        color="#999",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        optimized_throughput,
        width,
        label="Optimized",
        color="#2f6f9f",
    )
    axis.set_xticks(positions, labels)
    axis.set(title="Official throughput (PASS cases only)", ylabel="Tokens / second", xlabel="Published case")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "throughput-comparison.png", dpi=160)
    figure.savefig(output_dir / "throughput-comparison.svg")
    plt.close(figure)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="matrix summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return _make_plots(load_summary(args.summary), args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
