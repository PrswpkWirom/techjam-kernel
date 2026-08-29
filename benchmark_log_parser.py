"""Parse the human-readable output of the official benchmark.

The parser is deliberately separate from evaluation: the subprocess exit code
and raw log remain the source of truth, while this module only structures the
already printed summaries for reports and plots.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_CONFIG_RE = re.compile(
    r"TransformerConfig\(batch_size=(?P<batch>\d+), seq_len=(?P<seq>\d+), "
    r"d_model=(?P<d_model>\d+), num_heads=(?P<heads>\d+), "
    r"ffn_dim=(?P<ffn>\d+), num_layers=(?P<layers>\d+), causal=(?P<causal>True|False)\)"
)
_ENV_RE = re.compile(r"device=(?P<device>[^,]+), dtype=(?P<dtype>[^,]+), torch=(?P<torch>\S+)")
_GPU_RE = re.compile(r"gpu=(?P<gpu>.+)")
_ACCURACY_RE = re.compile(
    r"summary: (?P<status>PASS|FAIL) \| max_abs=(?P<max_abs>\S+) \| "
    r"max_rel=(?P<max_rel>\S+) \| failed=(?P<failed>\d+)/(?P<total>\d+)"
)
_TIMING_RE = re.compile(
    r"(?P<model>baseline|optimized)\s*: median=(?P<median>[\deE+\-.]+) ms \| "
    r"mean=(?P<mean>[\deE+\-.]+) ms \| p90=(?P<p90>[\deE+\-.]+) ms \| "
    r"min=(?P<min>[\deE+\-.]+) ms \| throughput=(?P<throughput>[\deE+\-.]+) token/s"
)
_SPEEDUP_RE = re.compile(r"speedup\s+: (?P<speedup>[\deE+\-.]+)x based on median latency")


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    mean_ms: float
    p90_ms: float
    min_ms: float
    throughput_tokens_per_second: float


@dataclass(frozen=True)
class ParsedBenchmarkLog:
    config: dict[str, Any] | None
    device: str | None
    dtype: str | None
    torch_version: str | None
    gpu: str | None
    accuracy_status: str | None
    accuracy_max_abs: float | None
    accuracy_max_rel: float | None
    accuracy_failed: int | None
    accuracy_total: int | None
    baseline: TimingSummary | None
    optimized: TimingSummary | None
    speedup: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: str) -> float:
    return float(value)


def parse_benchmark_log(text: str) -> ParsedBenchmarkLog:
    """Parse one official benchmark stdout/stderr capture."""

    config_match = _CONFIG_RE.search(text)
    config = None
    if config_match:
        groups = config_match.groupdict()
        config = {
            "batch_size": int(groups["batch"]),
            "seq_len": int(groups["seq"]),
            "d_model": int(groups["d_model"]),
            "heads": int(groups["heads"]),
            "ffn_dim": int(groups["ffn"]),
            "layers": int(groups["layers"]),
            "causal": groups["causal"] == "True",
        }

    env_match = _ENV_RE.search(text)
    gpu_match = _GPU_RE.search(text)
    accuracy_match = _ACCURACY_RE.search(text)
    timings: dict[str, TimingSummary] = {}
    for match in _TIMING_RE.finditer(text):
        groups = match.groupdict()
        timings[groups["model"]] = TimingSummary(
            median_ms=_float(groups["median"]),
            mean_ms=_float(groups["mean"]),
            p90_ms=_float(groups["p90"]),
            min_ms=_float(groups["min"]),
            throughput_tokens_per_second=_float(groups["throughput"]),
        )
    speedup_match = _SPEEDUP_RE.search(text)

    return ParsedBenchmarkLog(
        config=config,
        device=env_match.group("device") if env_match else None,
        dtype=env_match.group("dtype") if env_match else None,
        torch_version=env_match.group("torch") if env_match else None,
        gpu=gpu_match.group("gpu").strip() if gpu_match else None,
        accuracy_status=accuracy_match.group("status") if accuracy_match else None,
        accuracy_max_abs=(float(accuracy_match.group("max_abs")) if accuracy_match else None),
        accuracy_max_rel=(float(accuracy_match.group("max_rel")) if accuracy_match else None),
        accuracy_failed=(int(accuracy_match.group("failed")) if accuracy_match else None),
        accuracy_total=(int(accuracy_match.group("total")) if accuracy_match else None),
        baseline=timings.get("baseline"),
        optimized=timings.get("optimized"),
        speedup=(float(speedup_match.group("speedup")) if speedup_match else None),
    )


def evaluation_status(return_code: int, parsed: ParsedBenchmarkLog) -> str:
    """Map official process results to a conservative report status."""

    if parsed.accuracy_status == "FAIL":
        return "ACCURACY_FAIL"
    if return_code == 0 and parsed.accuracy_status == "PASS":
        return "PASS" if parsed.baseline and parsed.optimized else "REPORTING_ERROR"
    if return_code < 0 or return_code == 137:
        return "PROCESS_KILLED"
    if return_code != 0:
        return "ERROR"
    return "REPORTING_ERROR"
