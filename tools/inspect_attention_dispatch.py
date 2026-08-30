#!/usr/bin/env python3
"""Inspect central attention plans without constructing or timing a model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

# Make direct ``python tools/inspect_attention_dispatch.py`` invocation work
# from any current directory, matching the long-sequence smoke tool.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.attention_dispatch import (
    AttentionModelSpec,
    AttentionRuntime,
    AttentionStackPlan,
    select_attention_stack_plan,
)
from tools.benchmark_shape_matrix import ANNOUNCED_CASES, parse_case_ids, select_cases


def _capability(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "compute capability must look like MAJOR.MINOR, for example 12.0"
        ) from exc


def _cuda_capability() -> tuple[int, int] | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except (ImportError, RuntimeError):
        pass
    return None


def _plan_case(case, dtype: str, device_type: str, capability) -> AttentionStackPlan:
    spec = AttentionModelSpec(
        batch_size=case.batch_size,
        sequence_length=case.seq_len,
        d_model=case.d_model,
        num_heads=case.heads,
        ffn_dim=case.ffn_dim,
        num_layers=case.layers,
        causal=case.causal,
    )
    runtime = AttentionRuntime(
        batch_size=case.batch_size,
        sequence_length=case.seq_len,
        d_model=case.d_model,
        dtype=dtype,
        device_type=device_type,
        device_capability=capability,
        needs_autograd=False,
        causal=case.causal,
    )
    return select_attention_stack_plan(spec, runtime)


def _launch_dict(plan) -> dict[str, Any] | None:
    if plan.launch is None:
        return None
    return asdict(plan.launch)


def _json_record(case, dtype: str, plan: AttentionStackPlan) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "dtype": dtype,
        "execution": plan.execution,
        "layers": [
            {
                "layer_index": layer.layer_index,
                "family": layer.family.value,
                "mode": layer.mode.value,
                "launch": _launch_dict(layer),
                "reason": layer.reason.value if layer.reason else None,
                "explanation": layer.explanation,
            }
            for layer in plan.layers
        ],
    }


def _text_record(case, dtype: str, plan: AttentionStackPlan) -> str:
    lines = [f"case {case.case_id} / {dtype}", f"execution: {plan.execution}"]
    for layer in plan.layers:
        launch = _launch_dict(layer)
        launch_text = "" if launch is None else f" launch={launch}"
        reason = "" if layer.reason is None else f" reason={layer.reason.value}"
        lines.append(
            f"layer {layer.layer_index}: family={layer.family.value} "
            f"mode={layer.mode.value}{launch_text}{reason}"
        )
        if layer.explanation:
            lines.append(f"  explanation: {layer.explanation}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="comma-separated published case IDs; defaults to all")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        action="append",
        help="dtype to inspect; repeat for multiple dtypes (defaults to all)",
    )
    parser.add_argument("--device-type", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compute-capability", type=_capability)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    case_ids = parse_case_ids(args.case) if args.case else None
    cases = select_cases(case_ids)
    dtypes = tuple(args.dtype) if args.dtype else ("float16", "bfloat16", "float32")
    capability = args.compute_capability
    if capability is None and args.device_type == "cuda":
        capability = _cuda_capability()
    records = [
        (case, dtype, _plan_case(case, dtype, args.device_type, capability))
        for case in cases
        for dtype in dtypes
    ]
    if args.format == "json":
        print(json.dumps([_json_record(case, dtype, plan) for case, dtype, plan in records], indent=2))
    else:
        print("\n\n".join(_text_record(case, dtype, plan) for case, dtype, plan in records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
