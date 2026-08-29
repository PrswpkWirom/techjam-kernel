"""Published Transformer benchmark cases and shape-level preflight helpers.

This module contains problem data from the organizer's announced shape table.
It does not run models or alter the official evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ShapeCase:
    """One announced evaluator configuration."""

    case_id: int
    batch_size: int
    d_model: int
    heads: int
    seq_len: int
    layers: int
    causal: bool
    ffn_dim: int

    def benchmark_args(self) -> tuple[str, ...]:
        """Return only shape flags, in the evaluator's CLI vocabulary."""

        args = (
            "--batch-size",
            str(self.batch_size),
            "--seq-len",
            str(self.seq_len),
            "--d-model",
            str(self.d_model),
            "--heads",
            str(self.heads),
            "--ffn-dim",
            str(self.ffn_dim),
            "--layers",
            str(self.layers),
        )
        return args + (("--causal",) if self.causal else ())

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "case_id": self.case_id,
            "batch_size": self.batch_size,
            "d_model": self.d_model,
            "heads": self.heads,
            "seq_len": self.seq_len,
            "layers": self.layers,
            "causal": self.causal,
            "ffn_dim": self.ffn_dim,
        }

    def dense_attention_bytes(self, dtype_bytes: int = 2) -> int:
        """Bytes for one dense ``[B, H, S, S]`` score/probability tensor."""

        if dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        return self.batch_size * self.heads * self.seq_len**2 * dtype_bytes


# Appendix 3.7, in the published order.
ANNOUNCED_CASES: tuple[ShapeCase, ...] = (
    ShapeCase(1, 64, 128, 4, 128, 4, True, 128),
    ShapeCase(2, 1, 128, 4, 128, 4, True, 128),
    ShapeCase(3, 4, 128, 4, 128, 4, True, 128),
    ShapeCase(4, 16, 128, 4, 128, 4, True, 128),
    ShapeCase(5, 128, 128, 4, 128, 4, True, 128),
    ShapeCase(6, 10000, 128, 4, 128, 4, True, 128),
    ShapeCase(7, 64, 32, 4, 128, 4, True, 32),
    ShapeCase(8, 64, 1024, 4, 128, 4, True, 1024),
    ShapeCase(9, 64, 128, 1, 128, 4, True, 128),
    ShapeCase(10, 64, 128, 2, 128, 4, True, 128),
    ShapeCase(11, 64, 128, 16, 128, 4, True, 128),
    ShapeCase(12, 64, 128, 4, 32, 4, True, 128),
    ShapeCase(13, 64, 128, 4, 1024, 4, True, 128),
    ShapeCase(14, 32, 1024, 16, 100000, 2, True, 1024),
)


def select_cases(case_ids: Iterable[int] | None = None) -> tuple[ShapeCase, ...]:
    """Select cases by one-based ID, preserving announcement order."""

    if case_ids is None:
        return ANNOUNCED_CASES
    requested = tuple(case_ids)
    known = {case.case_id: case for case in ANNOUNCED_CASES}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"unknown case id(s): {', '.join(map(str, unknown))}")
    if not requested:
        raise ValueError("at least one case must be selected")
    return tuple(known[case_id] for case_id in requested)


def parse_case_ids(value: str) -> tuple[int, ...]:
    """Parse ``--case 1,4,14`` input without silently accepting bad IDs."""

    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("case IDs must be comma-separated integers") from exc
    return values


def format_bytes(value: int) -> str:
    """Human-readable binary size used in preflight warnings."""

    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} PiB"
