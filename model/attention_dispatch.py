"""Pure attention planning for the Transformer benchmark.

The planner deliberately contains no torch or Triton imports.  It describes
which already-validated executor a caller should run; executors remain in the
kernel modules so that planning can be tested on any machine and inspected
without compiling a GPU kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional


class _StringEnum(str, Enum):
    """String-valued enum that remains friendly to JSON and CLI output."""

    def __str__(self) -> str:
        return self.value


class PlanScope(_StringEnum):
    MODEL = "model"
    STANDALONE = "standalone"


class AttentionFamily(_StringEnum):
    EXACT = "exact"
    SHORT_FULL_ROW = "short_full_row"
    SHORT_SMALL_HEAD = "short_small_head"
    LONG_TILED = "long_tiled"
    D256_SPECIALIZED = "d256_specialized"


class AttentionMode(_StringEnum):
    EXACT_REFERENCE = "exact_reference"
    FP32_BASELINE_EXACT = "fp32_baseline_exact"
    FP16_BLOCKED_EXACT = "fp16_blocked_exact"
    BF16_BLOCKED_EXACT = "bf16_blocked_exact"
    FP16_FULL_ROW = "fp16_full_row"
    BF16_FULL_ROW = "bf16_full_row"
    FP32_FULL_ROW_TF32 = "fp32_full_row_tf32"
    FP32_SMALL_HEAD_TF32 = "fp32_small_head_tf32"
    FP16_TWO_PASS = "fp16_two_pass"
    FP16_LONG_TILED = "fp16_long_tiled"
    BF16_LONG_TILED = "bf16_long_tiled"
    FP32_LONG_TILED = "fp32_long_tiled"


class FallbackReason(_StringEnum):
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    UNSUPPORTED_DEVICE = "unsupported_device"
    CUDA_CAPABILITY = "cuda_capability"
    AUTOGRAD = "autograd"
    SHAPE_MISMATCH = "shape_mismatch"
    UNVALIDATED_SHAPE = "unvalidated_shape"
    FP32_D128_REJECTED = "fp32_d128_rejected"
    FP32_D256_REJECTED = "fp32_d256_rejected"
    BF16_D32_LONG_UNVALIDATED = "bf16_d32_long_unvalidated"
    RESIDUAL_STACK_POLICY = "residual_stack_policy"


@dataclass(frozen=True, slots=True)
class AttentionModelSpec:
    """The configured Transformer shape used for policy selection."""

    batch_size: int
    sequence_length: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.batch_size,
                self.sequence_length,
                self.d_model,
                self.num_heads,
                self.ffn_dim,
                self.num_layers,
            )
        ):
            raise ValueError("model dimensions and layer count must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            return 0
        return self.d_model // self.num_heads

    def key(self) -> tuple[object, ...]:
        return (
            self.batch_size,
            self.sequence_length,
            self.d_model,
            self.num_heads,
            self.ffn_dim,
            self.num_layers,
            self.causal,
        )


@dataclass(frozen=True, slots=True)
class AttentionRuntime:
    """Runtime facts that can make a configured plan ineligible."""

    batch_size: int
    sequence_length: int
    d_model: int
    dtype: str
    device_type: str
    device_capability: Optional[tuple[int, int]]
    needs_autograd: bool
    causal: bool
    bf16_fused_min_length: int = 100000

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (self.batch_size, self.sequence_length, self.d_model)
        ):
            raise ValueError("runtime dimensions must be positive")
        if self.bf16_fused_min_length <= 0:
            raise ValueError("bf16_fused_min_length must be positive")

    def key(self) -> tuple[object, ...]:
        return (
            self.batch_size,
            self.sequence_length,
            self.d_model,
            self.dtype,
            self.device_type,
            self.device_capability,
            self.needs_autograd,
            self.causal,
            self.bf16_fused_min_length,
        )


@dataclass(frozen=True, slots=True)
class KernelLaunch:
    """Offline-selected launch values carried by an attention plan."""

    block_m: Optional[int] = None
    block_n: Optional[int] = None
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    query_block_size: Optional[int] = None
    output_layout: str = "bshd"


@dataclass(frozen=True, slots=True)
class AttentionPlan:
    """One layer's selected family and executable mode."""

    layer_index: int
    family: AttentionFamily
    mode: AttentionMode
    launch: Optional[KernelLaunch] = None
    reason: Optional[FallbackReason] = None
    explanation: Optional[str] = None

    @property
    def label(self) -> str:
        """Compact stable label used by the inspection command."""

        return self.mode.value


@dataclass(frozen=True, slots=True)
class AttentionStackPlan:
    """All layer plans and the whole-stack execution policy."""

    layers: tuple[AttentionPlan, ...]
    execution: str = "batched"

    @property
    def microbatch_size(self) -> Optional[int]:
        return 1 if self.execution == "microbatch_one" else None


@dataclass(frozen=True, slots=True)
class _PolicyRule:
    """Declarative published-shape override."""

    dtype: str
    batch_sizes: tuple[int, ...]
    sequence_length: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool
    modes: tuple[AttentionMode, ...]

    def matches(self, spec: AttentionModelSpec) -> bool:
        return (
            spec.batch_size in self.batch_sizes
            and spec.sequence_length == self.sequence_length
            and spec.d_model == self.d_model
            and spec.num_heads == self.num_heads
            and spec.ffn_dim == self.ffn_dim
            and spec.num_layers == self.num_layers
            and spec.causal == self.causal
        )


def _rule(
    dtype: str,
    batches: tuple[int, ...],
    sequence_length: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
    modes: tuple[AttentionMode, ...],
    causal: bool = True,
) -> _PolicyRule:
    return _PolicyRule(
        dtype=dtype,
        batch_sizes=batches,
        sequence_length=sequence_length,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        causal=causal,
        modes=modes,
    )


_E = AttentionMode.FP32_BASELINE_EXACT
_F16 = AttentionMode.FP16_FULL_ROW
_BF16 = AttentionMode.BF16_FULL_ROW
_F32 = AttentionMode.FP32_FULL_ROW_TF32
_SMALL = AttentionMode.FP32_SMALL_HEAD_TF32
_T16 = AttentionMode.FP16_TWO_PASS
_O16 = AttentionMode.FP16_LONG_TILED
_OBF16 = AttentionMode.BF16_LONG_TILED
_T32 = AttentionMode.FP32_LONG_TILED


# These are the policies that previously lived partly in the benchmark model
# and partly in TritonFusedSelfAttention.forward.  Keeping them as data makes
# the residual-stack accuracy decisions visible without flattening dtype modes.
_PUBLISHED_POLICY_RULES: tuple[_PolicyRule, ...] = (
    _rule("float16", (1, 4, 16, 64, 128, 10000), 128, 128, 4, 128, 4, (_F16,) * 4),
    _rule("bfloat16", (1, 4, 16, 64, 128, 10000), 128, 128, 4, 128, 4, (_BF16,) * 4),
    _rule("float32", (1, 4, 16, 64, 128), 128, 128, 4, 128, 4, (_E, _F32, _F32, _F32)),
    _rule("float32", (10000,), 128, 128, 4, 128, 4, (_E, _E, _F32, _F32)),
    _rule("float32", (64,), 128, 32, 4, 32, 4, (_E, _SMALL, _SMALL, _SMALL)),
    _rule("float32", (64,), 128, 1024, 4, 1024, 4, (_E,) * 4),
    _rule("float32", (64,), 128, 128, 1, 128, 4, (_E,) * 4),
    _rule("float32", (64,), 128, 128, 2, 128, 4, (_F32,) * 4),
    _rule("float32", (64,), 128, 128, 16, 128, 4, (_SMALL,) * 4),
    _rule("float32", (64,), 32, 128, 4, 128, 4, (_E, _F32, _F32, _F32)),
    _rule("float32", (64,), 1024, 128, 4, 128, 4, (_T32,) * 4),
    _rule("float16", (64,), 1024, 128, 4, 128, 4, (AttentionMode.FP16_BLOCKED_EXACT, _T16, _T16, _T16)),
    _rule("float16", (32,), 100000, 1024, 16, 1024, 2, (_O16, _O16)),
    _rule("bfloat16", (32,), 100000, 1024, 16, 1024, 2, (_OBF16, _OBF16)),
    _rule("float32", (32,), 100000, 1024, 16, 1024, 2, (_T32, _T32)),
    # B=1 probes use the same attention mode without whole-stack microbatching.
    _rule("float32", (1,), 1024, 1024, 16, 1024, 2, (_T32, _T32)),
    _rule("float32", (1,), 100000, 1024, 16, 1024, 2, (_T32, _T32)),
)


def _exact_mode(dtype: str) -> AttentionMode:
    return (
        AttentionMode.FP32_BASELINE_EXACT
        if dtype == "float32"
        else AttentionMode.EXACT_REFERENCE
    )


def _exact_plan(
    layer_index: int,
    dtype: str,
    reason: FallbackReason,
    explanation: str,
    *,
    mode: Optional[AttentionMode] = None,
    launch: Optional[KernelLaunch] = None,
) -> AttentionPlan:
    return AttentionPlan(
        layer_index=layer_index,
        family=AttentionFamily.EXACT,
        mode=mode or _exact_mode(dtype),
        launch=launch,
        reason=reason,
        explanation=explanation,
    )


def _launch_for_mode(
    mode: AttentionMode,
    spec: AttentionModelSpec,
    *,
    scope: PlanScope,
) -> Optional[KernelLaunch]:
    sequence_length = spec.sequence_length
    head_dim = spec.head_dim
    if mode in (
        AttentionMode.FP16_FULL_ROW,
        AttentionMode.BF16_FULL_ROW,
    ):
        return KernelLaunch(
            block_m=32 if sequence_length <= 32 else 64,
            block_n=1 << (sequence_length - 1).bit_length(),
            num_warps=4,
        )
    if mode == AttentionMode.FP32_FULL_ROW_TF32:
        if sequence_length == 32 and head_dim == 32:
            block_m, warps = 32, 8
        elif sequence_length <= 32:
            block_m, warps = 16, 4
        elif sequence_length <= 64:
            block_m, warps = (16 if head_dim <= 32 else 32), 4
        else:
            block_m, warps = (32 if head_dim <= 32 else 64), 4
        return KernelLaunch(
            block_m=block_m,
            block_n=1 << (sequence_length - 1).bit_length(),
            num_warps=warps,
        )
    if mode == AttentionMode.FP32_SMALL_HEAD_TF32:
        if spec.num_heads == 4:
            return KernelLaunch(block_m=64, block_n=64, num_warps=2)
        if spec.num_heads == 16:
            return KernelLaunch(block_m=64, block_n=128, num_warps=4)
        return KernelLaunch(block_m=32, block_n=128, num_warps=4)
    if mode == AttentionMode.FP16_TWO_PASS:
        return KernelLaunch(block_m=64, block_n=64, num_warps=4, num_stages=3)
    if mode == AttentionMode.FP16_LONG_TILED:
        return KernelLaunch(block_m=64, block_n=64, num_warps=4, num_stages=3)
    if mode == AttentionMode.BF16_LONG_TILED:
        return KernelLaunch(block_m=32, block_n=32, num_warps=4, num_stages=1)
    if mode == AttentionMode.FP32_LONG_TILED:
        if head_dim == 32:
            return KernelLaunch(block_m=64, block_n=32, num_warps=4, num_stages=3)
        return KernelLaunch(block_m=64, block_n=32, num_warps=4, num_stages=2)
    if mode == AttentionMode.FP16_BLOCKED_EXACT:
        return KernelLaunch(query_block_size=16, output_layout="bhsd")
    if mode == AttentionMode.BF16_BLOCKED_EXACT:
        return KernelLaunch(query_block_size=16, output_layout="bhsd")
    return None


def _runtime_reason(spec: AttentionModelSpec, runtime: AttentionRuntime) -> Optional[FallbackReason]:
    if runtime.dtype not in {"float16", "bfloat16", "float32"}:
        return FallbackReason.UNSUPPORTED_DTYPE
    if runtime.device_type != "cuda":
        return FallbackReason.UNSUPPORTED_DEVICE
    if runtime.device_capability is None or runtime.device_capability[0] < 12:
        return FallbackReason.CUDA_CAPABILITY
    if runtime.needs_autograd:
        return FallbackReason.AUTOGRAD
    # During case-14 whole-stack microbatching each layer is intentionally
    # called with a single sample even though the configured stack has B=32.
    case14_microbatch_sample = (
        spec.batch_size == 32
        and spec.sequence_length == 100000
        and spec.d_model == 1024
        and spec.num_heads == 16
        and spec.ffn_dim == 1024
        and spec.num_layers == 2
        and spec.causal
        and runtime.batch_size == 1
    )
    if (
        not (runtime.batch_size == spec.batch_size or case14_microbatch_sample)
        or runtime.sequence_length != spec.sequence_length
        or runtime.d_model != spec.d_model
        or runtime.causal != spec.causal
        or runtime.batch_size <= 0
    ):
        return FallbackReason.SHAPE_MISMATCH
    return None


def _fallback_explanation(reason: FallbackReason) -> str:
    return {
        FallbackReason.UNSUPPORTED_DTYPE: "dtype is outside the validated attention modes",
        FallbackReason.UNSUPPORTED_DEVICE: "validated kernels require CUDA",
        FallbackReason.CUDA_CAPABILITY: "validated Gluon/Triton kernels require CUDA capability >= sm120",
        FallbackReason.AUTOGRAD: "inference-only kernels are disabled when autograd is required",
        FallbackReason.SHAPE_MISMATCH: "runtime shape or causal mode does not match the configured plan",
    }.get(reason, reason.value)


def _generic_mode(
    spec: AttentionModelSpec,
    runtime: AttentionRuntime,
    scope: PlanScope,
) -> tuple[Optional[AttentionMode], Optional[FallbackReason], str]:
    dtype = runtime.dtype
    sequence_length = spec.sequence_length
    head_dim = spec.head_dim

    if sequence_length >= 257:
        # The bounded long-sequence kernels preserve the published causal
        # masking path only.  Standalone non-causal calls must retain the
        # adapter's exact/reference behavior rather than widening dispatch.
        if not spec.causal:
            return (
                None,
                FallbackReason.UNVALIDATED_SHAPE,
                "long-sequence kernels are validated only for causal attention",
            )
        if dtype == "float16" and (
            head_dim == 64 or (head_dim == 32 and sequence_length == 1024)
        ):
            if head_dim == 32 and sequence_length == 1024:
                return (
                    AttentionMode.FP16_TWO_PASS,
                    None,
                    "validated D=32 long attention keeps the two-pass reduction",
                )
            return AttentionMode.FP16_LONG_TILED, None, "validated FP16 bounded long-sequence kernel"
        if dtype == "bfloat16" and head_dim == 64:
            if sequence_length >= runtime.bf16_fused_min_length:
                return AttentionMode.BF16_LONG_TILED, None, "validated BF16 bounded long-sequence kernel"
            return AttentionMode.BF16_BLOCKED_EXACT, None, "BF16 bounded exact reduction preserves baseline rounding below the fused threshold"
        # FP32 long tiling is intentionally restricted to the published case
        # 13/14 policy rules above; the old adapter never enabled it for an
        # arbitrary model shape.
        if dtype == "bfloat16" and head_dim == 32:
            return None, FallbackReason.BF16_D32_LONG_UNVALIDATED, "no validated BF16 D=32 long-sequence kernel"
        return None, FallbackReason.UNVALIDATED_SHAPE, "no validated long-sequence kernel for this dtype/head dimension"

    if sequence_length in (32, 64, 128):
        if dtype in ("float16", "bfloat16") and head_dim in (16, 32, 64, 128):
            return (
                AttentionMode.FP16_FULL_ROW if dtype == "float16" else AttentionMode.BF16_FULL_ROW,
                None,
                "validated short full-row Gluon kernel",
            )
        if dtype == "float32" and sequence_length == 128 and head_dim == 8:
            return AttentionMode.FP32_SMALL_HEAD_TF32, None, "validated padded D=8 TF32 small-head kernel"
        if dtype == "float32" and head_dim in (16, 32, 64, 128):
            if scope == PlanScope.STANDALONE:
                supported = {
                    (32, 32), (128, 32), (32, 64), (64, 64), (128, 64), (128, 8)
                }
                if (sequence_length, head_dim) in supported:
                    return AttentionMode.FP32_FULL_ROW_TF32, None, "standalone validated FP32 full-row kernel"
            return None, FallbackReason.UNVALIDATED_SHAPE, "FP32 model shape has no published residual-stack policy"

    if dtype == "float32" and head_dim == 256:
        return None, FallbackReason.FP32_D256_REJECTED, "experimental FP32 D=256 kernel is not correctness/performance certified"
    return None, FallbackReason.UNVALIDATED_SHAPE, "shape is outside the validated attention envelope"


def _policy_rule(spec: AttentionModelSpec, dtype: str, scope: PlanScope) -> Optional[_PolicyRule]:
    if scope == PlanScope.STANDALONE:
        return None
    for rule in _PUBLISHED_POLICY_RULES:
        if rule.dtype == dtype and rule.matches(spec):
            return rule
    return None


def _plan_for_mode(
    layer_index: int,
    mode: AttentionMode,
    spec: AttentionModelSpec,
    explanation: Optional[str] = None,
) -> AttentionPlan:
    family = {
        AttentionMode.FP16_FULL_ROW: AttentionFamily.SHORT_FULL_ROW,
        AttentionMode.BF16_FULL_ROW: AttentionFamily.SHORT_FULL_ROW,
        AttentionMode.FP32_FULL_ROW_TF32: AttentionFamily.SHORT_FULL_ROW,
        AttentionMode.FP32_SMALL_HEAD_TF32: AttentionFamily.SHORT_SMALL_HEAD,
        AttentionMode.FP16_TWO_PASS: AttentionFamily.LONG_TILED,
        AttentionMode.FP16_LONG_TILED: AttentionFamily.LONG_TILED,
        AttentionMode.BF16_BLOCKED_EXACT: AttentionFamily.EXACT,
        AttentionMode.BF16_LONG_TILED: AttentionFamily.LONG_TILED,
        AttentionMode.FP32_LONG_TILED: AttentionFamily.LONG_TILED,
        AttentionMode.FP16_BLOCKED_EXACT: AttentionFamily.EXACT,
        AttentionMode.EXACT_REFERENCE: AttentionFamily.EXACT,
        AttentionMode.FP32_BASELINE_EXACT: AttentionFamily.EXACT,
    }[mode]
    reason = None
    if mode in {AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP16_BLOCKED_EXACT}:
        if mode == AttentionMode.FP32_BASELINE_EXACT and spec.d_model == 1024 and spec.num_heads == 4:
            reason = FallbackReason.FP32_D256_REJECTED
        elif mode == AttentionMode.FP32_BASELINE_EXACT and spec.num_heads == 1:
            reason = FallbackReason.FP32_D128_REJECTED
        else:
            reason = FallbackReason.RESIDUAL_STACK_POLICY
    if reason == FallbackReason.FP32_D256_REJECTED:
        explanation = "FP32 D=256 chunked candidate is not correctness/performance certified"
    elif reason == FallbackReason.FP32_D128_REJECTED:
        explanation = "FP32 D=128 Gluon candidate regressed the protected benchmark"
    return AttentionPlan(
        layer_index=layer_index,
        family=family,
        mode=mode,
        launch=_launch_for_mode(mode, spec, scope=PlanScope.MODEL),
        reason=reason,
        explanation=explanation,
    )


@lru_cache(maxsize=256)
def _select_cached(
    spec: AttentionModelSpec,
    runtime: AttentionRuntime,
    scope: PlanScope,
) -> AttentionStackPlan:
    runtime_reason = _runtime_reason(spec, runtime)
    layer_count = spec.num_layers if spec.num_layers > 0 else 1
    if runtime_reason is not None:
        explanation = _fallback_explanation(runtime_reason)
        plans = tuple(
            _exact_plan(index, runtime.dtype, runtime_reason, explanation)
            for index in range(layer_count)
        )
        return AttentionStackPlan(plans)

    rule = _policy_rule(spec, runtime.dtype, scope)
    if rule is not None:
        plans: list[AttentionPlan] = []
        for index, mode in enumerate(rule.modes):
            if mode == AttentionMode.FP32_BASELINE_EXACT:
                explanation = "published residual-stack policy keeps this layer in organizer operation order"
            elif mode == AttentionMode.FP16_BLOCKED_EXACT:
                explanation = "first D=32 long layer keeps the exact FP16 score/probability boundary"
            else:
                explanation = None
            plans.append(_plan_for_mode(index, mode, spec, explanation))
        execution = (
            "microbatch_one"
            if spec.batch_size == 32
            and spec.sequence_length == 100000
            and spec.d_model == 1024
            and spec.num_heads == 16
            and spec.ffn_dim == 1024
            and spec.num_layers == 2
            and runtime.batch_size == 32
            else "batched"
        )
        return AttentionStackPlan(tuple(plans), execution)

    mode, reason, explanation = _generic_mode(spec, runtime, scope)
    plans = []
    for index in range(layer_count):
        if mode is not None:
            plans.append(_plan_for_mode(index, mode, spec, explanation))
        else:
            plans.append(_exact_plan(index, runtime.dtype, reason or FallbackReason.UNVALIDATED_SHAPE, explanation))
    return AttentionStackPlan(tuple(plans))


def select_attention_stack_plan(
    spec: AttentionModelSpec,
    runtime: AttentionRuntime,
    scope: PlanScope = PlanScope.MODEL,
) -> AttentionStackPlan:
    """Select all layer attention modes and whole-stack execution policy.

    The arguments are immutable values, so the result is cached by shape,
    dtype, device capability, and autograd mode.  This function is the only
    production policy authority; kernel modules merely execute its result.
    """

    return _select_cached(spec, runtime, PlanScope(scope))


def clear_plan_cache() -> None:
    """Clear cached plans for tests that need to inspect cache behavior."""

    _select_cached.cache_clear()
