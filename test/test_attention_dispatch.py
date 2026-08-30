from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

from model.attention_dispatch import (
    AttentionFamily,
    AttentionModelSpec,
    AttentionMode,
    AttentionRuntime,
    FallbackReason,
    PlanScope,
    select_attention_stack_plan,
)
from tools.benchmark_shape_matrix import ANNOUNCED_CASES
from tools.inspect_attention_dispatch import main as inspect_dispatch


def _spec(case) -> AttentionModelSpec:
    return AttentionModelSpec(
        batch_size=case.batch_size,
        sequence_length=case.seq_len,
        d_model=case.d_model,
        num_heads=case.heads,
        ffn_dim=case.ffn_dim,
        num_layers=case.layers,
        causal=case.causal,
    )


def _runtime(case, dtype: str, *, device_type: str = "cuda", autograd: bool = False):
    return AttentionRuntime(
        batch_size=case.batch_size,
        sequence_length=case.seq_len,
        d_model=case.d_model,
        dtype=dtype,
        device_type=device_type,
        device_capability=(12, 0) if device_type == "cuda" else None,
        needs_autograd=autograd,
        causal=case.causal,
    )


class AttentionDispatchTests(unittest.TestCase):
    def test_published_matrix_has_explicit_per_layer_modes_for_all_dtypes(self) -> None:
        expected = {
            "float16": {
                1: (AttentionMode.FP16_FULL_ROW,) * 4,
                2: (AttentionMode.FP16_FULL_ROW,) * 4,
                3: (AttentionMode.FP16_FULL_ROW,) * 4,
                4: (AttentionMode.FP16_FULL_ROW,) * 4,
                5: (AttentionMode.FP16_FULL_ROW,) * 4,
                6: (AttentionMode.FP16_FULL_ROW,) * 4,
                7: (AttentionMode.EXACT_REFERENCE,) * 4,
                8: (AttentionMode.EXACT_REFERENCE,) * 4,
                9: (AttentionMode.FP16_FULL_ROW,) * 4,
                10: (AttentionMode.FP16_FULL_ROW,) * 4,
                11: (AttentionMode.EXACT_REFERENCE,) * 4,
                12: (AttentionMode.FP16_FULL_ROW,) * 4,
                13: (
                    AttentionMode.FP16_BLOCKED_EXACT,
                    AttentionMode.FP16_TWO_PASS,
                    AttentionMode.FP16_TWO_PASS,
                    AttentionMode.FP16_TWO_PASS,
                ),
                14: (AttentionMode.FP16_LONG_TILED,) * 2,
            },
            "bfloat16": {
                1: (AttentionMode.BF16_FULL_ROW,) * 4,
                2: (AttentionMode.BF16_FULL_ROW,) * 4,
                3: (AttentionMode.BF16_FULL_ROW,) * 4,
                4: (AttentionMode.BF16_FULL_ROW,) * 4,
                5: (AttentionMode.BF16_FULL_ROW,) * 4,
                6: (AttentionMode.BF16_FULL_ROW,) * 4,
                7: (AttentionMode.EXACT_REFERENCE,) * 4,
                8: (AttentionMode.EXACT_REFERENCE,) * 4,
                9: (AttentionMode.BF16_FULL_ROW,) * 4,
                10: (AttentionMode.BF16_FULL_ROW,) * 4,
                11: (AttentionMode.EXACT_REFERENCE,) * 4,
                12: (AttentionMode.BF16_FULL_ROW,) * 4,
                13: (AttentionMode.EXACT_REFERENCE,) * 4,
                14: (AttentionMode.BF16_LONG_TILED,) * 2,
            },
            "float32": {
                1: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32),
                2: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32),
                3: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32),
                4: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32),
                5: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32),
                6: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32),
                7: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_SMALL_HEAD_TF32, AttentionMode.FP32_SMALL_HEAD_TF32, AttentionMode.FP32_SMALL_HEAD_TF32),
                8: (AttentionMode.FP32_BASELINE_EXACT,) * 4,
                9: (AttentionMode.FP32_BASELINE_EXACT,) * 4,
                10: (AttentionMode.FP32_FULL_ROW_TF32,) * 4,
                11: (AttentionMode.FP32_SMALL_HEAD_TF32,) * 4,
                12: (AttentionMode.FP32_BASELINE_EXACT, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32, AttentionMode.FP32_FULL_ROW_TF32),
                13: (AttentionMode.FP32_LONG_TILED,) * 4,
                14: (AttentionMode.FP32_LONG_TILED,) * 2,
            },
        }
        for dtype, cases in expected.items():
            for case in ANNOUNCED_CASES:
                with self.subTest(dtype=dtype, case=case.case_id):
                    plan = select_attention_stack_plan(_spec(case), _runtime(case, dtype))
                    self.assertEqual(tuple(layer.mode for layer in plan.layers), cases[case.case_id])
                    for layer in plan.layers:
                        if layer.mode in (AttentionMode.EXACT_REFERENCE, AttentionMode.FP32_BASELINE_EXACT):
                            self.assertIsNone(layer.launch)
                        elif layer.mode in (AttentionMode.FP16_BLOCKED_EXACT, AttentionMode.BF16_BLOCKED_EXACT):
                            self.assertEqual(layer.launch.query_block_size, 16)
                        else:
                            self.assertIsNotNone(layer.launch)

    def test_published_families_and_reasons_are_explicit(self) -> None:
        case11 = ANNOUNCED_CASES[10]
        plan = select_attention_stack_plan(_spec(case11), _runtime(case11, "float32"))
        self.assertEqual({layer.family for layer in plan.layers}, {AttentionFamily.SHORT_SMALL_HEAD})
        self.assertIsNone(plan.layers[0].reason)
        self.assertEqual(plan.layers[0].launch.block_n, 128)

        case13 = ANNOUNCED_CASES[12]
        bf16 = select_attention_stack_plan(_spec(case13), _runtime(case13, "bfloat16"))
        self.assertEqual(bf16.layers[0].reason, FallbackReason.BF16_D32_LONG_UNVALIDATED)
        fp16 = select_attention_stack_plan(_spec(case13), _runtime(case13, "float16"))
        self.assertEqual(fp16.layers[0].reason, FallbackReason.RESIDUAL_STACK_POLICY)
        self.assertEqual(fp16.layers[0].launch.query_block_size, 16)

        case14 = ANNOUNCED_CASES[13]
        extreme = select_attention_stack_plan(_spec(case14), _runtime(case14, "float16"))
        self.assertEqual(extreme.execution, "microbatch_one")
        self.assertEqual(extreme.microbatch_size, 1)

        case8 = ANNOUNCED_CASES[7]
        d256 = select_attention_stack_plan(_spec(case8), _runtime(case8, "float32"))
        self.assertEqual(d256.layers[0].reason, FallbackReason.FP32_D256_REJECTED)
        case9 = ANNOUNCED_CASES[8]
        d128 = select_attention_stack_plan(_spec(case9), _runtime(case9, "float32"))
        self.assertEqual(d128.layers[0].reason, FallbackReason.FP32_D128_REJECTED)

        probe_spec = AttentionModelSpec(1, 1024, 1024, 16, 1024, 2, True)
        probe_runtime = AttentionRuntime(1, 1024, 1024, "float32", "cuda", (12, 0), False, True)
        probe = select_attention_stack_plan(probe_spec, probe_runtime)
        self.assertEqual(tuple(layer.mode for layer in probe.layers), (AttentionMode.FP32_LONG_TILED,) * 2)
        self.assertEqual(probe.execution, "batched")

    def test_runtime_gates_choose_exact_without_changing_policy_table(self) -> None:
        case = ANNOUNCED_CASES[10]
        spec = _spec(case)
        for runtime in (
            _runtime(case, "float32", device_type="cpu"),
            _runtime(case, "float32", autograd=True),
        ):
            with self.subTest(runtime=runtime):
                plan = select_attention_stack_plan(spec, runtime)
                self.assertEqual(tuple(layer.mode for layer in plan.layers), (AttentionMode.FP32_BASELINE_EXACT,) * 4)
                self.assertIsNotNone(plan.layers[0].reason)

    def test_runtime_shape_and_hardware_gates_are_explicit(self) -> None:
        case = ANNOUNCED_CASES[12]
        spec = _spec(case)
        checks = (
            (
                AttentionRuntime(64, 1024, 128, "float32", "cuda", (11, 0), False, True),
                FallbackReason.CUDA_CAPABILITY,
            ),
            (
                AttentionRuntime(64, 1024, 128, "float32", "cuda", (12, 0), True, True),
                FallbackReason.AUTOGRAD,
            ),
            (
                AttentionRuntime(64, 1024, 128, "float32", "cuda", (12, 0), False, False),
                FallbackReason.SHAPE_MISMATCH,
            ),
            (
                AttentionRuntime(64, 2048, 128, "float32", "cuda", (12, 0), False, True),
                FallbackReason.SHAPE_MISMATCH,
            ),
            (
                AttentionRuntime(64, 1024, 128, "float64", "cuda", (12, 0), False, True),
                FallbackReason.UNSUPPORTED_DTYPE,
            ),
            (
                AttentionRuntime(1, 1024, 128, "float32", "cuda", (12, 0), False, True),
                FallbackReason.SHAPE_MISMATCH,
            ),
        )
        for runtime, reason in checks:
            with self.subTest(runtime=runtime):
                plan = select_attention_stack_plan(spec, runtime)
                self.assertEqual(plan.layers[0].reason, reason)
                self.assertEqual(plan.layers[0].family, AttentionFamily.EXACT)

        noncausal = AttentionModelSpec(64, 1024, 128, 4, 128, 4, False)
        runtime = AttentionRuntime(64, 1024, 128, "float16", "cuda", (12, 0), False, False)
        plan = select_attention_stack_plan(noncausal, runtime)
        self.assertEqual(plan.layers[0].reason, FallbackReason.UNVALIDATED_SHAPE)

        case14 = _spec(ANNOUNCED_CASES[13])
        sample_runtime = AttentionRuntime(1, 100000, 1024, "float16", "cuda", (12, 0), False, True)
        sample_plan = select_attention_stack_plan(case14, sample_runtime)
        self.assertEqual(sample_plan.layers[0].mode, AttentionMode.FP16_LONG_TILED)

    def test_invalid_model_layer_index_is_rejected_by_adapter(self) -> None:
        # The planner itself returns an ordered stack; the adapter owns the
        # layer-index boundary when consuming one layer from that stack.
        import torch
        from model.triton_fused_attention import TritonFusedSelfAttention

        spec = AttentionModelSpec(1, 32, 32, 2, 32, 1, True)
        adapter = TritonFusedSelfAttention(32, 2, model_spec=spec, layer_index=1)
        with self.assertRaises(ValueError):
            adapter._attention_plan(torch.randn(1, 32, 32), True, False)

    def test_invalid_specs_are_rejected_at_planning_boundary(self) -> None:
        with self.assertRaises(ValueError):
            AttentionModelSpec(1, 32, 33, 2, 32, 1, True)
        with self.assertRaises(ValueError):
            AttentionModelSpec(1, 32, 32, 2, 32, 0, True)

    def test_unpublished_fp32_long_shape_does_not_widen_tiled_dispatch(self) -> None:
        spec = AttentionModelSpec(1, 1024, 256, 4, 256, 2, True)
        runtime = AttentionRuntime(1, 1024, 256, "float32", "cuda", (12, 0), False, True)
        plan = select_attention_stack_plan(spec, runtime)
        self.assertEqual(tuple(layer.mode for layer in plan.layers), (AttentionMode.FP32_BASELINE_EXACT,) * 2)

    def test_standalone_scope_keeps_generic_validated_kernel_behavior(self) -> None:
        case = ANNOUNCED_CASES[0]
        plan = select_attention_stack_plan(
            _spec(case), _runtime(case, "float32"), scope=PlanScope.STANDALONE
        )
        self.assertEqual(tuple(layer.mode for layer in plan.layers), (AttentionMode.FP32_FULL_ROW_TF32,) * 4)
        self.assertIs(
            plan,
            select_attention_stack_plan(
                _spec(case), _runtime(case, "float32"), scope=PlanScope.STANDALONE
            ),
        )

    def test_inspection_cli_reports_case13_and_case14_without_model_execution(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                inspect_dispatch(
                    [
                        "--case",
                        "13,14",
                        "--dtype",
                        "float16",
                        "--dtype",
                        "bfloat16",
                        "--compute-capability",
                        "12.0",
                    ]
                ),
                0,
            )
        text = output.getvalue()
        self.assertIn("case 13 / float16", text)
        self.assertIn("mode=fp16_blocked_exact", text)
        self.assertIn("case 13 / bfloat16", text)
        self.assertIn("reason=bf16_d32_long_unvalidated", text)
        self.assertIn("execution: microbatch_one", text)

    def test_inspection_cli_json_is_machine_readable(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            inspect_dispatch(["--case", "11", "--dtype", "float32", "--compute-capability", "12.0", "--format", "json"])
        self.assertIn('"mode": "fp32_small_head_tf32"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
