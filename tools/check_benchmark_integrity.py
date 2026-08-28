#!/usr/bin/env python3
"""Reject changes to the official benchmark outside its solution seam.

The hashes below are derived from the Python AST of the benchmark supplied by
the competition. Formatting and comments may change without causing a false
alarm, while executable changes to protected classes/functions are detected.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "torch_transformer_benchmark.py"

# These are the only definitions in the official harness that competitors are
# explicitly invited to customize.
EDITABLE_DEFINITIONS = {"UserOptimizedTransformer", "copy_model_weights"}

EXPECTED_AST_HASHES = {
    "TransformerConfig": "7a403ad528d65496fade822a1441208c57f5ffeea3ab7a2664186747bad05948",
    "BaselineSelfAttention": "5492f3ac4671fda1ce47a7723a7d562abbbeb2ed554422c5fa147ce822ea43bd",
    "BaselineTransformerBlock": "0aa57dc4154915bbd220044567c7c642a0e151c1fae6ca716dc64ce0dfaf6123",
    "BaselineTransformer": "77611d1f59a5f570749b9e503405ce3f451a2136816071fe08e58e1979c3ca7d",
    "resolve_device": "dddb72ad85952a2408bccc22f31b5fefcde8cc996ddec7debe154c89920146c5",
    "resolve_dtype": "c52e0f1dba2fe55521770a843f8412417d468f1314921502266269cedc203bc8",
    "generate_random_case": "b138137e1dc04151de8bab133dc08ff6ca49568ed9e00f1e23db18ec72ba488a",
    "AccuracyResult": "729aa7673a3eb611da21121302b0dedff77efc2fa939209efe334c586850c504",
    "compare_outputs": "0d76c12a05016c8619ef836ea35ad436fc106d10a791ddc48417326c7c795d2d",
    "run_accuracy_tests": "10b75e24b5480b10e92257d54d7efbc9ccd06458cc4ff387ccdb812e4dd501ef",
    "percentile": "06710e58d55a6133a085ee10ae963bc9d0fe60b1f3a82d12a1546743d0e6da55",
    "TimingResult": "f9862b07fe5605b3c1f70fa6b397c86e9c4fc63d9b8b8843b05e9a11cfeefce9",
    "warmup_model": "f7dae173880a531002c2d65961169f3640589317c7567473d599104a28df57a0",
    "benchmark_once": "70973c5513491bdda2ddc72d0b5a4362588198bf2ca94eb3608da73b9966cf32",
    "benchmark_models": "4130d538759fb2af4dd265da4da3d6247a8e01d59bedcf3947ed80277665772a",
    "maybe_compile": "151c2092a249987ad5efdf562d5d4c73bcf5b142ebce3b54e854e8a3402133d3",
    "parse_args": "5d2d03a922b8b7086dce28ee541077b0705d66e78ec9b795dbed2b8db70067ae",
    "validate_args": "7db4ac4ea2273cb4974c2f1e479d76f8464cfd0841544aa2c7a75929ccff5c7d",
    "main": "0281b781dec49f677e9baf3aa29ad064d432ed9b09ec0635eda2e1e679a07c74",
}

EXPECTED_MAIN_GUARD_HASH = (
    "be6767dd3debdea09edccea6c2c2af628c6025cf732c8353f6063635e367dae8"
)


def node_hash(node: ast.AST) -> str:
    normalized = ast.dump(node, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    try:
        tree = ast.parse(BENCHMARK.read_text(encoding="utf-8"), filename=str(BENCHMARK))
    except (OSError, SyntaxError) as error:
        print(f"benchmark integrity: FAIL: {error}", file=sys.stderr)
        return 1

    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    failures: list[str] = []

    for name, expected_hash in EXPECTED_AST_HASHES.items():
        node = definitions.get(name)
        if node is None:
            failures.append(f"missing protected definition: {name}")
        elif node_hash(node) != expected_hash:
            failures.append(f"protected definition changed: {name}")

    expected_names = set(EXPECTED_AST_HASHES) | EDITABLE_DEFINITIONS
    unexpected = sorted(set(definitions) - expected_names)
    if unexpected:
        failures.append(
            "unexpected top-level definitions in benchmark (put solution code in a "
            f"separate module): {', '.join(unexpected)}"
        )

    main_guards = [node for node in tree.body if isinstance(node, ast.If)]
    if len(main_guards) != 1 or node_hash(main_guards[0]) != EXPECTED_MAIN_GUARD_HASH:
        failures.append("official __main__ guard changed")

    permitted_top_level = (
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.If,
    )
    disallowed = [
        type(node).__name__
        for node in tree.body
        if not isinstance(node, permitted_top_level)
    ]
    if disallowed:
        failures.append(
            "unexpected executable top-level statements: " + ", ".join(disallowed)
        )

    if failures:
        print("benchmark integrity: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "Restore protected code from the organizer-provided benchmark; do not "
            "update these hashes to bless a local change.",
            file=sys.stderr,
        )
        return 1

    print(
        "benchmark integrity: PASS "
        "(official evaluator protected; solution seam remains editable)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
