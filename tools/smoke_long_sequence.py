"""Optimized-only smoke test for the announced 100k-token configuration.

The protected benchmark cannot run its dense baseline for this case.  This
tool therefore exercises only ``UserOptimizedTransformer`` and reports the
memory, latency, and output properties needed to validate the long path.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

# Make ``python tools/smoke_long_sequence.py`` behave the same as module-style
# invocation without requiring an environment-specific PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch_transformer_benchmark import (
    TransformerConfig,
    UserOptimizedTransformer,
)


def _all_finite_by_batch(output: torch.Tensor) -> bool:
    """Check finiteness without allocating a full-output boolean tensor."""
    for batch_index in range(output.shape[0]):
        if not bool(torch.isfinite(output[batch_index]).all().item()):
            return False
    return True


def run_smoke(
    seed: int,
    dtype_name: str,
    batch_size: int,
    padding_ratio: float,
    warmup: int,
    repeats: int,
) -> int:
    if not torch.cuda.is_available():
        print("status: ERROR")
        print("reason: CUDA is unavailable")
        return 2

    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    required_full_fp32_bytes = 2 * 32 * 100_000 * 1024 * 4
    if (
        dtype == torch.float32
        and batch_size == 32
        and torch.cuda.get_device_properties(device).total_memory
        < required_full_fp32_bytes
    ):
        print("status: PRECONDITION_BLOCKED")
        print("reason: FP32 input and output alone require 24.41 GiB; "
              "run the full B=32 smoke on a >=32 GiB CUDA GPU")
        return 2
    config = TransformerConfig(
        batch_size=batch_size,
        seq_len=100_000,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        model = UserOptimizedTransformer(config).to(device=device, dtype=dtype).eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        # Allocate the fixed input once.  The shared benchmark helper multiplies
        # by input_scale and would transiently allocate a second 6.10 GiB tensor
        # for the default scale of 1.0, needlessly fragmenting this 16 GiB device.
        x = torch.randn(
            config.batch_size,
            config.seq_len,
            config.d_model,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        valid_length = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
        valid_token_mask = torch.arange(
            config.seq_len, device=device
        )[None, :] < valid_length
        valid_token_mask = valid_token_mask.expand(config.batch_size, -1)
        with torch.inference_mode():
            # Compile the long Triton specialization before measuring either
            # dtype. A single sample exercises both Transformer layers without
            # entering the model-level B=32 microbatch loop.
            warm_output = model(x[:1], valid_token_mask[:1])
            torch.cuda.synchronize(device)
            del warm_output
            torch.cuda.empty_cache()

            for _ in range(warmup):
                warm_output = model(x, valid_token_mask)
                del warm_output
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            elapsed_samples_ms: list[float] = []
            output = None
            for _ in range(repeats):
                if output is not None:
                    del output
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(x, valid_token_mask)
                end.record()
                torch.cuda.synchronize(device)
                elapsed_samples_ms.append(start.elapsed_time(end))
            peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2**20
            peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20
            finite = _all_finite_by_batch(output)
            padded_queries_zero = bool(
                (output[~valid_token_mask] == 0).all().item()
            )
    except torch.cuda.OutOfMemoryError as exc:
        print("status: OOM")
        print(f"error: {exc}")
        return 1

    print("status: PASS")
    print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")
    print(f"output_shape: {tuple(output.shape)}")
    print(f"output_dtype: {output.dtype}")
    print(f"output_finite: {finite}")
    print(f"padding_ratio: {padding_ratio}")
    print(f"padded_queries_zero: {padded_queries_zero}")
    print(f"runtime_samples_ms: {','.join(f'{value:.3f}' for value in elapsed_samples_ms)}")
    print(f"runtime_ms: {statistics.median(elapsed_samples_ms):.3f}")
    print(f"peak_allocated_mib: {peak_allocated_mib:.1f}")
    print(f"peak_reserved_mib: {peak_reserved_mib:.1f}")
    return 0 if finite else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1, 32),
        default=32,
        help="use B=1 for FP32 validation on GPUs below the full-output memory floor",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        choices=(0.0, 0.25),
        default=0.0,
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("--warmup must be non-negative and --repeats must be positive")
    return run_smoke(
        args.seed,
        args.dtype,
        args.batch_size,
        args.padding_ratio,
        args.warmup,
        args.repeats,
    )


if __name__ == "__main__":
    raise SystemExit(main())
