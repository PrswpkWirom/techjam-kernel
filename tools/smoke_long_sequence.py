"""Optimized-only smoke test for the announced 100k-token configuration.

The protected benchmark cannot run its dense baseline for this case.  This
tool therefore exercises only ``UserOptimizedTransformer`` and reports the
memory, latency, and output properties needed to validate the long path.
"""

from __future__ import annotations

import argparse
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


def run_smoke(seed: int, dtype_name: str) -> int:
    if not torch.cuda.is_available():
        print("status: ERROR")
        print("reason: CUDA is unavailable")
        return 2

    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    config = TransformerConfig(
        batch_size=32,
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
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        with torch.inference_mode():
            if dtype == torch.float16:
                # Compile the Triton specialization before measuring FP16.
                warm_output = model(x[:1], valid_token_mask[:1])
                torch.cuda.synchronize(device)
                del warm_output
                torch.cuda.empty_cache()
            # BF16 uses native bounded PyTorch operations rather than a JIT
            # kernel. Avoiding a redundant 100k-token warmup also leaves the
            # allocator unfragmented before this near-capacity smoke test.

            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(x, valid_token_mask)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
            peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2**20
            peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20
            finite = _all_finite_by_batch(output)
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
    print(f"runtime_ms: {elapsed_ms:.3f}")
    print(f"peak_allocated_mib: {peak_allocated_mib:.1f}")
    print(f"peak_reserved_mib: {peak_reserved_mib:.1f}")
    return 0 if finite else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    args = parser.parse_args()
    return run_smoke(args.seed, args.dtype)


if __name__ == "__main__":
    raise SystemExit(main())
