#!/usr/bin/env python3
"""Collect a single-expert vLLM fused-MoE latency curve on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=1408)
    parser.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def event_latency_ms(callable_: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.hidden_size <= 0 or args.intermediate_size <= 0:
        raise ValueError("model dimensions must be positive")
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    if any(value <= 0 for value in args.token_counts):
        raise ValueError("token counts must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size

    # One active expert isolates the latency quantity consumed by the current
    # M2 one-dimensional per-expert curve. Multi-expert grouped execution is a
    # separate calibration surface and must not be mixed into this CSV.
    w1 = torch.randn(
        1,
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    w2 = torch.randn(
        1,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    )

    rows: list[dict[str, object]] = []
    sample_id = 0
    for token_count in sorted(set(args.token_counts)):
        hidden = torch.randn(token_count, hidden_size, device=device, dtype=dtype)
        topk_ids = torch.zeros((token_count, 1), device=device, dtype=torch.int32)
        topk_weights = torch.ones((token_count, 1), device=device, dtype=torch.float32)

        def run(
            hidden_states: torch.Tensor = hidden,
            weights: torch.Tensor = topk_weights,
            expert_ids: torch.Tensor = topk_ids,
        ) -> torch.Tensor:
            return fused_experts(hidden_states, w1, w2, weights, expert_ids)

        for repetition in range(args.warmup + args.repetitions):
            latency_ms = event_latency_ms(run)
            rows.append(
                {
                    "sample_id": sample_id,
                    "gpu_id": 0,
                    "expert_id": 0,
                    "token_count": token_count,
                    "latency_ms": f"{latency_ms:.9f}",
                    "warmup": int(repetition < args.warmup),
                    "repetition": repetition,
                }
            )
            sample_id += 1

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "expert_kernel_samples.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "created_at_unix": time.time(),
        "collector": "single_expert_vllm_fused_experts",
        "active_expert_count": 1,
        "grouped_execution": False,
        "tensor_parallel_size": 1,
        "seed": args.seed,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": str(dtype).removeprefix("torch."),
        "warmup_count": args.warmup,
        "measurement_iterations": args.repetitions,
        "token_counts": sorted(set(args.token_counts)),
        "gpu_model": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": version("vllm"),
    }
    (args.output / "expert_kernel_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"csv": str(csv_path), "metadata": metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
