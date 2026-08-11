#!/usr/bin/env python3
"""Measure a dispatch+combine NCCL all-to-all path on four local ranks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument(
        "--tokens-per-peer",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world_size != 4:
        raise RuntimeError(f"EP4 collector requires world_size=4, found {world_size}")
    if any(value <= 0 for value in args.tokens_per_peer):
        raise ValueError("tokens-per-peer values must be positive")

    local_rows: list[dict[str, object]] = []
    bytes_per_token = args.hidden_size * args.bytes_per_element
    sample_id = 0
    for tokens_per_peer in sorted(set(args.tokens_per_peer)):
        elements_per_peer = tokens_per_peer * args.hidden_size
        element_count = elements_per_peer * world_size
        outbound = torch.randn(element_count, device="cuda", dtype=torch.bfloat16)
        dispatched = torch.empty_like(outbound)
        combined = torch.empty_like(outbound)
        for repetition in range(args.warmup + args.repetitions):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_to_all_single(dispatched, outbound)
            dist.all_to_all_single(combined, dispatched)
            end.record()
            end.synchronize()
            local_rows.append(
                {
                    "sample_id": f"rank{rank}-{sample_id}",
                    "collective": "ep_dispatch_combine",
                    "active_ranks": world_size,
                    "message_count": 2 * (world_size - 1),
                    "transferred_bytes": 2
                    * (world_size - 1)
                    * tokens_per_peer
                    * bytes_per_token,
                    "latency_ms": f"{start.elapsed_time(end):.9f}",
                    "warmup": int(repetition < args.warmup),
                    "repetition": repetition,
                    "gpu_id": rank,
                    "non_empty_peers": world_size - 1,
                    "max_peer_bytes": tokens_per_peer * bytes_per_token,
                    "tokens_per_peer": tokens_per_peer,
                }
            )
            sample_id += 1

    gathered: list[list[dict[str, object]]] | None = (
        [None] * world_size if rank == 0 else None  # type: ignore[list-item]
    )
    dist.gather_object(local_rows, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        rows = [row for rank_rows in gathered for row in rank_rows]
        args.output.mkdir(parents=True, exist_ok=True)
        csv_path = args.output / "network_samples.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "created_at_unix": time.time(),
            "collector": "torch_nccl_all_to_all_single_dispatch_combine",
            "active_ranks": world_size,
            "tensor_parallel_size": 1,
            "data_parallel_size": 4,
            "expert_parallel_size": 4,
            "hidden_size": args.hidden_size,
            "bytes_per_element": args.bytes_per_element,
            "tokens_per_peer": sorted(set(args.tokens_per_peer)),
            "warmup_count": args.warmup,
            "measurement_iterations": args.repetitions,
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "topology": subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout,
        }
        (args.output / "network_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
