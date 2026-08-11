#!/usr/bin/env python3
"""Benchmark native position-parallel work on a four-rank MoE EP path."""

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
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--requests-per-rank", type=int, default=4)
    parser.add_argument(
        "--active-positions", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def routes(
    *,
    mode: str,
    rank: int,
    world_size: int,
    requests: int,
    positions: int,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    experts_per_rank = num_experts // world_size
    rows: list[list[int]] = []
    for request_id in range(requests):
        for position_id in range(positions):
            global_token = (rank * requests + request_id) * positions + position_id
            if mode == "uniform":
                selected = [
                    (global_token * top_k + slot) % num_experts for slot in range(top_k)
                ]
            elif mode == "request_correlated":
                offset = (rank * requests + request_id) % experts_per_rank
                selected = [
                    (slot % world_size) * experts_per_rank
                    + (offset + slot // world_size) % experts_per_rank
                    for slot in range(top_k)
                ]
            elif mode == "hot_rank":
                selected = [
                    0,
                    experts_per_rank,
                    2 * experts_per_rank,
                    3 * experts_per_rank,
                    1,
                    2,
                    3,
                    4,
                ][:top_k]
            else:
                raise ValueError(f"unknown routing mode: {mode}")
            if len(set(selected)) != top_k:
                raise RuntimeError("route generator produced duplicate experts")
            rows.append(selected)
    return torch.tensor(rows, device="cuda", dtype=torch.int64)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"expected four EP ranks, found {world_size}")
    if args.num_experts % world_size:
        raise ValueError("num_experts must be divisible by world size")
    if args.top_k > 8:
        raise ValueError("the frozen route modes support top_k <= 8")

    torch.manual_seed(args.seed + rank)
    experts_per_rank = args.num_experts // world_size
    w1 = torch.randn(
        experts_per_rank,
        2 * args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        experts_per_rank,
        args.hidden_size,
        args.intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    )

    local_rows: list[dict[str, object]] = []
    for mode in ("uniform", "request_correlated", "hot_rank"):
        for positions in sorted(set(args.active_positions)):
            token_count = args.requests_per_rank * positions
            hidden = torch.randn(
                token_count, args.hidden_size, device=device, dtype=torch.bfloat16
            )
            global_ids = routes(
                mode=mode,
                rank=rank,
                world_size=world_size,
                requests=args.requests_per_rank,
                positions=positions,
                num_experts=args.num_experts,
                top_k=args.top_k,
            )
            expanded = (
                hidden[:, None, :]
                .expand(-1, args.top_k, -1)
                .reshape(-1, args.hidden_size)
            )
            flat_ids = global_ids.reshape(-1)
            destinations = flat_ids // experts_per_rank
            order = torch.argsort(destinations, stable=True)
            send_hidden = expanded[order].contiguous()
            send_ids = (flat_ids[order] % experts_per_rank).to(torch.int32).contiguous()
            send_counts_tensor = torch.bincount(
                destinations, minlength=world_size
            ).to(torch.int64)
            gathered_counts = [torch.empty_like(send_counts_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_counts, send_counts_tensor)
            send_counts = [int(value) for value in send_counts_tensor.cpu().tolist()]
            recv_counts = [int(values[rank].item()) for values in gathered_counts]
            recv_count = sum(recv_counts)
            recv_hidden = torch.empty(
                recv_count, args.hidden_size, device=device, dtype=torch.bfloat16
            )
            recv_ids = torch.empty(recv_count, device=device, dtype=torch.int32)
            combined = torch.empty_like(send_hidden)
            expert_weights = torch.ones((recv_count, 1), device=device, dtype=torch.float32)

            for repetition in range(args.warmup + args.repetitions):
                dist.barrier()
                total_start = torch.cuda.Event(enable_timing=True)
                dispatch_end = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                total_start.record()
                dist.all_to_all_single(
                    recv_hidden,
                    send_hidden,
                    output_split_sizes=recv_counts,
                    input_split_sizes=send_counts,
                )
                dist.all_to_all_single(
                    recv_ids,
                    send_ids,
                    output_split_sizes=recv_counts,
                    input_split_sizes=send_counts,
                )
                dispatch_end.record()
                expert_output = fused_experts(
                    recv_hidden,
                    w1,
                    w2,
                    expert_weights,
                    recv_ids[:, None],
                )
                compute_end.record()
                dist.all_to_all_single(
                    combined,
                    expert_output,
                    output_split_sizes=send_counts,
                    input_split_sizes=recv_counts,
                )
                total_end.record()
                total_end.synchronize()

                expert_counts = torch.bincount(
                    recv_ids.to(torch.int64), minlength=experts_per_rank
                )
                active_counts = expert_counts[expert_counts > 0].to(torch.float32)
                cross_assignments = sum(
                    count for destination, count in enumerate(send_counts) if destination != rank
                )
                local_rows.append(
                    {
                        "rank": rank,
                        "routing_mode": mode,
                        "active_positions": positions,
                        "requests_per_rank": args.requests_per_rank,
                        "source_tokens": token_count,
                        "source_assignments": token_count * args.top_k,
                        "received_assignments": recv_count,
                        "active_local_experts": int(active_counts.numel()),
                        "mean_tokens_per_active_expert": float(active_counts.mean().item()),
                        "max_tokens_per_active_expert": int(active_counts.max().item()),
                        "cross_gpu_bytes": cross_assignments
                        * args.hidden_size
                        * 2
                        * 2,
                        "total_ms": total_start.elapsed_time(total_end),
                        "dispatch_ms": total_start.elapsed_time(dispatch_end),
                        "expert_compute_ms": dispatch_end.elapsed_time(compute_end),
                        "combine_ms": compute_end.elapsed_time(total_end),
                        "warmup": int(repetition < args.warmup),
                        "repetition": repetition,
                    }
                )

    gathered_rows: list[list[dict[str, object]]] | None = (
        [None] * world_size if rank == 0 else None  # type: ignore[list-item]
    )
    dist.gather_object(local_rows, gathered_rows, dst=0)
    if rank == 0:
        assert gathered_rows is not None
        rows = [row for rank_rows in gathered_rows for row in rank_rows]
        args.output.mkdir(parents=True, exist_ok=True)
        csv_path = args.output / "position_parallel_ep_samples.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "created_at_unix": time.time(),
            "execution_kind": "native_position_parallel_ep_prototype",
            "full_diffusion_model": False,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 4,
            "world_size": world_size,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_experts": args.num_experts,
            "experts_per_rank": experts_per_rank,
            "top_k": args.top_k,
            "requests_per_rank": args.requests_per_rank,
            "active_positions": sorted(set(args.active_positions)),
            "routing_modes": ["uniform", "request_correlated", "hot_rank"],
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
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps({"csv": str(csv_path), "rows": len(rows)}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
