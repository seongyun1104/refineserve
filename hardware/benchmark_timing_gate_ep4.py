#!/usr/bin/env python3
"""Measure whether K-dependent EP data-plane cost is identifiable on EP=4.

This gate deliberately excludes online scheduling, dynamic count exchange, validation,
and metric extraction from its CUDA-event interval. It separates local execution,
NCCL launch/synchronization floor, and real payload cost before any scheduler matrix is
authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-positions", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--require-nccl-provenance", action="store_true")
    return parser.parse_args()


def balanced_routes(tokens: int, experts: int, top_k: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(tokens, device=device, dtype=torch.int64)
    routes = [((token_ids * top_k) + offset) % experts for offset in range(top_k)]
    return torch.stack(routes, dim=1)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    except ModuleNotFoundError as exc:
        raise RuntimeError("the H100 timing gate requires an installed vLLM build") from exc
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("timing gate requires exactly four ranks")
    if args.experts % world_size:
        raise ValueError("experts must divide evenly across ranks")
    if args.top_k != 2:
        raise ValueError("the controlled timing gate requires top-k=2")
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    recorded_nccl_env = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG", "NCCL_DEBUG_FILE")
    required_nccl_env = ("NCCL_DEBUG", "NCCL_DEBUG_FILE")
    missing_nccl_env = [key for key in required_nccl_env if not os.environ.get(key)]
    if args.require_nccl_provenance and missing_nccl_env:
        raise RuntimeError(f"missing required NCCL provenance: {missing_nccl_env}")
    experts_per_rank = args.experts // world_size
    torch.manual_seed(7000 + rank)
    weights = [
        (
            torch.empty(
                experts_per_rank,
                2 * args.intermediate_size,
                args.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            ).normal_(0.0, 0.02),
            torch.empty(
                experts_per_rank,
                args.hidden_size,
                args.intermediate_size,
                device=device,
                dtype=torch.bfloat16,
            ).normal_(0.0, 0.02),
        )
        for _ in range(args.layers)
    ]
    rows: list[dict[str, object]] = []
    for positions in sorted(set(args.active_positions)):
        tokens = args.batch_size * positions
        routes = balanced_routes(tokens, args.experts, args.top_k, device)
        flat_ids = routes.reshape(-1)
        destinations = flat_ids // experts_per_rank
        order = torch.argsort(destinations, stable=True)
        local_ids = (flat_ids[order] % experts_per_rank).to(torch.int32).contiguous()
        send_counts = [tokens * args.top_k // world_size] * world_size
        recv_counts = list(send_counts)
        if sum(send_counts) != tokens * args.top_k:
            raise ValueError("balanced route requires assignments divisible by world size")
        hidden_batches = [
            torch.randn(
                tokens,
                args.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(args.batches)
        ]
        expert_weights = torch.ones(
            (tokens * args.top_k, 1), device=device, dtype=torch.float32
        )
        minimal_counts = [1] * world_size
        minimal_bf16 = torch.ones(world_size, device=device, dtype=torch.bfloat16)
        minimal_i32 = torch.arange(world_size, device=device, dtype=torch.int32)
        recv_minimal_bf16 = torch.empty_like(minimal_bf16)
        recv_minimal_i32 = torch.empty_like(minimal_i32)
        combine_minimal = torch.empty_like(minimal_bf16)
        assignment_count = tokens * args.top_k
        recv_hidden_buffer = torch.empty(
            (assignment_count, args.hidden_size),
            device=device,
            dtype=torch.bfloat16,
        )
        recv_ids_buffer = torch.empty(
            assignment_count,
            device=device,
            dtype=torch.int32,
        )
        combined_buffer = torch.empty_like(recv_hidden_buffer)
        restored_buffer = torch.empty_like(recv_hidden_buffer)
        for repetition in range(args.warmup + args.repetitions):
            modes = ["local_copy", "nccl_minimal", "nccl_real"]
            offset = repetition % len(modes)
            modes = modes[offset:] + modes[:offset]
            for execution_index, mode in enumerate(modes):
                dist.barrier()
                run_start = torch.cuda.Event(enable_timing=True)
                run_end = torch.cuda.Event(enable_timing=True)
                stage_events: list[dict[str, torch.cuda.Event]] = [
                    {
                        name: torch.cuda.Event(enable_timing=True)
                        for name in (
                            "start",
                            "pack_end",
                            "dispatch_copy_end",
                            "dispatch_end",
                            "compute_end",
                            "combine_copy_end",
                            "transport_end",
                            "end",
                        )
                    }
                    for _ in range(args.batches * args.layers)
                ]
                final_hidden: list[torch.Tensor] = []
                run_start.record()
                for batch_index, batch_hidden in enumerate(hidden_batches):
                    hidden = batch_hidden
                    for layer in range(args.layers):
                        events = stage_events[batch_index * args.layers + layer]
                        layer_start = events["start"]
                        pack_end = events["pack_end"]
                        dispatch_copy_end = events["dispatch_copy_end"]
                        dispatch_end = events["dispatch_end"]
                        compute_end = events["compute_end"]
                        combine_copy_end = events["combine_copy_end"]
                        transport_end = events["transport_end"]
                        combine_end = events["end"]
                        layer_start.record()
                        expanded = (
                            hidden[:, None, :]
                            .expand(-1, args.top_k, -1)
                            .reshape(-1, args.hidden_size)
                        )
                        send_hidden = expanded[order].contiguous()
                        pack_end.record()
                        # Every mode performs identical shape-matched HBM copies. The
                        # real-NCCL mode deliberately retains these as control work so
                        # real - minimal isolates collective payload rather than
                        # subtracting local-copy bandwidth.
                        local_recv_hidden = send_hidden.clone()
                        local_recv_ids = local_ids.clone()
                        dispatch_copy_end.record()
                        if mode == "nccl_real":
                            dist.all_to_all_single(
                                recv_hidden_buffer,
                                send_hidden,
                                output_split_sizes=recv_counts,
                                input_split_sizes=send_counts,
                            )
                            dist.all_to_all_single(
                                recv_ids_buffer,
                                local_ids,
                                output_split_sizes=recv_counts,
                                input_split_sizes=send_counts,
                            )
                        elif mode == "nccl_minimal":
                            dist.all_to_all_single(
                                recv_minimal_bf16,
                                minimal_bf16,
                                output_split_sizes=minimal_counts,
                                input_split_sizes=minimal_counts,
                            )
                            dist.all_to_all_single(
                                recv_minimal_i32,
                                minimal_i32,
                                output_split_sizes=minimal_counts,
                                input_split_sizes=minimal_counts,
                            )
                        if mode == "nccl_real":
                            recv_hidden = recv_hidden_buffer
                            recv_ids = recv_ids_buffer
                        else:
                            recv_hidden = local_recv_hidden
                            recv_ids = local_recv_ids
                        dispatch_end.record()
                        expert_output = fused_experts(
                            recv_hidden,
                            weights[layer][0],
                            weights[layer][1],
                            expert_weights,
                            recv_ids[:, None],
                        )
                        compute_end.record()
                        local_combined = expert_output.clone()
                        combine_copy_end.record()
                        if mode == "nccl_real":
                            dist.all_to_all_single(
                                combined_buffer,
                                expert_output,
                                output_split_sizes=send_counts,
                                input_split_sizes=recv_counts,
                            )
                        elif mode == "nccl_minimal":
                            dist.all_to_all_single(
                                combine_minimal,
                                minimal_bf16,
                                output_split_sizes=minimal_counts,
                                input_split_sizes=minimal_counts,
                            )
                        combined = (
                            combined_buffer
                            if mode == "nccl_real"
                            else local_combined
                        )
                        transport_end.record()
                        restored = restored_buffer
                        restored[order] = combined
                        hidden = restored.reshape(
                            -1, args.top_k, args.hidden_size
                        ).mean(1)
                        combine_end.record()
                    final_hidden.append(hidden)
                run_end.record()
                run_end.synchronize()
                finite = all(
                    bool(torch.isfinite(value).all().item()) for value in final_hidden
                )
                if not finite:
                    raise RuntimeError("timing gate produced non-finite output")
                layer_ms = sum(
                    events["start"].elapsed_time(events["end"])
                    for events in stage_events
                )
                dispatch_ms = sum(
                    events["start"].elapsed_time(events["dispatch_end"])
                    for events in stage_events
                )
                compute_ms = sum(
                    events["dispatch_end"].elapsed_time(events["compute_end"])
                    for events in stage_events
                )
                combine_ms = sum(
                    events["compute_end"].elapsed_time(events["end"])
                    for events in stage_events
                )
                packing_ms = sum(
                    events["start"].elapsed_time(events["pack_end"])
                    for events in stage_events
                )
                local_copy_memory_ms = sum(
                    events["pack_end"].elapsed_time(events["dispatch_copy_end"])
                    + events["compute_end"].elapsed_time(
                        events["combine_copy_end"]
                    )
                    for events in stage_events
                )
                unpacking_ms = sum(
                    events["transport_end"].elapsed_time(events["end"])
                    for events in stage_events
                )
                rows.append(
                    {
                        "rank": rank,
                        "active_positions": positions,
                        "mode": mode,
                        "repetition": repetition,
                        "warmup": int(repetition < args.warmup),
                        "execution_index": execution_index,
                        "gpu_path_ms": run_start.elapsed_time(run_end),
                        "summed_layer_ms": layer_ms,
                        "dispatch_ms": dispatch_ms,
                        "expert_compute_ms": compute_ms,
                        "combine_ms": combine_ms,
                        "packing_ms": packing_ms,
                        "local_copy_memory_ms": local_copy_memory_ms,
                        "unpacking_ms": unpacking_ms,
                        "global_hidden_payload_bytes_per_layer": (
                            world_size
                            * tokens
                            * args.top_k
                            * args.hidden_size
                            * 2
                        ),
                        "communicated_payload_bytes_per_layer": (
                            0
                            if mode == "local_copy"
                            else (
                                world_size * world_size * (2 + 4 + 2)
                                if mode == "nccl_minimal"
                                else (
                                    world_size
                                    * tokens
                                    * args.top_k
                                    * (2 * args.hidden_size * 2 + 4)
                                )
                            )
                        ),
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / f"rank{rank}_timing_gate.csv", rows)
    dist.barrier()
    if rank == 0:
        fused_source = inspect.getsourcefile(fused_experts)
        fused_path = Path(fused_source).resolve() if fused_source else None
        config_hashes = {}
        if fused_path is not None:
            for config_path in fused_path.parent.rglob("*.json"):
                if "config" in str(config_path).lower():
                    config_hashes[str(config_path.resolve())] = hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest()
        metadata = {
            "measurement_protocol": "timing_identifiability_v2",
            "clock_preflight_required": True,
            "world_size": world_size,
            "layers": args.layers,
            "experts": args.experts,
            "top_k": args.top_k,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "batch_size": args.batch_size,
            "batches": args.batches,
            "active_positions": sorted(set(args.active_positions)),
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "validation_outside_cuda_interval": True,
            "dynamic_count_exchange_in_cuda_interval": False,
            "non_collective_work_symmetric_across_modes": True,
            "real_mode_retains_shape_matched_local_copy_control": True,
            "packing_and_unpacking_in_cuda_interval": True,
            "origin_slot_transmitted": False,
            "origin_restoration": "implicit_stable_assignment_order_plus_local_inverse",
            "same_rank_topk_hidden_deduplication": False,
            "data_plane_collectives_per_layer": {
                "hidden_dispatch": 1,
                "expert_id_dispatch": 1,
                "hidden_combine": 1,
            },
            "collective_api": "torch.distributed.all_to_all_single",
            "nccl_execution_path_verification": (
                "parse NCCL_DEBUG logs; NCCL_ALGO/PROTO are recorded but are not "
                "assumed to control grouped-P2P all-to-all behavior"
            ),
            "mode_rotation": "repetition_level_cyclic",
            "modes": ["local_copy", "nccl_minimal", "nccl_real"],
            "fused_moe_source_path": str(fused_path) if fused_path else None,
            "fused_moe_source_sha256": (
                hashlib.sha256(fused_path.read_bytes()).hexdigest()
                if fused_path is not None and fused_path.is_file()
                else None
            ),
            "fused_moe_config_hashes": config_hashes,
            "nccl_environment": {
                key: os.environ.get(key) for key in recorded_nccl_env
            },
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
        }
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps({"output": str(args.output), "status": "PASS"}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
