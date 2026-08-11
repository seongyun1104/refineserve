#!/usr/bin/env python3
"""Validate that the planner load objective maps to EP data-plane time on EP=4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from coordinated_scheduling import plan_cost
from proxy_validation_contract import (
    ARMS,
    BALANCED_PLAN,
    FIFO_PLAN,
    PREFERENCES,
    constructed_split_contract,
    global_plans,
    validate_split_contract,
)

CONTROL_ARMS = ("fifo_local_copy_control", "fifo_nccl_minimal_control")
RUN_ARMS = (*CONTROL_ARMS, *ARMS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-positions", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--require-nccl-provenance", action="store_true")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def request_routes(
    *,
    positions: int,
    experts_per_rank: int,
    device: torch.device,
) -> torch.Tensor:
    routes = torch.empty((8, positions, 2), device=device, dtype=torch.int64)
    for request, preferred_rank in enumerate(PREFERENCES):
        routes[request, :, 0] = preferred_rank * experts_per_rank
        routes[request, :, 1] = (
            ((preferred_rank + 1) % 4) * experts_per_rank + 1
        )
    return routes


def objective_values(
    *,
    positions: int,
    layers: int,
    experts: int,
    experts_per_rank: int,
) -> dict[str, float]:
    counts = np.zeros((4, 8, layers, experts), dtype=np.int64)
    for source in range(4):
        for request, preferred_rank in enumerate(PREFERENCES):
            counts[source, request, :, preferred_rank * experts_per_rank] = positions
            counts[
                source,
                request,
                :,
                ((preferred_rank + 1) % 4) * experts_per_rank + 1,
            ] = positions
    return {
        arm: plan_cost(counts, plan, experts_per_rank)
        for arm, plan in global_plans().items()
    }


def main() -> None:
    args = parse_args()
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    except ModuleNotFoundError as exc:
        raise RuntimeError("the H100 proxy gate requires an installed vLLM build") from exc
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("proxy validation requires exactly four ranks")
    if (args.layers, args.experts, args.top_k, args.hidden_size) != (8, 16, 2, 2048):
        raise RuntimeError("proxy validation must preserve the controlled EP contract")
    if args.experts % world_size:
        raise ValueError("experts must divide evenly across ranks")
    if args.active_positions <= 0:
        raise ValueError("active positions must be positive")
    required_nccl_env = ("NCCL_DEBUG", "NCCL_DEBUG_FILE")
    if args.require_nccl_provenance:
        missing = [key for key in required_nccl_env if not os.environ.get(key)]
        if missing:
            raise RuntimeError(f"missing required NCCL provenance: {missing}")
    experts_per_rank = args.experts // world_size
    objectives = objective_values(
        positions=args.active_positions,
        layers=args.layers,
        experts=args.experts,
        experts_per_rank=experts_per_rank,
    )
    fifo_objective = objectives["fifo_constructed"]
    objective_reductions = {
        arm: (fifo_objective - objective) / fifo_objective
        for arm, objective in objectives.items()
    }
    if not np.isclose(
        objective_reductions["dose_083_constructed"], 1.0 / 12.0
    ):
        raise RuntimeError("low-dose proxy cell must reduce the objective by one twelfth")
    if not np.isclose(objective_reductions["balanced_constructed"], 1.0 / 3.0):
        raise RuntimeError("constructed proxy cell must reduce the objective by one third")
    torch.manual_seed(8100 + rank)
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
    routes = request_routes(
        positions=args.active_positions,
        experts_per_rank=experts_per_rank,
        device=device,
    )
    split_contract = constructed_split_contract(
        positions=args.active_positions,
        world_size=world_size,
    )
    validate_split_contract(split_contract, positions=args.active_positions)
    request_hidden = torch.randn(
        8,
        args.active_positions,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    minimal_counts = [1] * world_size
    minimal_bf16 = torch.ones(world_size, device=device, dtype=torch.bfloat16)
    minimal_i32 = torch.arange(world_size, device=device, dtype=torch.int32)
    recv_minimal_bf16 = torch.empty_like(minimal_bf16)
    recv_minimal_i32 = torch.empty_like(minimal_i32)
    combine_minimal = torch.empty_like(minimal_bf16)
    rows: list[dict[str, object]] = []
    local_plans = {
        "fifo_constructed": FIFO_PLAN,
        "dose_083_constructed": BALANCED_PLAN if rank == 0 else FIFO_PLAN,
        "balanced_constructed": BALANCED_PLAN,
    }
    # Validate the static split contract once, outside every CUDA timing interval.
    for arm, plan in local_plans.items():
        for batch_index, selected_requests in enumerate(plan):
            selected_routes = routes[list(selected_requests)].reshape(-1, args.top_k)
            actual = torch.bincount(
                selected_routes.reshape(-1) // experts_per_rank,
                minlength=world_size,
            ).cpu().tolist()
            expected = split_contract[arm][rank][batch_index]
            if actual != expected:
                raise RuntimeError(
                    f"static send split mismatch for {arm}, rank {rank}, "
                    f"batch {batch_index}: {actual} != {expected}"
                )
    shape_indexes: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    scratch: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    send_total = 4 * args.active_positions * args.top_k
    all_plans = global_plans()
    for arm, plan in local_plans.items():
        for batch_index in range(len(plan)):
            recv_total = sum(
                split_contract[arm][source][batch_index][rank]
                for source in range(world_size)
            )
            shape_indexes[(arm, batch_index)] = (
                torch.arange(recv_total, device=device) % send_total,
                torch.arange(send_total, device=device) % recv_total,
            )
            recv_id_parts = []
            for source in range(world_size):
                source_requests = list(all_plans[arm][source][batch_index])
                source_flat_ids = routes[source_requests].reshape(-1)
                source_destinations = source_flat_ids // experts_per_rank
                recv_id_parts.append(
                    (source_flat_ids[source_destinations == rank] % experts_per_rank)
                    .to(torch.int32)
                    .contiguous()
                )
            expected_recv_ids = torch.cat(recv_id_parts)
            if expected_recv_ids.numel() != recv_total:
                raise RuntimeError("constructed receive-ID contract has wrong length")
            scratch[(arm, batch_index)] = {
                "recv_hidden": torch.empty(
                    (recv_total, args.hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                ),
                "recv_ids": torch.empty(
                    recv_total,
                    device=device,
                    dtype=torch.int32,
                ),
                "combined": torch.empty(
                    (send_total, args.hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                ),
                "restored": torch.empty(
                    (send_total, args.hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                ),
                "expert_weights": torch.ones(
                    (recv_total, 1),
                    device=device,
                    dtype=torch.float32,
                ),
                "local_recv_ids": expected_recv_ids,
            }
    for repetition in range(args.warmup + args.repetitions):
        arm_order = list(RUN_ARMS)
        offset = repetition % len(arm_order)
        arm_order = arm_order[offset:] + arm_order[:offset]
        for execution_index, run_arm in enumerate(arm_order):
            composition_arm = (
                "fifo_constructed" if run_arm in CONTROL_ARMS else run_arm
            )
            transport_mode = {
                "fifo_local_copy_control": "local_copy",
                "fifo_nccl_minimal_control": "nccl_minimal",
            }.get(run_arm, "nccl_real")
            plan = local_plans[composition_arm]
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dispatch_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            combine_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            stage_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(len(plan) * args.layers)
            ]
            outputs: list[torch.Tensor] = []
            start.record()
            for batch_index, selected_requests in enumerate(plan):
                request_indexes = list(selected_requests)
                hidden = request_hidden[request_indexes].reshape(-1, args.hidden_size)
                selected_routes = routes[request_indexes].reshape(-1, args.top_k)
                if selected_routes.numel() != (
                    4 * args.active_positions * args.top_k
                ):
                    raise RuntimeError("constructed arms must keep expert-ID length fixed")
                for layer in range(args.layers):
                    (
                        dispatch_start,
                        dispatch_end,
                        compute_end,
                        combine_end,
                    ) = stage_events[batch_index * args.layers + layer]
                    flat_ids = selected_routes.reshape(-1)
                    destinations = flat_ids // experts_per_rank
                    order = torch.argsort(destinations, stable=True)
                    send_counts = split_contract[composition_arm][rank][batch_index]
                    recv_counts = [
                        split_contract[composition_arm][source][batch_index][rank]
                        for source in range(world_size)
                    ]
                    expanded = (
                        hidden[:, None, :]
                        .expand(-1, args.top_k, -1)
                        .reshape(-1, args.hidden_size)
                    )
                    send_hidden = expanded[order].contiguous()
                    send_ids = (
                        flat_ids[order] % experts_per_rank
                    ).to(torch.int32).contiguous()
                    buffers = scratch[(composition_arm, batch_index)]
                    recv_total = sum(recv_counts)
                    if buffers["recv_hidden"].shape[0] != recv_total:
                        raise RuntimeError("preallocated receive buffer has wrong length")
                    recv_hidden_buffer = buffers["recv_hidden"]
                    recv_ids_buffer = buffers["recv_ids"]
                    combined_buffer = buffers["combined"]
                    recv_shape_index, combine_shape_index = shape_indexes[
                        (composition_arm, batch_index)
                    ]
                    local_recv_hidden = send_hidden[recv_shape_index].contiguous()
                    local_recv_ids = buffers["local_recv_ids"].clone()
                    dispatch_start.record()
                    if transport_mode == "nccl_real":
                        dist.all_to_all_single(
                            recv_hidden_buffer,
                            send_hidden,
                            output_split_sizes=recv_counts,
                            input_split_sizes=send_counts,
                        )
                        dist.all_to_all_single(
                            recv_ids_buffer,
                            send_ids,
                            output_split_sizes=recv_counts,
                            input_split_sizes=send_counts,
                        )
                        recv_hidden = recv_hidden_buffer
                        recv_ids = recv_ids_buffer
                    elif transport_mode == "nccl_minimal":
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
                        recv_hidden = local_recv_hidden
                        recv_ids = local_recv_ids
                    else:
                        recv_hidden = local_recv_hidden
                        recv_ids = local_recv_ids
                    dispatch_end.record()
                    expert_output = fused_experts(
                        recv_hidden,
                        weights[layer][0],
                        weights[layer][1],
                        buffers["expert_weights"],
                        recv_ids[:, None],
                    )
                    compute_end.record()
                    local_combined = expert_output[combine_shape_index].contiguous()
                    if transport_mode == "nccl_real":
                        dist.all_to_all_single(
                            combined_buffer,
                            expert_output,
                            output_split_sizes=send_counts,
                            input_split_sizes=recv_counts,
                        )
                        combined = combined_buffer
                    elif transport_mode == "nccl_minimal":
                        dist.all_to_all_single(
                            combine_minimal,
                            minimal_bf16,
                            output_split_sizes=minimal_counts,
                            input_split_sizes=minimal_counts,
                        )
                        combined = local_combined
                    else:
                        combined = local_combined
                    restored = buffers["restored"]
                    restored[order] = combined
                    hidden = restored.reshape(
                        -1, args.top_k, args.hidden_size
                    ).mean(1)
                    combine_end.record()
                    dispatch_events.append((dispatch_start, dispatch_end))
                    compute_events.append((dispatch_end, compute_end))
                    combine_events.append((compute_end, combine_end))
                outputs.append(hidden)
            end.record()
            end.synchronize()
            if not all(bool(torch.isfinite(value).all().item()) for value in outputs):
                raise RuntimeError("proxy validation produced non-finite output")
            objective = objectives[composition_arm]
            rows.append(
                {
                    "rank": rank,
                    "arm": run_arm,
                    "composition_arm": composition_arm,
                    "transport_mode": transport_mode,
                    "active_positions": args.active_positions,
                    "repetition": repetition,
                    "warmup": int(repetition < args.warmup),
                    "execution_index": execution_index,
                    "planner_objective": objective,
                    "objective_reduction_fraction": objective_reductions[
                        composition_arm
                    ],
                    "expert_id_elements_per_layer": (
                        4 * args.active_positions * args.top_k
                    ),
                    "send_split_vectors": json.dumps(
                        split_contract[composition_arm][rank]
                    ),
                    "recv_split_vectors": json.dumps(
                        [
                            [
                                split_contract[composition_arm][source][batch_index][
                                    rank
                                ]
                                for source in range(world_size)
                            ]
                            for batch_index in range(len(plan))
                        ]
                    ),
                    "gpu_path_ms": start.elapsed_time(end),
                    "dispatch_ms": sum(a.elapsed_time(b) for a, b in dispatch_events),
                    "expert_compute_ms": sum(a.elapsed_time(b) for a, b in compute_events),
                    "combine_ms": sum(a.elapsed_time(b) for a, b in combine_events),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / f"rank{rank}_proxy_validation.csv", rows)
    dist.barrier()
    if rank == 0:
        source = inspect.getsourcefile(fused_experts)
        source_path = Path(source).resolve() if source else None
        metadata = {
            "measurement_protocol": (
                "constructed_objective_to_time_proxy_v3_two_dose_local_minimal_real"
            ),
            "world_size": world_size,
            "layers": args.layers,
            "experts": args.experts,
            "top_k": args.top_k,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "active_positions": args.active_positions,
            "requests_per_rank": 8,
            "batch_size": 4,
            "batches": 2,
            "fifo_objective": fifo_objective,
            "objectives": objectives,
            "constructed_objective_reduction_fractions": objective_reductions,
            "constructed_route_preferences": list(PREFERENCES),
            "global_plans": global_plans(),
            "global_split_contract": split_contract,
            "run_arms": list(RUN_ARMS),
            "transport_controls": {
                "fifo_local_copy_control": "FIFO composition, local-copy transport",
                "fifo_nccl_minimal_control": (
                    "FIFO composition, three minimal-payload collectives per layer"
                ),
                "fifo_constructed": (
                    "FIFO composition, three full-payload collectives per layer"
                ),
            },
            "route_semantics": (
                "each request routes one assignment to its preferred rank and one "
                "to the next rank; every rank remains non-empty in every plan"
            ),
            "payload_invariant": (
                "every arm executes four requests per local batch, top-2 assignments, "
                "and the same int32 expert-ID element count; only split distributions "
                "and cross-source batch alignment change"
            ),
            "dynamic_count_exchange_in_cuda_interval": False,
            "split_source": "preregistered global constructed plan",
            "non_collective_work_symmetric_across_transport_modes": True,
            "nccl_operation": "torch.distributed.all_to_all_single",
            "nccl_path_claim": (
                "NCCL_DEBUG logs are authoritative; NCCL_ALGO/PROTO are provenance "
                "only and are not assumed to select grouped-P2P all-to-all behavior"
            ),
            "nccl_environment": {
                key: os.environ.get(key)
                for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG", "NCCL_DEBUG_FILE")
            },
            "fused_moe_source_sha256": (
                hashlib.sha256(source_path.read_bytes()).hexdigest()
                if source_path is not None and source_path.is_file()
                else None
            ),
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
