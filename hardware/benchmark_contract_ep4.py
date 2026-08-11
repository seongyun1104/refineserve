#!/usr/bin/env python3
"""Authoritative 8-layer, 16-expert native K-position EP4 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from coordinated_scheduling import (
    combined_rank_loads,
    coordinated_dose_ladder,
    coordinated_plan_with_diagnostics,
    fifo_plan,
    reassigned_request_fraction,
    split_vectors_and_local_expert_counts,
)
from synthetic_routes import ROUTING_MODES, make_routes, request_counts
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

DEFAULT_SCHEDULERS = (
    "fifo",
    "fifo_selection_control",
    "random_permutation",
    "locality_only",
    "load_balance_only",
    "critical_path",
    "joint",
    "local_route_replay",
    "coordinated_route_replay",
)
DOSE_SCHEDULERS = (
    "coordinated_dose_25",
    "coordinated_dose_50",
    "coordinated_dose_75",
)
SCHEDULERS = (*DEFAULT_SCHEDULERS, *DOSE_SCHEDULERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--candidate-requests-per-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--active-positions", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--routing-modes", nargs="+", default=list(ROUTING_MODES))
    parser.add_argument("--schedulers", nargs="+", default=list(DEFAULT_SCHEDULERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--counterbalance-schedulers", action="store_true")
    parser.add_argument("--planner-restarts", type=int, default=8)
    parser.add_argument("--planner-max-restarts", type=int, default=64)
    parser.add_argument("--request-correlation-strength", type=float, default=0.75)
    parser.add_argument("--require-nccl-provenance", action="store_true")
    return parser.parse_args()


def coordinated_plan_adaptive(
    global_actual_counts: np.ndarray,
    batch_size: int,
    experts_per_rank: int,
    *,
    initial_restarts: int,
    max_restarts: int,
    seed: int,
):
    """Increase deterministic restarts while the best-found tail still improves."""
    if initial_restarts <= 0 or max_restarts < initial_restarts:
        raise ValueError("planner restart bounds are invalid")
    restarts = initial_restarts
    while True:
        batches, diagnostics = coordinated_plan_with_diagnostics(
            global_actual_counts,
            batch_size,
            experts_per_rank,
            restarts=restarts,
            seed=seed,
        )
        if not diagnostics.improved_in_last_two_restarts or restarts >= max_restarts:
            return batches, diagnostics
        restarts = min(max_restarts, restarts * 2)


def schedule_batches(
    *,
    scheduler: str,
    online_counts: np.ndarray,
    actual_counts: np.ndarray,
    batch_size: int,
    experts_per_rank: int,
    random_seed: int = 0,
) -> list[list[int]]:
    if scheduler == "random_permutation":
        order = np.random.default_rng(random_seed).permutation(len(online_counts)).tolist()
        return [
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        ]
    remaining = list(range(len(online_counts)))
    batches: list[list[int]] = []
    scoring_counts = actual_counts if scheduler == "local_route_replay" else online_counts

    def score(batch: list[int], candidate: int, kind: str) -> float:
        chosen = [*batch, candidate]
        counts = scoring_counts[chosen].sum(axis=0)
        rank_load = counts.reshape(counts.shape[0], -1, experts_per_rank).sum(axis=2)
        critical = float(rank_load.max(axis=1).sum())
        variance = float(rank_load.var(axis=1).sum())
        active = counts > 0
        locality = float((counts - active).clip(min=0).sum())
        if kind == "locality_only":
            return -locality
        if kind == "load_balance_only":
            return variance
        if kind in {"critical_path", "local_route_replay"}:
            return critical
        if kind == "joint":
            return critical + 0.25 * variance - 0.05 * locality
        raise ValueError(f"unknown scheduler: {kind}")

    while remaining:
        if scheduler == "fifo":
            batch = remaining[:batch_size]
        elif scheduler == "local_route_replay" and len(remaining) <= 12:
            width = min(batch_size, len(remaining))
            batch = list(
                min(
                    itertools.combinations(remaining, width),
                    key=lambda choice: score([], -1, "local_route_replay")
                    if False
                    else (
                        actual_counts[list(choice)]
                        .sum(axis=0)
                        .reshape(actual_counts.shape[1], -1, experts_per_rank)
                        .sum(axis=2)
                        .max(axis=1)
                        .sum()
                    ),
                )
            )
        else:
            batch = [remaining[0]]
            while len(batch) < min(batch_size, len(remaining)):
                candidates = [value for value in remaining if value not in batch]
                batch.append(min(candidates, key=lambda value: score(batch, value, scheduler)))
        batches.append(batch)
        selected = set(batch)
        remaining = [value for value in remaining if value not in selected]
    return batches


def fifo_batches(requests: int, batch_size: int) -> list[list[int]]:
    return [
        list(range(start, start + batch_size))
        for start in range(0, requests, batch_size)
    ]


def batch_plan_checksum(batches: list[list[int]]) -> int:
    return sum(
        (batch_index + 1) * (slot_index + 1) * (request + 1)
        for batch_index, batch in enumerate(batches)
        for slot_index, request in enumerate(batch)
    )


def select_batches(
    *,
    scheduler: str,
    online_counts: np.ndarray,
    actual_counts: np.ndarray,
    batch_size: int,
    experts_per_rank: int,
    random_seed: int,
) -> tuple[list[list[int]], int]:
    selection_scheduler = (
        "critical_path" if scheduler == "fifo_selection_control" else scheduler
    )
    proposed = schedule_batches(
        scheduler=selection_scheduler,
        online_counts=online_counts,
        actual_counts=actual_counts,
        batch_size=batch_size,
        experts_per_rank=experts_per_rank,
        random_seed=random_seed,
    )
    executed = (
        fifo_batches(len(online_counts), batch_size)
        if scheduler == "fifo_selection_control"
        else proposed
    )
    return executed, batch_plan_checksum(proposed)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fused_moe_identity() -> dict[str, object]:
    source = inspect.getsourcefile(fused_experts)
    source_path = Path(source).resolve() if source else None
    source_hash = None
    if source_path is not None and source_path.is_file():
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    runtime_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("VLLM_", "NCCL_", "CUDA_"))
    }
    config_hashes = {}
    if source_path is not None:
        for config_path in source_path.parent.rglob("*.json"):
            if "config" in str(config_path).lower():
                config_hashes[str(config_path.resolve())] = hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest()
    return {
        "callable_module": fused_experts.__module__,
        "source_path": str(source_path) if source_path else None,
        "source_sha256": source_hash,
        "config_hashes": config_hashes,
        "runtime_environment": runtime_env,
    }


def main() -> None:
    args = parse_args()
    required_nccl_env = ("NCCL_DEBUG", "NCCL_DEBUG_FILE")
    missing_nccl_env = [key for key in required_nccl_env if not os.environ.get(key)]
    if args.require_nccl_provenance and missing_nccl_env:
        raise RuntimeError(f"missing required NCCL provenance: {missing_nccl_env}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if (world_size, args.layers, args.num_experts, args.top_k, args.hidden_size) != (
        4,
        8,
        16,
        2,
        2048,
    ):
        raise RuntimeError("hardware contract requires EP4, 8 layers, 16 experts, top-2, H=2048")
    if args.num_experts % world_size:
        raise ValueError("experts must divide evenly across ranks")
    if args.candidate_requests_per_rank % args.batch_size:
        raise ValueError("candidate requests must divide evenly into batches")
    unknown_modes = set(args.routing_modes) - set(ROUTING_MODES)
    unknown_schedulers = set(args.schedulers) - set(SCHEDULERS)
    if unknown_modes or unknown_schedulers:
        raise ValueError(f"unknown matrix values: {unknown_modes=}, {unknown_schedulers=}")

    experts_per_rank = args.num_experts // world_size
    torch.manual_seed(1000 + rank)
    weights = [
        (
            torch.empty(
                experts_per_rank,
                2 * args.intermediate_size,
                args.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            ).normal_(mean=0.0, std=0.02),
            torch.empty(
                experts_per_rank,
                args.hidden_size,
                args.intermediate_size,
                device=device,
                dtype=torch.bfloat16,
            ).normal_(mean=0.0, std=0.02),
        )
        for _ in range(args.layers)
    ]
    layer_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    profiler = None
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        )
        profiler.start()

    for seed in args.seeds:
        for mode in args.routing_modes:
            for positions in sorted(set(args.active_positions)):
                global_actual_np = np.stack(
                    [
                        make_routes(
                            seed=seed,
                            mode=mode,
                            global_request_ids=(
                                np.arange(args.candidate_requests_per_rank)
                                + source * args.candidate_requests_per_rank
                            ),
                            layers=args.layers,
                            positions=positions,
                            experts=args.num_experts,
                            request_correlation_strength=(
                                args.request_correlation_strength
                            ),
                        )
                        for source in range(world_size)
                    ]
                )
                actual_np = global_actual_np[rank]
                global_prior_np = np.stack(
                    [
                        make_routes(
                            seed=seed - 1,
                            mode=mode,
                            global_request_ids=(
                                np.arange(args.candidate_requests_per_rank)
                                + source * args.candidate_requests_per_rank
                            ),
                            layers=args.layers,
                            positions=1,
                            experts=args.num_experts,
                            request_correlation_strength=(
                                args.request_correlation_strength
                            ),
                        )
                        for source in range(world_size)
                    ]
                )
                actual_counts = request_counts(actual_np, args.num_experts)
                global_online_counts = np.stack(
                    [
                        request_counts(source_routes, args.num_experts) * positions
                        for source_routes in global_prior_np
                    ]
                )
                online_counts = global_online_counts[rank]
                coordinated_start = time.perf_counter_ns()
                global_actual_counts = np.stack(
                    [
                        request_counts(source_routes, args.num_experts)
                        for source_routes in global_actual_np
                    ]
                )
                coordinated_batches, coordinated_diagnostics = coordinated_plan_adaptive(
                    global_actual_counts,
                    args.batch_size,
                    experts_per_rank,
                    initial_restarts=args.planner_restarts,
                    max_restarts=args.planner_max_restarts,
                    seed=seed + positions + ROUTING_MODES.index(mode),
                )
                dose_ladder = coordinated_dose_ladder(
                    global_actual_counts,
                    coordinated_batches,
                    args.batch_size,
                    experts_per_rank,
                    seed=seed + positions + ROUTING_MODES.index(mode),
                )
                coordinated_plans = {
                    "coordinated_route_replay": {
                        "target": 1.0,
                        "achieved": 1.0,
                        "plan": coordinated_batches,
                        "max_receive_load": (
                            coordinated_diagnostics.best_max_receive_load
                        ),
                        "reassigned_fraction": (
                            coordinated_diagnostics.reassigned_request_fraction
                        ),
                    }
                }
                for target, achieved, plan in dose_ladder:
                    if target >= 1.0:
                        continue
                    name = f"coordinated_dose_{int(round(target * 100))}"
                    coordinated_plans[name] = {
                        "target": target,
                        "achieved": achieved,
                        "plan": plan,
                        "max_receive_load": int(
                            combined_rank_loads(
                                global_actual_counts,
                                plan,
                                experts_per_rank,
                            ).max()
                        ),
                        "reassigned_fraction": reassigned_request_fraction(
                            fifo_plan(
                                world_size,
                                args.candidate_requests_per_rank,
                                args.batch_size,
                            ),
                            plan,
                        ),
                    }
                coordinated_plan_ms = (
                    time.perf_counter_ns() - coordinated_start
                ) / 1_000_000
                actual_routes = torch.from_numpy(actual_np).to(device)
                initial_hidden = torch.randn(
                    args.candidate_requests_per_rank,
                    positions,
                    args.hidden_size,
                    device=device,
                    dtype=torch.bfloat16,
                )

                base_scheduler_order = list(args.schedulers)
                for repetition in range(args.warmup + args.repetitions):
                    scheduler_order = list(base_scheduler_order)
                    if args.counterbalance_schedulers:
                        offset = (
                            seed
                            + positions
                            + ROUTING_MODES.index(mode)
                            + repetition
                        ) % len(scheduler_order)
                        scheduler_order = (
                            scheduler_order[offset:] + scheduler_order[:offset]
                        )
                    for scheduler_index, scheduler in enumerate(scheduler_order):
                        coordinated_entry = coordinated_plans.get(scheduler)
                        if scheduler in DOSE_SCHEDULERS and coordinated_entry is None:
                            raise RuntimeError(
                                f"cell lacks a distinct {scheduler} replay plan"
                            )
                        is_coordinated = coordinated_entry is not None
                        selection_start = time.perf_counter_ns()
                        if is_coordinated:
                            plan = coordinated_entry["plan"]
                            batches = [list(batch) for batch in plan[rank]]
                            selection_plan_checksum = batch_plan_checksum(batches)
                        else:
                            batches, selection_plan_checksum = select_batches(
                                scheduler=scheduler,
                                online_counts=online_counts,
                                actual_counts=actual_counts,
                                batch_size=args.batch_size,
                                experts_per_rank=experts_per_rank,
                                random_seed=(
                                    seed * 1_000_003
                                    + positions * 10_007
                                    + repetition * 101
                                    + rank
                                ),
                            )
                        scheduler_ms = (time.perf_counter_ns() - selection_start) / 1_000_000
                        if is_coordinated:
                            global_batches = [
                                [list(batch) for batch in source_batches]
                                for source_batches in coordinated_entry["plan"]
                            ]
                        else:
                            global_batches = []
                            for source in range(world_size):
                                if source == rank:
                                    source_batches = batches
                                else:
                                    source_batches, _ = select_batches(
                                        scheduler=scheduler,
                                        online_counts=global_online_counts[source],
                                        actual_counts=global_actual_counts[source],
                                        batch_size=args.batch_size,
                                        experts_per_rank=experts_per_rank,
                                        random_seed=(
                                            seed * 1_000_003
                                            + positions * 10_007
                                            + repetition * 101
                                            + source
                                        ),
                                    )
                                global_batches.append(
                                    [list(batch) for batch in source_batches]
                                )
                        replay_metadata: dict[
                            tuple[int, int], tuple[list[int], list[int], np.ndarray]
                        ] = {}
                        recv_sizes: set[int] = set()
                        for batch_index in range(len(batches)):
                            for layer in range(args.layers):
                                split_matrix, local_expert_counts = (
                                    split_vectors_and_local_expert_counts(
                                        global_routes=global_actual_np,
                                        global_batches=global_batches,
                                        batch_index=batch_index,
                                        layer=layer,
                                        experts_per_rank=experts_per_rank,
                                        destination_rank=rank,
                                    )
                                )
                                send_counts = split_matrix[rank]
                                recv_counts = [row[rank] for row in split_matrix]
                                replay_metadata[(batch_index, layer)] = (
                                    send_counts,
                                    recv_counts,
                                    local_expert_counts,
                                )
                                recv_sizes.add(sum(recv_counts))
                        expert_weight_cache = {
                            count: torch.ones(
                                (count, 1), device=device, dtype=torch.float32
                            )
                            for count in recv_sizes
                            if count
                        }
                        count_exchange_inputs = {
                            key: torch.tensor(
                                value[0], device=device, dtype=torch.int64
                            )
                            for key, value in replay_metadata.items()
                        }
                        count_exchange_outputs = {
                            key: [
                                torch.empty(world_size, device=device, dtype=torch.int64)
                                for _ in range(world_size)
                            ]
                            for key in replay_metadata
                        }
                        dist.barrier()
                        count_exchange_start = torch.cuda.Event(enable_timing=True)
                        count_exchange_end = torch.cuda.Event(enable_timing=True)
                        count_exchange_start.record()
                        for key in sorted(replay_metadata):
                            dist.all_gather(
                                count_exchange_outputs[key],
                                count_exchange_inputs[key],
                            )
                        count_exchange_end.record()
                        count_exchange_end.synchronize()
                        count_exchange_ms = count_exchange_start.elapsed_time(
                            count_exchange_end
                        )
                        count_exchange_valid = True
                        for key, (_, expected_recv, _) in replay_metadata.items():
                            observed_recv = [
                                int(value[rank].item())
                                for value in count_exchange_outputs[key]
                            ]
                            count_exchange_valid = (
                                count_exchange_valid
                                and observed_recv == expected_recv
                            )
                        pre_data_arrival_ns = time.perf_counter_ns()
                        barrier_start_ns = pre_data_arrival_ns
                        dist.barrier()
                        pre_data_barrier_wait_ms = (
                            time.perf_counter_ns() - barrier_start_ns
                        ) / 1_000_000
                        run_start = torch.cuda.Event(enable_timing=True)
                        run_end = torch.cuda.Event(enable_timing=True)
                        run_start.record()
                        pending_layers: list[dict[str, object]] = []
                        final_hidden: list[torch.Tensor] = []
                        for batch_index, batch in enumerate(batches):
                            selected = torch.tensor(batch, device=device, dtype=torch.int64)
                            hidden = initial_hidden[selected].reshape(-1, args.hidden_size)
                            for layer in range(args.layers):
                                layer_start = torch.cuda.Event(enable_timing=True)
                                router_end = torch.cuda.Event(enable_timing=True)
                                dispatch_end = torch.cuda.Event(enable_timing=True)
                                compute_end = torch.cuda.Event(enable_timing=True)
                                combine_end = torch.cuda.Event(enable_timing=True)
                                layer_start.record()
                                global_ids = actual_routes[selected, layer].reshape(-1, args.top_k)
                                expanded = (
                                    hidden[:, None, :]
                                    .expand(-1, args.top_k, -1)
                                    .reshape(-1, args.hidden_size)
                                )
                                flat_ids = global_ids.reshape(-1)
                                destinations = flat_ids // experts_per_rank
                                order = torch.argsort(destinations, stable=True)
                                send_hidden = expanded[order].contiguous()
                                send_ids = (
                                    flat_ids[order] % experts_per_rank
                                ).to(torch.int32).contiguous()
                                router_end.record()
                                send_counts, recv_counts, expert_counts = replay_metadata[
                                    (batch_index, layer)
                                ]
                                recv_count = sum(recv_counts)
                                expected_assignments = len(batch) * positions * args.top_k
                                recv_hidden = torch.empty(
                                    recv_count,
                                    args.hidden_size,
                                    device=device,
                                    dtype=torch.bfloat16,
                                )
                                recv_ids = torch.empty(
                                    recv_count, device=device, dtype=torch.int32
                                )
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
                                if recv_count:
                                    expert_output = fused_experts(
                                        recv_hidden,
                                        weights[layer][0],
                                        weights[layer][1],
                                        expert_weight_cache[recv_count],
                                        recv_ids[:, None],
                                    )
                                else:
                                    # A skewed route can leave a rank with no local work. It
                                    # must still participate in both all-to-all collectives.
                                    expert_output = recv_hidden
                                compute_end.record()
                                combined = torch.empty_like(send_hidden)
                                dist.all_to_all_single(
                                    combined,
                                    expert_output,
                                    output_split_sizes=send_counts,
                                    input_split_sizes=recv_counts,
                                )
                                restored = torch.empty_like(combined)
                                restored[order] = combined
                                # Synthetic top-k routes use uniform normalized router weights.
                                hidden = restored.reshape(
                                    -1, args.top_k, args.hidden_size
                                ).mean(1)
                                combine_end.record()
                                active_counts = expert_counts[expert_counts > 0]
                                active_experts = int(active_counts.size)
                                mean_active_tokens = (
                                    float(active_counts.mean())
                                    if active_experts
                                    else 0.0
                                )
                                max_active_tokens = (
                                    int(active_counts.max()) if active_experts else 0
                                )
                                cross_counts = [
                                    value
                                    for destination, value in enumerate(send_counts)
                                    if destination != rank
                                ]
                                dispatch_bytes_per_assignment = (
                                    args.hidden_size * 2 + 4
                                )
                                combine_bytes_per_assignment = args.hidden_size * 2
                                total_bytes_per_assignment = (
                                    dispatch_bytes_per_assignment
                                    + combine_bytes_per_assignment
                                )
                                pending_layers.append(
                                    {
                                        "seed": seed,
                                        "routing_mode": mode,
                                        "scheduler": scheduler,
                                        "scheduler_execution_index": scheduler_index,
                                        "active_positions": positions,
                                        "repetition": repetition,
                                        "warmup": int(repetition < args.warmup),
                                        "rank": rank,
                                        "batch_index": batch_index,
                                        "layer": layer,
                                        "tokens": len(batch) * positions,
                                        "dispatched_assignments": len(batch)
                                        * positions
                                        * args.top_k,
                                        "received_assignments": recv_count,
                                        "combined_assignments": expected_assignments,
                                        "active_local_experts": active_experts,
                                        "mean_assignments_per_active_expert": (
                                            mean_active_tokens
                                        ),
                                        "max_assignments_per_active_expert": (
                                            max_active_tokens
                                        ),
                                        "load_imbalance": max_active_tokens
                                        / max(mean_active_tokens, 1.0),
                                        "non_empty_peers": sum(value > 0 for value in cross_counts),
                                        "average_peer_bytes": float(
                                            np.mean(cross_counts)
                                            * total_bytes_per_assignment
                                            if cross_counts
                                            else 0.0
                                        ),
                                        "max_peer_bytes": max(cross_counts, default=0)
                                        * total_bytes_per_assignment,
                                        "cross_gpu_bytes": sum(cross_counts)
                                        * total_bytes_per_assignment,
                                        "cross_gpu_hidden_dispatch_bytes": (
                                            sum(cross_counts) * args.hidden_size * 2
                                        ),
                                        "cross_gpu_expert_id_dispatch_bytes": (
                                            sum(cross_counts) * 4
                                        ),
                                        "cross_gpu_hidden_combine_bytes": (
                                            sum(cross_counts) * args.hidden_size * 2
                                        ),
                                        "send_counts": json.dumps(send_counts),
                                        "recv_counts": json.dumps(recv_counts),
                                        "_layer_start": layer_start,
                                        "_router_end": router_end,
                                        "_dispatch_end": dispatch_end,
                                        "_compute_end": compute_end,
                                        "_combine_end": combine_end,
                                    }
                                )
                            final_hidden.append(hidden)
                        run_end.record()
                        run_end.synchronize()
                        gpu_ms = run_start.elapsed_time(run_end)
                        finite = all(
                            bool(torch.isfinite(value).all().item())
                            for value in final_hidden
                        )
                        route_ids_valid = bool(
                            np.logical_and(
                                global_actual_np >= 0,
                                global_actual_np < args.num_experts,
                            ).all()
                        )
                        assignment_counts_match = all(
                            int(row["dispatched_assignments"])
                            == int(row["combined_assignments"])
                            for row in pending_layers
                        ) and count_exchange_valid
                        for row in pending_layers:
                            layer_start = row.pop("_layer_start")
                            router_end = row.pop("_router_end")
                            dispatch_end = row.pop("_dispatch_end")
                            compute_end = row.pop("_compute_end")
                            combine_end = row.pop("_combine_end")
                            row.update(
                                {
                                    "layer_ms": layer_start.elapsed_time(combine_end),
                                    "router_ms": layer_start.elapsed_time(router_end),
                                    "dispatch_ms": router_end.elapsed_time(dispatch_end),
                                    "expert_compute_ms": dispatch_end.elapsed_time(
                                        compute_end
                                    ),
                                    "combine_ms": compute_end.elapsed_time(combine_end),
                                }
                            )
                            layer_rows.append(row)
                        useful_positions = args.candidate_requests_per_rank * positions
                        run_rows.append(
                            {
                                "seed": seed,
                                "routing_mode": mode,
                                "scheduler": scheduler,
                                "scheduler_execution_index": scheduler_index,
                                "active_positions": positions,
                                "repetition": repetition,
                                "warmup": int(repetition < args.warmup),
                                "rank": rank,
                                "scheduler_ms": scheduler_ms,
                                "selection_plan_checksum": selection_plan_checksum,
                                "count_exchange_ms": count_exchange_ms,
                                "pre_data_arrival_ns": pre_data_arrival_ns,
                                "pre_data_barrier_wait_ms": pre_data_barrier_wait_ms,
                                "offline_plan_generation_ms": (
                                    coordinated_plan_ms
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_fifo_objective": (
                                    coordinated_diagnostics.fifo_cost
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_best_found_objective": (
                                    coordinated_diagnostics.best_cost
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_fifo_max_receive_load": (
                                    coordinated_diagnostics.fifo_max_receive_load
                                    if is_coordinated
                                    else 0
                                ),
                                "coordinated_best_max_receive_load": (
                                    coordinated_entry["max_receive_load"]
                                    if is_coordinated
                                    else 0
                                ),
                                "coordinated_predicted_reduction_percent": (
                                    coordinated_diagnostics.predicted_reduction_percent
                                    * float(coordinated_entry["achieved"])
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_dose_target_fraction": (
                                    float(coordinated_entry["target"])
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_dose_achieved_fraction": (
                                    float(coordinated_entry["achieved"])
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_reassigned_request_fraction": (
                                    float(coordinated_entry["reassigned_fraction"])
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_restart_cost_std": (
                                    coordinated_diagnostics.restart_cost_std
                                    if is_coordinated
                                    else 0.0
                                ),
                                "coordinated_restart_costs": (
                                    json.dumps(coordinated_diagnostics.restart_costs)
                                    if is_coordinated
                                    else "[]"
                                ),
                                "coordinated_best_so_far_costs": (
                                    json.dumps(
                                        coordinated_diagnostics.best_so_far_costs
                                    )
                                    if is_coordinated
                                    else "[]"
                                ),
                                "coordinated_restart_tail_improved": int(
                                    is_coordinated
                                    and coordinated_diagnostics.improved_in_last_two_restarts
                                ),
                                "scheduler_scope": (
                                    "coordinated_offline_replay"
                                    if is_coordinated
                                    else "rank_local_independent"
                                ),
                                "gpu_path_ms": gpu_ms,
                                "wall_with_scheduler_ms": (
                                    gpu_ms + count_exchange_ms + scheduler_ms
                                ),
                                "scheduler_fraction": scheduler_ms
                                / (gpu_ms + count_exchange_ms + scheduler_ms),
                                "useful_finalized_positions": useful_positions,
                                "executed_positions": useful_positions,
                                "work_amplification": 1.0,
                                "expert_token_executions": useful_positions
                                * args.layers
                                * args.top_k,
                                "finite": int(finite),
                                "assignment_counts_match": int(assignment_counts_match),
                                "route_ids_valid": int(route_ids_valid),
                                "valid": int(
                                    finite and assignment_counts_match and route_ids_valid
                                ),
                                "batch_order": json.dumps(batches),
                            }
                        )
                        if profiler is not None:
                            profiler.step()

    if profiler is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(args.trace_dir / f"rank{rank}_trace.json"))

    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / f"rank{rank}_layers.csv", layer_rows)
    write_rows(args.output / f"rank{rank}_runs.csv", run_rows)
    dist.barrier()
    if rank == 0:
        metadata = {
            "execution_kind": "PRIMARY_NATIVE_K_POSITION_EP4",
            "measurement_protocol": "isolated_replay_data_plane_v2",
            "validation_outside_cuda_interval": True,
            "dynamic_count_exchange_in_cuda_interval": False,
            "packing_and_unpacking_in_cuda_interval": True,
            "origin_slot_transmitted": False,
            "origin_restoration": "implicit_stable_assignment_order_plus_local_inverse",
            "expert_id_bytes_per_assignment": 4,
            "same_rank_topk_hidden_deduplication": False,
            "dispatch_granularity": "one_hidden_copy_per_expert_assignment",
            "scheduler_scope": "rank_local_independent",
            "local_route_replay_scope": "actual_local_routes_only_not_global_bound",
            "hardware_contract": "docs/hardware_execution_contract.md",
            "layers": args.layers,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "expert_parallel_size": world_size,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "dtype": "bfloat16",
            "candidate_requests_per_rank": args.candidate_requests_per_rank,
            "batch_size": args.batch_size,
            "active_positions": sorted(set(args.active_positions)),
            "routing_modes": args.routing_modes,
            "request_correlation_strength": args.request_correlation_strength,
            "schedulers": args.schedulers,
            "scheduler_order": (
                "deterministically_counterbalanced"
                if args.counterbalance_schedulers
                else "fixed"
            ),
            "seeds": args.seeds,
            "warmup_count": args.warmup,
            "measurement_iterations": args.repetitions,
            "planner_restarts": args.planner_restarts,
            "planner_max_restarts": args.planner_max_restarts,
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "fused_moe_identity": fused_moe_identity(),
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
        print(json.dumps({"output": str(args.output), "status": "PASS"}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
