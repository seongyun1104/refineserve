#!/usr/bin/env python3
"""Screen measured native router IDs for request-composition opportunity.

This is a route-only supporting analysis. It does not turn a single-GPU router
observation into EP timing evidence and it cannot authorize a scheduler run without
the matching measured data-plane accessibility gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .coordinated_scheduling import (
        combined_rank_loads,
        composition_invariant_cost_lower_bound,
        coordinated_dose_ladder,
        coordinated_plan_with_diagnostics,
        fifo_plan,
    )
except ImportError:
    from coordinated_scheduling import (  # type: ignore[no-redef]
        combined_rank_loads,
        composition_invariant_cost_lower_bound,
        coordinated_dose_ladder,
        coordinated_plan_with_diagnostics,
        fifo_plan,
    )


REQUIRED_ROUTE_COLUMNS = {
    "trace_phase",
    "segment_id",
    "seed",
    "request_id",
    "active_positions",
    "block_width",
    "iteration",
    "layer_id",
    "position_id",
    "route_slot",
    "expert_id",
    "compute_positions",
    "masked_positions_before_step",
    "masked_positions_after_step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--requests-per-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument(
        "--trace-phase",
        choices=["initial_width_ablation", "native_denoising"],
        default="native_denoising",
    )
    parser.add_argument(
        "--all-iterations",
        action="store_true",
        help="Screen every complete denoising iteration instead of one iteration.",
    )
    parser.add_argument(
        "--expert-to-rank-mapping",
        type=Path,
        help="Optional JSON list mapping every expert ID to an EP rank.",
    )
    return parser.parse_args()


def _load_metadata(trace: Path) -> dict[str, object]:
    metadata_path = trace / "route_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("eligible_for_scheduler_opportunity_screening", False):
        raise ValueError("trace is not eligible for route-opportunity screening")
    return metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(
    *,
    num_experts: int,
    world_size: int,
    mapping_path: Path | None,
) -> tuple[np.ndarray, str]:
    if num_experts % world_size:
        raise ValueError("experts must divide evenly across EP ranks")
    experts_per_rank = num_experts // world_size
    if mapping_path is None:
        mapping = np.repeat(np.arange(world_size), experts_per_rank)
        source = "hypothetical_contiguous_rank_ownership"
    else:
        mapping = np.asarray(json.loads(mapping_path.read_text()), dtype=np.int64)
        source = str(mapping_path)
    if mapping.shape != (num_experts,):
        raise ValueError("expert-to-rank mapping must contain one rank per expert")
    if np.any(mapping < 0) or np.any(mapping >= world_size):
        raise ValueError("expert-to-rank mapping contains an invalid rank")
    rank_counts = np.bincount(mapping, minlength=world_size)
    if not np.all(rank_counts == experts_per_rank):
        raise ValueError("every EP rank must own the same number of experts")
    return mapping, source


def _planner_expert_ids(mapping: np.ndarray) -> np.ndarray:
    """Map arbitrary ownership into contiguous rank-major planner expert IDs."""
    local_slots = np.empty_like(mapping)
    for rank in np.unique(mapping):
        expert_ids = np.flatnonzero(mapping == rank)
        local_slots[expert_ids] = np.arange(len(expert_ids))
    experts_per_rank = int(np.bincount(mapping).max())
    return mapping * experts_per_rank + local_slots


def _validate_routes(
    routes: pd.DataFrame,
    *,
    metadata: dict[str, object],
    world_size: int,
    requests_per_rank: int,
    iteration: int,
    trace_phase: str,
    all_iterations: bool,
) -> pd.DataFrame:
    missing = REQUIRED_ROUTE_COLUMNS - set(routes.columns)
    if missing:
        raise ValueError(f"route CSV is missing columns: {sorted(missing)}")
    routes = routes[routes["trace_phase"] == trace_phase].copy()
    if not all_iterations:
        routes = routes[routes["iteration"] == iteration].copy()
    if routes.empty:
        raise ValueError(f"trace contains no rows for iteration {iteration}")
    key = [
        "segment_id",
        "trace_phase",
        "request_id",
        "active_positions",
        "iteration",
        "layer_id",
        "position_id",
        "route_slot",
    ]
    if routes.duplicated(key).any():
        raise ValueError("route trace contains duplicate composite keys")
    num_experts = int(metadata["num_experts"])
    top_k = int(metadata["top_k"])
    if not routes["expert_id"].between(0, num_experts - 1).all():
        raise ValueError("route trace contains an out-of-range expert ID")
    if set(routes["route_slot"].unique()) != set(range(top_k)):
        raise ValueError("route slots do not match checkpoint top-k")
    work_key = [
        "segment_id",
        "request_id",
        "active_positions",
        "iteration",
        "layer_id",
        "position_id",
    ]
    route_group_sizes = routes.groupby(work_key, sort=False).size()
    route_group_unique_experts = routes.groupby(work_key, sort=False)[
        "expert_id"
    ].nunique()
    if not (route_group_sizes == top_k).all():
        raise ValueError("every position/layer must contain exactly top-k rows")
    if not (route_group_unique_experts == top_k).all():
        raise ValueError("top-k expert IDs must be unique within every position/layer")
    expected_requests = world_size * requests_per_rank
    request_counts = routes.groupby(
        ["trace_phase", "segment_id", "active_positions", "iteration"],
        sort=False,
    )["request_id"].nunique()
    complete_keys = request_counts[request_counts == expected_requests].index
    if len(complete_keys) == 0:
        raise ValueError(
            f"no phase/segment/K/iteration cell contains {expected_requests} requests"
        )
    indexed = routes.set_index(
        ["trace_phase", "segment_id", "active_positions", "iteration"]
    )
    routes = indexed[indexed.index.isin(complete_keys)].reset_index()
    if not (routes["block_width"] == routes["active_positions"]).all():
        raise ValueError("this screen requires block_width == active_positions")
    position_counts = routes.groupby(
        [
            "trace_phase",
            "segment_id",
            "request_id",
            "active_positions",
            "iteration",
            "layer_id",
        ],
        sort=False,
    )["position_id"].nunique()
    expected_positions = position_counts.index.get_level_values("active_positions")
    if not np.array_equal(
        position_counts.to_numpy(dtype=np.int64),
        expected_positions.to_numpy(dtype=np.int64),
    ):
        raise ValueError("position cardinality does not match active_positions")
    return routes


def _counts_for_cell(
    cell: pd.DataFrame,
    *,
    mapping: np.ndarray,
    world_size: int,
    requests_per_rank: int,
    top_k: int,
) -> tuple[np.ndarray, list[int]]:
    request_ids = sorted(int(value) for value in cell["request_id"].unique())
    expected_requests = world_size * requests_per_rank
    if len(request_ids) != expected_requests:
        raise ValueError("cell request count does not match the screening contract")
    layers = sorted(int(value) for value in cell["layer_id"].unique())
    positions = int(cell["active_positions"].iloc[0])
    planner_ids = _planner_expert_ids(mapping)
    num_experts = len(mapping)
    counts = np.zeros(
        (world_size, requests_per_rank, len(layers), num_experts),
        dtype=np.int64,
    )
    layer_index = {layer: index for index, layer in enumerate(layers)}
    request_index = {request: index for index, request in enumerate(request_ids)}
    expected_rows = expected_requests * len(layers) * positions * top_k
    if len(cell) != expected_rows:
        raise ValueError(
            f"cell has {len(cell)} rows, expected {expected_rows} from its shape"
        )
    for row in cell.itertuples(index=False):
        global_request = request_index[int(row.request_id)]
        source = global_request // requests_per_rank
        local_request = global_request % requests_per_rank
        mapped_expert = int(planner_ids[int(row.expert_id)])
        counts[source, local_request, layer_index[int(row.layer_id)], mapped_expert] += 1
    return counts, layers


def _screen_cell(
    counts: np.ndarray,
    *,
    batch_size: int,
    experts_per_rank: int,
    restarts: int,
    planner_seed: int,
) -> dict[str, object]:
    sources, requests = counts.shape[:2]
    best_plan, diagnostics = coordinated_plan_with_diagnostics(
        counts,
        batch_size,
        experts_per_rank,
        restarts=restarts,
        seed=planner_seed,
    )
    fifo = fifo_plan(sources, requests, batch_size)
    fifo_loads = combined_rank_loads(counts, fifo, experts_per_rank)
    best_loads = combined_rank_loads(counts, best_plan, experts_per_rank)
    lower_bound = composition_invariant_cost_lower_bound(
        counts,
        batch_size,
        experts_per_rank,
    )
    objective_opportunity = max(diagnostics.fifo_cost - lower_bound, 0.0)
    objective_achievability = (
        (diagnostics.fifo_cost - diagnostics.best_cost) / objective_opportunity
        if objective_opportunity > 0
        else 0.0
    )
    realized_reduction = (
        (diagnostics.fifo_cost - diagnostics.best_cost)
        / max(diagnostics.fifo_cost, 1.0)
    )
    objective_opportunity_fraction = objective_opportunity / max(
        diagnostics.fifo_cost, 1.0
    )
    dose_ladder = coordinated_dose_ladder(
        counts,
        best_plan,
        batch_size,
        experts_per_rank,
        seed=planner_seed,
    )
    mean_load = float(fifo_loads.mean())
    fifo_max = float(fifo_loads.max())
    best_max = float(best_loads.max())
    global_opportunity = max(fifo_max - mean_load, 0.0)
    global_achievability = (
        max(fifo_max - best_max, 0.0) / global_opportunity
        if global_opportunity > 0
        else 0.0
    )
    return {
        "mean_receive_load": mean_load,
        "fifo_max_receive_load": fifo_max,
        "best_found_max_receive_load": best_max,
        "fifo_rank_load_imbalance": fifo_max / max(mean_load, 1.0),
        "global_max_achievability_diagnostic": global_achievability,
        "objective_lower_bound": lower_bound,
        "fifo_objective": diagnostics.fifo_cost,
        "best_found_objective": diagnostics.best_cost,
        "objective_opportunity": objective_opportunity,
        "objective_opportunity_fraction": objective_opportunity_fraction,
        "objective_physically_zero": int(np.isclose(objective_opportunity, 0.0)),
        "objective_achievability": objective_achievability,
        "realized_objective_reduction_fraction": realized_reduction,
        "restart_tail_improved": int(diagnostics.improved_in_last_two_restarts),
        "restart_cost_std": diagnostics.restart_cost_std,
        "dose_distinct_count": len(dose_ladder),
        "dose_target_fractions": json.dumps(
            [target for target, _, _ in dose_ladder]
        ),
        "dose_achieved_fractions": json.dumps(
            [achieved for _, achieved, _ in dose_ladder]
        ),
    }


def _dense_trace_records(
    *,
    trace: Path,
    metadata: dict[str, object],
    mapping: np.ndarray,
    world_size: int,
    requests_per_rank: int,
    batch_size: int,
    restarts: int,
    trace_phase: str,
    iteration: int,
    all_iterations: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_path = trace / "route_observations.csv"
    arrays_path = trace / "routes_dense.npz"
    recorded_hashes = metadata.get("artifact_sha256", {})
    for name, path in (
        ("route_observations.csv", manifest_path),
        ("routes_dense.npz", arrays_path),
    ):
        if recorded_hashes and recorded_hashes.get(name) != _file_sha256(path):
            raise ValueError(f"{name} checksum does not match trace metadata")
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["trace_phase"] == trace_phase].copy()
    if not all_iterations:
        manifest = manifest[manifest["iteration"] == iteration].copy()
    if manifest.empty:
        raise ValueError("dense trace contains no matching observations")
    expected_requests = world_size * requests_per_rank
    num_experts = int(metadata["num_experts"])
    top_k = int(metadata["top_k"])
    experts_per_rank = num_experts // world_size
    planner_ids = _planner_expert_ids(mapping)
    records: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    group_columns = [
        "trace_phase",
        "segment_id",
        "seed",
        "active_positions",
        "block_width",
        "position_width_source",
        "model_forward_positions",
        "iteration",
    ]
    with np.load(arrays_path) as arrays:
        for group_key, observations in manifest.groupby(group_columns, sort=True):
            (
                phase,
                segment_id,
                seed,
                active_positions,
                block_width,
                position_width_source,
                model_forward_positions,
                step,
            ) = group_key
            route_parts: list[np.ndarray] = []
            request_ids: list[int] = []
            masked_before: list[int] = []
            masked_after: list[int] = []
            for observation in observations.itertuples(index=False):
                key = str(observation.array_key)
                if key not in arrays:
                    raise ValueError(f"manifest references missing route array: {key}")
                route_part = np.asarray(arrays[key], dtype=np.int64)
                part_requests = [int(value) for value in json.loads(observation.request_ids)]
                before = [
                    int(value)
                    for value in json.loads(observation.masked_positions_before_step)
                ]
                raw_after = observation.masked_positions_after_step
                after = (
                    []
                    if pd.isna(raw_after) or raw_after == ""
                    else [int(value) for value in json.loads(raw_after)]
                )
                if route_part.shape[0] != len(part_requests):
                    raise ValueError("route array request dimension disagrees with manifest")
                if len(before) != len(part_requests):
                    raise ValueError("masked-before vector disagrees with request IDs")
                if after and len(after) != len(part_requests):
                    raise ValueError("masked-after vector disagrees with request IDs")
                route_parts.append(route_part)
                request_ids.extend(part_requests)
                masked_before.extend(before)
                masked_after.extend(after)
            coverage.append(
                {
                    "trace_phase": str(phase),
                    "segment_id": int(segment_id),
                    "seed": int(seed),
                    "active_positions": int(active_positions),
                    "block_width": int(block_width),
                    "position_width_source": str(position_width_source),
                    "model_forward_positions": int(model_forward_positions),
                    "iteration": int(step),
                    "observed_requests": len(request_ids),
                    "expected_requests": expected_requests,
                    "complete_for_fixed_pool_screen": int(
                        len(request_ids) == expected_requests
                    ),
                }
            )
            if len(request_ids) != expected_requests:
                continue
            if len(set(request_ids)) != expected_requests:
                raise ValueError("dense cell contains duplicate request IDs")
            order = np.argsort(request_ids)
            routes = np.concatenate(route_parts, axis=0)[order]
            before_array = np.asarray(masked_before, dtype=np.int64)[order]
            after_array = (
                np.asarray(masked_after, dtype=np.int64)[order]
                if masked_after
                else np.full(expected_requests, np.nan)
            )
            if routes.ndim != 4 or routes.shape[2:] != (
                int(model_forward_positions),
                top_k,
            ):
                raise ValueError("dense route array shape violates trace metadata")
            if np.any(routes < 0) or np.any(routes >= num_experts):
                raise ValueError("dense trace contains an out-of-range expert ID")
            sorted_routes = np.sort(routes, axis=-1)
            if np.any(np.diff(sorted_routes, axis=-1) == 0):
                raise ValueError("dense trace contains duplicate top-k expert IDs")
            layers = routes.shape[1]
            counts = np.zeros(
                (world_size, requests_per_rank, layers, num_experts),
                dtype=np.int64,
            )
            mapped_routes = planner_ids[routes]
            for global_request in range(expected_requests):
                source = global_request // requests_per_rank
                local_request = global_request % requests_per_rank
                for layer in range(layers):
                    counts[source, local_request, layer] = np.bincount(
                        mapped_routes[global_request, layer].reshape(-1),
                        minlength=num_experts,
                    )
            destination_routes = mapping[routes]
            sorted_destinations = np.sort(destination_routes, axis=-1)
            collision = float(
                (np.diff(sorted_destinations, axis=-1) == 0).any(axis=-1).mean()
            )
            record = {
                "segment_id": int(segment_id),
                "seed": int(seed),
                "active_positions": int(active_positions),
                "block_width": int(block_width),
                "position_width_source": str(position_width_source),
                "model_forward_positions": int(model_forward_positions),
                "iteration": int(step),
                "trace_phase": str(phase),
                "routing_mode": f"measured_llada2_{phase}",
                "sparse_layer_count": layers,
                "first_sparse_layer": int(metadata["first_sparse_layer"]),
                "last_sparse_layer": int(metadata["num_layers"]) - 1,
                "same_rank_multi_expert_collision_fraction": collision,
                "masked_positions_before_step_mean": float(before_array.mean()),
                "masked_positions_after_step_mean": float(
                    np.nanmean(after_array)
                ),
            }
            record.update(
                _screen_cell(
                    counts,
                    batch_size=batch_size,
                    experts_per_rank=experts_per_rank,
                    restarts=restarts,
                    planner_seed=int(seed) + int(active_positions) + int(step),
                )
            )
            records.append(record)
    if not records:
        raise ValueError(
            "no complete 32-request dense observation cells survived screening"
        )
    return records, coverage


def main() -> None:
    args = parse_args()
    metadata = _load_metadata(args.trace)
    num_experts = int(metadata["num_experts"])
    mapping, mapping_source = _load_mapping(
        num_experts=num_experts,
        world_size=args.world_size,
        mapping_path=args.expert_to_rank_mapping,
    )
    dense_path = args.trace / "routes_dense.npz"
    dense_coverage: list[dict[str, object]] | None = None
    if dense_path.exists():
        records, dense_coverage = _dense_trace_records(
            trace=args.trace,
            metadata=metadata,
            mapping=mapping,
            world_size=args.world_size,
            requests_per_rank=args.requests_per_rank,
            batch_size=args.batch_size,
            restarts=args.restarts,
            trace_phase=args.trace_phase,
            iteration=args.iteration,
            all_iterations=args.all_iterations,
        )
        routes_hash = _file_sha256(dense_path)
    else:
        routes_path = args.trace / "routes_observational.csv"
        recorded_hashes = metadata.get("artifact_sha256", {})
        if recorded_hashes:
            expected_hash = recorded_hashes.get("routes_observational.csv")
            if expected_hash != _file_sha256(routes_path):
                raise ValueError("route CSV checksum does not match trace metadata")
        routes = _validate_routes(
            pd.read_csv(routes_path),
            metadata=metadata,
            world_size=args.world_size,
            requests_per_rank=args.requests_per_rank,
            iteration=args.iteration,
            trace_phase=args.trace_phase,
            all_iterations=args.all_iterations,
        )
        top_k = int(metadata["top_k"])
        experts_per_rank = num_experts // args.world_size
        records = []
        group_columns = [
            "trace_phase",
            "segment_id",
            "seed",
            "active_positions",
            "block_width",
            "iteration",
        ]
        for group_key, cell in routes.groupby(group_columns, sort=True):
            (
                trace_phase,
                segment_id,
                seed,
                active_positions,
                block_width,
                iteration,
            ) = group_key
            counts, layers = _counts_for_cell(
                cell,
                mapping=mapping,
                world_size=args.world_size,
                requests_per_rank=args.requests_per_rank,
                top_k=top_k,
            )
            destination_routes = mapping[cell["expert_id"].to_numpy(dtype=np.int64)]
            assignments = cell.assign(destination_rank=destination_routes)
            collision = (
                assignments.groupby(
                    ["request_id", "layer_id", "position_id"], sort=False
                )["destination_rank"].nunique()
                < top_k
            ).mean()
            record = {
                "segment_id": int(segment_id),
                "seed": int(seed),
                "active_positions": int(active_positions),
                "block_width": int(block_width),
                "position_width_source": "legacy_row_trace_unspecified",
                "model_forward_positions": int(
                    cell["compute_positions"].iloc[0]
                ),
                "iteration": int(iteration),
                "trace_phase": str(trace_phase),
                "routing_mode": f"measured_llada2_{trace_phase}",
                "sparse_layer_count": len(layers),
                "first_sparse_layer": min(layers),
                "last_sparse_layer": max(layers),
                "same_rank_multi_expert_collision_fraction": float(collision),
                "masked_positions_before_step_mean": float(
                    cell["masked_positions_before_step"].mean()
                ),
                "masked_positions_after_step_mean": float(
                    pd.to_numeric(
                        cell["masked_positions_after_step"], errors="coerce"
                    ).mean()
                ),
            }
            record.update(
                _screen_cell(
                    counts,
                    batch_size=args.batch_size,
                    experts_per_rank=experts_per_rank,
                    restarts=args.restarts,
                    planner_seed=(
                        int(seed) + int(active_positions) + int(iteration)
                    ),
                )
            )
            records.append(record)
        routes_hash = _file_sha256(routes_path)
    cells = pd.DataFrame.from_records(records)
    by_cell = (
        cells.groupby(
            [
                "trace_phase",
                "position_width_source",
                "iteration",
                "active_positions",
                "model_forward_positions",
                "routing_mode",
            ],
            sort=True,
        )
        .agg(
            samples=("objective_achievability", "size"),
            fifo_imbalance_p25=(
                "fifo_rank_load_imbalance",
                lambda values: np.percentile(values, 25),
            ),
            fifo_imbalance_median=("fifo_rank_load_imbalance", "median"),
            objective_achievability_p25=(
                "objective_achievability",
                lambda values: np.percentile(values, 25),
            ),
            objective_achievability_median=("objective_achievability", "median"),
            objective_opportunity_fraction_p25=(
                "objective_opportunity_fraction",
                lambda values: np.percentile(values, 25),
            ),
            objective_opportunity_fraction_median=(
                "objective_opportunity_fraction",
                "median",
            ),
            objective_physically_zero_fraction=(
                "objective_physically_zero",
                "mean",
            ),
            realized_objective_reduction_fraction_p25=(
                "realized_objective_reduction_fraction",
                lambda values: np.percentile(values, 25),
            ),
            realized_objective_reduction_fraction_median=(
                "realized_objective_reduction_fraction",
                "median",
            ),
            same_rank_multi_expert_collision_fraction_median=(
                "same_rank_multi_expert_collision_fraction",
                "median",
            ),
            dose_distinct_count_min=("dose_distinct_count", "min"),
            dose_distinct_count_median=("dose_distinct_count", "median"),
            masked_positions_before_step_mean=(
                "masked_positions_before_step_mean",
                "mean",
            ),
            masked_positions_after_step_mean=(
                "masked_positions_after_step_mean",
                "mean",
            ),
        )
        .reset_index()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "measured_screening_cells.csv", index=False)
    by_cell.to_csv(args.output / "measured_screening_by_cell.csv", index=False)
    if dense_coverage is not None:
        pd.DataFrame.from_records(dense_coverage).to_csv(
            args.output / "measured_screening_coverage.csv",
            index=False,
        )
    output_metadata = {
        "source_trace": str(args.trace),
        "source_model": metadata["model_identifier"],
        "source_revision": metadata["model_revision"],
        "source_trace_kind": metadata["trace_kind"],
        "trace_phase": args.trace_phase,
        "all_iterations": args.all_iterations,
        "source_routes_sha256": routes_hash,
        "evidence_class": "MEASURED_ROUTE_SUPPORTING_CALIBRATION",
        "eligible_for_gate3_timing_authorization": False,
        "world_size": args.world_size,
        "requests_per_rank": args.requests_per_rank,
        "batch_size": args.batch_size,
        "expert_to_rank_mapping_source": mapping_source,
        "source_rank_assignment": "contiguous_groups_of_sorted_request_ids",
        "planner_objective": "sum_over_batch_layer_of_max_destination_rank_load",
        "screening_metric": "realized_objective_reduction_fraction",
        "fixed_pool_coverage_policy": (
            "Only iterations retaining all expected requests enter the fixed-pool "
            "composition screen. Coverage for later partial-pool steps is retained "
            "separately and is not silently promoted to a 32-request result."
        ),
        "semantic_limit": (
            "This route-only screen measures composition opportunity under a "
            "hypothetical EP placement. It does not contain measured EP timing and "
            "cannot be multiplied by the toy EP timing gate as if the shapes matched."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(by_cell.to_string(index=False))


if __name__ == "__main__":
    main()
