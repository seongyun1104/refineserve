#!/usr/bin/env python3
"""Analyze native LLaDA2 route structure without making a timing claim.

The analyzer preserves the distinction between routed positions, current-block
masked positions, and newly finalized positions. It evaluates two hypothetical
EP placements and compares FIFO, best-found current-route composition, and a
previous-step plan evaluated on the current route.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .coordinated_scheduling import (
        coordinated_plan_with_diagnostics,
        plan_cost,
    )
except ImportError:
    from coordinated_scheduling import (  # type: ignore[no-redef]
        coordinated_plan_with_diagnostics,
        plan_cost,
    )


ROLE_PREFIX = 0
ROLE_CURRENT_FINALIZED = 1
ROLE_CURRENT_MASKED = 2


@dataclass(frozen=True)
class NativeObservation:
    segment_id: int
    seed: int
    workload_class: str
    block_id: int
    denoise_step: int
    iteration: int
    block_width: int
    request_ids: tuple[int, ...]
    routes: np.ndarray
    roles: np.ndarray
    masked_before: np.ndarray
    masked_after: np.ndarray
    finalized: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--requests-per-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--previous-route-gain-threshold", type=float, default=0.8)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ints(value: object) -> list[int]:
    if pd.isna(value) or value == "":
        return []
    return [int(item) for item in json.loads(str(value))]


def _load_observations(trace: Path) -> tuple[dict[str, object], list[NativeObservation]]:
    metadata_path = trace / "route_metadata.json"
    manifest_path = trace / "route_observations.csv"
    routes_path = trace / "routes_dense.npz"
    roles_path = trace / "position_roles_dense.npz"
    for path in (metadata_path, manifest_path, routes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("trace_kind") != "native_llada2_dense_router_observational":
        raise ValueError("unsupported native route trace kind")
    recorded = metadata.get("artifact_sha256", {})
    for name, path in (
        ("route_observations.csv", manifest_path),
        ("routes_dense.npz", routes_path),
    ):
        if recorded and recorded.get(name) != _sha256(path):
            raise ValueError(f"{name} checksum does not match metadata")
    if roles_path.exists() and recorded and recorded.get(
        "position_roles_dense.npz"
    ) != _sha256(roles_path):
        raise ValueError("position_roles_dense.npz checksum does not match metadata")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["trace_phase"] == "native_denoising"].copy()
    required = {
        "segment_id",
        "seed",
        "iteration",
        "block_width",
        "array_key",
        "request_ids",
        "masked_positions_before_step",
        "masked_positions_after_step",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"route manifest missing columns: {sorted(missing)}")
    group_columns = [
        "segment_id",
        "seed",
        "workload_class",
        "block_id",
        "denoise_step",
        "iteration",
        "block_width",
    ]
    for column, default in (
        ("workload_class", "unspecified"),
        ("block_id", 0),
        ("denoise_step", manifest["iteration"]),
    ):
        if column not in manifest:
            manifest[column] = default

    observations: list[NativeObservation] = []
    roles_context = np.load(roles_path) if roles_path.exists() else None
    try:
        with np.load(routes_path) as route_arrays:
            for key, rows in manifest.groupby(group_columns, sort=True):
                route_parts: list[np.ndarray] = []
                role_parts: list[np.ndarray] = []
                request_ids: list[int] = []
                before: list[int] = []
                after: list[int] = []
                finalized: list[int] = []
                for row in rows.itertuples(index=False):
                    array_key = str(row.array_key)
                    routes = np.asarray(route_arrays[array_key], dtype=np.int64)
                    ids = _json_ints(row.request_ids)
                    if routes.shape[0] != len(ids):
                        raise ValueError("request IDs disagree with route array")
                    if roles_context is None:
                        roles = np.full(
                            (len(ids), routes.shape[2]), ROLE_PREFIX, dtype=np.uint8
                        )
                        roles[:, -int(row.block_width) :] = ROLE_CURRENT_MASKED
                    else:
                        roles = np.asarray(roles_context[array_key], dtype=np.uint8)
                    if roles.shape != (len(ids), routes.shape[2]):
                        raise ValueError("position roles disagree with route array")
                    row_before = _json_ints(row.masked_positions_before_step)
                    row_after = _json_ints(row.masked_positions_after_step)
                    if hasattr(row, "finalized_positions_this_step"):
                        row_finalized = _json_ints(row.finalized_positions_this_step)
                    else:
                        row_finalized = [
                            a - b for a, b in zip(row_before, row_after, strict=True)
                        ]
                    route_parts.append(routes)
                    role_parts.append(roles)
                    request_ids.extend(ids)
                    before.extend(row_before)
                    after.extend(row_after)
                    finalized.extend(row_finalized)
                order = np.argsort(request_ids)
                routes = np.concatenate(route_parts, axis=0)[order]
                roles = np.concatenate(role_parts, axis=0)[order]
                sorted_ids = tuple(np.asarray(request_ids, dtype=np.int64)[order].tolist())
                if len(set(sorted_ids)) != len(sorted_ids):
                    raise ValueError("observation contains duplicate request IDs")
                observations.append(
                    NativeObservation(
                        segment_id=int(key[0]),
                        seed=int(key[1]),
                        workload_class=str(key[2]),
                        block_id=int(key[3]),
                        denoise_step=int(key[4]),
                        iteration=int(key[5]),
                        block_width=int(key[6]),
                        request_ids=sorted_ids,
                        routes=routes,
                        roles=roles,
                        masked_before=np.asarray(before, dtype=np.int64)[order],
                        masked_after=np.asarray(after, dtype=np.int64)[order],
                        finalized=np.asarray(finalized, dtype=np.int64)[order],
                    )
                )
    finally:
        if roles_context is not None:
            roles_context.close()
    return metadata, observations


def _route_bitsets(routes: np.ndarray, num_experts: int) -> np.ndarray:
    """Encode expert sets as uint64 words while preserving exact set semantics."""
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if routes.ndim < 1 or routes.shape[-1] <= 0:
        raise ValueError("routes must have a non-empty top-k dimension")
    if np.any(routes < 0) or np.any(routes >= num_experts):
        raise ValueError("route contains an out-of-range expert")
    leading_shape = routes.shape[:-1]
    flat_routes = routes.reshape(-1, routes.shape[-1]).astype(np.int64, copy=False)
    words = (num_experts + 63) // 64
    encoded = np.zeros((len(flat_routes), words), dtype=np.uint64)
    rows = np.repeat(np.arange(len(flat_routes)), routes.shape[-1])
    experts = flat_routes.reshape(-1)
    word_ids = experts // 64
    bit_ids = (experts % 64).astype(np.uint64, copy=False)
    bit_values = np.left_shift(np.uint64(1), bit_ids)
    np.bitwise_or.at(encoded, (rows, word_ids), bit_values)
    return encoded.reshape(*leading_shape, words)


def _paired_bitset_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("paired bitsets must have identical shapes")
    intersection = np.bitwise_count(np.bitwise_and(left, right)).sum(axis=-1)
    left_size = np.bitwise_count(left).sum(axis=-1)
    right_size = np.bitwise_count(right).sum(axis=-1)
    union = left_size + right_size - intersection
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union != 0,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def _histogram(routes: np.ndarray, num_experts: int) -> np.ndarray:
    return np.bincount(routes.reshape(-1), minlength=num_experts).astype(np.float64)


def _spatial_jaccard(
    observation: NativeObservation, role: int, num_experts: int
) -> float:
    values: list[float] = []
    for request in range(len(observation.request_ids)):
        positions = np.flatnonzero(observation.roles[request] == role)
        if len(positions) < 2:
            continue
        for layer in range(observation.routes.shape[1]):
            bitsets = _route_bitsets(
                observation.routes[request, layer, positions], num_experts
            )
            intersection = np.bitwise_count(
                np.bitwise_and(bitsets[:, None, :], bitsets[None, :, :])
            ).sum(axis=-1)
            sizes = np.bitwise_count(bitsets).sum(axis=-1)
            union = sizes[:, None] + sizes[None, :] - intersection
            upper = np.triu_indices(len(positions), k=1)
            values.extend((intersection[upper] / union[upper]).tolist())
    return float(np.mean(values)) if values else float("nan")


def _temporal_metrics(
    current: NativeObservation,
    previous: NativeObservation | None,
    num_experts: int,
) -> tuple[float, float]:
    if previous is None:
        return float("nan"), float("nan")
    current_index = {request_id: index for index, request_id in enumerate(current.request_ids)}
    previous_index = {
        request_id: index for index, request_id in enumerate(previous.request_ids)
    }
    common_requests = sorted(set(current_index) & set(previous_index))
    if not common_requests:
        return float("nan"), float("nan")
    common_positions = min(current.routes.shape[2], previous.routes.shape[2])
    current_requests = [current_index[request_id] for request_id in common_requests]
    previous_requests = [previous_index[request_id] for request_id in common_requests]
    current_routes = current.routes[current_requests, :, :common_positions]
    previous_routes = previous.routes[previous_requests, :, :common_positions]
    jaccards = _paired_bitset_jaccard(
        _route_bitsets(current_routes, num_experts),
        _route_bitsets(previous_routes, num_experts),
    )
    cosine_values = [
        _cosine(
            _histogram(current.routes[current_request], num_experts),
            _histogram(previous.routes[previous_request], num_experts),
        )
        for current_request, previous_request in zip(
            current_requests, previous_requests, strict=True
        )
    ]
    return float(np.mean(jaccards)), float(np.mean(cosine_values))


def _between_request_cosine(
    observation: NativeObservation, num_experts: int
) -> float:
    histograms = [
        _histogram(observation.routes[index], num_experts)
        for index in range(len(observation.request_ids))
    ]
    values = [
        _cosine(histograms[left], histograms[right])
        for left, right in itertools.combinations(range(len(histograms)), 2)
    ]
    return float(np.mean(values)) if values else float("nan")


def _mapping(name: str, num_experts: int, world_size: int) -> np.ndarray:
    if num_experts % world_size:
        raise ValueError("experts must divide evenly across ranks")
    if name == "contiguous":
        return np.repeat(np.arange(world_size), num_experts // world_size)
    if name == "round_robin":
        return np.arange(num_experts) % world_size
    raise ValueError(name)


def _planner_ids(mapping: np.ndarray) -> np.ndarray:
    local_slots = np.empty_like(mapping)
    for rank in np.unique(mapping):
        experts = np.flatnonzero(mapping == rank)
        local_slots[experts] = np.arange(len(experts))
    experts_per_rank = int(np.bincount(mapping).max())
    return mapping * experts_per_rank + local_slots


def _counts(
    observation: NativeObservation,
    mapping: np.ndarray,
    requests_per_rank: int,
) -> np.ndarray:
    world_size = len(np.unique(mapping))
    expected_requests = world_size * requests_per_rank
    if len(observation.request_ids) != expected_requests:
        raise ValueError("observation does not contain the fixed request pool")
    num_experts = len(mapping)
    planner_routes = _planner_ids(mapping)[observation.routes]
    counts = np.zeros(
        (world_size, requests_per_rank, observation.routes.shape[1], num_experts),
        dtype=np.int64,
    )
    for request in range(expected_requests):
        source = request // requests_per_rank
        local = request % requests_per_rank
        for layer in range(observation.routes.shape[1]):
            counts[source, local, layer] = np.bincount(
                planner_routes[request, layer].reshape(-1), minlength=num_experts
            )
    return counts


def _scheduling_record(
    observation: NativeObservation,
    previous: NativeObservation | None,
    *,
    mapping_name: str,
    mapping: np.ndarray,
    requests_per_rank: int,
    batch_size: int,
    restarts: int,
) -> dict[str, object]:
    counts = _counts(observation, mapping, requests_per_rank)
    world_size = counts.shape[0]
    experts_per_rank = len(mapping) // world_size
    best_plan, diagnostics = coordinated_plan_with_diagnostics(
        counts,
        batch_size,
        experts_per_rank,
        restarts=restarts,
        seed=observation.seed + observation.iteration,
    )
    del best_plan
    fifo_cost = diagnostics.fifo_cost
    best_cost = diagnostics.best_cost
    oracle_gain = fifo_cost - best_cost
    previous_cost = float("nan")
    previous_gain = float("nan")
    gain_capture = float("nan")
    if previous is not None and previous.request_ids == observation.request_ids:
        previous_counts = _counts(previous, mapping, requests_per_rank)
        previous_plan, _ = coordinated_plan_with_diagnostics(
            previous_counts,
            batch_size,
            experts_per_rank,
            restarts=restarts,
            seed=previous.seed + previous.iteration,
        )
        previous_cost = plan_cost(counts, previous_plan, experts_per_rank)
        previous_gain = fifo_cost - previous_cost
        if oracle_gain > 0:
            gain_capture = previous_gain / oracle_gain
    rank_routes = mapping[observation.routes]
    rank_loads = np.bincount(rank_routes.reshape(-1), minlength=world_size)
    return {
        "segment_id": observation.segment_id,
        "seed": observation.seed,
        "workload_class": observation.workload_class,
        "block_id": observation.block_id,
        "denoise_step": observation.denoise_step,
        "iteration": observation.iteration,
        "mapping": mapping_name,
        "routed_positions": observation.routes.shape[2],
        "masked_positions_before_mean": float(observation.masked_before.mean()),
        "finalized_positions_this_step_mean": float(observation.finalized.mean()),
        "unique_experts": int(np.unique(observation.routes).size),
        "max_rank_load": int(rank_loads.max()),
        "mean_rank_load": float(rank_loads.mean()),
        "rank_load_imbalance": float(rank_loads.max() / rank_loads.mean()),
        "fifo_objective": fifo_cost,
        "best_found_objective": best_cost,
        "best_found_headroom_fraction": oracle_gain / max(fifo_cost, 1.0),
        "previous_route_objective": previous_cost,
        "previous_route_gain_fraction": previous_gain / max(fifo_cost, 1.0),
        "previous_route_oracle_gain_capture": gain_capture,
        "best_found_reassigned_request_fraction": (
            diagnostics.reassigned_request_fraction
        ),
    }


def _rank_projection_record(
    observation: NativeObservation,
    *,
    mapping_name: str,
    mapping: np.ndarray,
) -> dict[str, object]:
    rank_routes = mapping[observation.routes]
    rank_loads = np.bincount(rank_routes.reshape(-1), minlength=len(np.unique(mapping)))
    return {
        "segment_id": observation.segment_id,
        "seed": observation.seed,
        "workload_class": observation.workload_class,
        "block_id": observation.block_id,
        "denoise_step": observation.denoise_step,
        "iteration": observation.iteration,
        "mapping": mapping_name,
        "observed_requests": len(observation.request_ids),
        "routed_positions": observation.routes.shape[2],
        "masked_positions_before_mean": float(observation.masked_before.mean()),
        "finalized_positions_this_step_mean": float(observation.finalized.mean()),
        "unique_experts": int(np.unique(observation.routes).size),
        "max_rank_load": int(rank_loads.max()),
        "mean_rank_load": float(rank_loads.mean()),
        "rank_load_imbalance": float(rank_loads.max() / rank_loads.mean()),
    }


def _plot_progress(
    correlations: pd.DataFrame,
    projections: pd.DataFrame,
    scheduling: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 1, figsize=(9, 15), sharex=True)
    correlation_progress = correlations.groupby("iteration", sort=True).median(
        numeric_only=True
    )
    axes[0].plot(
        correlation_progress.index,
        correlation_progress["masked_positions_before_mean"],
        marker="o",
    )
    axes[0].set_ylabel("masked positions")
    axes[1].plot(
        correlation_progress.index,
        correlation_progress["unique_experts"],
        marker="o",
    )
    axes[1].set_ylabel("unique experts")
    for mapping_name, rows in projections.groupby("mapping", sort=True):
        progress = rows.groupby("iteration", sort=True)["max_rank_load"].median()
        axes[2].plot(progress.index, progress, marker="o", label=mapping_name)
    axes[2].set_ylabel("max-rank load")
    axes[2].legend()
    axes[3].plot(
        correlation_progress.index,
        correlation_progress["position_temporal_jaccard"],
        marker="o",
    )
    axes[3].set_ylabel("route persistence")
    for mapping_name, rows in scheduling.groupby("mapping", sort=True):
        progress = rows.groupby("iteration", sort=True)[
            "best_found_headroom_fraction"
        ].median()
        axes[4].plot(progress.index, progress, marker="o", label=mapping_name)
    axes[4].set_ylabel("batching headroom")
    axes[4].set_xlabel("denoising iteration")
    axes[4].legend()
    figure.suptitle("Native LLaDA2 denoising route progression")
    figure.tight_layout()
    figure.savefig(output / "native_denoising_progress.png", dpi=160)
    plt.close(figure)


def analyze(
    trace: Path,
    output: Path,
    *,
    world_size: int,
    requests_per_rank: int,
    batch_size: int,
    restarts: int,
    previous_route_gain_threshold: float,
) -> dict[str, object]:
    metadata, observations = _load_observations(trace)
    num_experts = int(metadata["num_experts"])
    expected_requests = world_size * requests_per_rank
    complete = [
        observation
        for observation in observations
        if len(observation.request_ids) == expected_requests
    ]
    if not complete:
        raise ValueError("no complete native observations are available")
    observations.sort(
        key=lambda item: (item.segment_id, item.block_id, item.denoise_step)
    )
    previous_by_segment: dict[int, NativeObservation] = {}
    correlation_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    scheduling_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for observation in observations:
        previous = previous_by_segment.get(observation.segment_id)
        temporal_jaccard, temporal_cosine = _temporal_metrics(
            observation, previous, num_experts
        )
        between_cosine = _between_request_cosine(observation, num_experts)
        correlation_rows.append(
            {
                "segment_id": observation.segment_id,
                "seed": observation.seed,
                "workload_class": observation.workload_class,
                "block_id": observation.block_id,
                "denoise_step": observation.denoise_step,
                "iteration": observation.iteration,
                "observed_requests": len(observation.request_ids),
                "complete_for_fixed_pool_screen": int(
                    len(observation.request_ids) == expected_requests
                ),
                "routed_positions": observation.routes.shape[2],
                "masked_positions_before_mean": float(
                    observation.masked_before.mean()
                ),
                "finalized_positions_this_step_mean": float(
                    observation.finalized.mean()
                ),
                "masked_position_spatial_jaccard": _spatial_jaccard(
                    observation, ROLE_CURRENT_MASKED, num_experts
                ),
                "finalized_position_spatial_jaccard": _spatial_jaccard(
                    observation, ROLE_CURRENT_FINALIZED, num_experts
                ),
                "position_temporal_jaccard": temporal_jaccard,
                "within_request_temporal_signature_cosine": temporal_cosine,
                "between_request_signature_cosine": between_cosine,
                "within_minus_between_signature_cosine": (
                    temporal_cosine - between_cosine
                ),
                "unique_experts": int(np.unique(observation.routes).size),
            }
        )
        for mapping_name in ("contiguous", "round_robin"):
            mapping = _mapping(mapping_name, num_experts, world_size)
            projection_rows.append(
                _rank_projection_record(
                    observation, mapping_name=mapping_name, mapping=mapping
                )
            )
            if len(observation.request_ids) == expected_requests:
                scheduling_rows.append(
                    _scheduling_record(
                        observation,
                        previous,
                        mapping_name=mapping_name,
                        mapping=mapping,
                        requests_per_rank=requests_per_rank,
                        batch_size=batch_size,
                        restarts=restarts,
                    )
                )
        coverage_rows.append(
            {
                "segment_id": observation.segment_id,
                "seed": observation.seed,
                "workload_class": observation.workload_class,
                "block_id": observation.block_id,
                "denoise_step": observation.denoise_step,
                "observed_requests": len(observation.request_ids),
                "expected_requests": expected_requests,
                "complete_for_fixed_pool_screen": int(
                    len(observation.request_ids) == expected_requests
                ),
            }
        )
        previous_by_segment[observation.segment_id] = observation

    correlations = pd.DataFrame(correlation_rows)
    projections = pd.DataFrame(projection_rows)
    scheduling = pd.DataFrame(scheduling_rows)
    summary_rows: list[dict[str, object]] = []
    for mapping_name, cells in scheduling.groupby("mapping", sort=True):
        headroom = cells["best_found_headroom_fraction"].to_numpy(dtype=float)
        capture = cells["previous_route_oracle_gain_capture"].to_numpy(dtype=float)
        capture = capture[np.isfinite(capture)]
        positive_headroom = headroom > 0
        if not positive_headroom.any():
            status = "NO_COMPOSITION_HEADROOM"
        elif capture.size and float(np.percentile(capture, 25)) >= (
            previous_route_gain_threshold
        ):
            status = "PREVIOUS_ROUTE_PASS"
        else:
            status = "PREDICTION_GAP"
        summary_rows.append(
            {
                "mapping": mapping_name,
                "status": status,
                "cells": len(cells),
                "positive_headroom_fraction": float(positive_headroom.mean()),
                "best_found_headroom_fraction_median": float(np.median(headroom)),
                "best_found_headroom_fraction_p25": float(np.percentile(headroom, 25)),
                "previous_route_gain_capture_median": (
                    float(np.median(capture)) if capture.size else float("nan")
                ),
                "previous_route_gain_capture_p25": (
                    float(np.percentile(capture, 25)) if capture.size else float("nan")
                ),
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    statuses = set(summary_table["status"])
    if statuses == {"PREVIOUS_ROUTE_PASS"}:
        overall_status = "ROUTE_SPACE_PASS"
    elif statuses == {"NO_COMPOSITION_HEADROOM"}:
        overall_status = "ROUTE_SPACE_NO_HEADROOM"
    else:
        overall_status = "ROUTE_SPACE_MAPPING_OR_PREDICTION_DEPENDENT"
    result = {
        "status": overall_status,
        "evidence_class": "MEASURED_ROUTE_SUPPORTING_CALIBRATION",
        "eligible_for_hardware_speedup_claim": False,
        "source_trace": str(trace),
        "source_model": metadata["model_identifier"],
        "source_revision": metadata["model_revision"],
        "complete_observations": len(complete),
        "mappings": ["contiguous", "round_robin"],
        "planner_objective": "sum_over_batch_layer_of_max_destination_rank_load",
        "best_found_is_exact_oracle": False,
        "previous_route_gain_threshold": previous_route_gain_threshold,
        "mapping_summaries": summary_rows,
        "semantic_limit": (
            "Route-space opportunity only. This result contains no measured EP "
            "dispatch, expert-compute, combine, or scheduler wall time."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(output / "native_route_correlations.csv", index=False)
    projections.to_csv(output / "native_rank_projection.csv", index=False)
    scheduling.to_csv(output / "native_scheduling_opportunity.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "native_trace_coverage.csv", index=False)
    summary_table.to_csv(output / "native_opportunity_by_mapping.csv", index=False)
    _plot_progress(correlations, projections, scheduling, output)
    (output / "native_opportunity_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    result = analyze(
        args.trace,
        args.output,
        world_size=args.world_size,
        requests_per_rank=args.requests_per_rank,
        batch_size=args.batch_size,
        restarts=args.restarts,
        previous_route_gain_threshold=args.previous_route_gain_threshold,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
