from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hardware import screen_measured_router_trace
from hardware.analyze_native_route_opportunity import (
    _paired_bitset_jaccard,
    _route_bitsets,
    analyze,
)


def _write_trace(path: Path) -> None:
    path.mkdir()
    metadata = {
        "trace_kind": "native_llada2_initial_block_router_observational",
        "eligible_for_scheduler_opportunity_screening": True,
        "model_identifier": "fixture",
        "model_revision": "immutable-fixture",
        "num_experts": 4,
        "top_k": 2,
    }
    (path / "route_metadata.json").write_text(json.dumps(metadata))
    rows: list[dict[str, int]] = []
    for segment_id, seed in enumerate((17, 29)):
        for request_id in range(8):
            preferred_rank = (request_id + segment_id) % 2
            experts = (preferred_rank * 2, preferred_rank * 2 + 1)
            for layer_id in (1, 2):
                for route_slot, expert_id in enumerate(experts):
                    rows.append(
                        {
                            "trace_phase": "native_denoising",
                            "segment_id": segment_id,
                            "seed": seed,
                            "request_id": request_id,
                            "active_positions": 1,
                            "block_width": 1,
                            "iteration": 0,
                            "layer_id": layer_id,
                            "position_id": 0,
                            "route_slot": route_slot,
                            "expert_id": expert_id,
                            "compute_positions": 1,
                            "masked_positions_before_step": 1,
                            "masked_positions_after_step": 0,
                        }
                    )
    with (path / "routes_observational.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_route_bitsets_preserve_exact_jaccard() -> None:
    left = np.asarray([[[1, 2, 65], [4, 5, 6]]], dtype=np.int64)
    right = np.asarray([[[2, 3, 65], [4, 7, 8]]], dtype=np.int64)

    values = _paired_bitset_jaccard(
        _route_bitsets(left, 128),
        _route_bitsets(right, 128),
    )

    assert np.allclose(values, [[0.5, 0.2]])


def _write_dense_trace(path: Path) -> None:
    path.mkdir()
    arrays: dict[str, np.ndarray] = {}
    observations: list[dict[str, object]] = []
    for microbatch, request_start in enumerate((0, 4)):
        routes = np.empty((4, 2, 1, 2), dtype=np.uint16)
        for local_request in range(4):
            request_id = request_start + local_request
            preferred_rank = request_id % 2
            routes[local_request, :, 0, :] = (
                preferred_rank * 2,
                preferred_rank * 2 + 1,
            )
        key = f"denoise_s0_mb{microbatch}_step0"
        arrays[key] = routes
        observations.append(
            {
                "trace_phase": "native_denoising",
                "segment_id": 0,
                "seed": 17,
                "active_positions": 1,
                "block_width": 1,
                "position_width_source": "native_trajectory_fixed_block_compute",
                "model_forward_positions": 1,
                "iteration": 0,
                "array_key": key,
                "request_ids": json.dumps(list(range(request_start, request_start + 4))),
                "masked_positions_before_step": json.dumps([1] * 4),
                "masked_positions_after_step": json.dumps([0] * 4),
            }
        )
    arrays_path = path / "routes_dense.npz"
    manifest_path = path / "route_observations.csv"
    np.savez_compressed(arrays_path, **arrays)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    metadata = {
        "trace_kind": "native_llada2_dense_router_observational",
        "eligible_for_scheduler_opportunity_screening": True,
        "model_identifier": "fixture",
        "model_revision": "immutable-fixture",
        "num_layers": 3,
        "first_sparse_layer": 1,
        "num_experts": 4,
        "top_k": 2,
        "artifact_sha256": {
            "routes_dense.npz": _sha256(arrays_path),
            "route_observations.csv": _sha256(manifest_path),
        },
    }
    (path / "route_metadata.json").write_text(json.dumps(metadata))


def test_measured_route_screen_uses_planner_objective(tmp_path: Path, monkeypatch) -> None:
    trace = tmp_path / "trace"
    output = tmp_path / "screen"
    _write_trace(trace)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_measured_router_trace.py",
            str(trace),
            "--output",
            str(output),
            "--world-size",
            "2",
            "--requests-per-rank",
            "4",
            "--batch-size",
            "2",
            "--restarts",
            "4",
        ],
    )

    screen_measured_router_trace.main()

    cells = pd.read_csv(output / "measured_screening_cells.csv")
    by_cell = pd.read_csv(output / "measured_screening_by_cell.csv")
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(cells) == 2
    assert len(by_cell) == 1
    assert (cells["objective_lower_bound"] <= cells["best_found_objective"]).all()
    assert (
        cells["realized_objective_reduction_fraction"].between(0.0, 1.0).all()
    )
    assert metadata["eligible_for_gate3_timing_authorization"] is False
    assert metadata["planner_objective"].startswith("sum_over_batch_layer")


def test_dense_measured_route_screen_preserves_step_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    trace = tmp_path / "dense-trace"
    output = tmp_path / "dense-screen"
    _write_dense_trace(trace)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_measured_router_trace.py",
            str(trace),
            "--output",
            str(output),
            "--world-size",
            "2",
            "--requests-per-rank",
            "4",
            "--batch-size",
            "2",
            "--restarts",
            "4",
            "--all-iterations",
        ],
    )

    screen_measured_router_trace.main()

    cells = pd.read_csv(output / "measured_screening_cells.csv")
    coverage = pd.read_csv(output / "measured_screening_coverage.csv")
    assert len(cells) == 1
    assert len(coverage) == 1
    assert coverage.loc[0, "complete_for_fixed_pool_screen"] == 1
    assert cells.loc[0, "trace_phase"] == "native_denoising"
    assert cells.loc[0, "iteration"] == 0
    assert cells.loc[0, "position_width_source"] == (
        "native_trajectory_fixed_block_compute"
    )
    assert cells.loc[0, "masked_positions_before_step_mean"] == 1.0
    assert cells.loc[0, "masked_positions_after_step_mean"] == 0.0
    assert cells.loc[0, "objective_lower_bound"] <= cells.loc[0, "fifo_objective"]


def test_native_opportunity_analysis_checks_both_mappings(tmp_path: Path) -> None:
    trace = tmp_path / "dense-trace"
    output = tmp_path / "native-opportunity"
    _write_dense_trace(trace)

    result = analyze(
        trace,
        output,
        world_size=2,
        requests_per_rank=4,
        batch_size=2,
        restarts=4,
        previous_route_gain_threshold=0.8,
    )

    scheduling = pd.read_csv(output / "native_scheduling_opportunity.csv")
    correlations = pd.read_csv(output / "native_route_correlations.csv")
    projections = pd.read_csv(output / "native_rank_projection.csv")
    coverage = pd.read_csv(output / "native_trace_coverage.csv")
    assert result["eligible_for_hardware_speedup_claim"] is False
    assert set(scheduling["mapping"]) == {"contiguous", "round_robin"}
    assert len(correlations) == 1
    assert correlations.loc[0, "routed_positions"] == 1
    assert len(projections) == 2
    assert coverage.loc[0, "complete_for_fixed_pool_screen"] == 1
