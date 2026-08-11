from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from refineserve.calibration import (
    CalibrationArtifact,
    CalibrationRangeError,
    fit_calibration,
)
from refineserve.config import (
    CalibrationConfig,
    ModelConfig,
    SimulationConfig,
    WorkloadConfig,
)
from refineserve.simulator import Simulator
from refineserve.trace_bundle import RouteTraceBundle, TraceValidationError


def _write_bundle(root: Path, model: ModelConfig, *, complete: bool = True) -> None:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "trace_kind": "native_position_parallel",
                "created_at_utc": "2026-08-04T00:00:00Z",
                "source": "pytest-fixture",
                "model_identifier": "pytest-moe",
                "model_revision": "test-revision",
                "random_seed": 17,
                "model": asdict(model),
                "expert_to_rank_mapping": [
                    [expert_id % model.num_gpus for expert_id in range(model.num_experts)]
                    for _ in range(model.num_layers)
                ],
                "measurement_environment": {
                    "gpu_model": "synthetic",
                    "gpu_count": model.num_gpus,
                    "topology": "synthetic",
                    "node_scope": "single_node",
                    "cuda_version": "not_applicable",
                    "nccl_version": "not_applicable",
                    "pytorch_version": "pytest",
                    "kernel_backend": "synthetic",
                    "dtype": "bf16",
                    "intermediate_size": 256,
                    "concurrent_streams": 1,
                    "warmup_count": 0,
                    "measurement_iterations": 1,
                },
                "units": {"latency": "ms", "size": "bytes"},
            }
        )
    )
    route_rows: list[dict[str, int]] = []
    widths = (2, 1)
    for iteration, width in enumerate(widths):
        for layer_id in range(model.num_layers):
            for position_id in range(width):
                for slot in range(model.top_k):
                    if not complete and iteration == 1 and layer_id == 1 and slot == 1:
                        continue
                    route_rows.append(
                        {
                            "request_id": 0,
                            "iteration": iteration,
                            "layer_id": layer_id,
                            "position_id": position_id,
                            "route_slot": slot,
                            "expert_id": (layer_id + position_id + slot) % model.num_experts,
                            "routing_weight": 1.0 / model.top_k,
                            "batch_size": 1,
                            "active_position_count": width,
                            "context_length": 16 + iteration,
                        }
                    )
    with (root / "routes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(route_rows[0]))
        writer.writeheader()
        writer.writerows(route_rows)
    prior_rows = [
        {
            "request_id": 0,
            "layer_id": layer_id,
            "route_slot": slot,
            "expert_id": (layer_id + slot) % model.num_experts,
        }
        for layer_id in range(model.num_layers)
        for slot in range(model.top_k)
    ]
    with (root / "route_priors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prior_rows[0]))
        writer.writeheader()
        writer.writerows(prior_rows)


def _write_timing_samples(root: Path) -> None:
    expert_rows = [
        dict(
            sample_id="e0", gpu_id=0, expert_id=0, token_count=1,
            latency_ms=9.0, warmup=1, repetition=0,
        ),
        dict(
            sample_id="e1", gpu_id=0, expert_id=0, token_count=1,
            latency_ms=0.10, warmup=0, repetition=1,
        ),
        dict(
            sample_id="e2", gpu_id=1, expert_id=1, token_count=1,
            latency_ms=0.20, warmup=0, repetition=1,
        ),
        dict(
            sample_id="e3", gpu_id=0, expert_id=0, token_count=4,
            latency_ms=0.30, warmup=0, repetition=2,
        ),
    ]
    with (root / "expert_kernel_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expert_rows[0]))
        writer.writeheader()
        writer.writerows(expert_rows)
    network_rows = [
        dict(
            sample_id="n1", collective="all_to_all", active_ranks=2,
            message_count=2, transferred_bytes=128, latency_ms=0.05,
            warmup=0, repetition=1,
        ),
        dict(
            sample_id="n2", collective="all_to_all", active_ranks=2,
            message_count=2, transferred_bytes=256, latency_ms=0.08,
            warmup=0, repetition=2,
        ),
    ]
    with (root / "network_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(network_rows[0]))
        writer.writeheader()
        writer.writerows(network_rows)


def test_trace_bundle_replays_strict_routes_and_records_checksum(tmp_path: Path) -> None:
    model = ModelConfig(
        num_layers=2,
        num_experts=4,
        top_k=2,
        num_gpus=2,
        hidden_size=128,
    )
    trace_root = tmp_path / "trace"
    _write_bundle(trace_root, model)
    base = SimulationConfig(
        model=model,
        workload=WorkloadConfig(
            num_requests=1,
            output_tokens=2,
            max_batch_size=1,
            diffusion_block_size=2,
            active_position_schedule=(2, 1),
        ),
    )
    config = replace(
        base,
        router=replace(base.router, source="trace", trace_path=str(trace_root)),
    )

    result = Simulator(config, "diffusion").run()
    bundle = RouteTraceBundle.load(trace_root, expected_model=model)

    assert result.summary.finalized_tokens == 2
    assert result.summary.trace_bundle_sha256 == bundle.bundle_sha256
    assert len(bundle.bundle_sha256) == 64


def test_trace_bundle_rejects_incomplete_top_k_group(tmp_path: Path) -> None:
    model = ModelConfig(num_layers=2, num_experts=4, top_k=2, num_gpus=2)
    trace_root = tmp_path / "trace"
    _write_bundle(trace_root, model, complete=False)

    with pytest.raises(TraceValidationError, match="must contain slots"):
        RouteTraceBundle.load(trace_root, expected_model=model)


def test_trace_bundle_rejects_model_mismatch(tmp_path: Path) -> None:
    model = ModelConfig(num_layers=2, num_experts=4, top_k=2, num_gpus=2)
    trace_root = tmp_path / "trace"
    _write_bundle(trace_root, model)

    with pytest.raises(TraceValidationError, match="does not match"):
        RouteTraceBundle.load(
            trace_root,
            expected_model=replace(model, num_layers=3),
        )


def test_calibration_ignores_warmup_and_forbids_extrapolation(tmp_path: Path) -> None:
    model = ModelConfig(num_layers=2, num_experts=4, top_k=2, num_gpus=2)
    trace_root = tmp_path / "trace"
    _write_bundle(trace_root, model)
    _write_timing_samples(trace_root)
    artifact = fit_calibration(RouteTraceBundle.load(trace_root, expected_model=model))

    assert artifact.expert_kernel_curve is not None
    assert artifact.expert_kernel_curve.points[0].raw_median_ms == pytest.approx(0.15)
    assert artifact.expert_kernel_curve.latency_ms(2) == pytest.approx(0.20)
    assert len(artifact.network_curves) == 1
    assert artifact.network_curves[0].latency_by_bytes.latency_ms(192) == pytest.approx(0.065)
    with pytest.raises(CalibrationRangeError, match="outside measured range"):
        artifact.expert_kernel_curve.latency_ms(8)

    artifact_path = artifact.write(tmp_path / "calibration.json")
    restored = CalibrationArtifact.load(
        artifact_path,
        expected_bundle_sha256=artifact.source_bundle_sha256,
    )
    assert restored == artifact


def test_expert_calibration_drives_replay_cost_and_records_provenance(tmp_path: Path) -> None:
    model = ModelConfig(
        num_layers=2,
        num_experts=4,
        top_k=2,
        num_gpus=2,
        hidden_size=128,
    )
    trace_root = tmp_path / "trace"
    _write_bundle(trace_root, model)
    _write_timing_samples(trace_root)
    bundle = RouteTraceBundle.load(trace_root, expected_model=model)
    artifact_path = fit_calibration(bundle).write(tmp_path / "calibration.json")
    base = SimulationConfig(
        model=model,
        workload=WorkloadConfig(
            num_requests=1,
            output_tokens=2,
            max_batch_size=1,
            diffusion_block_size=2,
            active_position_schedule=(2, 1),
        ),
    )
    traced = replace(
        base,
        router=replace(base.router, source="trace", trace_path=str(trace_root)),
    )
    calibrated = replace(
        traced,
        calibration=CalibrationConfig(
            artifact_path=str(artifact_path),
            require_trace_checksum_match=True,
        ),
    )

    synthetic_result = Simulator(traced, "diffusion").run()
    calibrated_result = Simulator(calibrated, "diffusion").run()

    assert calibrated_result.summary.makespan_ms != synthetic_result.summary.makespan_ms
    assert calibrated_result.summary.calibration_source_bundle_sha256 == bundle.bundle_sha256
