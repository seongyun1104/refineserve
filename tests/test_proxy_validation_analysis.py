from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from hardware import analyze_proxy_validation_ep4


def test_proxy_validation_reports_aligned_direction(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "analysis"
    timing_path = tmp_path / "timing"
    input_path.mkdir()
    timing_path.mkdir()
    (input_path / "metadata.json").write_text(
        json.dumps(
            {
                "measurement_protocol": (
                    "constructed_objective_to_time_proxy_v3_two_dose_"
                    "local_minimal_real"
                ),
                "active_positions": 64,
                "run_arms": [
                    "fifo_local_copy_control",
                    "fifo_nccl_minimal_control",
                    "fifo_constructed",
                    "dose_083_constructed",
                    "balanced_constructed",
                ],
            }
        )
    )
    pd.DataFrame(
        {
            "active_positions": [64],
            "accessible_fraction_p50": [0.2],
        }
    ).to_csv(timing_path / "scheduler_accessible_time.csv", index=False)
    for rank in range(4):
        rows: list[dict[str, object]] = []
        for repetition in range(10):
            for arm, gpu_ms, objective, reduction in (
                ("fifo_local_copy_control", 6.0, 300.0, 0.0),
                ("fifo_nccl_minimal_control", 7.0, 300.0, 0.0),
                ("fifo_constructed", 10.0, 300.0, 0.0),
                ("dose_083_constructed", 9.5, 275.0, 1.0 / 12.0),
                ("balanced_constructed", 8.0, 200.0, 1.0 / 3.0),
            ):
                rows.append(
                    {
                        "rank": rank,
                        "arm": arm,
                        "active_positions": 64,
                        "repetition": repetition,
                        "warmup": 0,
                        "execution_index": 0,
                        "planner_objective": objective,
                        "objective_reduction_fraction": reduction,
                        "gpu_path_ms": gpu_ms + rank * 0.001,
                        "dispatch_ms": gpu_ms * 0.3,
                        "expert_compute_ms": gpu_ms * 0.4,
                        "combine_ms": gpu_ms * 0.3,
                    }
                )
        with (input_path / f"rank{rank}_proxy_validation.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_proxy_validation_ep4.py",
            str(input_path),
            "--output",
            str(output_path),
            "--bootstrap-samples",
            "100",
            "--timing-gate-analysis",
            str(timing_path),
        ],
    )

    analyze_proxy_validation_ep4.main()

    report = json.loads((output_path / "report.json").read_text())
    assert report["status"] == "PROXY_TIME_ALIGNED"
    assert report["objective_reduction_fraction"] == 1.0 / 3.0
    assert report["measured_latency_reduction_median_fraction"] > 0.19
    assert report["transmission_slope_median"] is not None
    assert len(report["dose_arms"]) == 2
    assert report["constructed_accessibility"]["identified"] is True
    assert report["accessible_fraction"] == pytest.approx(0.3, rel=1e-3)
    assert report["gate2_balanced_route_accessible_fraction_crosscheck"] == 0.2
