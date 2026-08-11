from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from hardware import analyze_timing_gate_ep4


def write_rank(path: Path, rank: int) -> None:
    rows: list[dict[str, object]] = []
    base_by_mode = {
        "local_copy": 1.0,
        "nccl_minimal": 1.5,
    }
    real_by_k = {1: 1.55, 16: 2.5, 64: 4.0}
    for positions in (1, 16, 64):
        for repetition in range(10):
            for mode in ("local_copy", "nccl_minimal", "nccl_real"):
                base = real_by_k[positions] if mode == "nccl_real" else base_by_mode[mode]
                gpu_ms = base + rank * 0.001 + repetition * 0.0001
                rows.append(
                    {
                        "rank": rank,
                        "active_positions": positions,
                        "mode": mode,
                        "repetition": repetition,
                        "warmup": 0,
                        "gpu_path_ms": gpu_ms,
                        "summed_layer_ms": gpu_ms * 0.98,
                        "dispatch_ms": gpu_ms * 0.3,
                        "expert_compute_ms": gpu_ms * 0.4,
                        "combine_ms": gpu_ms * 0.28,
                        "packing_ms": gpu_ms * 0.05,
                        "local_copy_memory_ms": gpu_ms * 0.03,
                        "unpacking_ms": gpu_ms * 0.02,
                        "communicated_payload_bytes_per_layer": positions,
                    }
                )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_timing_gate_distinguishes_powered_k(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "analysis"
    input_path.mkdir()
    for rank in range(4):
        write_rank(input_path / f"rank{rank}_timing_gate.csv", rank)
    (input_path / "metadata.json").write_text(
        json.dumps(
            {
                "measurement_protocol": "timing_identifiability_v2",
                "modes": ["local_copy", "nccl_minimal", "nccl_real"],
                "active_positions": [1, 16, 64],
            }
        )
    )
    screening_path = tmp_path / "scheduler_screening_by_k.csv"
    pd.DataFrame(
        {
            "active_positions": [1, 16, 64],
            "routing_mode": ["test", "test", "test"],
            "fifo_imbalance_p25": [1.1, 1.3, 1.5],
            "objective_achievability_p25": [0.5, 0.6, 0.7],
            "realized_objective_reduction_fraction_p25": [0.2, 0.2, 0.2],
        }
    ).to_csv(screening_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_timing_gate_ep4.py",
            str(input_path),
            "--output",
            str(output_path),
            "--bootstrap-samples",
            "100",
            "--screening-profile",
            str(screening_path),
        ],
    )

    analyze_timing_gate_ep4.main()

    report = json.loads((output_path / "report.json").read_text())
    assert report["status"] == "PASS-POWERED"
    assert report["powered_active_positions"] == [16, 64]
    assert report["phase_attribution_eligible"] is True
