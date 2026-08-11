#!/usr/bin/env python3
"""Analyze timing cleanliness and scheduler-accessible EP time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ("local_copy", "nccl_minimal", "nccl_real")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-mde-percent", type=float, default=2.0)
    parser.add_argument("--screening-profile", type=Path, required=True)
    parser.add_argument("--power-safety-multiplier", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-unattributed-fraction", type=float, default=0.15)
    parser.add_argument("--phase-max-unattributed-fraction", type=float, default=0.05)
    parser.add_argument("--phase-max-mode-gap-fraction", type=float, default=0.01)
    return parser.parse_args()


def bootstrap_median_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return tuple(float(value) for value in np.percentile(medians, [2.5, 97.5]))


def main() -> None:
    args = parse_args()
    if args.target_mde_percent <= 0:
        raise ValueError("target MDE must be positive")
    screening = pd.read_csv(args.screening_profile)
    metadata_path = args.input / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("timing gate metadata.json is required")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("measurement_protocol") != "timing_identifiability_v2":
        raise ValueError("unsupported timing-gate measurement protocol")
    if metadata.get("modes") != list(MODES):
        raise ValueError("timing-gate metadata mode contract is inconsistent")
    required_screening_columns = {
        "active_positions",
        "routing_mode",
        "fifo_imbalance_p25",
        "objective_achievability_p25",
        "realized_objective_reduction_fraction_p25",
    }
    missing_screening = required_screening_columns - set(screening.columns)
    if missing_screening:
        raise ValueError(f"screening profile lacks columns: {missing_screening}")
    paths = sorted(args.input.glob("rank*_timing_gate.csv"))
    if len(paths) != 4:
        raise ValueError(f"expected four rank files, found {len(paths)}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw = raw[raw["warmup"] == 0].copy()
    if sorted(raw["active_positions"].unique().tolist()) != sorted(
        metadata.get("active_positions", [])
    ):
        raise ValueError("timing rows disagree with metadata active positions")
    key_columns = ["rank", "active_positions", "mode", "repetition"]
    if raw.duplicated(key_columns).any():
        raise ValueError("timing gate contains duplicate rank/K/mode/repetition rows")
    if set(raw["rank"].unique()) != {0, 1, 2, 3}:
        raise ValueError("timing gate must contain ranks 0..3")
    critical = (
        raw.groupby(["active_positions", "mode", "repetition"], sort=True)
        .agg(
            gpu_path_ms=("gpu_path_ms", "max"),
            summed_layer_ms=("summed_layer_ms", "max"),
            dispatch_ms=("dispatch_ms", "max"),
            expert_compute_max_ms=("expert_compute_ms", "max"),
            expert_compute_mean_ms=("expert_compute_ms", "mean"),
            combine_ms=("combine_ms", "max"),
            packing_max_ms=("packing_ms", "max"),
            local_copy_memory_max_ms=("local_copy_memory_ms", "max"),
            local_copy_memory_mean_ms=("local_copy_memory_ms", "mean"),
            unpacking_max_ms=("unpacking_ms", "max"),
            communicated_payload_bytes=(
                "communicated_payload_bytes_per_layer",
                "max",
            ),
        )
        .reset_index()
    )
    observed_modes = set(critical["mode"])
    if observed_modes != set(MODES):
        raise ValueError(f"expected modes {MODES}, found {sorted(observed_modes)}")
    critical["unattributed_ms"] = (
        critical["gpu_path_ms"] - critical["summed_layer_ms"]
    )
    critical["unattributed_fraction"] = (
        critical["unattributed_ms"] / critical["gpu_path_ms"]
    )
    critical["unattributed_fraction_abs"] = critical[
        "unattributed_fraction"
    ].abs()
    critical["expert_compute_imbalance_ms"] = (
        critical["expert_compute_max_ms"] - critical["expert_compute_mean_ms"]
    )
    rng = np.random.default_rng(20260805)
    summaries: list[dict[str, object]] = []
    for (positions, mode), group in critical.groupby(
        ["active_positions", "mode"], sort=True
    ):
        values = group["gpu_path_ms"].to_numpy(float)
        low, high = bootstrap_median_ci(values, args.bootstrap_samples, rng)
        summaries.append(
            {
                "active_positions": int(positions),
                "mode": mode,
                "samples": len(values),
                "p50_gpu_ms": float(np.median(values)),
                "bootstrap_p50_low_ms": low,
                "bootstrap_p50_high_ms": high,
                "cv": float(np.std(values, ddof=1) / np.mean(values)),
                "p50_unattributed_fraction": float(
                    np.median(group["unattributed_fraction"])
                ),
                "p50_unattributed_fraction_abs": float(
                    np.median(group["unattributed_fraction_abs"])
                ),
                "p50_dispatch_ms": float(np.median(group["dispatch_ms"])),
                "p50_compute_max_ms": float(
                    np.median(group["expert_compute_max_ms"])
                ),
                "p50_compute_mean_ms": float(
                    np.median(group["expert_compute_mean_ms"])
                ),
                "p50_compute_imbalance_ms": float(
                    np.median(group["expert_compute_imbalance_ms"])
                ),
                "p50_combine_ms": float(np.median(group["combine_ms"])),
                "p50_packing_max_ms": float(np.median(group["packing_max_ms"])),
                "p50_local_copy_memory_max_ms": float(
                    np.median(group["local_copy_memory_max_ms"])
                ),
                "p50_local_copy_memory_mean_ms": float(
                    np.median(group["local_copy_memory_mean_ms"])
                ),
                "p50_unpacking_max_ms": float(
                    np.median(group["unpacking_max_ms"])
                ),
                "communicated_payload_bytes": int(
                    group["communicated_payload_bytes"].iloc[0]
                ),
            }
        )
    summary = pd.DataFrame.from_records(summaries)
    paired = critical.pivot(
        index=["active_positions", "repetition"],
        columns="mode",
        values="gpu_path_ms",
    ).reset_index()
    if paired[list(MODES)].isna().any().any():
        raise ValueError("timing gate has an incomplete paired mode set")
    paired["launch_floor_ms"] = paired["nccl_minimal"] - paired["local_copy"]
    paired["accessible_payload_ms"] = paired["nccl_real"] - paired["nccl_minimal"]
    paired["total_nccl_premium_ms"] = paired["nccl_real"] - paired["local_copy"]
    paired["accessible_fraction"] = (
        paired["accessible_payload_ms"] / paired["nccl_real"]
    )
    target_fraction = args.target_mde_percent / 100.0
    required_fraction = target_fraction * args.power_safety_multiplier
    comparisons: list[dict[str, object]] = []
    for positions, group in paired.groupby("active_positions", sort=True):
        reference_mean = float(group["nccl_real"].mean())
        paired_sd_fraction = float(
            group["accessible_payload_ms"].std(ddof=1) / reference_mean
        )
        paired_repetitions = int(
            np.ceil(
                (1.96 + 0.84) ** 2
                * paired_sd_fraction**2
                / target_fraction**2
            )
        )
        accessible_fraction = float(np.median(group["accessible_fraction"]))
        launch_low, launch_high = bootstrap_median_ci(
            group["launch_floor_ms"].to_numpy(float),
            args.bootstrap_samples,
            rng,
        )
        total_low, total_high = bootstrap_median_ci(
            group["total_nccl_premium_ms"].to_numpy(float),
            args.bootstrap_samples,
            rng,
        )
        accessible_low, accessible_high = bootstrap_median_ci(
            group["accessible_payload_ms"].to_numpy(float),
            args.bootstrap_samples,
            rng,
        )
        comparisons.append(
            {
                "active_positions": int(positions),
                "local_copy_p50_ms": float(np.median(group["local_copy"])),
                "nccl_minimal_p50_ms": float(np.median(group["nccl_minimal"])),
                "nccl_real_p50_ms": float(np.median(group["nccl_real"])),
                "launch_floor_p50_ms": float(np.median(group["launch_floor_ms"])),
                "launch_floor_ci_low_ms": launch_low,
                "launch_floor_ci_high_ms": launch_high,
                "accessible_payload_p50_ms": float(
                    np.median(group["accessible_payload_ms"])
                ),
                "accessible_payload_ci_low_ms": accessible_low,
                "accessible_payload_ci_high_ms": accessible_high,
                "total_nccl_premium_ci_low_ms": total_low,
                "total_nccl_premium_ci_high_ms": total_high,
                "accessible_fraction_p50": accessible_fraction,
                "paired_accessible_sd_fraction": paired_sd_fraction,
                "screening_paired_repetitions": paired_repetitions,
            }
        )
    comparison = pd.DataFrame.from_records(comparisons).merge(
        screening[sorted(required_screening_columns)],
        on="active_positions",
        validate="one_to_many",
    )
    comparison["screened_recoverable_fraction"] = (
        comparison["accessible_fraction_p50"]
        * comparison["realized_objective_reduction_fraction_p25"]
    )
    comparison["required_recoverable_fraction"] = required_fraction
    comparison["scheduler_powered"] = (
        comparison["screened_recoverable_fraction"] >= required_fraction
    ).astype(int)
    max_unattributed = float(summary["p50_unattributed_fraction_abs"].max())
    mode_gap = (
        summary.groupby("active_positions")["p50_unattributed_fraction_abs"]
        .agg(lambda values: float(values.max() - values.min()))
        .max()
    )
    total_nccl_identified = bool(
        (comparison["total_nccl_premium_ci_low_ms"] > 0).all()
    )
    launch_floor_identified = bool(
        (comparison["launch_floor_ci_low_ms"] > 0).all()
    )
    harness_valid = (
        total_nccl_identified
        and launch_floor_identified
        and max_unattributed <= args.max_unattributed_fraction
    )
    powered = comparison[comparison["scheduler_powered"].astype(bool)]
    powered_k = sorted(int(value) for value in powered["active_positions"].unique())
    powered_cells = [
        {
            "active_positions": int(row.active_positions),
            "routing_mode": str(row.routing_mode),
        }
        for row in powered.itertuples(index=False)
    ]
    if not harness_valid:
        status = "FAIL"
    elif powered_k:
        status = "PASS-POWERED"
    else:
        status = "PASS-UNPOWERED"
    phase_attribution_eligible = bool(
        max_unattributed <= args.phase_max_unattributed_fraction
        and mode_gap <= args.phase_max_mode_gap_fraction
    )
    output = args.output or (args.input / "timing_gate_analysis")
    output.mkdir(parents=True, exist_ok=True)
    critical.to_csv(output / "critical_samples.csv", index=False)
    paired.to_csv(output / "paired_mode_differences.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparison.to_csv(output / "scheduler_accessible_time.csv", index=False)
    report = {
        "status": status,
        "harness_valid": harness_valid,
        "powered_active_positions": powered_k,
        "powered_scheduler_cells": powered_cells,
        "total_nccl_premium_positive_at_all_k": total_nccl_identified,
        "launch_floor_positive_at_all_k": launch_floor_identified,
        "max_p50_unattributed_fraction": max_unattributed,
        "max_unattributed_mode_gap_fraction": float(mode_gap),
        "phase_attribution_eligible": phase_attribution_eligible,
        "target_mde_percent": args.target_mde_percent,
        "target_mde_denominator": "nccl_real_gpu_interval",
        "power_safety_multiplier": args.power_safety_multiplier,
        "screening_profile": str(args.screening_profile),
        "measurement_protocol": metadata["measurement_protocol"],
        "interpretation": {
            "PASS-POWERED": "Run only scheduler cells whose measured screen passes.",
            "PASS-UNPOWERED": (
                "Harness is clean; skip the scheduler matrix and move to native "
                "adapter correctness."
            ),
            "FAIL": "Stop performance work and repair the measurement harness.",
        }[status],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
