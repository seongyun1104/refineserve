#!/usr/bin/env python3
"""Estimate within-cell dose response for coordinated composition replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_KEYS = ["seed", "routing_mode", "active_positions", "repetition"]
REPLAY_KEYS = [*PAIR_KEYS, "scheduler"]
CELL_KEYS = ["seed", "routing_mode", "active_positions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregated", type=Path)
    parser.add_argument("--timing-gate-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--minimum-seeds-for-inference", type=int, default=5)
    return parser.parse_args()


def within_cell_slope(frame: pd.DataFrame) -> float:
    x = frame["coordinated_predicted_reduction_percent"].astype(float)
    y = frame["paired_latency_change_percent"].astype(float)
    centered_x = x - frame.assign(_x=x).groupby(CELL_KEYS)["_x"].transform("mean")
    centered_y = y - frame.assign(_y=y).groupby(CELL_KEYS)["_y"].transform("mean")
    denominator = float(np.square(centered_x).sum())
    if denominator <= 0:
        raise ValueError("dose design has no within-cell predicted-reduction variation")
    return float((centered_x * centered_y).sum() / denominator)


def main() -> None:
    args = parse_args()
    runs = pd.read_csv(args.aggregated / "aggregated_runs.csv")
    layers = pd.read_csv(args.aggregated / "aggregated_layers.csv")
    fifo = runs[runs["scheduler"] == "fifo"].copy()
    coordinated = runs[runs["scheduler"].str.startswith("coordinated_")].copy()
    if coordinated.empty:
        raise ValueError("coordinated replay rows are required")

    fifo_reference = fifo[PAIR_KEYS + ["gpu_critical_path_ms"]].rename(
        columns={"gpu_critical_path_ms": "fifo_gpu_ms"}
    )
    coordinated = coordinated.merge(
        fifo_reference, on=PAIR_KEYS, validate="many_to_one"
    )
    coordinated["paired_latency_change_percent"] = (
        coordinated["gpu_critical_path_ms"] / coordinated["fifo_gpu_ms"] - 1.0
    ) * 100.0
    fifo["fifo_gpu_ms"] = fifo["gpu_critical_path_ms"]
    fifo["paired_latency_change_percent"] = 0.0
    fifo["coordinated_predicted_reduction_percent"] = 0.0
    fifo["coordinated_dose_target_fraction"] = 0.0
    fifo["coordinated_dose_achieved_fraction"] = 0.0
    analysis = pd.concat([fifo, coordinated], ignore_index=True, sort=False)

    layer_imbalance = (
        layers[
            (layers["scheduler"] == "fifo")
            | layers["scheduler"].str.startswith("coordinated_")
        ]
        .groupby(REPLAY_KEYS, sort=True)["rank_load_imbalance"]
        .mean()
        .rename("measured_rank_load_imbalance")
        .reset_index()
    )
    analysis = analysis.merge(layer_imbalance, on=REPLAY_KEYS, validate="one_to_one")
    accessible = pd.read_csv(
        args.timing_gate_analysis / "scheduler_accessible_time.csv"
    )[["active_positions", "accessible_fraction_p50"]].drop_duplicates()
    analysis = analysis.merge(
        accessible, on="active_positions", how="left", validate="many_to_one"
    )
    if analysis["accessible_fraction_p50"].isna().any():
        raise ValueError("timing gate lacks an active-position value used by replay")

    distinct_positive_doses = (
        coordinated.groupby(CELL_KEYS)["coordinated_dose_achieved_fraction"]
        .nunique()
        .rename("distinct_positive_doses")
        .reset_index()
    )
    eligible_cells = distinct_positive_doses[
        distinct_positive_doses["distinct_positive_doses"] >= 3
    ][CELL_KEYS]
    analysis = analysis.merge(eligible_cells, on=CELL_KEYS, validate="many_to_one")
    if analysis.empty:
        raise ValueError("no cell has at least three distinct positive replay doses")

    slope = within_cell_slope(analysis)
    seeds = sorted(int(value) for value in analysis["seed"].unique())
    rng = np.random.default_rng(20260807)
    bootstrap_slopes = []
    for _ in range(args.bootstrap_samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sample = pd.concat(
            [analysis[analysis["seed"] == seed] for seed in sampled_seeds],
            ignore_index=True,
        )
        bootstrap_slopes.append(within_cell_slope(sample))
    inference_ready = len(seeds) >= args.minimum_seeds_for_inference
    report = {
        "samples": len(analysis),
        "eligible_cells": len(eligible_cells),
        "seed_count": len(seeds),
        "minimum_seeds_for_inference": args.minimum_seeds_for_inference,
        "inference_ready": inference_ready,
        "primary_model": (
            "cell-fixed-effect slope: paired latency change percent per one "
            "percentage point of predicted objective reduction"
        ),
        "within_cell_slope": slope,
        "seed_cluster_bootstrap_low": float(
            np.percentile(bootstrap_slopes, 2.5)
        ),
        "seed_cluster_bootstrap_high": float(
            np.percentile(bootstrap_slopes, 97.5)
        ),
        "interpretation": (
            "final clustered interval"
            if inference_ready
            else "descriptive only; measure additional seeds before inference"
        ),
        "mediator_note": (
            "measured rank-load imbalance and accessible fraction are retained in "
            "the pair artifact, but are not used to identify the primary dose slope; "
            "cell fixed effects remove K/routing level confounding"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(args.output / "composition_calibration_pairs.csv", index=False)
    distinct_positive_doses.to_csv(
        args.output / "dose_design_by_cell.csv", index=False
    )
    (args.output / "composition_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
