#!/usr/bin/env python3
"""Analyze constructed planner-objective doses against measured EP time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIFO_ARM = "fifo_constructed"
FIFO_LOCAL_ARM = "fifo_local_copy_control"
FIFO_MINIMAL_ARM = "fifo_nccl_minimal_control"
DOSE_ARMS = ("dose_083_constructed", "balanced_constructed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--timing-gate-analysis",
        type=Path,
        help=(
            "Directory containing scheduler_accessible_time.csv, or the CSV itself. "
            "Optional Gate 2 accessibility cross-check. Gate 2B transmission uses "
            "its own constructed-FIFO local/minimal/real controls."
        ),
    )
    return parser.parse_args()


def bootstrap_median(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    return tuple(float(value) for value in np.percentile(np.median(draws, axis=1), [2.5, 97.5]))


def load_accessible_fraction(path: Path | None, active_positions: int) -> float | None:
    if path is None:
        return None
    csv_path = path if path.is_file() else path / "scheduler_accessible_time.csv"
    timing = pd.read_csv(csv_path)
    rows = timing[timing["active_positions"] == active_positions]
    if rows.empty:
        raise ValueError("timing gate contains no matching active-position row")
    values = rows["accessible_fraction_p50"].drop_duplicates().to_numpy(float)
    if len(values) != 1:
        raise ValueError("timing gate reports inconsistent accessible fractions")
    return float(values[0])


def main() -> None:
    args = parse_args()
    metadata_path = args.input / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("proxy validation metadata.json is required")
    metadata = json.loads(metadata_path.read_text())
    expected_protocol = (
        "constructed_objective_to_time_proxy_v3_two_dose_local_minimal_real"
    )
    if metadata.get("measurement_protocol") != expected_protocol:
        raise ValueError("unsupported proxy-validation measurement protocol")
    paths = sorted(args.input.glob("rank*_proxy_validation.csv"))
    if len(paths) != 4:
        raise ValueError("expected four rank proxy-validation CSV files")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    measured = raw[raw["warmup"] == 0].copy()
    key_columns = ["rank", "arm", "repetition"]
    if measured.duplicated(key_columns).any():
        raise ValueError("proxy validation contains duplicate rank/arm/repetition rows")
    if set(measured["rank"].unique()) != {0, 1, 2, 3}:
        raise ValueError("proxy validation must contain ranks 0..3")
    if "expert_id_elements_per_layer" in measured:
        payload_counts = measured.groupby("arm")[
            "expert_id_elements_per_layer"
        ].first()
        if payload_counts.nunique() != 1:
            raise ValueError("proxy arms do not preserve expert-ID payload length")
    observed_arms = set(measured["arm"])
    required_arms = {FIFO_LOCAL_ARM, FIFO_MINIMAL_ARM, FIFO_ARM, *DOSE_ARMS}
    if observed_arms != required_arms:
        raise ValueError(f"expected arms {sorted(required_arms)}, found {sorted(observed_arms)}")
    if set(metadata.get("run_arms", [])) != required_arms:
        raise ValueError("proxy metadata run-arm contract is inconsistent")
    active_positions_values = measured["active_positions"].unique()
    if len(active_positions_values) != 1:
        raise ValueError("proxy validation must contain one active-position width")
    active_positions = int(active_positions_values[0])
    if int(metadata.get("active_positions", -1)) != active_positions:
        raise ValueError("proxy rows disagree with metadata active positions")
    critical = (
        measured.groupby(["arm", "repetition"], sort=True)
        .agg(
            gpu_path_ms=("gpu_path_ms", "max"),
            dispatch_ms=("dispatch_ms", "max"),
            expert_compute_ms=("expert_compute_ms", "max"),
            combine_ms=("combine_ms", "max"),
            planner_objective=("planner_objective", "first"),
            objective_reduction_fraction=(
                "objective_reduction_fraction",
                "first",
            ),
        )
        .reset_index()
    )
    control_pivot = critical.pivot(
        index="repetition",
        columns="arm",
        values="gpu_path_ms",
    )
    if control_pivot[list(required_arms)].isna().any().any():
        raise ValueError("proxy validation has an incomplete paired arm set")
    transport_control = pd.DataFrame(index=control_pivot.index)
    transport_control["local_copy_ms"] = control_pivot[FIFO_LOCAL_ARM]
    transport_control["nccl_minimal_ms"] = control_pivot[FIFO_MINIMAL_ARM]
    transport_control["nccl_real_ms"] = control_pivot[FIFO_ARM]
    transport_control["launch_floor_ms"] = (
        transport_control["nccl_minimal_ms"]
        - transport_control["local_copy_ms"]
    )
    transport_control["accessible_payload_ms"] = (
        transport_control["nccl_real_ms"]
        - transport_control["nccl_minimal_ms"]
    )
    transport_control["total_nccl_premium_ms"] = (
        transport_control["nccl_real_ms"]
        - transport_control["local_copy_ms"]
    )
    transport_control["accessible_fraction"] = (
        transport_control["accessible_payload_ms"]
        / transport_control["nccl_real_ms"]
    )
    phase_columns = (
        "gpu_path_ms",
        "dispatch_ms",
        "expert_compute_ms",
        "combine_ms",
    )
    repetitions = sorted(critical["repetition"].unique())
    paired_phase = pd.DataFrame(index=repetitions)
    rng = np.random.default_rng(20260807)
    launch_low, launch_high = bootstrap_median(
        transport_control["launch_floor_ms"].to_numpy(float),
        samples=args.bootstrap_samples,
        rng=rng,
    )
    accessible_ms_low, accessible_ms_high = bootstrap_median(
        transport_control["accessible_payload_ms"].to_numpy(float),
        samples=args.bootstrap_samples,
        rng=rng,
    )
    total_low, total_high = bootstrap_median(
        transport_control["total_nccl_premium_ms"].to_numpy(float),
        samples=args.bootstrap_samples,
        rng=rng,
    )
    accessible_fraction_low, accessible_fraction_high = bootstrap_median(
        transport_control["accessible_fraction"].to_numpy(float),
        samples=args.bootstrap_samples,
        rng=rng,
    )
    constructed_accessible_fraction = float(
        np.median(transport_control["accessible_fraction"])
    )
    constructed_accessibility_identified = bool(
        launch_low > 0 and accessible_ms_low > 0 and total_low > 0
    )
    gate2_accessible_fraction = load_accessible_fraction(
        args.timing_gate_analysis,
        active_positions,
    )
    accessible_fraction = (
        constructed_accessible_fraction
        if constructed_accessibility_identified
        else None
    )
    dose_records: list[dict[str, object]] = []
    gpu_reductions: dict[str, np.ndarray] = {}
    objective_reductions: dict[str, float] = {}
    for arm in DOSE_ARMS:
        objective_reduction = float(
            critical.loc[
                critical["arm"] == arm,
                "objective_reduction_fraction",
            ].iloc[0]
        )
        objective_reductions[arm] = objective_reduction
        phase_medians: dict[str, float] = {}
        for column in phase_columns:
            pivot = critical.pivot(index="repetition", columns="arm", values=column)
            reduction = (pivot[FIFO_ARM] - pivot[arm]) / pivot[FIFO_ARM]
            output_name = f"{arm}_{column.removesuffix('_ms')}_reduction_fraction"
            paired_phase[output_name] = reduction
            phase_medians[column.removesuffix("_ms")] = float(np.median(reduction))
            if column == "gpu_path_ms":
                gpu_reductions[arm] = reduction.to_numpy(float)
        values = gpu_reductions[arm]
        low, high = bootstrap_median(
            values,
            samples=args.bootstrap_samples,
            rng=rng,
        )
        median_reduction = float(np.median(values))
        transmission_values = None
        transmission_low = None
        transmission_high = None
        transmission_median = None
        if accessible_fraction is not None:
            denominator = accessible_fraction * objective_reduction
            transmission_values = values / denominator
            transmission_low, transmission_high = bootstrap_median(
                transmission_values,
                samples=args.bootstrap_samples,
                rng=rng,
            )
            transmission_median = float(np.median(transmission_values))
        dose_records.append(
            {
                "arm": arm,
                "objective_reduction_fraction": objective_reduction,
                "measured_latency_reduction_median_fraction": median_reduction,
                "measured_latency_reduction_ci_low": low,
                "measured_latency_reduction_ci_high": high,
                "latency_response_per_objective_reduction": (
                    median_reduction / objective_reduction
                ),
                "accessible_fraction": accessible_fraction,
                "transmission_fraction_median": transmission_median,
                "transmission_fraction_ci_low": transmission_low,
                "transmission_fraction_ci_high": transmission_high,
                "phase_median_reductions": json.dumps(phase_medians, sort_keys=True),
            }
        )
    high_values = gpu_reductions["balanced_constructed"]
    high_low, high_high = bootstrap_median(
        high_values,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    if not constructed_accessibility_identified:
        status = "PROXY_TIME_UNRESOLVED"
    elif high_low > 0:
        status = "PROXY_TIME_ALIGNED"
    elif high_high < 0:
        status = "PROXY_TIME_DISCONFIRMED"
    else:
        status = "PROXY_TIME_UNRESOLVED"
    x = np.asarray([objective_reductions[arm] for arm in DOSE_ARMS])
    y = np.stack([gpu_reductions[arm] for arm in DOSE_ARMS], axis=1)
    slopes = (y @ x) / float(x @ x)
    slope_low, slope_high = bootstrap_median(
        slopes,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    slope_median = float(np.median(slopes))
    transmission_slope = (
        slope_median / accessible_fraction
        if accessible_fraction is not None
        else None
    )
    transmission_slope_ci = (
        [slope_low / accessible_fraction, slope_high / accessible_fraction]
        if accessible_fraction is not None
        else None
    )
    output = args.output or (args.input / "proxy_validation_analysis")
    output.mkdir(parents=True, exist_ok=True)
    critical.to_csv(output / "critical_samples.csv", index=False)
    transport_control.reset_index().to_csv(
        output / "transport_control_samples.csv",
        index=False,
    )
    paired_phase.reset_index(names="repetition").to_csv(
        output / "paired_phase_reductions.csv",
        index=False,
    )
    pd.DataFrame.from_records(dose_records).to_csv(
        output / "dose_response.csv",
        index=False,
    )
    high_record = next(
        record for record in dose_records if record["arm"] == "balanced_constructed"
    )
    report = {
        "status": status,
        "measurement_protocol": expected_protocol,
        "active_positions": active_positions,
        "objective_reduction_fraction": high_record["objective_reduction_fraction"],
        "measured_latency_reduction_median_fraction": high_record[
            "measured_latency_reduction_median_fraction"
        ],
        "measured_latency_reduction_bootstrap_ci": [high_low, high_high],
        "accessible_fraction": accessible_fraction,
        "constructed_accessibility": {
            "identified": constructed_accessibility_identified,
            "accessible_fraction_median": constructed_accessible_fraction,
            "accessible_fraction_bootstrap_ci": [
                accessible_fraction_low,
                accessible_fraction_high,
            ],
            "launch_floor_bootstrap_ci_ms": [launch_low, launch_high],
            "accessible_payload_bootstrap_ci_ms": [
                accessible_ms_low,
                accessible_ms_high,
            ],
            "total_nccl_premium_bootstrap_ci_ms": [total_low, total_high],
        },
        "gate2_balanced_route_accessible_fraction_crosscheck": (
            gate2_accessible_fraction
        ),
        "dose_arms": dose_records,
        "objective_to_latency_slope_median": slope_median,
        "objective_to_latency_slope_bootstrap_ci": [slope_low, slope_high],
        "transmission_slope_median": transmission_slope,
        "transmission_slope_bootstrap_ci": transmission_slope_ci,
        "transmission_definition": (
            "measured latency reduction / (Gate 2B constructed-FIFO accessible "
            "fraction * objective reduction); the general Gate 2 value is a "
            "cross-check only"
        ),
        "interpretation": {
            "PROXY_TIME_ALIGNED": (
                "The high constructed dose lowers measured time. Use the continuous "
                "transmission estimate, not direction alone, in later screens."
            ),
            "PROXY_TIME_DISCONFIRMED": (
                "The high constructed dose increases measured time; do not use the "
                "load proxy as a scheduler opportunity gate."
            ),
            "PROXY_TIME_UNRESOLVED": (
                "The high-dose direction is unresolved; load-only negative results "
                "cannot be promoted to timing conclusions."
            ),
        }[status],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
