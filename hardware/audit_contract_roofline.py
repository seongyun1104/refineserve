#!/usr/bin/env python3
"""Create a transparent FLOP/byte accounting table for the EP4 contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-tflops", type=float, default=500.0)
    parser.add_argument("--effective-bandwidth-gbps", type=float, default=400.0)
    parser.add_argument("--collective-fixed-us", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition = pd.read_csv(args.input / "aggregated" / "condition_summary.csv")
    fifo = condition[condition["scheduler"] == "fifo"]
    observed = (
        fifo.groupby("active_positions", sort=True)
        .agg(
            observed_p50_ms=("latency_p50_ms", "mean"),
            observed_gpu_p50_ms=("gpu_critical_path_p50_ms", "mean"),
        )
        .reset_index()
    )
    world_size = 4
    batch_size = 4
    batches = 2
    layers = 8
    top_k = 2
    hidden = 2048
    intermediate = 8192
    bytes_per_element = 2
    data_collectives_per_layer = 3
    control_collectives_per_layer = 1
    records: list[dict[str, float | int]] = []
    for row in observed.itertuples(index=False):
        positions = int(row.active_positions)
        global_tokens_per_batch = world_size * batch_size * positions
        global_assignments = global_tokens_per_batch * top_k
        expert_flops_global = (
            global_assignments * 6 * hidden * intermediate * batches * layers
        )
        expert_flops_per_rank = expert_flops_global / world_size
        logical_data_plane_bytes_global = (
            global_assignments
            * (2 * hidden * bytes_per_element + 4)
            * batches
            * layers
        )
        cross_data_plane_bytes_global = (
            logical_data_plane_bytes_global * (world_size - 1) / world_size
        )
        ideal_compute_ms = (
            expert_flops_per_rank / (args.effective_tflops * 1e12) * 1000.0
        )
        ideal_bandwidth_ms = (
            cross_data_plane_bytes_global
            / (args.effective_bandwidth_gbps * 1e9)
            * 1000.0
        )
        logical_expert_major_bytes_global = (
            global_assignments
            * (2 * hidden * bytes_per_element)
            * batches
            * layers
        )
        cross_expert_major_bytes_global = (
            logical_expert_major_bytes_global * (world_size - 1) / world_size
        )
        expert_major_bandwidth_ms = (
            cross_expert_major_bytes_global
            / (args.effective_bandwidth_gbps * 1e9)
            * 1000.0
        )
        data_collective_fixed_ms = (
            data_collectives_per_layer
            * batches
            * layers
            * args.collective_fixed_us
            / 1000.0
        )
        control_collective_fixed_ms = (
            control_collectives_per_layer
            * batches
            * layers
            * args.collective_fixed_us
            / 1000.0
        )
        ideal_data_plane_ms = (
            ideal_compute_ms + ideal_bandwidth_ms + data_collective_fixed_ms
        )
        ideal_with_control_ms = ideal_data_plane_ms + control_collective_fixed_ms
        expert_major_data_fixed_ms = (
            2 * batches * layers * args.collective_fixed_us / 1000.0
        )
        expert_major_ideal_with_control_ms = (
            ideal_compute_ms
            + expert_major_bandwidth_ms
            + expert_major_data_fixed_ms
            + control_collective_fixed_ms
        )
        expert_major_accessible_fraction = (
            expert_major_bandwidth_ms / expert_major_ideal_with_control_ms
        )
        substrate_sensitivity = {}
        for reduction in (1, 2, 3, 5):
            reduced_total = (
                ideal_compute_ms
                + ideal_bandwidth_ms
                + data_collective_fixed_ms / reduction
                + control_collective_fixed_ms
            )
            accessible_fraction = ideal_bandwidth_ms / reduced_total
            substrate_sensitivity[reduction] = (
                reduced_total,
                accessible_fraction,
                accessible_fraction / 3.0,
            )
        records.append(
            {
                "active_positions": positions,
                "global_tokens_per_batch": global_tokens_per_batch,
                "global_topk_assignments_per_batch": global_assignments,
                "expert_flops_global_full_path": expert_flops_global,
                "expert_flops_per_rank_full_path": expert_flops_per_rank,
                "logical_data_plane_bytes_global_full_path": logical_data_plane_bytes_global,
                "estimated_cross_data_plane_bytes_global_full_path": (
                    cross_data_plane_bytes_global
                ),
                "assumed_effective_compute_tflops": args.effective_tflops,
                "assumed_effective_bandwidth_gbps": args.effective_bandwidth_gbps,
                "assumed_collective_fixed_us": args.collective_fixed_us,
                "ideal_compute_ms": ideal_compute_ms,
                "ideal_bandwidth_ms": ideal_bandwidth_ms,
                "data_collective_fixed_ms": data_collective_fixed_ms,
                "control_collective_fixed_ms": control_collective_fixed_ms,
                "ideal_data_plane_ms": ideal_data_plane_ms,
                "ideal_with_control_ms": ideal_with_control_ms,
                "expert_major_2_collective_fixed_ms": expert_major_data_fixed_ms,
                "expert_major_2_collective_bandwidth_ms": (
                    expert_major_bandwidth_ms
                ),
                "expert_major_2_collective_ideal_with_control_ms": (
                    expert_major_ideal_with_control_ms
                ),
                "expert_major_2_collective_accessible_fraction": (
                    expert_major_accessible_fraction
                ),
                "observed_v1_p50_ms": float(row.observed_p50_ms),
                "observed_v1_gpu_p50_ms": float(row.observed_gpu_p50_ms),
                "observed_to_ideal_ratio": float(row.observed_gpu_p50_ms)
                / ideal_with_control_ms,
                **{
                    f"fixed_reduction_{factor}x_total_ms": values[0]
                    for factor, values in substrate_sensitivity.items()
                },
                **{
                    f"fixed_reduction_{factor}x_accessible_fraction": values[1]
                    for factor, values in substrate_sensitivity.items()
                },
                **{
                    f"fixed_reduction_{factor}x_recoverable_fraction_at_1p5": values[2]
                    for factor, values in substrate_sensitivity.items()
                },
            }
        )
    audit = pd.DataFrame.from_records(records)
    args.output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output / "roofline_accounting.csv", index=False)
    selected_columns = [
        "active_positions",
        "ideal_compute_ms",
        "ideal_bandwidth_ms",
        "data_collective_fixed_ms",
        "control_collective_fixed_ms",
        "ideal_with_control_ms",
        "observed_v1_gpu_p50_ms",
        "observed_to_ideal_ratio",
    ]
    header = "| " + " | ".join(selected_columns) + " |"
    divider = "|" + "|".join(["---"] * len(selected_columns)) + "|"
    table_rows = []
    for row in audit[selected_columns].itertuples(index=False):
        values = [str(int(row[0]))]
        values.extend(f"{float(value):.4f}" for value in row[1:])
        table_rows.append("| " + " | ".join(values) + " |")
    markdown = [
        "# EP4 contract FLOP/byte accounting",
        "",
        "This is a sensitivity table, not a calibrated roofline. Effective TFLOPS,",
        "bandwidth, and collective latency are explicit assumptions. The v1 observed",
        "interval additionally contains control, synchronization, validation, and metric",
        "work, so `observed_to_ideal_ratio` is diagnostic only.",
        "",
        "Data-plane accounting contains three collectives per layer execution: BF16 hidden",
        "dispatch, int32 expert-ID dispatch, and BF16 hidden combine. Split-count exchange",
        "is a fourth control-plane collective and is reported separately. The byte term",
        "contains all three data-plane payloads. Fixed-cost reduction columns in the CSV",
        "show substrate sensitivity at 1x, 2x, 3x, and 5x; recoverable share assumes",
        "max/mean rank imbalance 1.5.",
        "The CSV also includes an explicit expert-major two-collective scenario that",
        "reconstructs local expert IDs from expert-level counts, removing the int32 ID",
        "collective and its payload. It retains one control-collective fixed cost but",
        "does not model the larger expert-level count payload, so it is sensitivity",
        "analysis rather than a measured or complete optimized path.",
        "",
        header,
        divider,
        *table_rows,
        "",
    ]
    (args.output / "roofline_accounting.md").write_text("\n".join(markdown))


if __name__ == "__main__":
    main()
