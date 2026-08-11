#!/usr/bin/env python3
"""Produce paired FIFO deltas for the H100 EP4 scheduler failure analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

INDEX = ["seed", "routing_mode", "active_positions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregated", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def paired_deltas(conditions: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "latency_p50_ms",
        "gpu_critical_path_p50_ms",
        "scheduler_selection_p50_ms",
        "communication_fraction_mean",
        "rank_load_imbalance_mean",
        "total_rank_idle_mean_ms",
        "tokens_per_active_expert_mean",
        "active_unique_experts_mean",
        "cross_gpu_bytes_mean",
    ]
    fifo = conditions[conditions["scheduler"] == "fifo"].set_index(INDEX)
    records: list[dict[str, object]] = []
    for row in conditions[conditions["scheduler"] != "fifo"].itertuples():
        baseline = fifo.loc[(row.seed, row.routing_mode, row.active_positions)]
        record: dict[str, object] = {
            "seed": row.seed,
            "routing_mode": row.routing_mode,
            "active_positions": row.active_positions,
            "scheduler": row.scheduler,
        }
        for metric in metrics:
            value = float(getattr(row, metric))
            base = float(baseline[metric])
            record[f"{metric}_delta"] = value - base
            record[f"{metric}_delta_percent"] = (
                (value / base - 1.0) * 100.0 if base else float("nan")
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def layer_deltas(layers: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "layer_critical_path_ms",
        "dispatch_ms",
        "expert_compute_ms",
        "combine_ms",
        "total_rank_idle_ms",
        "max_rank_assignments",
        "rank_load_imbalance",
        "maximum_peer_bytes",
        "non_empty_peer_count",
    ]
    grouped = layers.groupby([*INDEX, "scheduler"], sort=True)[metrics].mean().reset_index()
    fifo = grouped[grouped["scheduler"] == "fifo"].set_index(INDEX)
    records: list[dict[str, object]] = []
    for row in grouped[grouped["scheduler"] != "fifo"].itertuples():
        baseline = fifo.loc[(row.seed, row.routing_mode, row.active_positions)]
        record: dict[str, object] = {
            "seed": row.seed,
            "routing_mode": row.routing_mode,
            "active_positions": row.active_positions,
            "scheduler": row.scheduler,
        }
        for metric in metrics:
            value = float(getattr(row, metric))
            base = float(baseline[metric])
            record[f"{metric}_delta"] = value - base
            record[f"{metric}_delta_percent"] = (
                (value / base - 1.0) * 100.0 if base else float("nan")
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def write_report(run_deltas: pd.DataFrame, layer_changes: pd.DataFrame, output: Path) -> None:
    run_summary = run_deltas.groupby("scheduler", sort=True).mean(numeric_only=True)
    layer_summary = layer_changes.groupby("scheduler", sort=True).mean(numeric_only=True)
    critical = run_summary.loc["critical_path"]
    oracle = run_summary.loc["routing_oracle"]
    critical_layer = layer_summary.loc["critical_path"]
    oracle_layer = layer_summary.loc["routing_oracle"]

    def run_row(name: str, row: pd.Series) -> str:
        values = [
            name,
            f"{row['latency_p50_ms_delta_percent']:.3f}%",
            f"{row['gpu_critical_path_p50_ms_delta_percent']:.3f}%",
            f"{row['scheduler_selection_p50_ms_delta']:.4f} ms above FIFO",
        ]
        return "| " + " | ".join(values) + " |"

    def layer_row(name: str, row: pd.Series) -> str:
        values = [
            name,
            f"{row['dispatch_ms_delta_percent']:.3f}%",
            f"{row['expert_compute_ms_delta_percent']:.3f}%",
            f"{row['combine_ms_delta_percent']:.3f}%",
            f"{row['max_rank_assignments_delta_percent']:.3f}%",
        ]
        return "| " + " | ".join(values) + " |"

    critical_run_row = run_row("Critical path", critical)
    oracle_run_row = run_row("Routing oracle", oracle)
    critical_layer_row = layer_row("Critical path", critical_layer)
    oracle_layer_row = layer_row("Routing oracle", oracle_layer)
    content = f"""# H100 EP4 scheduler root-cause analysis

## Direct observations

| Arm | End-to-end vs FIFO | GPU path vs FIFO | Selection P50 |
|---|---:|---:|---:|
{critical_run_row}
{oracle_run_row}

The critical-path arm does not improve any of the 49 routing-mode/K cells in the
three-seed mean. Online selection overhead is the largest established component.

## GPU-path attribution

| Arm | Dispatch | Expert compute | Combine | Max rank assignments |
|---|---:|---:|---:|---:|
{critical_layer_row}
{oracle_layer_row}

Expert compute is effectively unchanged. The measured GPU-path penalty is localized to
dispatch/combine. The routing oracle reduces maximum rank assignments but still slows
communication, so route knowledge alone is not the missing component.

## What is not yet proven

The current aggregate network fields do not explain the communication penalty: total
cross-GPU bytes are invariant and mean maximum peer bytes do not increase for the
critical-path or routing-oracle arms. Therefore the data does not yet distinguish:

1. an omitted per-peer split-vector or synchronization cost;
2. insufficient batch-formation freedom in the 8-candidate/4-request setup;
3. fixed scheduler-arm order and three-repetition measurement noise.

The hardware runner also schedules independently on each source rank. Its
`routing_oracle` sees actual routes for that rank only; it is not a coordinated global
oracle and not an upper bound on four-rank batch formation. Independent choices can
collide on the same destination rank after all-to-all. The measured result therefore
rejects the current rank-local policy, not coordinated critical-path scheduling.

## Required next measurement design

1. Counterbalance or randomize scheduler-arm execution order.
2. Use at least 20 measured repetitions for a small representative matrix.
3. Separate `offline schedule replay` from `online selection` to isolate selection cost.
4. Record the full 4x4 rank-pair split vector per batch and layer.
5. Compare FIFO, offline routing oracle, online routing oracle, and a measured-NCCL-cost
   scheduler before carrying the policy into LLaDA2.0-mini.
6. Add a bypass when predicted savings do not exceed measured selection cost plus a
   safety margin.
7. Add a coordinated offline replay arm that scores the combined traffic from all four
   source ranks; keep the current local oracle labeled as local-route knowledge only.

The current rank-local critical-path policy is rejected for this measured regime. The
native-model EP adapter may proceed, but FIFO remains its initial runtime baseline.
"""
    (output / "scheduler_root_cause.md").write_text(content)


def main() -> None:
    args = parse_args()
    output = args.output or args.aggregated
    output.mkdir(parents=True, exist_ok=True)
    conditions = pd.read_csv(args.aggregated / "condition_summary.csv")
    layers = pd.read_csv(args.aggregated / "aggregated_layers.csv")
    runs = paired_deltas(conditions)
    layer_changes = layer_deltas(layers)
    runs.to_csv(output / "scheduler_run_deltas.csv", index=False)
    layer_changes.to_csv(output / "scheduler_layer_deltas.csv", index=False)
    write_report(runs, layer_changes, output)
    print(output / "scheduler_root_cause.md")


if __name__ == "__main__":
    main()
