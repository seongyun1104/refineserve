#!/usr/bin/env python3
"""Aggregate the authoritative native K-position EP4 hardware matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_KEYS = ["seed", "routing_mode", "scheduler", "active_positions", "repetition"]
LAYER_KEYS = [*RUN_KEYS, "batch_index", "layer"]
CONDITION_KEYS = ["seed", "routing_mode", "scheduler", "active_positions"]
SUMMARY_KEYS = ["routing_mode", "scheduler", "active_positions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(series: pd.Series, q: float) -> float:
    return float(np.percentile(series.to_numpy(dtype=float), q))


def load_rank_csvs(root: Path, suffix: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"rank*_{suffix}.csv"))
    if len(paths) != 4:
        raise ValueError(f"expected four rank {suffix} files, found {len(paths)}")
    frames = [pd.read_csv(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result = result[result["warmup"] == 0].copy()
    if result.empty:
        raise ValueError(f"no measured rows in {suffix} files")
    return result


def optional_max(group: pd.DataFrame, column: str) -> float:
    return float(group[column].max()) if column in group else 0.0


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, group in runs.groupby(RUN_KEYS, sort=True):
        if set(group["rank"].astype(int)) != {0, 1, 2, 3}:
            raise ValueError(f"incomplete rank set for run {key}")
        useful = int(group["useful_finalized_positions"].sum())
        executed = int(group["executed_positions"].sum())
        selection_ms = float(group["scheduler_ms"].max())
        gpu_ms = float(group["gpu_path_ms"].max())
        count_exchange_ms = optional_max(group, "count_exchange_ms")
        end_to_end_ms = selection_ms + count_exchange_ms + gpu_ms
        records.append(
            {
                **dict(zip(RUN_KEYS, key, strict=True)),
                "rank_count": int(group["rank"].nunique()),
                "scheduler_selection_ms": selection_ms,
                "gpu_critical_path_ms": gpu_ms,
                "count_exchange_critical_path_ms": count_exchange_ms,
                "pre_data_arrival_spread_ms": float(
                    (
                        group["pre_data_arrival_ns"].max()
                        - group["pre_data_arrival_ns"].min()
                    )
                    / 1_000_000
                )
                if "pre_data_arrival_ns" in group
                else 0.0,
                "pre_data_barrier_wait_max_ms": optional_max(
                    group, "pre_data_barrier_wait_ms"
                ),
                "end_to_end_ms": end_to_end_ms,
                "scheduler_fraction": selection_ms / end_to_end_ms,
                "useful_finalized_positions": useful,
                "executed_positions": executed,
                "work_amplification": executed / useful,
                "useful_positions_per_second": useful * 1000.0 / end_to_end_ms,
                "executed_positions_per_second": executed * 1000.0 / end_to_end_ms,
                "expert_token_executions": int(group["expert_token_executions"].sum()),
                "all_ranks_valid": int(group["valid"].astype(bool).all()),
                "all_ranks_finite": int(group["finite"].astype(bool).all()),
                "assignment_counts_match": int(
                    group["assignment_counts_match"].astype(bool).all()
                ),
                "route_ids_valid": int(group["route_ids_valid"].astype(bool).all()),
                "rank_gpu_time_spread_ms": float(
                    group["gpu_path_ms"].max() - group["gpu_path_ms"].min()
                ),
                "coordinated_fifo_objective": optional_max(
                    group,
                    "coordinated_fifo_objective"
                ),
                "coordinated_best_found_objective": optional_max(
                    group,
                    "coordinated_best_found_objective"
                ),
                "coordinated_fifo_max_receive_load": optional_max(
                    group,
                    "coordinated_fifo_max_receive_load",
                ),
                "coordinated_best_max_receive_load": optional_max(
                    group,
                    "coordinated_best_max_receive_load",
                ),
                "coordinated_predicted_reduction_percent": optional_max(
                    group,
                    "coordinated_predicted_reduction_percent"
                ),
                "coordinated_dose_target_fraction": optional_max(
                    group,
                    "coordinated_dose_target_fraction",
                ),
                "coordinated_dose_achieved_fraction": optional_max(
                    group,
                    "coordinated_dose_achieved_fraction",
                ),
                "coordinated_reassigned_request_fraction": optional_max(
                    group,
                    "coordinated_reassigned_request_fraction"
                ),
                "coordinated_restart_cost_std": optional_max(
                    group,
                    "coordinated_restart_cost_std"
                ),
                "coordinated_restart_tail_improved": int(
                    optional_max(group, "coordinated_restart_tail_improved")
                ),
                "selection_plan_checksum_sum": float(
                    group["selection_plan_checksum"].sum()
                )
                if "selection_plan_checksum" in group
                else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_layers(layers: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, group in layers.groupby(LAYER_KEYS, sort=True):
        if set(group["rank"].astype(int)) != {0, 1, 2, 3}:
            raise ValueError(f"incomplete rank set for layer {key}")
        critical_index = group["layer_ms"].idxmax()
        critical = group.loc[critical_index]
        critical_ms = float(critical["layer_ms"])
        received = group["received_assignments"].to_numpy(dtype=float)
        total_received = int(received.sum())
        active_experts = int(group["active_local_experts"].sum())
        records.append(
            {
                **dict(zip(LAYER_KEYS, key, strict=True)),
                "critical_rank": int(critical["rank"]),
                "layer_critical_path_ms": critical_ms,
                "router_ms": float(critical["router_ms"]),
                "dispatch_ms": float(critical["dispatch_ms"]),
                "expert_compute_ms": float(critical["expert_compute_ms"]),
                "expert_compute_rank_max_ms": float(
                    group["expert_compute_ms"].max()
                ),
                "expert_compute_rank_mean_ms": float(
                    group["expert_compute_ms"].mean()
                ),
                "expert_compute_rank_imbalance_ms": float(
                    group["expert_compute_ms"].max()
                    - group["expert_compute_ms"].mean()
                ),
                "combine_ms": float(critical["combine_ms"]),
                "communication_ms": float(critical["dispatch_ms"] + critical["combine_ms"]),
                "communication_fraction": float(
                    (critical["dispatch_ms"] + critical["combine_ms"]) / critical_ms
                ),
                "total_rank_idle_ms": float((critical_ms - group["layer_ms"]).sum()),
                "mean_rank_idle_ms": float((critical_ms - group["layer_ms"]).mean()),
                "max_rank_idle_ms": float(critical_ms - group["layer_ms"].min()),
                "global_tokens": int(group["tokens"].sum()),
                "global_expert_assignments": total_received,
                "active_unique_experts": active_experts,
                "assignments_per_active_expert": total_received
                / max(active_experts, 1),
                "expert_invocation_count": active_experts,
                "max_rank_assignments": int(received.max()),
                "mean_rank_assignments": float(received.mean()),
                "rank_load_imbalance": float(received.max() / max(received.mean(), 1.0)),
                "cross_gpu_bytes": int(group["cross_gpu_bytes"].sum()),
                "non_empty_peer_count": int(group["non_empty_peers"].sum()),
                "average_peer_bytes": float(group["average_peer_bytes"].mean()),
                "maximum_peer_bytes": int(group["max_peer_bytes"].max()),
            }
        )
    return pd.DataFrame.from_records(records)


def condition_summary(runs: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    run_records: list[dict[str, object]] = []
    for key, group in runs.groupby(CONDITION_KEYS, sort=True):
        run_records.append(
            {
                **dict(zip(CONDITION_KEYS, key, strict=True)),
                "samples": len(group),
                "latency_p50_ms": percentile(group["end_to_end_ms"], 50),
                "latency_p95_ms": percentile(group["end_to_end_ms"], 95),
                "latency_p99_ms": percentile(group["end_to_end_ms"], 99),
                "gpu_critical_path_p50_ms": percentile(
                    group["gpu_critical_path_ms"], 50
                ),
                "count_exchange_p50_ms": percentile(
                    group["count_exchange_critical_path_ms"], 50
                ),
                "useful_positions_per_second_p50": percentile(
                    group["useful_positions_per_second"], 50
                ),
                "executed_positions_per_second_p50": percentile(
                    group["executed_positions_per_second"], 50
                ),
                "scheduler_selection_p50_ms": percentile(
                    group["scheduler_selection_ms"], 50
                ),
                "scheduler_selection_p95_ms": percentile(
                    group["scheduler_selection_ms"], 95
                ),
                "scheduler_fraction_p50": percentile(group["scheduler_fraction"], 50),
                "valid_fraction": float(group["all_ranks_valid"].mean()),
                "coordinated_predicted_reduction_percent": float(
                    group["coordinated_predicted_reduction_percent"].max()
                ),
                "coordinated_dose_target_fraction": float(
                    group["coordinated_dose_target_fraction"].max()
                ),
                "coordinated_dose_achieved_fraction": float(
                    group["coordinated_dose_achieved_fraction"].max()
                ),
                "coordinated_fifo_max_receive_load": float(
                    group["coordinated_fifo_max_receive_load"].max()
                ),
                "coordinated_best_max_receive_load": float(
                    group["coordinated_best_max_receive_load"].max()
                ),
                "coordinated_reassigned_request_fraction": float(
                    group["coordinated_reassigned_request_fraction"].max()
                ),
                "coordinated_restart_cost_std": float(
                    group["coordinated_restart_cost_std"].max()
                ),
                "coordinated_restart_tail_improved": int(
                    group["coordinated_restart_tail_improved"].max()
                ),
            }
        )
    result = pd.DataFrame.from_records(run_records)
    layer_summary = (
        layers.groupby(CONDITION_KEYS, sort=True)
        .agg(
            layer_critical_path_mean_ms=("layer_critical_path_ms", "mean"),
            dispatch_mean_ms=("dispatch_ms", "mean"),
            expert_compute_mean_ms=("expert_compute_ms", "mean"),
            expert_compute_rank_max_mean_ms=(
                "expert_compute_rank_max_ms",
                "mean",
            ),
            expert_compute_rank_mean_mean_ms=(
                "expert_compute_rank_mean_ms",
                "mean",
            ),
            expert_compute_rank_imbalance_mean_ms=(
                "expert_compute_rank_imbalance_ms",
                "mean",
            ),
            combine_mean_ms=("combine_ms", "mean"),
            communication_fraction_mean=("communication_fraction", "mean"),
            assignments_per_active_expert_mean=(
                "assignments_per_active_expert",
                "mean",
            ),
            active_unique_experts_mean=("active_unique_experts", "mean"),
            rank_load_imbalance_mean=("rank_load_imbalance", "mean"),
            total_rank_idle_mean_ms=("total_rank_idle_ms", "mean"),
            cross_gpu_bytes_mean=("cross_gpu_bytes", "mean"),
        )
        .reset_index()
    )
    result = result.merge(layer_summary, on=CONDITION_KEYS, validate="one_to_one")
    fifo = result[result["scheduler"] == "fifo"][
        [
            "seed",
            "routing_mode",
            "active_positions",
            "latency_p50_ms",
            "gpu_critical_path_p50_ms",
        ]
    ].rename(
        columns={
            "latency_p50_ms": "fifo_latency_p50_ms",
            "gpu_critical_path_p50_ms": "fifo_gpu_critical_path_p50_ms",
        }
    )
    result = result.merge(
        fifo,
        on=["seed", "routing_mode", "active_positions"],
        how="left",
        validate="many_to_one",
    )
    result["latency_change_vs_fifo_percent"] = (
        (result["latency_p50_ms"] / result["fifo_latency_p50_ms"]) - 1.0
    ) * 100.0
    result["gpu_path_change_vs_fifo_percent"] = (
        result["gpu_critical_path_p50_ms"]
        / result["fifo_gpu_critical_path_p50_ms"]
        - 1.0
    ) * 100.0
    return result


def overall_summary(conditions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, group in conditions.groupby(SUMMARY_KEYS, sort=True):
        records.append(
            {
                **dict(zip(SUMMARY_KEYS, key, strict=True)),
                "seed_count": int(group["seed"].nunique()),
                "latency_p50_ms_mean": float(group["latency_p50_ms"].mean()),
                "latency_p95_ms_mean": float(group["latency_p95_ms"].mean()),
                "latency_p99_ms_mean": float(group["latency_p99_ms"].mean()),
                "latency_change_vs_fifo_percent_mean": float(
                    group["latency_change_vs_fifo_percent"].mean()
                ),
                "latency_change_vs_fifo_percent_max": float(
                    group["latency_change_vs_fifo_percent"].max()
                ),
                "gpu_path_change_vs_fifo_percent_mean": float(
                    group["gpu_path_change_vs_fifo_percent"].mean()
                ),
                "count_exchange_p50_ms_mean": float(
                    group["count_exchange_p50_ms"].mean()
                ),
                "useful_positions_per_second_p50_mean": float(
                    group["useful_positions_per_second_p50"].mean()
                ),
                "scheduler_selection_p95_ms_mean": float(
                    group["scheduler_selection_p95_ms"].mean()
                ),
                "scheduler_fraction_p50_mean": float(
                    group["scheduler_fraction_p50"].mean()
                ),
                "communication_fraction_mean": float(
                    group["communication_fraction_mean"].mean()
                ),
                "assignments_per_active_expert_mean": float(
                    group["assignments_per_active_expert_mean"].mean()
                ),
                "active_unique_experts_mean": float(
                    group["active_unique_experts_mean"].mean()
                ),
                "rank_load_imbalance_mean": float(
                    group["rank_load_imbalance_mean"].mean()
                ),
                "valid_fraction": float(group["valid_fraction"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def write_plots(summary: pd.DataFrame, output: Path) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "latency_p50_ms_mean": ("P50 end-to-end latency (ms)", "latency"),
        "useful_positions_per_second_p50_mean": (
            "Useful finalized positions/s",
            "throughput",
        ),
        "assignments_per_active_expert_mean": (
            "Mean top-k assignments per active expert",
            "expert_density",
        ),
        "communication_fraction_mean": (
            "Mean communication fraction",
            "communication_fraction",
        ),
    }
    for routing_mode, routing_group in summary.groupby("routing_mode", sort=True):
        for metric, (ylabel, suffix) in metrics.items():
            figure, axis = plt.subplots()
            for scheduler, scheduler_group in routing_group.groupby("scheduler", sort=True):
                ordered = scheduler_group.sort_values("active_positions")
                axis.plot(
                    ordered["active_positions"],
                    ordered[metric],
                    marker="o",
                    label=scheduler,
                )
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Native active positions K")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{routing_mode}: {ylabel}")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize="small")
            figure.tight_layout()
            figure.savefig(plot_dir / f"{routing_mode}_{suffix}.png", dpi=160)
            plt.close(figure)


def write_report(summary: pd.DataFrame, report: dict[str, object], output: Path) -> None:
    online = summary[summary["scheduler"] == "critical_path"].copy()
    best = online.nsmallest(5, "latency_change_vs_fifo_percent_mean")
    worst = online.nlargest(5, "latency_change_vs_fifo_percent_mean")

    def rows(frame: pd.DataFrame) -> list[str]:
        return [
            "| "
            + " | ".join(
                [
                    str(row.routing_mode),
                    str(int(row.active_positions)),
                    f"{row.latency_change_vs_fifo_percent_mean:.3f}%",
                    f"{row.scheduler_selection_p95_ms_mean:.4f}",
                    f"{100.0 * row.scheduler_fraction_p50_mean:.3f}%",
                ]
            )
            + " |"
            for row in frame.itertuples()
        ]

    header = [
        "| Routing | K | P50 latency vs FIFO | Selection P95 (ms) | Scheduler % |",
        "|---|---:|---:|---:|---:|",
    ]
    improved_cases = int((online["latency_change_vs_fifo_percent_mean"] < 0).sum())
    content = [
        "# H100×4 native K-position EP4 report",
        "",
        f"- Measured runs: {report['measured_run_count']}",
        f"- Aggregated layer records: {report['aggregated_layer_count']}",
        f"- All runs valid: {report['all_runs_valid']}",
        f"- Seeds: {report['seed_count']}",
        (
            "- Critical-path cases improving three-seed mean latency: "
            f"{improved_cases}/{len(online)}"
        ),
        (
            "- Mean critical-path end-to-end change vs FIFO: "
            f"{online['latency_change_vs_fifo_percent_mean'].mean():.3f}%"
        ),
        (
            "- Mean critical-path GPU-only change vs FIFO: "
            f"{online['gpu_path_change_vs_fifo_percent_mean'].mean():.3f}%"
        ),
        (
            "- Mean critical-path scheduler fraction: "
            f"{100.0 * online['scheduler_fraction_p50_mean'].mean():.3f}%"
        ),
        "",
        "## Lowest critical-path changes",
        "",
        *header,
        *rows(best),
        "",
        "## Highest critical-path slowdowns",
        "",
        *header,
        *rows(worst),
        "",
        "Negative percentages improve latency. All values include online scheduler selection time.",
        "P95/P99 from three repetitions per seed are descriptive, not strong tail estimates.",
        "",
    ]
    (output / "report.md").write_text("\n".join(content))


def main() -> None:
    args = parse_args()
    output = args.output or args.input / "aggregated"
    output.mkdir(parents=True, exist_ok=True)
    runs = aggregate_runs(load_rank_csvs(args.input, "runs"))
    layers = aggregate_layers(load_rank_csvs(args.input, "layers"))
    conditions = condition_summary(runs, layers)
    summary = overall_summary(conditions)
    runs.to_csv(output / "aggregated_runs.csv", index=False)
    layers.to_csv(output / "aggregated_layers.csv", index=False)
    conditions.to_csv(output / "condition_summary.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    metadata = json.loads((args.input / "metadata.json").read_text())
    report = {
        "execution_kind": metadata["execution_kind"],
        "measured_run_count": len(runs),
        "aggregated_layer_count": len(layers),
        "condition_count": len(conditions),
        "all_runs_valid": bool(runs["all_ranks_valid"].all()),
        "seed_count": int(runs["seed"].nunique()),
        "routing_mode_count": int(runs["routing_mode"].nunique()),
        "scheduler_count": int(runs["scheduler"].nunique()),
        "active_position_values": sorted(
            int(value) for value in runs["active_positions"].unique()
        ),
    }
    write_plots(summary, output)
    write_report(summary, report, output)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
