from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refineserve.config import ExecutionMode, load_config
from refineserve.scenarios import SCENARIOS, ScenarioName, scenario_config
from refineserve.simulator import Simulator

matplotlib.use("Agg")

DEFAULT_POOLS = (1, 2, 4, 8, 16, None)


def _pool_label(pool_size: int | None) -> str:
    return "all" if pool_size is None else str(pool_size)


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in runs["scenario"].drop_duplicates():
        scenario_runs = runs[runs["scenario"] == scenario]
        fifo = scenario_runs[scenario_runs["policy"] == "fifo"][
            ["seed", "makespan_ms", "latency_p95_ms"]
        ].rename(
            columns={
                "makespan_ms": "fifo_makespan_ms",
                "latency_p95_ms": "fifo_p95_ms",
            }
        )
        unlimited = scenario_runs[scenario_runs["pool_label"] == "all"][
            ["seed", "makespan_ms"]
        ].rename(columns={"makespan_ms": "unlimited_makespan_ms"})
        for pool_label in [_pool_label(pool) for pool in DEFAULT_POOLS]:
            group = scenario_runs[scenario_runs["pool_label"] == pool_label]
            paired = group.merge(fifo, on="seed", validate="one_to_one").merge(
                unlimited,
                on="seed",
                validate="one_to_one",
            )
            unlimited_gain = paired["fifo_makespan_ms"] - paired["unlimited_makespan_ms"]
            pool_gain = paired["fifo_makespan_ms"] - paired["makespan_ms"]
            retention = np.where(unlimited_gain > 1e-9, pool_gain / unlimited_gain, np.nan)
            wall_fraction = (
                paired["scheduler_total_wall_time_ms"] / paired["makespan_ms"]
            )
            online_pass = (paired["scheduler_selection_p95_ms"] <= 1.0) & (
                wall_fraction <= 0.05
            )
            makespan_change = paired["makespan_ms"] / paired["fifo_makespan_ms"] - 1.0
            p95_change = paired["latency_p95_ms"] / paired["fifo_p95_ms"] - 1.0
            rows.append(
                {
                    "scenario": scenario,
                    "pool_label": pool_label,
                    "pool_size": None if pool_label == "all" else int(pool_label),
                    "seed_count": int(group["seed"].nunique()),
                    "paired_makespan_change_mean": float(makespan_change.mean()),
                    "paired_makespan_change_std": float(makespan_change.std(ddof=1)),
                    "paired_p95_change_mean": float(p95_change.mean()),
                    "quality_retention_mean": float(np.nanmean(retention)),
                    "scheduler_wall_fraction_mean": float(wall_fraction.mean()),
                    "scheduler_selection_p95_ms_mean": float(
                        paired["scheduler_selection_p95_ms"].mean()
                    ),
                    "scheduler_profile_update_p95_ms_mean": float(
                        paired["scheduler_profile_update_p95_ms"].mean()
                    ),
                    "candidate_evaluations_mean": float(
                        paired["scheduler_candidate_evaluations"].mean()
                    ),
                    "online_gate_pass_rate": float(online_pass.mean()),
                    "quality_gate_pass": bool(
                        makespan_change.mean() <= 0.0 and np.nanmean(retention) >= 0.8
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_by_pool(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    labels = [_pool_label(pool) for pool in DEFAULT_POOLS]
    x = np.arange(len(labels), dtype=float)
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for scenario in summary["scenario"].drop_duplicates():
        values = summary[summary["scenario"] == scenario].set_index("pool_label").reindex(labels)
        axis.plot(x, values[metric], marker="o", label=scenario)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Candidate pool size")
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep joint-scheduler candidate pool size.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m1_scheduler_diagnostic.yaml"),
    )
    parser.add_argument("--mode", choices=("autoregressive", "diffusion"), default="diffusion")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--pools", nargs="+", default=["1", "2", "4", "8", "16", "all"])
    parser.add_argument("--output", type=Path, default=Path("results/candidate_pool"))
    args = parser.parse_args()

    pools: list[int | None] = [None if value == "all" else int(value) for value in args.pools]
    if tuple(pools) != DEFAULT_POOLS:
        raise ValueError(f"the gate report currently requires pools {DEFAULT_POOLS}, got {pools}")
    mode = cast(ExecutionMode, args.mode)
    base = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scenario_value in args.scenarios:
        scenario = cast(ScenarioName, scenario_value)
        scenario_base = scenario_config(base, scenario)
        for seed in args.seeds:
            seeded = replace(scenario_base, seed=seed)
            fifo_config = replace(
                seeded,
                scheduler=replace(seeded.scheduler, name="fifo", candidate_pool_size=None),
            )
            fifo = Simulator(fifo_config, mode).run()
            fifo.write(args.output / scenario / f"seed_{seed}" / "fifo")
            rows.append(
                {
                    "scenario": scenario,
                    "policy": "fifo",
                    "pool_label": "fifo",
                    **asdict(fifo.summary),
                    **asdict(fifo.runtime_diagnostics),
                }
            )
            for pool_size in pools:
                label = _pool_label(pool_size)
                config = replace(
                    seeded,
                    scheduler=replace(
                        seeded.scheduler,
                        name="joint",
                        candidate_pool_size=pool_size,
                    ),
                )
                result = Simulator(config, mode).run()
                result.write(args.output / scenario / f"seed_{seed}" / f"pool_{label}")
                rows.append(
                    {
                        "scenario": scenario,
                        "policy": "joint",
                        "pool_label": label,
                        **asdict(result.summary),
                        **asdict(result.runtime_diagnostics),
                    }
                )

    runs = pd.DataFrame(rows)
    summary = _summarize(runs)
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    gates = {
        f"{row.scenario}/pool_{row.pool_label}": {
            "quality_gate_pass": bool(row.quality_gate_pass),
            "online_gate_pass_rate": float(row.online_gate_pass_rate),
        }
        for row in summary.itertuples(index=False)
    }
    (args.output / "experiment.json").write_text(
        json.dumps(
            {
                "status": "SIMULATION_RESULT",
                "config": str(args.config),
                "mode": mode,
                "seeds": args.seeds,
                "scenarios": args.scenarios,
                "pools": args.pools,
                "online_gate": {
                    "selection_p95_ms_max": 1.0,
                    "total_scheduler_wall_fraction_max": 0.05,
                },
                "quality_gate": {
                    "fifo_makespan_change_max": 0.0,
                    "unlimited_gain_retention_min": 0.8,
                },
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _plot_by_pool(
        summary,
        "paired_makespan_change_mean",
        "Mean makespan change vs FIFO",
        args.output / "makespan_change.png",
    )
    _plot_by_pool(
        summary,
        "scheduler_selection_p95_ms_mean",
        "Mean scheduler selection P95 (ms)",
        args.output / "selection_p95.png",
    )
    print(
        summary[
            [
                "scenario",
                "pool_label",
                "paired_makespan_change_mean",
                "quality_retention_mean",
                "scheduler_selection_p95_ms_mean",
                "scheduler_wall_fraction_mean",
                "online_gate_pass_rate",
                "quality_gate_pass",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
