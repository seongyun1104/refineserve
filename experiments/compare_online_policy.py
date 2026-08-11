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
from refineserve.policies import POLICIES, PolicyName, policy_config
from refineserve.scenarios import SCENARIOS, ScenarioName, scenario_config
from refineserve.simulator import Simulator

matplotlib.use("Agg")

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
        full = scenario_runs[scenario_runs["policy"] == "full_joint"][
            ["seed", "makespan_ms"]
        ].rename(columns={"makespan_ms": "full_makespan_ms"})
        for policy in POLICIES:
            group = scenario_runs[scenario_runs["policy"] == policy]
            paired = group.merge(fifo, on="seed", validate="one_to_one").merge(
                full,
                on="seed",
                validate="one_to_one",
            )
            makespan_change = paired["makespan_ms"] / paired["fifo_makespan_ms"] - 1.0
            p95_change = paired["latency_p95_ms"] / paired["fifo_p95_ms"] - 1.0
            full_gain = paired["fifo_makespan_ms"] - paired["full_makespan_ms"]
            policy_gain = paired["fifo_makespan_ms"] - paired["makespan_ms"]
            retention = np.where(full_gain > 1e-9, policy_gain / full_gain, np.nan)
            scheduler_fraction = (
                paired["scheduler_total_wall_time_ms"] / paired["makespan_ms"]
            )
            online_pass = (paired["scheduler_selection_p95_ms"] <= 1.0) & (
                scheduler_fraction <= 0.05
            )
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "seed_count": int(group["seed"].nunique()),
                    "paired_makespan_change_mean": float(makespan_change.mean()),
                    "paired_makespan_change_std": float(makespan_change.std(ddof=1)),
                    "paired_p95_change_mean": float(p95_change.mean()),
                    "full_joint_gain_retention_mean": float(np.nanmean(retention)),
                    "selection_p95_ms_mean": float(
                        paired["scheduler_selection_p95_ms"].mean()
                    ),
                    "profile_update_p95_ms_mean": float(
                        paired["scheduler_profile_update_p95_ms"].mean()
                    ),
                    "scheduler_wall_fraction_mean": float(scheduler_fraction.mean()),
                    "online_gate_pass_rate": float(online_pass.mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    scenarios = list(summary["scenario"].drop_duplicates())
    policies = list(summary["policy"].drop_duplicates())
    x = np.arange(len(scenarios), dtype=float)
    width = 0.8 / len(policies)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for index, policy in enumerate(policies):
        values = (
            summary[summary["policy"] == policy]
            .set_index("scenario")
            .reindex(scenarios)[metric]
        )
        axis.bar(x + (index - (len(policies) - 1) / 2) * width, values, width, label=policy)
    axis.set_xticks(x, scenarios)
    axis.set_xlabel("Synthetic scenario")
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full and online M1 policies.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m1_scheduler_diagnostic.yaml"),
    )
    parser.add_argument("--mode", choices=("autoregressive", "diffusion"), default="diffusion")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--output", type=Path, default=Path("results/online_policy"))
    args = parser.parse_args()

    base = load_config(args.config)
    mode = cast(ExecutionMode, args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scenario_value in args.scenarios:
        scenario = cast(ScenarioName, scenario_value)
        scenario_base = scenario_config(base, scenario)
        for seed in args.seeds:
            for policy in POLICIES:
                config = replace(
                    scenario_base,
                    seed=seed,
                    scheduler=policy_config(
                        scenario_base.scheduler,
                        cast(PolicyName, policy),
                    ),
                )
                result = Simulator(config, mode).run()
                result.write(args.output / scenario / f"seed_{seed}" / policy)
                rows.append(
                    {
                        "scenario": scenario,
                        "policy": policy,
                        **asdict(result.summary),
                        **asdict(result.runtime_diagnostics),
                    }
                )

    runs = pd.DataFrame(rows)
    summary = _summarize(runs)
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "experiment.json").write_text(
        json.dumps(
            {
                "status": "SIMULATION_RESULT",
                "config": str(args.config),
                "mode": mode,
                "seeds": args.seeds,
                "scenarios": args.scenarios,
                "policies": list(POLICIES),
                "online_gate": {
                    "selection_p95_ms_max": 1.0,
                    "total_scheduler_wall_fraction_max": 0.05,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _plot(
        summary,
        "paired_makespan_change_mean",
        "Mean makespan change vs FIFO",
        args.output / "makespan_change.png",
    )
    _plot(
        summary,
        "selection_p95_ms_mean",
        "Mean selection P95 (ms)",
        args.output / "selection_p95.png",
    )
    print(
        summary[
            [
                "scenario",
                "policy",
                "paired_makespan_change_mean",
                "full_joint_gain_retention_mean",
                "selection_p95_ms_mean",
                "scheduler_wall_fraction_mean",
                "online_gate_pass_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
