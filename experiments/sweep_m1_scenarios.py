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

from refineserve.config import ExecutionMode, SchedulerName, load_config
from refineserve.scenarios import SCENARIOS, ScenarioName, scenario_config
from refineserve.simulator import Simulator

matplotlib.use("Agg")

SCHEDULERS = ("fifo", "locality_only", "joint", "routing_oracle")


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "makespan_ms",
        "latency_p95_ms",
        "finalized_tokens_per_second",
        "expert_invocations",
        "network_messages",
        "underfilled_batch_ratio",
        "mean_rank_layer_load_cv",
        "scheduler_selection_wall_time_ms",
        "scheduler_profile_update_wall_time_ms",
        "scheduler_total_wall_time_ms",
    )
    rows: list[dict[str, object]] = []
    for (scenario, scheduler), group in frame.groupby(["scenario", "scheduler"], sort=False):
        row: dict[str, object] = {
            "scenario": scenario,
            "scheduler": scheduler,
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        fifo = frame[(frame["scenario"] == scenario) & (frame["scheduler"] == "fifo")][
            ["seed", "makespan_ms", "latency_p95_ms"]
        ].rename(
            columns={
                "makespan_ms": "fifo_makespan_ms",
                "latency_p95_ms": "fifo_latency_p95_ms",
            }
        )
        paired = group.merge(fifo, on="seed", validate="one_to_one")
        row["paired_makespan_change_mean"] = float(
            (paired["makespan_ms"] / paired["fifo_makespan_ms"] - 1.0).mean()
        )
        row["paired_p95_change_mean"] = float(
            (paired["latency_p95_ms"] / paired["fifo_latency_p95_ms"] - 1.0).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    scenarios = list(summary["scenario"].drop_duplicates())
    schedulers = list(summary["scheduler"].drop_duplicates())
    x = np.arange(len(scenarios), dtype=float)
    width = 0.8 / len(schedulers)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for index, scheduler in enumerate(schedulers):
        values = (
            summary[summary["scheduler"] == scheduler]
            .set_index("scenario")
            .reindex(scenarios)[metric]
        )
        axis.bar(x + (index - (len(schedulers) - 1) / 2) * width, values, width, label=scheduler)
    axis.set_xticks(x, scenarios)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Synthetic scenario")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired-seed M1 scenario sweeps.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m1_scheduler_diagnostic.yaml"),
    )
    parser.add_argument("--mode", choices=("autoregressive", "diffusion"), default="diffusion")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--schedulers", nargs="+", choices=SCHEDULERS, default=list(SCHEDULERS))
    parser.add_argument("--output", type=Path, default=Path("results/m1_scenarios"))
    args = parser.parse_args()

    base = load_config(args.config)
    mode = cast(ExecutionMode, args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scenario in args.scenarios:
        scenario_base = scenario_config(base, cast(ScenarioName, scenario))
        for seed in args.seeds:
            for scheduler_name in args.schedulers:
                scheduler = cast(SchedulerName, scheduler_name)
                config = replace(
                    scenario_base,
                    seed=seed,
                    scheduler=replace(scenario_base.scheduler, name=scheduler),
                )
                result = Simulator(config, mode).run()
                run_name = f"{scenario}/seed_{seed}/{scheduler_name}"
                result.write(args.output / run_name)
                rows.append(
                    {
                        "scenario": scenario,
                        **asdict(result.summary),
                        **asdict(result.runtime_diagnostics),
                    }
                )

    runs = pd.DataFrame(rows)
    summary = _summarize(runs)
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status": "SIMULATION_RESULT",
        "config": str(args.config),
        "mode": mode,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "schedulers": args.schedulers,
        "calibrated": False,
    }
    (args.output / "experiment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    _plot(summary, "makespan_ms_mean", "Mean makespan (ms)", args.output / "makespan.png")
    _plot(
        summary,
        "latency_p95_ms_mean",
        "Mean P95 latency (ms)",
        args.output / "p95_latency.png",
    )
    _plot(
        summary,
        "scheduler_selection_wall_time_ms_mean",
        "Measured Python scheduler wall time (ms)",
        args.output / "scheduler_wall_time.png",
    )
    print(
        summary[
            [
                "scenario",
                "scheduler",
                "seed_count",
                "paired_makespan_change_mean",
                "paired_p95_change_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
