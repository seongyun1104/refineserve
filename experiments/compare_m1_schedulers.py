from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from refineserve.config import ExecutionMode, SchedulerName, load_config
from refineserve.simulator import Simulator

matplotlib.use("Agg")

SCHEDULERS = (
    "fifo",
    "locality_only",
    "load_balance_only",
    "critical_path_only",
    "locality_plus_load",
    "joint",
    "routing_oracle",
    "runtime_oracle",
)


def _plot(frame: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(frame["scheduler"], frame[metric])
    axis.set_xlabel("Scheduler")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=28)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the eight M1 scheduler baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m1_scheduler_diagnostic.yaml"),
    )
    parser.add_argument(
        "--mode",
        choices=("autoregressive", "diffusion"),
        default="diffusion",
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        choices=SCHEDULERS,
        default=list(SCHEDULERS),
    )
    parser.add_argument("--output", type=Path, default=Path("results/m1_schedulers"))
    args = parser.parse_args()

    mode = cast(ExecutionMode, args.mode)
    base = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scheduler_name in args.schedulers:
        scheduler = cast(SchedulerName, scheduler_name)
        config = base.with_overrides(mode_scheduler=scheduler)
        result = Simulator(config, mode).run()
        result.write(args.output / scheduler_name)
        rows.append({**asdict(result.summary), **asdict(result.runtime_diagnostics)})

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "aggregate.csv", index=False)
    fifo = frame.loc[frame["scheduler"] == "fifo"].iloc[0]
    comparisons: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        comparisons[str(row["scheduler"])] = {
            "makespan_change_vs_fifo": float(row["makespan_ms"] / fifo["makespan_ms"] - 1.0),
            "p95_change_vs_fifo": float(row["latency_p95_ms"] / fifo["latency_p95_ms"] - 1.0),
            "batch_count_change_vs_fifo": float(row["batch_count"] - fifo["batch_count"]),
        }
    metadata = {
        "status": "SIMULATION_RESULT",
        "config": str(args.config),
        "mode": mode,
        "schedulers": list(args.schedulers),
        "comparisons": comparisons,
        "oracle_note": (
            "routing_oracle and runtime_oracle are intentionally identical until "
            "trace-driven runtime variation is implemented"
        ),
    }
    (args.output / "experiment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "report.md").write_text(
        "# M1 scheduler comparison\n\n"
        "Status: `SIMULATION_RESULT` under an uncalibrated deterministic cost model.\n\n"
        "The routing and runtime oracles coincide by construction in M1; their gap "
        "becomes meaningful only after trace-driven runtime variation is added.\n\n"
        "```json\n"
        + json.dumps(comparisons, indent=2, sort_keys=True)
        + "\n```\n"
    )

    for metric, ylabel, filename in (
        ("makespan_ms", "Makespan (ms)", "makespan.png"),
        ("latency_p95_ms", "P95 request latency (ms)", "p95_latency.png"),
        ("batch_count", "Executed batches", "batch_count.png"),
        ("mean_rank_layer_load_cv", "Mean EP-rank layer-time CV", "rank_load_cv.png"),
        (
            "scheduler_modeled_overhead_ms",
            "Modeled scheduling overhead (ms)",
            "scheduler_overhead.png",
        ),
        (
            "scheduler_selection_wall_time_ms",
            "Measured Python scheduler wall time (ms)",
            "scheduler_wall_time.png",
        ),
    ):
        _plot(frame, metric, ylabel, args.output / filename)
    print(
        frame[
            [
                "scheduler",
                "makespan_ms",
                "latency_p95_ms",
                "batch_count",
                "mean_rank_layer_load_cv",
                "scheduler_modeled_overhead_ms",
                "scheduler_selection_wall_time_ms",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
