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


ARMS = (
    ("ar_fifo", "autoregressive", "fifo"),
    ("ar_previous_route", "autoregressive", "previous_route"),
    ("diffusion_fifo", "diffusion", "fifo"),
    ("diffusion_previous_route", "diffusion", "previous_route"),
)


def _plot_metric(frame: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(frame["arm"], frame[metric])
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Execution arm")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _relative_gain(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def _relative_reduction(candidate: float, baseline: float) -> float:
    return 1.0 - candidate / baseline


def _analyze(frame: pd.DataFrame) -> dict[str, object]:
    rows = frame.set_index("arm")
    ar_fifo = rows.loc["ar_fifo"]
    ar_local = rows.loc["ar_previous_route"]
    diffusion_fifo = rows.loc["diffusion_fifo"]
    diffusion_local = rows.loc["diffusion_previous_route"]

    diffusion_batch_gain = (
        diffusion_fifo["mean_expert_batch_size"] / ar_fifo["mean_expert_batch_size"]
    )
    diffusion_tps_gain = _relative_gain(
        diffusion_fifo["finalized_tokens_per_second"],
        ar_fifo["finalized_tokens_per_second"],
    )
    diffusion_p95_reduction = _relative_reduction(
        diffusion_fifo["latency_p95_ms"], ar_fifo["latency_p95_ms"]
    )

    locality: dict[str, dict[str, float]] = {}
    for mode, baseline, candidate in (
        ("autoregressive", ar_fifo, ar_local),
        ("diffusion", diffusion_fifo, diffusion_local),
    ):
        locality[mode] = {
            "kernel_launch_reduction": _relative_reduction(
                candidate["kernel_launch_count"], baseline["kernel_launch_count"]
            ),
            "network_message_reduction": _relative_reduction(
                candidate["network_messages"], baseline["network_messages"]
            ),
            "finalized_tps_gain": _relative_gain(
                candidate["finalized_tokens_per_second"],
                baseline["finalized_tokens_per_second"],
            ),
            "p95_latency_reduction": _relative_reduction(
                candidate["latency_p95_ms"], baseline["latency_p95_ms"]
            ),
        }

    return {
        "diffusion_vs_ar_fifo": {
            "mean_expert_batch_size_multiplier": diffusion_batch_gain,
            "processed_work_multiplier": (
                diffusion_fifo["processed_positions"] / ar_fifo["processed_positions"]
            ),
            "finalized_tps_gain": diffusion_tps_gain,
            "p95_latency_reduction": diffusion_p95_reduction,
        },
        "locality_vs_fifo": locality,
        "gates": {
            "diffusion_expert_batch_2x": bool(diffusion_batch_gain >= 2.0),
            "diffusion_tps_or_p95_10pct": bool(
                diffusion_tps_gain >= 0.10 or diffusion_p95_reduction >= 0.10
            ),
            "locality_kernel_or_message_20pct": bool(
                any(
                    metrics["kernel_launch_reduction"] >= 0.20
                    or metrics["network_message_reduction"] >= 0.20
                    for metrics in locality.values()
                )
            ),
            "previous_route_vs_oracle_80pct": None,
        },
        "notes": [
            "A null locality gain can mean max_wait_ms forced deadline fallback.",
            "Oracle comparison and three-seed confirmation are later experiment gates.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the initial AR/diffusion and FIFO/locality comparison."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/four_arms"))
    args = parser.parse_args()

    base_config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for arm, mode, scheduler in ARMS:
        config = base_config.with_overrides(mode_scheduler=cast(SchedulerName, scheduler))
        result = Simulator(config, cast(ExecutionMode, mode)).run()
        result.write(args.output / arm)
        summaries.append(
            {
                "arm": arm,
                **asdict(result.summary),
                **asdict(result.runtime_diagnostics),
            }
        )

    frame = pd.DataFrame(summaries)
    frame.to_csv(args.output / "aggregate.csv", index=False)
    analysis = _analyze(frame)
    (args.output / "experiment.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "arms": [arm for arm, _, _ in ARMS],
                "success_gates": {
                    "mean_expert_batch_size_gain": 2.0,
                    "kernel_or_message_reduction": 0.20,
                    "finalized_tps_gain_or_p95_reduction": 0.10,
                },
                "analysis": analysis,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output / "report.md").write_text(
        "# Four-arm experiment report\n\n"
        "The machine-readable gate evaluation is stored in `experiment.json`.\n\n"
        "```json\n" + json.dumps(analysis, indent=2, sort_keys=True) + "\n```\n"
    )

    _plot_metric(
        frame,
        "finalized_tokens_per_second",
        "Finalized tokens / second",
        args.output / "finalized_throughput.png",
    )
    _plot_metric(
        frame,
        "latency_p95_ms",
        "P95 request latency (ms)",
        args.output / "p95_latency.png",
    )
    _plot_metric(
        frame,
        "mean_expert_batch_size",
        "Mean tokens / expert invocation",
        args.output / "expert_batch_size.png",
    )
    _plot_metric(
        frame,
        "communication_time_fraction",
        "Communication time / makespan",
        args.output / "communication_fraction.png",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
