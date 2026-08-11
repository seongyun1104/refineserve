from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from refineserve.config import ExecutionMode, load_config
from refineserve.policies import policy_config
from refineserve.scenarios import SCENARIOS, ScenarioName, scenario_config
from refineserve.simulator import Simulator

matplotlib.use("Agg")

LAYERS = (4, 8, 16, 32)


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario, layers), group in runs.groupby(["scenario", "num_layers"], sort=False):
        fifo = group[group["policy"] == "fifo"][
            ["seed", "makespan_ms", "latency_p95_ms"]
        ].rename(
            columns={
                "makespan_ms": "fifo_makespan_ms",
                "latency_p95_ms": "fifo_p95_ms",
            }
        )
        online = group[group["policy"] == "adaptive_online"].merge(
            fifo,
            on="seed",
            validate="one_to_one",
        )
        makespan_change = online["makespan_ms"] / online["fifo_makespan_ms"] - 1.0
        p95_change = online["latency_p95_ms"] / online["fifo_p95_ms"] - 1.0
        wall_fraction = online["scheduler_total_wall_time_ms"] / online["makespan_ms"]
        online_pass = (online["scheduler_selection_p95_ms"] <= 1.0) & (
            wall_fraction <= 0.05
        )
        rows.append(
            {
                "scenario": scenario,
                "num_layers": int(layers),
                "seed_count": int(online["seed"].nunique()),
                "paired_makespan_change_mean": float(makespan_change.mean()),
                "paired_makespan_change_std": float(makespan_change.std(ddof=1)),
                "paired_p95_change_mean": float(p95_change.mean()),
                "selection_p95_ms_mean": float(
                    online["scheduler_selection_p95_ms"].mean()
                ),
                "profile_update_p95_ms_mean": float(
                    online["scheduler_profile_update_p95_ms"].mean()
                ),
                "scheduler_wall_fraction_mean": float(wall_fraction.mean()),
                "online_gate_pass_rate": float(online_pass.mean()),
                "non_regression_all_seeds": bool((makespan_change <= 1e-12).all()),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for scenario in summary["scenario"].drop_duplicates():
        group = summary[summary["scenario"] == scenario].sort_values("num_layers")
        axis.plot(group["num_layers"], group[metric], marker="o", label=scenario)
    axis.set_xticks(LAYERS)
    axis.set_xlabel("Logical attention/MoE layers")
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive-policy layer sensitivity.")
    parser.add_argument("--config", type=Path, default=Path("configs/m1_online.yaml"))
    parser.add_argument("--mode", choices=("autoregressive", "diffusion"), default="diffusion")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--output", type=Path, default=Path("results/layer_sensitivity"))
    args = parser.parse_args()

    mode = cast(ExecutionMode, args.mode)
    base = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scenario_value in args.scenarios:
        scenario = cast(ScenarioName, scenario_value)
        scenario_base = scenario_config(base, scenario)
        for layers in args.layers:
            layered = replace(
                scenario_base,
                model=replace(scenario_base.model, num_layers=layers),
            )
            for seed in args.seeds:
                for policy in ("fifo", "adaptive_online"):
                    config = replace(
                        layered,
                        seed=seed,
                        scheduler=policy_config(layered.scheduler, policy),
                    )
                    result = Simulator(config, mode).run()
                    result.write(
                        args.output
                        / scenario
                        / f"layers_{layers}"
                        / f"seed_{seed}"
                        / policy
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "num_layers": layers,
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
                "layers": args.layers,
                "scenarios": args.scenarios,
                "policy": "adaptive_online",
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
                "num_layers",
                "paired_makespan_change_mean",
                "selection_p95_ms_mean",
                "scheduler_wall_fraction_mean",
                "online_gate_pass_rate",
                "non_regression_all_seeds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
