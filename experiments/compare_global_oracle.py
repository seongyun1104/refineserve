from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from refineserve.config import load_config
from refineserve.global_oracle import ExactGlobalMakespanOracle
from refineserve.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare greedy policies with the exact small-workload makespan oracle."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/oracle_micro.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/global_oracle"))
    args = parser.parse_args()

    base = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    action_traces: dict[str, list[list[int]]] = {}
    for mode in ("autoregressive", "diffusion"):
        oracle = ExactGlobalMakespanOracle(base, mode).solve()
        action_traces[mode] = [list(action) for action in oracle.actions]
        rows.append(
            {
                "mode": mode,
                "scheduler": "exact_global",
                "makespan_ms": oracle.optimal_makespan_ms,
                "batch_count": oracle.batch_count,
                "explored_states": oracle.explored_states,
            }
        )
        for scheduler in ("fifo", "joint", "routing_oracle"):
            config = base.with_overrides(mode_scheduler=scheduler)
            summary = Simulator(config, mode).run().summary
            rows.append(
                {
                    "mode": mode,
                    "scheduler": scheduler,
                    "makespan_ms": summary.makespan_ms,
                    "batch_count": summary.batch_count,
                    "explored_states": 0,
                }
            )

    frame = pd.DataFrame(rows)
    for mode in frame["mode"].unique():
        mask = frame["mode"] == mode
        optimum = float(
            frame.loc[mask & (frame["scheduler"] == "exact_global"), "makespan_ms"].iloc[0]
        )
        frame.loc[mask, "optimality_gap"] = frame.loc[mask, "makespan_ms"] / optimum - 1.0
    frame.to_csv(args.output / "aggregate.csv", index=False)
    metadata = {
        "status": "SIMULATION_RESULT",
        "config": str(args.config),
        "objective": "offline deterministic makespan",
        "scope": "exact only for the configured micro workload",
        "actions": action_traces,
    }
    (args.output / "experiment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
