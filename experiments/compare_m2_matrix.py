from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pandas as pd

from refineserve.calibration import CalibrationRangeError
from refineserve.config import (
    CalibrationConfig,
    ExecutionMode,
    load_config,
)
from refineserve.policies import policy_config
from refineserve.simulator import Simulator
from refineserve.trace_bundle import RouteTraceBundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare synthetic/measured routes and expert-kernel costs."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--mode", choices=("autoregressive", "diffusion"), default="diffusion")
    parser.add_argument("--use-network-curves", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mode = cast(ExecutionMode, args.mode)
    base = load_config(args.config)
    bundle = RouteTraceBundle.load(args.trace, expected_model=base.model)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    cells = (
        ("synthetic_routes_synthetic_cost", False, False),
        ("trace_routes_synthetic_cost", True, False),
        ("synthetic_routes_measured_expert_cost", False, True),
        ("trace_routes_measured_expert_cost", True, True),
    )
    for cell, use_trace, use_calibration in cells:
        router = replace(
            base.router,
            source="trace" if use_trace else "synthetic",
            trace_path=str(args.trace) if use_trace else None,
        )
        calibration = CalibrationConfig(
            artifact_path=str(args.calibration) if use_calibration else None,
            use_expert_kernel_curve=True,
            use_network_curves=args.use_network_curves,
            require_trace_checksum_match=use_trace and use_calibration,
        )
        for policy in ("fifo", "adaptive_online"):
            config = replace(
                base,
                router=router,
                calibration=calibration,
                scheduler=policy_config(base.scheduler, policy),
            )
            try:
                result = Simulator(config, mode).run()
            except CalibrationRangeError as error:
                rejections.append(
                    {
                        "cell": cell,
                        "policy": policy,
                        "input_name": error.input_name,
                        "observed_min": error.observed_min,
                        "observed_max": error.observed_max,
                        "calibrated_min": error.calibrated_min,
                        "calibrated_max": error.calibrated_max,
                        "max_observed_overflow": error.maximum_overflow,
                        "error": str(error),
                    }
                )
                continue
            result.write(args.output / cell / policy)
            rows.append(
                {
                    "cell": cell,
                    "route_source": "trace" if use_trace else "synthetic",
                    "expert_cost_source": "measured_curve" if use_calibration else "synthetic",
                    "network_cost_source": (
                        "measured_curve"
                        if use_calibration and args.use_network_curves
                        else "synthetic"
                    ),
                    "policy": policy,
                    **asdict(result.summary),
                    **asdict(result.runtime_diagnostics),
                }
            )

    runs = pd.DataFrame(rows)
    fifo = runs[runs["policy"] == "fifo"][["cell", "makespan_ms", "latency_p95_ms"]].rename(
        columns={"makespan_ms": "fifo_makespan_ms", "latency_p95_ms": "fifo_p95_ms"}
    )
    summary = runs[runs["policy"] == "adaptive_online"].merge(
        fifo,
        on="cell",
        validate="one_to_one",
    )
    summary["makespan_change_vs_fifo"] = (
        summary["makespan_ms"] / summary["fifo_makespan_ms"] - 1.0
    )
    summary["p95_change_vs_fifo"] = summary["latency_p95_ms"] / summary["fifo_p95_ms"] - 1.0
    summary["scheduler_wall_fraction"] = (
        summary["scheduler_total_wall_time_ms"] / summary["makespan_ms"]
    )
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(rejections).to_csv(args.output / "rejections.csv", index=False)
    (args.output / "experiment.json").write_text(
        json.dumps(
            {
                "status": "SIMULATION_RESULT",
                "mode": mode,
                "config": str(args.config),
                "trace": str(args.trace),
                "trace_bundle_sha256": bundle.bundle_sha256,
                "calibration": str(args.calibration),
                "measured_cost_scope": (
                    "expert_kernel_and_rank_local_network"
                    if args.use_network_curves
                    else "expert_kernel_only"
                ),
                "network_cost_source": (
                    "measured_in_measured_cost_cells"
                    if args.use_network_curves
                    else "synthetic_in_all_cells"
                ),
                "rejected_experiment_count": len(rejections),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        summary[
            [
                "cell",
                "makespan_change_vs_fifo",
                "p95_change_vs_fifo",
                "scheduler_selection_p95_ms",
                "scheduler_wall_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
