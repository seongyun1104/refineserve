#!/usr/bin/env python3
"""Measure best-found objective convergence as planner restart budget grows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .coordinated_scheduling import (
        composition_invariant_cost_lower_bound,
        coordinated_plan_with_diagnostics,
    )
    from .synthetic_routes import make_routes, request_counts
except ImportError:
    from coordinated_scheduling import (  # type: ignore[no-redef]
        composition_invariant_cost_lower_bound,
        coordinated_plan_with_diagnostics,
    )
    from synthetic_routes import make_routes, request_counts  # type: ignore[no-redef]


CANONICAL_ROUTING_MODES = (
    "uniform",
    "mild_skew",
    "strong_skew",
    "request_correlated",
    "temporally_unstable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests-per-rank", type=int, default=8)
    parser.add_argument("--batch-counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--restart-counts", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--active-positions", type=int, nargs="+", default=[64])
    parser.add_argument(
        "--routing-modes",
        nargs="+",
        default=["uniform", "request_correlated", "temporally_unstable"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41, 53, 67])
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--request-correlation-strength", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restart_counts = sorted(set(args.restart_counts))
    if not restart_counts or restart_counts[0] <= 0:
        raise ValueError("restart counts must be positive")
    if args.experts % args.world_size:
        raise ValueError("experts must divide evenly across ranks")
    for batch_count in args.batch_counts:
        if batch_count <= 0 or args.requests_per_rank % batch_count:
            raise ValueError("every batch count must divide requests per rank")
    experts_per_rank = args.experts // args.world_size
    records: list[dict[str, object]] = []
    max_restarts = max(restart_counts)
    for positions in args.active_positions:
        for routing_mode in args.routing_modes:
            for seed in args.seeds:
                routes = np.stack(
                    [
                        make_routes(
                            seed=seed,
                            mode=routing_mode,
                            global_request_ids=(
                                np.arange(args.requests_per_rank)
                                + source * args.requests_per_rank
                            ),
                            layers=args.layers,
                            positions=positions,
                            experts=args.experts,
                            request_correlation_strength=(
                                args.request_correlation_strength
                            ),
                        )
                        for source in range(args.world_size)
                    ]
                )
                counts = np.stack(
                    [request_counts(source, args.experts) for source in routes]
                )
                for batch_count in args.batch_counts:
                    batch_size = args.requests_per_rank // batch_count
                    # Match the screening seed contract even when this audit
                    # intentionally runs only a subset of routing modes.
                    routing_index = CANONICAL_ROUTING_MODES.index(routing_mode)
                    planner_seed = seed + positions + routing_index
                    if batch_count != 2:
                        planner_seed += 1000 * batch_count
                    _, diagnostics = coordinated_plan_with_diagnostics(
                        counts,
                        batch_size,
                        experts_per_rank,
                        restarts=max_restarts,
                        seed=planner_seed,
                    )
                    lower_bound = composition_invariant_cost_lower_bound(
                        counts,
                        batch_size,
                        experts_per_rank,
                    )
                    opportunity = max(diagnostics.fifo_cost - lower_bound, 0.0)
                    for restarts in restart_counts:
                        best_cost = diagnostics.best_so_far_costs[restarts - 1]
                        realized = max(diagnostics.fifo_cost - best_cost, 0.0)
                        records.append(
                            {
                                "active_positions": positions,
                                "routing_mode": routing_mode,
                                "seed": seed,
                                "requests_per_rank": args.requests_per_rank,
                                "batch_count": batch_count,
                                "batch_size": batch_size,
                                "restarts": restarts,
                                "objective_lower_bound": lower_bound,
                                "fifo_objective": diagnostics.fifo_cost,
                                "best_found_objective": best_cost,
                                "objective_opportunity_fraction": (
                                    opportunity / max(diagnostics.fifo_cost, 1.0)
                                ),
                                "realized_reduction_fraction": (
                                    realized / max(diagnostics.fifo_cost, 1.0)
                                ),
                                "objective_achievability": (
                                    realized / opportunity
                                    if opportunity > 0
                                    else 0.0
                                ),
                                "objective_physically_zero": int(
                                    np.isclose(opportunity, 0.0)
                                ),
                            }
                        )
    cells = pd.DataFrame.from_records(records)
    summary = (
        cells.groupby(
            [
                "requests_per_rank",
                "batch_count",
                "batch_size",
                "restarts",
            ],
            sort=True,
        )
        .agg(
            samples=("objective_achievability", "size"),
            physically_zero_fraction=("objective_physically_zero", "mean"),
            opportunity_fraction_median=(
                "objective_opportunity_fraction",
                "median",
            ),
            realized_reduction_fraction_median=(
                "realized_reduction_fraction",
                "median",
            ),
            achievability_p25=(
                "objective_achievability",
                lambda values: np.percentile(values, 25),
            ),
            achievability_median=("objective_achievability", "median"),
        )
        .reset_index()
    )
    by_condition = (
        cells.groupby(
            ["active_positions", "routing_mode", "batch_count", "restarts"],
            sort=True,
        )
        .agg(
            samples=("objective_achievability", "size"),
            physically_zero_fraction=("objective_physically_zero", "mean"),
            realized_reduction_fraction_p25=(
                "realized_reduction_fraction",
                lambda values: np.percentile(values, 25),
            ),
            realized_reduction_fraction_median=(
                "realized_reduction_fraction",
                "median",
            ),
            achievability_p25=(
                "objective_achievability",
                lambda values: np.percentile(values, 25),
            ),
            achievability_median=("objective_achievability", "median"),
        )
        .reset_index()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "restart_sensitivity_cells.csv", index=False)
    summary.to_csv(args.output / "restart_sensitivity_summary.csv", index=False)
    by_condition.to_csv(
        args.output / "restart_sensitivity_by_condition.csv",
        index=False,
    )
    metadata = {
        "evidence_class": "CPU_PLANNER_SEARCH_SENSITIVITY",
        "same_setting_anchor": (
            "requests_per_rank=8,batch_count=2,restarts=64 matches the v7 synthetic "
            "screening planner budget and seed formula for the selected K/routing cells"
        ),
        "restart_derivation": (
            "one deterministic max-restart run per cell; each lower budget reads the "
            "corresponding prefix of the retained best-so-far curve"
        ),
        "optimality_claim": False,
        "requests_per_rank": args.requests_per_rank,
        "batch_counts": args.batch_counts,
        "restart_counts": restart_counts,
        "active_positions": args.active_positions,
        "routing_modes": args.routing_modes,
        "seeds": args.seeds,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
