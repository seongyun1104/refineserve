#!/usr/bin/env python3
"""Audit how the planner objective's physical opportunity changes with batch count."""

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
        fifo_plan,
        plan_cost,
    )
    from .synthetic_routes import make_routes, request_counts
except ImportError:
    from coordinated_scheduling import (  # type: ignore[no-redef]
        composition_invariant_cost_lower_bound,
        coordinated_plan_with_diagnostics,
        fifo_plan,
        plan_cost,
    )
    from synthetic_routes import make_routes, request_counts  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests-per-rank", type=int, default=16)
    parser.add_argument("--batch-counts", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--active-positions", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument(
        "--routing-modes",
        nargs="+",
        default=[
            "uniform",
            "mild_skew",
            "strong_skew",
            "request_correlated",
            "temporally_unstable",
        ],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41, 53, 67])
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--request-correlation-strength", type=float, default=0.75)
    parser.add_argument("--restarts", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experts % args.world_size:
        raise ValueError("experts must divide evenly across ranks")
    for batch_count in args.batch_counts:
        if batch_count <= 0 or args.requests_per_rank % batch_count:
            raise ValueError("every batch count must divide requests per rank")
    experts_per_rank = args.experts // args.world_size
    records: list[dict[str, object]] = []
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
                    fifo = fifo_plan(
                        args.world_size,
                        args.requests_per_rank,
                        batch_size,
                    )
                    fifo_objective = plan_cost(counts, fifo, experts_per_rank)
                    lower_bound = composition_invariant_cost_lower_bound(
                        counts,
                        batch_size,
                        experts_per_rank,
                    )
                    _, diagnostics = coordinated_plan_with_diagnostics(
                        counts,
                        batch_size,
                        experts_per_rank,
                        restarts=args.restarts,
                        seed=seed + positions + batch_count,
                    )
                    if not np.isclose(diagnostics.fifo_cost, fifo_objective):
                        raise RuntimeError("planner and audit FIFO objectives disagree")
                    opportunity = max(fifo_objective - lower_bound, 0.0)
                    realized_reduction = max(
                        fifo_objective - diagnostics.best_cost,
                        0.0,
                    )
                    records.append(
                        {
                            "active_positions": positions,
                            "routing_mode": routing_mode,
                            "seed": seed,
                            "requests_per_rank": args.requests_per_rank,
                            "batch_count": batch_count,
                            "batch_size": batch_size,
                            "fifo_objective": fifo_objective,
                            "composition_invariant_lower_bound": lower_bound,
                            "best_found_objective": diagnostics.best_cost,
                            "objective_opportunity": opportunity,
                            "objective_opportunity_fraction": (
                                opportunity / max(fifo_objective, 1.0)
                            ),
                            "objective_physically_zero": int(
                                np.isclose(opportunity, 0.0)
                            ),
                            "realized_objective_reduction": realized_reduction,
                            "realized_objective_reduction_fraction": (
                                realized_reduction / max(fifo_objective, 1.0)
                            ),
                            "objective_achievability": (
                                realized_reduction / opportunity
                                if opportunity > 0
                                else 0.0
                            ),
                            "restart_tail_improved": int(
                                diagnostics.improved_in_last_two_restarts
                            ),
                        }
                    )
    cells = pd.DataFrame.from_records(records)
    summary = (
        cells.groupby(
            ["requests_per_rank", "batch_count", "batch_size"], sort=True
        )
        .agg(
            samples=("objective_opportunity_fraction", "size"),
            opportunity_fraction_p25=(
                "objective_opportunity_fraction",
                lambda values: np.percentile(values, 25),
            ),
            opportunity_fraction_median=(
                "objective_opportunity_fraction",
                "median",
            ),
            opportunity_fraction_p75=(
                "objective_opportunity_fraction",
                lambda values: np.percentile(values, 75),
            ),
            physically_zero_fraction=("objective_physically_zero", "mean"),
            realized_reduction_fraction_p25=(
                "realized_objective_reduction_fraction",
                lambda values: np.percentile(values, 25),
            ),
            realized_reduction_fraction_median=(
                "realized_objective_reduction_fraction",
                "median",
            ),
            achievability_median=("objective_achievability", "median"),
        )
        .reset_index()
    )
    by_condition = (
        cells.groupby(
            ["active_positions", "routing_mode", "batch_count", "batch_size"],
            sort=True,
        )
        .agg(
            samples=("objective_opportunity_fraction", "size"),
            opportunity_fraction_p25=(
                "objective_opportunity_fraction",
                lambda values: np.percentile(values, 25),
            ),
            opportunity_fraction_median=(
                "objective_opportunity_fraction",
                "median",
            ),
            physically_zero_fraction=("objective_physically_zero", "mean"),
            realized_reduction_fraction_p25=(
                "realized_objective_reduction_fraction",
                lambda values: np.percentile(values, 25),
            ),
            realized_reduction_fraction_median=(
                "realized_objective_reduction_fraction",
                "median",
            ),
            achievability_median=("objective_achievability", "median"),
        )
        .reset_index()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "batch_count_cells.csv", index=False)
    summary.to_csv(args.output / "batch_count_summary.csv", index=False)
    by_condition.to_csv(args.output / "batch_count_by_condition.csv", index=False)
    metadata = {
        "metric": (
            "physical opportunity upper bound = "
            "(FIFO objective - composition-invariant lower bound) / FIFO objective"
        ),
        "planner_executed": True,
        "planner_restarts": args.restarts,
        "interpretation": (
            "Physical opportunity is an invariant upper bound. Realized reduction is "
            "reported separately for the current best-found planner and is not an "
            "optimality claim."
        ),
        "requests_per_rank": args.requests_per_rank,
        "batch_counts": args.batch_counts,
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
