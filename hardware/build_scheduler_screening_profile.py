#!/usr/bin/env python3
"""Build empirical imbalance/achievability screens without renting a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from coordinated_scheduling import (
    combined_rank_loads,
    composition_invariant_cost_lower_bound,
    coordinated_dose_ladder,
    coordinated_plan_with_diagnostics,
    fifo_plan,
)
from synthetic_routes import make_routes, request_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--requests-per-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--request-correlation-strength", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experts % args.world_size:
        raise ValueError("experts must divide evenly across ranks")
    experts_per_rank = args.experts // args.world_size
    rows: list[dict[str, object]] = []
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
                plan, diagnostics = coordinated_plan_with_diagnostics(
                    counts,
                    args.batch_size,
                    experts_per_rank,
                    restarts=args.restarts,
                    seed=seed + positions + args.routing_modes.index(routing_mode),
                )
                fifo_loads = combined_rank_loads(
                    counts,
                    fifo_plan(
                        args.world_size,
                        args.requests_per_rank,
                        args.batch_size,
                    ),
                    experts_per_rank,
                )
                best_loads = combined_rank_loads(counts, plan, experts_per_rank)
                dose_ladder = coordinated_dose_ladder(
                    counts,
                    plan,
                    args.batch_size,
                    experts_per_rank,
                    seed=seed + positions + args.routing_modes.index(routing_mode),
                )
                mean_load = float(fifo_loads.mean())
                fifo_max = float(fifo_loads.max())
                best_max = float(best_loads.max())
                opportunity = max(fifo_max - mean_load, 0.0)
                achievability = (
                    max(fifo_max - best_max, 0.0) / opportunity
                    if opportunity > 0
                    else 0.0
                )
                fifo_imbalance = fifo_max / max(mean_load, 1.0)
                global_max_recoverable_fraction = (
                    (fifo_imbalance - 1.0) / fifo_imbalance * achievability
                )
                objective_lower_bound = composition_invariant_cost_lower_bound(
                    counts,
                    args.batch_size,
                    experts_per_rank,
                )
                objective_opportunity = max(
                    diagnostics.fifo_cost - objective_lower_bound,
                    0.0,
                )
                objective_achievability = (
                    (diagnostics.fifo_cost - diagnostics.best_cost)
                    / objective_opportunity
                    if objective_opportunity > 0
                    else 0.0
                )
                realized_objective_reduction_fraction = (
                    (diagnostics.fifo_cost - diagnostics.best_cost)
                    / max(diagnostics.fifo_cost, 1.0)
                )
                objective_opportunity_fraction = (
                    objective_opportunity / max(diagnostics.fifo_cost, 1.0)
                )
                destination_routes = routes // experts_per_rank
                same_rank_top2_collision_fraction = float(
                    (destination_routes[..., 0] == destination_routes[..., 1]).mean()
                )
                rows.append(
                    {
                        "active_positions": positions,
                        "routing_mode": routing_mode,
                        "seed": seed,
                        "mean_receive_load": mean_load,
                        "fifo_max_receive_load": fifo_max,
                        "best_found_max_receive_load": best_max,
                        "fifo_rank_load_imbalance": fifo_imbalance,
                        "achievability": achievability,
                        "global_max_recoverable_fraction": (
                            global_max_recoverable_fraction
                        ),
                        "objective_lower_bound": objective_lower_bound,
                        "fifo_objective": diagnostics.fifo_cost,
                        "best_found_objective": diagnostics.best_cost,
                        "objective_opportunity": objective_opportunity,
                        "objective_opportunity_fraction": (
                            objective_opportunity_fraction
                        ),
                        "objective_physically_zero": int(
                            np.isclose(objective_opportunity, 0.0)
                        ),
                        "objective_achievability": objective_achievability,
                        "realized_objective_reduction_fraction": (
                            realized_objective_reduction_fraction
                        ),
                        "same_rank_top2_collision_fraction": (
                            same_rank_top2_collision_fraction
                        ),
                        "predicted_objective_reduction_percent": (
                            diagnostics.predicted_reduction_percent
                        ),
                        "restart_tail_improved": int(
                            diagnostics.improved_in_last_two_restarts
                        ),
                        "dose_distinct_count": len(dose_ladder),
                        "dose_target_fractions": json.dumps(
                            [target for target, _, _ in dose_ladder]
                        ),
                        "dose_achieved_fractions": json.dumps(
                            [achieved for _, achieved, _ in dose_ladder]
                        ),
                    }
                )
    cells = pd.DataFrame.from_records(rows)
    by_cell = (
        cells.groupby(["active_positions", "routing_mode"], sort=True)
        .agg(
            samples=("achievability", "size"),
            fifo_imbalance_p25=("fifo_rank_load_imbalance", lambda x: np.percentile(x, 25)),
            fifo_imbalance_median=("fifo_rank_load_imbalance", "median"),
            achievability_p25=("achievability", lambda x: np.percentile(x, 25)),
            achievability_median=("achievability", "median"),
            global_max_recoverable_fraction_p25=(
                "global_max_recoverable_fraction",
                lambda x: np.percentile(x, 25),
            ),
            global_max_recoverable_fraction_median=(
                "global_max_recoverable_fraction",
                "median",
            ),
            objective_achievability_p25=(
                "objective_achievability",
                lambda x: np.percentile(x, 25),
            ),
            objective_achievability_median=("objective_achievability", "median"),
            objective_opportunity_fraction_p25=(
                "objective_opportunity_fraction",
                lambda x: np.percentile(x, 25),
            ),
            objective_opportunity_fraction_median=(
                "objective_opportunity_fraction",
                "median",
            ),
            objective_physically_zero_fraction=(
                "objective_physically_zero",
                "mean",
            ),
            realized_objective_reduction_fraction_p25=(
                "realized_objective_reduction_fraction",
                lambda x: np.percentile(x, 25),
            ),
            realized_objective_reduction_fraction_median=(
                "realized_objective_reduction_fraction",
                "median",
            ),
            same_rank_top2_collision_fraction_median=(
                "same_rank_top2_collision_fraction",
                "median",
            ),
            dose_distinct_count_min=("dose_distinct_count", "min"),
            dose_distinct_count_median=("dose_distinct_count", "median"),
        )
        .reset_index()
    )
    by_k = (
        cells.groupby("active_positions", sort=True)
        .agg(
            samples=("achievability", "size"),
            fifo_imbalance_p25=(
                "fifo_rank_load_imbalance",
                lambda x: np.percentile(x, 25),
            ),
            achievability_p25=("achievability", lambda x: np.percentile(x, 25)),
            realized_objective_reduction_fraction_p25=(
                "realized_objective_reduction_fraction",
                lambda x: np.percentile(x, 25),
            ),
        )
        .reset_index()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "scheduler_screening_cells.csv", index=False)
    by_cell.to_csv(args.output / "scheduler_screening_by_cell.csv", index=False)
    by_k.to_csv(args.output / "scheduler_screening_by_k.csv", index=False)
    metadata = {
        "semantics": (
            "Gate screening uses realized summed-critical-load objective reduction. "
            "Global-single-max achievability is retained as a diagnostic only."
        ),
        "active_positions": args.active_positions,
        "routing_modes": args.routing_modes,
        "seeds": args.seeds,
        "restarts": args.restarts,
        "request_correlation_strength": args.request_correlation_strength,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(by_cell.to_string(index=False))


if __name__ == "__main__":
    main()
