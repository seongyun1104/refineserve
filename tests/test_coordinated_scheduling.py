from __future__ import annotations

import numpy as np
import pytest

from hardware.coordinated_scheduling import (
    combined_rank_loads,
    composition_invariant_cost_lower_bound,
    coordinated_dose_ladder,
    coordinated_plan,
    coordinated_plan_with_diagnostics,
    fifo_plan,
    plan_cost,
    split_vectors_and_local_expert_counts,
)


def test_combined_loads_sum_all_source_contributions() -> None:
    counts = np.zeros((2, 4, 1, 4), dtype=np.int64)
    counts[0, :, 0, 0] = [4, 3, 1, 0]
    counts[1, :, 0, 0] = [4, 3, 1, 0]
    counts[0, :, 0, 2] = [0, 1, 3, 4]
    counts[1, :, 0, 2] = [0, 1, 3, 4]

    loads = combined_rank_loads(counts, fifo_plan(2, 4, 2), experts_per_rank=2)

    assert loads.shape == (2, 1, 2)
    assert loads[:, 0].tolist() == [[14, 2], [2, 14]]


def test_coordinated_plan_improves_colliding_local_fifo_choices() -> None:
    counts = np.zeros((2, 4, 1, 4), dtype=np.int64)
    for source in range(2):
        counts[source, 0, 0, 0] = 8
        counts[source, 1, 0, 0] = 8
        counts[source, 2, 0, 2] = 8
        counts[source, 3, 0, 2] = 8
    fifo = fifo_plan(2, 4, 2)

    planned = coordinated_plan(counts, batch_size=2, experts_per_rank=2)

    assert plan_cost(counts, planned, experts_per_rank=2) < plan_cost(
        counts, fifo, experts_per_rank=2
    )
    assert sorted(request for batch in planned[0] for request in batch) == [0, 1, 2, 3]
    assert sorted(request for batch in planned[1] for request in batch) == [0, 1, 2, 3]


def test_coordinated_plan_reports_non_vacuity_and_restart_dispersion() -> None:
    counts = np.zeros((2, 4, 1, 4), dtype=np.int64)
    for source in range(2):
        counts[source, 0:2, 0, 0] = 8
        counts[source, 2:4, 0, 2] = 8

    planned, diagnostics = coordinated_plan_with_diagnostics(
        counts,
        batch_size=2,
        experts_per_rank=2,
        restarts=4,
        seed=17,
    )

    assert diagnostics.best_cost == plan_cost(counts, planned, experts_per_rank=2)
    assert diagnostics.best_cost < diagnostics.fifo_cost
    assert diagnostics.best_max_receive_load <= diagnostics.fifo_max_receive_load
    assert diagnostics.predicted_reduction_percent > 0
    assert diagnostics.reassigned_request_fraction > 0
    assert len(diagnostics.restart_costs) == 4
    assert len(diagnostics.best_so_far_costs) == 4
    assert all(
        later <= earlier
        for earlier, later in zip(
            diagnostics.best_so_far_costs,
            diagnostics.best_so_far_costs[1:],
            strict=False,
        )
    )


def test_coordinated_plan_supports_more_than_two_batches() -> None:
    counts = np.zeros((2, 8, 1, 4), dtype=np.int64)
    for source in range(2):
        counts[source, 0:4, 0, 0] = 8
        counts[source, 4:8, 0, 2] = 8
    fifo = fifo_plan(2, 8, 2)

    planned, diagnostics = coordinated_plan_with_diagnostics(
        counts,
        batch_size=2,
        experts_per_rank=2,
        restarts=4,
        seed=17,
    )

    assert diagnostics.best_cost == plan_cost(counts, planned, experts_per_rank=2)
    assert diagnostics.best_cost < plan_cost(counts, fifo, experts_per_rank=2)
    for source_plan in planned:
        assert len(source_plan) == 4
        assert sorted(request for batch in source_plan for request in batch) == list(
            range(8)
        )
    assert diagnostics.restart_cost_std >= 0
    lower_bound = composition_invariant_cost_lower_bound(
        counts,
        batch_size=2,
        experts_per_rank=2,
    )
    assert lower_bound <= diagnostics.best_cost < diagnostics.fifo_cost

    ladder = coordinated_dose_ladder(
        counts,
        planned,
        batch_size=2,
        experts_per_rank=2,
    )
    assert ladder[-1][0] == pytest.approx(1.0)
    assert ladder[-1][1] == pytest.approx(1.0)
    assert all(0.0 <= achieved <= 1.0 for _, achieved, _ in ladder)
    assert all(
        later > earlier
        for earlier, later in zip(
            [target for target, _, _ in ladder],
            [target for target, _, _ in ladder][1:],
            strict=False,
        )
    )


def test_invalid_plan_shape_is_rejected() -> None:
    counts = np.zeros((2, 4, 1, 4), dtype=np.int64)

    with pytest.raises(ValueError, match="every request exactly once"):
        combined_rank_loads(
            counts,
            [[[0, 1], [1, 2]], [[0, 1], [2, 3]]],
            experts_per_rank=2,
        )


def test_replay_split_vectors_and_local_expert_counts_match_routes() -> None:
    routes = np.array(
        [
            [[[[0, 2]]], [[[1, 3]]]],
            [[[[0, 1]]], [[[2, 3]]]],
        ],
        dtype=np.int64,
    )
    batches = [[[0], [1]], [[0], [1]]]

    split_matrix, expert_counts = split_vectors_and_local_expert_counts(
        global_routes=routes,
        global_batches=batches,
        batch_index=0,
        layer=0,
        experts_per_rank=2,
        destination_rank=0,
    )

    assert split_matrix == [[1, 1], [2, 0]]
    assert expert_counts.tolist() == [2, 1]
