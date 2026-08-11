"""Pure NumPy coordinated batch planning for multi-source EP diagnostics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanDiagnostics:
    """Non-optimality-aware diagnostics for an offline coordinated replay plan."""

    fifo_cost: float
    best_cost: float
    fifo_max_receive_load: int
    best_max_receive_load: int
    predicted_reduction_percent: float
    reassigned_request_fraction: float
    restart_costs: tuple[float, ...]
    best_so_far_costs: tuple[float, ...]

    @property
    def restart_cost_std(self) -> float:
        return float(np.std(self.restart_costs))

    @property
    def improved_in_last_two_restarts(self) -> bool:
        if len(self.best_so_far_costs) < 3:
            return False
        return self.best_so_far_costs[-1] < self.best_so_far_costs[-3]


def validate_counts(counts: np.ndarray, batch_size: int, experts_per_rank: int) -> None:
    if counts.ndim != 4:
        raise ValueError("counts must have shape [sources, requests, layers, experts]")
    sources, requests, _, experts = counts.shape
    if sources <= 0 or requests <= 0:
        raise ValueError("counts must contain sources and requests")
    if batch_size <= 0 or requests % batch_size:
        raise ValueError("requests must divide evenly by batch_size")
    if experts_per_rank <= 0 or experts % experts_per_rank:
        raise ValueError("experts must divide evenly by experts_per_rank")
    if experts // experts_per_rank != sources:
        raise ValueError("destination rank count must equal source rank count")
    if np.any(counts < 0):
        raise ValueError("route counts must be non-negative")


def fifo_plan(sources: int, requests: int, batch_size: int) -> list[list[list[int]]]:
    return [
        [
            list(range(start, start + batch_size))
            for start in range(0, requests, batch_size)
        ]
        for _ in range(sources)
    ]


def combined_rank_loads(
    counts: np.ndarray,
    plan: list[list[list[int]]],
    experts_per_rank: int,
) -> np.ndarray:
    """Return [batch, layer, destination_rank] loads from every source rank."""
    sources, requests, layers, experts = counts.shape
    validate_counts(counts, len(plan[0][0]), experts_per_rank)
    if len(plan) != sources:
        raise ValueError("plan source count does not match route counts")
    batch_count = requests // len(plan[0][0])
    loads = np.zeros((batch_count, layers, sources), dtype=np.int64)
    seen_by_source: list[list[int]] = []
    for source, batches in enumerate(plan):
        if len(batches) != batch_count:
            raise ValueError("every source must have the same batch count")
        seen = [request for batch in batches for request in batch]
        if sorted(seen) != list(range(requests)):
            raise ValueError("each source plan must use every request exactly once")
        seen_by_source.append(seen)
        for batch_index, batch in enumerate(batches):
            expert_load = counts[source, batch].sum(axis=0)
            loads[batch_index] += expert_load.reshape(
                layers, sources, experts_per_rank
            ).sum(axis=2)
    if len(seen_by_source) != sources:
        raise RuntimeError("source validation failed")
    return loads


def plan_cost(
    counts: np.ndarray,
    plan: list[list[list[int]]],
    experts_per_rank: int,
) -> float:
    loads = combined_rank_loads(counts, plan, experts_per_rank)
    return float(loads.max(axis=2).sum())


def composition_invariant_cost_lower_bound(
    counts: np.ndarray,
    batch_size: int,
    experts_per_rank: int,
) -> float:
    """Lower-bound summed batch/layer max load using invariant destination totals."""
    validate_counts(counts, batch_size, experts_per_rank)
    sources, requests, layers, experts = counts.shape
    del sources, requests
    destination_totals = (
        counts.sum(axis=(0, 1))
        .reshape(layers, experts // experts_per_rank, experts_per_rank)
        .sum(axis=2)
    )
    return float(destination_totals.max(axis=1).sum())


def split_vectors_and_local_expert_counts(
    *,
    global_routes: np.ndarray,
    global_batches: list[list[list[int]]],
    batch_index: int,
    layer: int,
    experts_per_rank: int,
    destination_rank: int,
) -> tuple[list[list[int]], np.ndarray]:
    """Derive replay split vectors without a timed count-exchange collective."""

    sources = global_routes.shape[0]
    if not 0 <= destination_rank < sources:
        raise ValueError("destination rank is out of range")
    split_matrix = np.zeros((sources, sources), dtype=np.int64)
    local_ids: list[np.ndarray] = []
    for source in range(sources):
        selected = global_batches[source][batch_index]
        ids = global_routes[source, selected, layer].reshape(-1)
        destinations = ids // experts_per_rank
        if np.any(destinations >= sources):
            raise ValueError("route contains an out-of-range expert")
        split_matrix[source] = np.bincount(destinations, minlength=sources)
        destination_ids = ids[destinations == destination_rank] % experts_per_rank
        if destination_ids.size:
            local_ids.append(destination_ids)
    concatenated = (
        np.concatenate(local_ids) if local_ids else np.empty(0, dtype=np.int64)
    )
    expert_counts = np.bincount(concatenated, minlength=experts_per_rank)
    return split_matrix.tolist(), expert_counts


def _two_batch_choices(requests: int, batch_size: int) -> list[list[list[int]]]:
    if requests != 2 * batch_size:
        raise ValueError("coordinate-descent planner currently requires exactly two batches")
    all_requests = set(range(requests))
    choices: list[list[list[int]]] = []
    for first_tuple in itertools.combinations(range(requests), batch_size):
        first = list(first_tuple)
        second = sorted(all_requests - set(first))
        choices.append([first, second])
    return choices


def coordinated_plan(
    counts: np.ndarray,
    batch_size: int,
    experts_per_rank: int,
    *,
    max_sweeps: int = 4,
    restarts: int = 1,
    seed: int = 0,
) -> list[list[list[int]]]:
    """Coordinate-descent plan using combined loads from every source rank.

    This is a diagnostic greedy planner, not a global optimum or an online policy.
    """
    plan, _ = coordinated_plan_with_diagnostics(
        counts,
        batch_size,
        experts_per_rank,
        max_sweeps=max_sweeps,
        restarts=restarts,
        seed=seed,
    )
    return plan


def _coordinate_descent(
    counts: np.ndarray,
    plan: list[list[list[int]]],
    choices: list[list[list[int]]],
    experts_per_rank: int,
    max_sweeps: int,
) -> tuple[list[list[list[int]]], float]:
    sources, requests, layers, experts = counts.shape
    request_rank_loads = counts.reshape(
        sources,
        requests,
        layers,
        experts // experts_per_rank,
        experts_per_rank,
    ).sum(axis=4)
    loads = combined_rank_loads(counts, plan, experts_per_rank)
    current_cost = float(loads.max(axis=2).sum())

    def source_load(source: int, source_plan: list[list[int]]) -> np.ndarray:
        return np.stack(
            [request_rank_loads[source, batch].sum(axis=0) for batch in source_plan]
        )

    for _ in range(max_sweeps):
        changed = False
        for source in range(sources):
            best_batches = plan[source]
            best_cost = current_cost
            current_source_load = source_load(source, plan[source])
            best_source_load = current_source_load
            for choice in choices:
                candidate_source_load = source_load(source, choice)
                candidate_loads = loads - current_source_load + candidate_source_load
                candidate_cost = float(candidate_loads.max(axis=2).sum())
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_batches = choice
                    best_source_load = candidate_source_load
            if best_cost < current_cost:
                plan[source] = best_batches
                loads = loads - current_source_load + best_source_load
                current_cost = best_cost
                changed = True
        if not changed:
            break
    return plan, current_cost


def _coordinate_descent_swaps(
    counts: np.ndarray,
    plan: list[list[list[int]]],
    experts_per_rank: int,
    max_sweeps: int,
) -> tuple[list[list[list[int]]], float]:
    """Coordinate descent using request swaps for three or more batches.

    This is a diagnostic local search. It preserves batch cardinality and provides no
    optimality guarantee; its result must remain labeled best-found.
    """
    sources, requests, layers, experts = counts.shape
    request_rank_loads = counts.reshape(
        sources,
        requests,
        layers,
        experts // experts_per_rank,
        experts_per_rank,
    ).sum(axis=4)
    loads = combined_rank_loads(counts, plan, experts_per_rank)
    current_cost = float(loads.max(axis=2).sum())
    for _ in range(max_sweeps):
        changed = False
        for source in range(sources):
            best_source = [list(batch) for batch in plan[source]]
            best_cost = current_cost
            best_swap: tuple[int, int, int, int] | None = None
            batch_count = len(plan[source])
            for left_batch in range(batch_count):
                for right_batch in range(left_batch + 1, batch_count):
                    old_pair_cost = float(
                        loads[[left_batch, right_batch]].max(axis=2).sum()
                    )
                    for left_slot in range(len(plan[source][left_batch])):
                        for right_slot in range(len(plan[source][right_batch])):
                            left_request = plan[source][left_batch][left_slot]
                            right_request = plan[source][right_batch][right_slot]
                            delta = (
                                request_rank_loads[source, right_request]
                                - request_rank_loads[source, left_request]
                            )
                            new_left = loads[left_batch] + delta
                            new_right = loads[right_batch] - delta
                            new_pair_cost = float(
                                new_left.max(axis=1).sum()
                                + new_right.max(axis=1).sum()
                            )
                            candidate_cost = current_cost - old_pair_cost + new_pair_cost
                            if candidate_cost < best_cost:
                                best_cost = candidate_cost
                                best_source = [list(batch) for batch in plan[source]]
                                (
                                    best_source[left_batch][left_slot],
                                    best_source[right_batch][right_slot],
                                ) = (right_request, left_request)
                                best_swap = (
                                    left_batch,
                                    right_batch,
                                    left_request,
                                    right_request,
                                )
            if best_cost < current_cost:
                plan[source] = best_source
                if best_swap is None:
                    raise RuntimeError("improving swap was not retained")
                left_batch, right_batch, left_request, right_request = best_swap
                delta = (
                    request_rank_loads[source, right_request]
                    - request_rank_loads[source, left_request]
                )
                loads[left_batch] += delta
                loads[right_batch] -= delta
                current_cost = best_cost
                changed = True
        if not changed:
            break
    return plan, current_cost


def _random_plan(
    sources: int,
    requests: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[list[list[int]]]:
    plan: list[list[list[int]]] = []
    for _ in range(sources):
        order = rng.permutation(requests).tolist()
        plan.append(
            [order[start : start + batch_size] for start in range(0, requests, batch_size)]
        )
    return plan


def _reassigned_request_fraction(
    reference: list[list[list[int]]],
    candidate: list[list[list[int]]],
) -> float:
    moved = 0
    total = 0
    for reference_source, candidate_source in zip(reference, candidate, strict=True):
        reference_batch = {
            request: batch_index
            for batch_index, batch in enumerate(reference_source)
            for request in batch
        }
        candidate_batch = {
            request: batch_index
            for batch_index, batch in enumerate(candidate_source)
            for request in batch
        }
        total += len(reference_batch)
        moved += sum(
            reference_batch[request] != candidate_batch[request]
            for request in reference_batch
        )
    return moved / total


def reassigned_request_fraction(
    reference: list[list[list[int]]],
    candidate: list[list[list[int]]],
) -> float:
    """Return the request fraction assigned to a different batch index."""
    return _reassigned_request_fraction(reference, candidate)


def coordinated_plan_with_diagnostics(
    counts: np.ndarray,
    batch_size: int,
    experts_per_rank: int,
    *,
    max_sweeps: int = 4,
    restarts: int = 8,
    seed: int = 0,
) -> tuple[list[list[list[int]]], PlanDiagnostics]:
    """Return a best-found plan and continuous search-quality diagnostics.

    Random restarts measure sensitivity to the local search initialization. They do not
    turn coordinate descent into a global optimum or an oracle.
    """

    validate_counts(counts, batch_size, experts_per_rank)
    sources, requests = counts.shape[:2]
    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be positive")
    if restarts <= 0:
        raise ValueError("restarts must be positive")
    batch_count = requests // batch_size
    # Exact source-coordinate enumeration is useful for the controlled eight-request
    # contract but grows combinatorially (C(16, 8)=12,870). Larger pools use the same
    # cardinality-preserving swap search as multi-batch audits.
    choices = (
        _two_batch_choices(requests, batch_size)
        if batch_count == 2 and requests <= 8
        else None
    )
    fifo = fifo_plan(sources, requests, batch_size)
    fifo_cost = plan_cost(counts, fifo, experts_per_rank)
    rng = np.random.default_rng(seed)
    initial_plans = [fifo]
    initial_plans.extend(
        _random_plan(sources, requests, batch_size, rng) for _ in range(restarts - 1)
    )
    candidates: list[tuple[list[list[list[int]]], float]] = []
    for initial in initial_plans:
        copied = [[list(batch) for batch in source] for source in initial]
        if choices is not None:
            candidates.append(
                _coordinate_descent(
                    counts,
                    copied,
                    choices,
                    experts_per_rank,
                    max_sweeps,
                )
            )
        else:
            candidates.append(
                _coordinate_descent_swaps(
                    counts,
                    copied,
                    experts_per_rank,
                    max_sweeps,
                )
            )
    best_plan, best_cost = min(candidates, key=lambda value: value[1])
    reduction = 100.0 * (fifo_cost - best_cost) / max(fifo_cost, 1.0)
    restart_costs = tuple(cost for _, cost in candidates)
    running_best: list[float] = []
    for cost in restart_costs:
        running_best.append(min(cost, running_best[-1] if running_best else cost))
    diagnostics = PlanDiagnostics(
        fifo_cost=fifo_cost,
        best_cost=best_cost,
        fifo_max_receive_load=int(
            combined_rank_loads(counts, fifo, experts_per_rank).max()
        ),
        best_max_receive_load=int(
            combined_rank_loads(counts, best_plan, experts_per_rank).max()
        ),
        predicted_reduction_percent=reduction,
        reassigned_request_fraction=_reassigned_request_fraction(fifo, best_plan),
        restart_costs=restart_costs,
        best_so_far_costs=tuple(running_best),
    )
    return best_plan, diagnostics


def coordinated_dose_ladder(
    counts: np.ndarray,
    best_plan: list[list[list[int]]],
    batch_size: int,
    experts_per_rank: int,
    *,
    targets: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    random_samples: int = 10_000,
    seed: int = 0,
) -> list[tuple[float, float, list[list[list[int]]]]]:
    """Return valid replay plans nearest requested objective-reduction doses.

    The returned float is the achieved fraction of the FIFO-to-best objective
    reduction. Candidates include source-wise FIFO/best mixtures and deterministic
    random valid plans. A caller must inspect distinct achieved doses; request
    granularity can still make an intermediate ladder impossible for some cells.
    """
    validate_counts(counts, batch_size, experts_per_rank)
    sources, requests = counts.shape[:2]
    fifo = fifo_plan(sources, requests, batch_size)
    fifo_cost = plan_cost(counts, fifo, experts_per_rank)
    best_cost = plan_cost(counts, best_plan, experts_per_rank)
    opportunity = fifo_cost - best_cost
    if opportunity <= 0:
        return [(0.0, 0.0, fifo)]
    candidates: dict[
        tuple[tuple[tuple[int, ...], ...], ...],
        tuple[float, list[list[list[int]]]],
    ] = {}
    candidate_plans: list[list[list[list[int]]]] = []
    for mask in range(1 << sources):
        candidate_plans.append(
            [
                [
                    list(batch)
                    for batch in (
                        best_plan[source] if mask & (1 << source) else fifo[source]
                    )
                ]
                for source in range(sources)
            ]
        )
    rng = np.random.default_rng(seed)
    candidate_plans.extend(
        _random_plan(sources, requests, batch_size, rng)
        for _ in range(random_samples)
    )
    for mixed in candidate_plans:
        cost = plan_cost(counts, mixed, experts_per_rank)
        if best_cost <= cost <= fifo_cost:
            signature = tuple(
                tuple(tuple(batch) for batch in source_plan) for source_plan in mixed
            )
            candidates[signature] = (cost, mixed)
    selected: list[tuple[float, float, list[list[list[int]]]]] = []
    used: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    for target in targets:
        if not 0.0 <= target <= 1.0:
            raise ValueError("dose targets must lie in [0, 1]")
        eligible = candidates.items()
        if target < 1.0:
            intermediate = [
                item
                for item in eligible
                if item[1][0] > best_cost and item[0] not in used
            ]
            if intermediate:
                eligible = intermediate
        signature, (cost, plan) = min(
            eligible,
            key=lambda item: abs(((fifo_cost - item[1][0]) / opportunity) - target),
        )
        if signature in used:
            continue
        used.add(signature)
        selected.append((target, (fifo_cost - cost) / opportunity, plan))
    return sorted(selected, key=lambda value: value[0])
