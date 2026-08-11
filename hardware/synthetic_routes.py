"""Deterministic synthetic route generation shared by GPU and CPU diagnostics."""

from __future__ import annotations

import numpy as np

ROUTING_MODES = (
    "uniform",
    "balanced_round_robin",
    "mild_skew",
    "strong_skew",
    "hot_expert",
    "request_correlated",
    "temporally_stable",
    "temporally_unstable",
)


def _weighted_pair(
    rng: np.random.Generator, experts: int, exponent: float
) -> list[int]:
    weights = 1.0 / np.power(np.arange(1, experts + 1, dtype=float), exponent)
    weights /= weights.sum()
    return [int(value) for value in rng.choice(experts, 2, replace=False, p=weights)]


def make_routes(
    *,
    seed: int,
    mode: str,
    global_request_ids: np.ndarray,
    layers: int,
    positions: int,
    experts: int,
    request_correlation_strength: float = 0.75,
    ep_ranks: int = 4,
) -> np.ndarray:
    if not 0.0 <= request_correlation_strength <= 1.0:
        raise ValueError("request correlation strength must lie in [0, 1]")
    if experts % ep_ranks:
        raise ValueError("experts must divide evenly across EP ranks")
    experts_per_rank = experts // ep_ranks
    output = np.empty((len(global_request_ids), layers, positions, 2), dtype=np.int64)
    for local_request, request_id in enumerate(global_request_ids.tolist()):
        for layer in range(layers):
            base_rng = np.random.default_rng(
                np.random.SeedSequence([seed, request_id, layer, 991])
            )
            stable_pair = [
                int(value) for value in base_rng.choice(experts, 2, replace=False)
            ]
            preferred_rank = int(base_rng.integers(ep_ranks))
            preferred_experts = np.arange(
                preferred_rank * experts_per_rank,
                (preferred_rank + 1) * experts_per_rank,
            )
            for position in range(positions):
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, request_id, layer, position])
                )
                if mode == "uniform":
                    pair = [
                        int(value) for value in rng.choice(experts, 2, replace=False)
                    ]
                elif mode == "balanced_round_robin":
                    first = (request_id * positions + position + layer) % experts
                    pair = [first, (first + 7) % experts]
                elif mode == "mild_skew":
                    pair = _weighted_pair(rng, experts, 0.5)
                elif mode == "strong_skew":
                    pair = _weighted_pair(rng, experts, 1.5)
                elif mode == "hot_expert":
                    first = 0 if rng.random() < 0.85 else int(rng.integers(experts))
                    second = int(rng.integers(experts - 1))
                    if second >= first:
                        second += 1
                    pair = [first, second]
                elif mode == "request_correlated":
                    population = (
                        preferred_experts
                        if rng.random() < request_correlation_strength
                        else np.arange(experts)
                    )
                    pair = [
                        int(value) for value in rng.choice(population, 2, replace=False)
                    ]
                elif mode == "temporally_stable":
                    pair = (
                        stable_pair
                        if position == 0 or rng.random() < 0.9
                        else [
                            int(value)
                            for value in rng.choice(experts, 2, replace=False)
                        ]
                    )
                elif mode == "temporally_unstable":
                    pair = [
                        int(value) for value in rng.choice(experts, 2, replace=False)
                    ]
                else:
                    raise ValueError(f"unknown routing mode: {mode}")
                if pair[0] == pair[1]:
                    raise RuntimeError("route generator selected a duplicate expert")
                output[local_request, layer, position] = pair
    return output


def request_counts(route_data: np.ndarray, experts: int) -> np.ndarray:
    requests, layers = route_data.shape[:2]
    counts = np.zeros((requests, layers, experts), dtype=np.int64)
    for request in range(requests):
        for layer in range(layers):
            counts[request, layer] = np.bincount(
                route_data[request, layer].reshape(-1), minlength=experts
            )
    return counts
