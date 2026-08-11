from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import ModelConfig, RouterConfig
from .models import Request
from .placement import ExpertPlacement
from .trace_bundle import RouteTraceBundle, TraceValidationError
from .workloads.base import DecodeWorkload


class RouterLike(Protocol):
    placement: ExpertPlacement

    def route(
        self,
        request_id: int,
        iteration: int,
        layer_id: int,
        position_id: int,
    ) -> tuple[int, ...]: ...

    def prior_route(self, request_id: int, layer_id: int) -> tuple[int, ...]: ...

    def signature(
        self, request: Request, workload: DecodeWorkload, *, oracle: bool
    ) -> np.ndarray: ...

    def routing_stability(
        self, request: Request, workload: DecodeWorkload
    ) -> list[float]: ...


def _mix_seed(seed: int, *values: int) -> int:
    """Stable integer mixer independent of Python's randomized hash()."""
    value = seed & 0xFFFFFFFFFFFFFFFF
    for item in values:
        value ^= item + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2)
        value &= 0xFFFFFFFFFFFFFFFF
    return value


class SyntheticRouter:
    def __init__(self, model: ModelConfig, config: RouterConfig, seed: int):
        self.model = model
        self.config = config
        self.seed = seed
        self.placement = ExpertPlacement.round_robin(model)
        self._route_cache: dict[tuple[int, int, int, int], tuple[int, ...]] = {}

    def route(
        self,
        request_id: int,
        iteration: int,
        layer_id: int,
        position_id: int,
    ) -> tuple[int, ...]:
        key = (request_id, iteration, layer_id, position_id)
        if key not in self._route_cache:
            self._route_cache[key] = self._compute_route(*key)
        return self._route_cache[key]

    def prior_route(self, request_id: int, layer_id: int) -> tuple[int, ...]:
        """Deterministic top-k prior used when no route history exists."""
        probabilities = self._probabilities(request_id, layer_id)
        ranked = np.argsort(-probabilities, kind="stable")[: self.model.top_k]
        return tuple(int(expert_id) for expert_id in ranked)

    def _compute_route(
        self,
        request_id: int,
        iteration: int,
        layer_id: int,
        position_id: int,
    ) -> tuple[int, ...]:
        stability = self.config.temporal_stability
        if self.config.distribution == "temporally_stable":
            stability = max(stability, 0.9)
        elif self.config.distribution == "temporally_unstable":
            stability = min(stability, 0.1)

        if iteration > 0:
            stable_rng = np.random.default_rng(
                _mix_seed(self.seed, 11, request_id, iteration, layer_id, position_id)
            )
            if stable_rng.random() < stability:
                return self.route(request_id, iteration - 1, layer_id, position_id)

        rng = np.random.default_rng(
            _mix_seed(self.seed, 29, request_id, iteration, layer_id, position_id)
        )
        probabilities = self._probabilities(request_id, layer_id)
        chosen = rng.choice(
            self.model.num_experts,
            size=self.model.top_k,
            replace=False,
            p=probabilities,
        )
        return tuple(int(v) for v in chosen)

    def _probabilities(self, request_id: int, layer_id: int) -> np.ndarray:
        count = self.model.num_experts
        distribution = self.config.distribution
        if distribution in {"uniform", "temporally_stable", "temporally_unstable"}:
            weights = np.ones(count, dtype=float)
        elif distribution == "zipf":
            ranks = np.arange(1, count + 1, dtype=float)
            weights = 1.0 / np.power(ranks, self.config.zipf_alpha)
        elif distribution == "hot_expert":
            weights = np.full(count, (1.0 - self.config.hot_expert_probability) / (count - 1))
            weights[(layer_id * 3) % count] = self.config.hot_expert_probability
        elif distribution == "request_correlated":
            cluster_size = max(self.model.top_k, count // self.model.num_gpus)
            cluster_start = ((request_id * 5 + layer_id * 3) % count // cluster_size) * cluster_size
            weights = np.full(
                count,
                (1.0 - self.config.request_cluster_probability) / max(1, count - cluster_size),
            )
            weights[cluster_start : cluster_start + cluster_size] = (
                self.config.request_cluster_probability / cluster_size
            )
        else:
            raise ValueError(f"unsupported router distribution: {distribution}")
        return weights / weights.sum()

    def signature(self, request: Request, workload: DecodeWorkload, *, oracle: bool) -> np.ndarray:
        counts = np.zeros(self.model.num_experts, dtype=float)
        positions = (
            workload.active_positions(request)
            if oracle
            else workload.previous_active_positions(request)
        )
        for layer_id in range(self.model.num_layers):
            for position_id in range(positions):
                if oracle or request.iteration > 0:
                    iteration = request.iteration if oracle else request.iteration - 1
                    experts = self.route(
                        request.request_id,
                        iteration,
                        layer_id,
                        position_id,
                    )
                else:
                    experts = self.prior_route(request.request_id, layer_id)
                for expert_id in experts:
                    counts[expert_id] += 1.0
        return counts

    def routing_stability(self, request: Request, workload: DecodeWorkload) -> list[float]:
        if request.iteration == 0:
            return []
        values: list[float] = []
        positions = min(
            workload.active_positions(request),
            workload.previous_active_positions(request),
        )
        for layer_id in range(self.model.num_layers):
            for position_id in range(positions):
                current = set(
                    self.route(request.request_id, request.iteration, layer_id, position_id)
                )
                previous = set(
                    self.route(
                        request.request_id,
                        request.iteration - 1,
                        layer_id,
                        position_id,
                    )
                )
                values.append(len(current & previous) / len(current | previous))
        return values


class TraceRouter:
    """Strict replay router backed by a validated native-route trace bundle."""

    def __init__(self, bundle: RouteTraceBundle):
        self.bundle = bundle
        self.model = bundle.metadata.model
        self.bundle_sha256 = bundle.bundle_sha256
        self.placement = bundle.metadata.placement

    def route(
        self,
        request_id: int,
        iteration: int,
        layer_id: int,
        position_id: int,
    ) -> tuple[int, ...]:
        key = (request_id, iteration, layer_id, position_id)
        try:
            return self.bundle.routes[key]
        except KeyError as error:
            raise TraceValidationError(f"trace route lookup is not covered: {key}") from error

    def prior_route(self, request_id: int, layer_id: int) -> tuple[int, ...]:
        key = (request_id, layer_id)
        try:
            return self.bundle.priors[key]
        except KeyError as error:
            raise TraceValidationError(f"trace prior lookup is not covered: {key}") from error

    def signature(self, request: Request, workload: DecodeWorkload, *, oracle: bool) -> np.ndarray:
        counts = np.zeros(self.model.num_experts, dtype=float)
        positions = (
            workload.active_positions(request)
            if oracle
            else workload.previous_active_positions(request)
        )
        for layer_id in range(self.model.num_layers):
            for position_id in range(positions):
                if oracle or request.iteration > 0:
                    iteration = request.iteration if oracle else request.iteration - 1
                    experts = self.route(request.request_id, iteration, layer_id, position_id)
                else:
                    experts = self.prior_route(request.request_id, layer_id)
                for expert_id in experts:
                    counts[expert_id] += 1.0
        return counts

    def routing_stability(self, request: Request, workload: DecodeWorkload) -> list[float]:
        if request.iteration == 0:
            return []
        values: list[float] = []
        positions = min(
            workload.active_positions(request),
            workload.previous_active_positions(request),
        )
        for layer_id in range(self.model.num_layers):
            for position_id in range(positions):
                current = set(
                    self.route(request.request_id, request.iteration, layer_id, position_id)
                )
                previous = set(
                    self.route(
                        request.request_id,
                        request.iteration - 1,
                        layer_id,
                        position_id,
                    )
                )
                values.append(len(current & previous) / len(current | previous))
        return values


def make_router(model: ModelConfig, config: RouterConfig, seed: int) -> RouterLike:
    if config.source == "synthetic":
        return SyntheticRouter(model, config, seed)
    if config.trace_path is None:
        raise ValueError("trace router requires router.trace_path")
    bundle = RouteTraceBundle.load(config.trace_path, expected_model=model)
    return TraceRouter(bundle)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)
