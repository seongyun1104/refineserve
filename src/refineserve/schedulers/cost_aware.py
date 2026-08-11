from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..calibration import CalibrationArtifact
from ..config import ComputeConfig, ModelConfig, NetworkConfig, SchedulerConfig
from ..cost import LayerCostModel
from ..models import Request
from ..placement import ExpertPlacement
from ..router import RouterLike, cosine_similarity
from ..workloads.base import DecodeWorkload
from .base import Scheduler

CostObjective = Literal[
    "load_balance_only",
    "critical_path_only",
    "locality_plus_load",
    "joint",
]
RouteKnowledge = Literal["previous", "routing_oracle", "runtime_oracle"]


class _PredictedRouter:
    """Expose either previous-iteration or current (oracle) routes."""

    def __init__(
        self,
        router: RouterLike,
        workload: DecodeWorkload,
        requests: list[Request],
        knowledge: RouteKnowledge,
    ) -> None:
        self.router = router
        self.workload = workload
        self.requests = {request.request_id: request for request in requests}
        self.knowledge = knowledge

    def route(
        self,
        request_id: int,
        iteration: int,
        layer_id: int,
        position_id: int,
    ) -> tuple[int, ...]:
        if self.knowledge != "previous":
            return self.router.route(request_id, iteration, layer_id, position_id)
        if iteration == 0:
            return self.router.prior_route(request_id, layer_id)
        request = self.requests[request_id]
        previous_width = self.workload.previous_active_positions(request)
        predicted_position = position_id % max(1, previous_width)
        return self.router.route(
            request_id,
            iteration - 1,
            layer_id,
            predicted_position,
        )


@dataclass(frozen=True)
class BatchEstimate:
    critical_path_ms: float
    mean_rank_expert_load_cv: float
    locality: float
    kv_rank_load_cv: float
    progress_span: int
    unique_expert_invocations: int
    network_messages: int


class ReferenceBatchCostEstimator:
    """Readable LayerCostModel replay used as the estimator correctness oracle."""

    def __init__(
        self,
        model: ModelConfig,
        compute: ComputeConfig,
        network: NetworkConfig,
        calibration: CalibrationArtifact | None = None,
        placement: ExpertPlacement | None = None,
    ) -> None:
        self.model = model
        self.cost = LayerCostModel(
            model,
            compute,
            replace(network, use_rank_critical_path=True),
            calibration,
            placement,
        )
        self._cache: dict[
            tuple[RouteKnowledge, tuple[tuple[int, int, int], ...]], BatchEstimate
        ] = {}

    def estimate(
        self,
        requests: list[Request],
        workload: DecodeWorkload,
        router: RouterLike,
        knowledge: RouteKnowledge,
    ) -> BatchEstimate:
        cache_key = (
            knowledge,
            tuple(
                sorted(
                    (request.request_id, request.iteration, request.kv_location)
                    for request in requests
                )
            ),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        predicted_router = _PredictedRouter(router, workload, requests, knowledge)
        work_items = workload.ready_work_items(requests)
        critical_path_ms = 0.0
        load_cvs: list[float] = []
        unique_expert_invocations = 0
        network_messages = 0
        signatures = {
            request.request_id: np.zeros(self.model.num_experts, dtype=float)
            for request in requests
        }

        for layer_id in range(self.model.num_layers):
            layer = self.cost.execute_layer(
                requests,
                work_items,
                layer_id,
                predicted_router,
            )
            critical_path_ms += layer.rank_critical_path_ms
            rank_tokens = np.asarray(
                [execution.expert_tokens for execution in layer.rank_executions],
                dtype=float,
            )
            mean_tokens = float(rank_tokens.mean())
            load_cvs.append(
                float(rank_tokens.std() / mean_tokens) if mean_tokens > 0.0 else 0.0
            )
            unique_expert_invocations += layer.expert_invocations
            network_messages += layer.network_messages
            for item in work_items:
                for expert_id in predicted_router.route(
                    item.request_id,
                    item.iteration,
                    layer_id,
                    item.position_id,
                ):
                    signatures[item.request_id][expert_id] += 1.0

        locality = _mean_pairwise_similarity(list(signatures.values()))
        kv_tokens = np.zeros(self.model.num_gpus, dtype=float)
        requests_by_id = {request.request_id: request for request in requests}
        for item in work_items:
            kv_tokens[requests_by_id[item.request_id].kv_location] += 1.0
        kv_mean = float(kv_tokens.mean())
        kv_rank_load_cv = float(kv_tokens.std() / kv_mean) if kv_mean > 0.0 else 0.0
        estimate = BatchEstimate(
            critical_path_ms=critical_path_ms,
            mean_rank_expert_load_cv=float(np.mean(load_cvs)) if load_cvs else 0.0,
            locality=locality,
            kv_rank_load_cv=kv_rank_load_cv,
            progress_span=(
                max(request.iteration for request in requests)
                - min(request.iteration for request in requests)
            ),
            unique_expert_invocations=unique_expert_invocations,
            network_messages=network_messages,
        )
        self._cache[cache_key] = estimate
        return estimate


@dataclass(frozen=True)
class _RequestRouteProfile:
    expert_counts: np.ndarray
    pair_counts: np.ndarray
    signature: np.ndarray
    positions: int
    source_gpu: int


@dataclass(frozen=True)
class _BatchHistogram:
    expert_counts: np.ndarray
    pair_counts: np.ndarray
    source_tokens: np.ndarray
    signatures: tuple[np.ndarray, ...]
    signature_sum: np.ndarray
    iterations: tuple[int, ...]


class BatchCostEstimator:
    """Histogram-based estimator equivalent to full modeled-layer replay."""

    def __init__(
        self,
        model: ModelConfig,
        compute: ComputeConfig,
        network: NetworkConfig,
        calibration: CalibrationArtifact | None = None,
        placement: ExpertPlacement | None = None,
    ) -> None:
        self.model = model
        self.compute = compute
        self.network = replace(network, use_rank_critical_path=True)
        self.calibration = calibration
        self.placement = placement or ExpertPlacement.round_robin(model)
        self.placement_one_hot = self.placement.one_hot()
        self._profile_cache: dict[
            tuple[RouteKnowledge, int, int, int], _RequestRouteProfile
        ] = {}
        self._estimate_cache: dict[
            tuple[RouteKnowledge, tuple[tuple[int, int, int], ...]], BatchEstimate
        ] = {}

    def estimate(
        self,
        requests: list[Request],
        workload: DecodeWorkload,
        router: RouterLike,
        knowledge: RouteKnowledge,
    ) -> BatchEstimate:
        cache_key = (
            knowledge,
            tuple(
                sorted(
                    (request.request_id, request.iteration, request.kv_location)
                    for request in requests
                )
            ),
        )
        if cache_key in self._estimate_cache:
            return self._estimate_cache[cache_key]
        profiles = [self._profile(request, workload, router, knowledge) for request in requests]
        histogram = self._histogram(profiles, requests)
        estimate = self._estimate_histogram(histogram)
        self._estimate_cache[cache_key] = estimate
        return estimate

    def histogram(
        self,
        requests: list[Request],
        workload: DecodeWorkload,
        router: RouterLike,
        knowledge: RouteKnowledge,
    ) -> _BatchHistogram:
        profiles = [self._profile(request, workload, router, knowledge) for request in requests]
        return self._histogram(profiles, requests)

    def profile(
        self,
        request: Request,
        workload: DecodeWorkload,
        router: RouterLike,
        knowledge: RouteKnowledge,
    ) -> _RequestRouteProfile:
        return self._profile(request, workload, router, knowledge)

    def extend(
        self,
        histogram: _BatchHistogram,
        profile: _RequestRouteProfile,
        iteration: int,
    ) -> _BatchHistogram:
        source_tokens = histogram.source_tokens.copy()
        source_tokens[profile.source_gpu] += profile.positions
        return _BatchHistogram(
            expert_counts=histogram.expert_counts + profile.expert_counts,
            pair_counts=histogram.pair_counts + profile.pair_counts,
            source_tokens=source_tokens,
            signatures=(*histogram.signatures, profile.signature),
            signature_sum=histogram.signature_sum + profile.signature,
            iterations=(*histogram.iterations, iteration),
        )

    def estimate_extension(
        self,
        histogram: _BatchHistogram,
        profile: _RequestRouteProfile,
        iteration: int,
    ) -> BatchEstimate:
        return self._estimate_histogram(self.extend(histogram, profile, iteration))

    def cheap_extension_score(
        self,
        histogram: _BatchHistogram,
        profile: _RequestRouteProfile,
        iteration: int,
        scheduler: SchedulerConfig,
    ) -> float:
        return float(
            self.cheap_extension_scores(
                histogram,
                [profile],
                [iteration],
                scheduler,
            )[0]
        )

    def cheap_extension_scores(
        self,
        histogram: _BatchHistogram,
        profiles: list[_RequestRouteProfile],
        iterations: list[int],
        scheduler: SchedulerConfig,
    ) -> np.ndarray:
        candidate_expert_counts = np.stack(
            [profile.expert_counts for profile in profiles],
            axis=0,
        )
        expert_counts = candidate_expert_counts + histogram.expert_counts[None, :, :]
        rank_tokens = np.einsum(
            "cle,leg->clg",
            expert_counts,
            self.placement_one_hot,
        )
        rank_unique = np.einsum(
            "cle,leg->clg",
            expert_counts > 0,
            self.placement_one_hot,
        )
        if self.calibration is not None and self.calibration.expert_kernel_curve is not None:
            expert_elapsed_ms = self.calibration.expert_kernel_curve.latencies_ms(expert_counts)
            rank_compute_ms = np.einsum(
                "cle,leg->clg",
                expert_elapsed_ms,
                self.placement_one_hot,
            )
        else:
            weight_read_ms = self.compute.expert_weight_bytes / (
                self.compute.expert_memory_bandwidth_gb_per_s * 1_000_000.0
            )
            rank_compute_ms = (
                rank_tokens / self.compute.expert_peak_tokens_per_ms
                + rank_unique * (self.compute.expert_launch_ms + weight_read_ms)
            )
        candidate_pair_counts = np.stack(
            [profile.pair_counts for profile in profiles],
            axis=0,
        )
        pair_counts = candidate_pair_counts + histogram.pair_counts[None, :, :, :]
        endpoint_tokens = pair_counts.sum(axis=3) + pair_counts.sum(axis=2)
        endpoint_pairs = (pair_counts > 0).sum(axis=3) + (pair_counts > 0).sum(axis=2)
        endpoint_messages = (
            2 * endpoint_pairs
            if self.network.aggregate_messages
            else 2 * endpoint_tokens
        )
        endpoint_bytes = (
            endpoint_tokens * self.model.hidden_size * self.model.bytes_per_element * 2
        )
        if self.calibration is not None and self.calibration.network_curves:
            rank_communication_ms = self.calibration.network_latencies_ms(
                collective="ep_dispatch_combine",
                active_ranks=self.model.num_gpus,
                message_counts=endpoint_messages,
                transferred_bytes=endpoint_bytes,
            )
        else:
            rank_communication_ms = (
                endpoint_messages * self.network.fixed_message_latency_ms
                + endpoint_bytes / (self.network.bandwidth_gb_per_s * 1_000_000.0)
            )
        source_tokens = np.repeat(
            histogram.source_tokens[None, :],
            len(profiles),
            axis=0,
        )
        for candidate_index, profile in enumerate(profiles):
            source_tokens[candidate_index, profile.source_gpu] += profile.positions
        attention_ms = self.compute.attention_base_ms + (
            source_tokens.max(axis=1) * self.compute.attention_token_ms
        )
        proxy_critical_path = np.max(
            attention_ms[:, None, None] + rank_compute_ms + rank_communication_ms,
            axis=2,
        ).sum(axis=1)
        current_min = min(histogram.iterations)
        current_max = max(histogram.iterations)
        iteration_array = np.asarray(iterations, dtype=np.int64)
        progress_span = np.maximum(current_max, iteration_array) - np.minimum(
            current_min,
            iteration_array,
        )
        signature_matrix = np.stack([profile.signature for profile in profiles], axis=0)
        histogram_norm = float(np.linalg.norm(histogram.signature_sum))
        candidate_norms = np.linalg.norm(signature_matrix, axis=1)
        denominators = histogram_norm * candidate_norms
        locality = np.divide(
            signature_matrix @ histogram.signature_sum,
            denominators,
            out=np.zeros(len(profiles), dtype=float),
            where=denominators > 0.0,
        )
        return (
            proxy_critical_path
            + scheduler.progress_fragmentation_weight_ms * progress_span
            - scheduler.locality_benefit_ms * locality
        )

    def _histogram(
        self,
        profiles: list[_RequestRouteProfile],
        requests: list[Request],
    ) -> _BatchHistogram:
        source_tokens = np.zeros(self.model.num_gpus, dtype=np.int64)
        for profile in profiles:
            source_tokens[profile.source_gpu] += profile.positions
        return _BatchHistogram(
            expert_counts=np.sum(
                [profile.expert_counts for profile in profiles],
                axis=0,
                dtype=np.int64,
            ),
            pair_counts=np.sum(
                [profile.pair_counts for profile in profiles],
                axis=0,
                dtype=np.int64,
            ),
            source_tokens=source_tokens,
            signatures=tuple(profile.signature for profile in profiles),
            signature_sum=np.sum(
                [profile.signature for profile in profiles],
                axis=0,
            ),
            iterations=tuple(request.iteration for request in requests),
        )

    def _estimate_histogram(self, histogram: _BatchHistogram) -> BatchEstimate:
        bytes_per_activation = self.model.hidden_size * self.model.bytes_per_element
        bandwidth_bytes_per_ms = self.network.bandwidth_gb_per_s * 1_000_000.0
        source_tokens = histogram.source_tokens
        attention_ms = self.compute.attention_base_ms + (
            int(source_tokens.max(initial=0)) * self.compute.attention_token_ms
        )
        expert_counts = histogram.expert_counts
        active_experts = expert_counts > 0
        utilization = np.clip(
            expert_counts / self.compute.expert_saturation_tokens,
            0.05,
            1.0,
        )
        token_compute_ms = expert_counts / (
            self.compute.expert_peak_tokens_per_ms * utilization
        )
        weight_read_ms = self.compute.expert_weight_bytes / (
            self.compute.expert_memory_bandwidth_gb_per_s * 1_000_000.0
        )
        if self.calibration is not None and self.calibration.expert_kernel_curve is not None:
            expert_elapsed_ms = self.calibration.expert_kernel_curve.latencies_ms(expert_counts)
        else:
            expert_elapsed_ms = np.where(
                active_experts,
                self.compute.expert_launch_ms + np.maximum(token_compute_ms, weight_read_ms),
                0.0,
            )
        rank_expert_tokens = np.einsum(
            "le,leg->lg",
            expert_counts,
            self.placement_one_hot,
        )
        rank_expert_ms = np.einsum(
            "le,leg->lg",
            expert_elapsed_ms,
            self.placement_one_hot,
        )

        pair_counts = histogram.pair_counts
        nonzero_pairs = pair_counts > 0
        if self.network.aggregate_messages:
            one_way_messages = nonzero_pairs.sum(axis=(1, 2))
            rank_messages = 2 * (
                nonzero_pairs.sum(axis=2) + nonzero_pairs.sum(axis=1)
            )
        else:
            one_way_messages = pair_counts.sum(axis=(1, 2))
            rank_messages = 2 * (pair_counts.sum(axis=2) + pair_counts.sum(axis=1))
        endpoint_tokens = pair_counts.sum(axis=2) + pair_counts.sum(axis=1)
        rank_bytes = endpoint_tokens * bytes_per_activation * 2
        mean_endpoint_bytes = rank_bytes.mean(axis=1)
        byte_ratio = np.divide(
            rank_bytes,
            mean_endpoint_bytes[:, None],
            out=np.zeros_like(rank_bytes, dtype=float),
            where=mean_endpoint_bytes[:, None] > 0.0,
        )
        pressure = np.maximum(0.0, byte_ratio - 1.0)
        rank_bandwidth_ms = rank_bytes / bandwidth_bytes_per_ms
        if self.calibration is not None and self.calibration.network_curves:
            rank_communication_ms = self.calibration.network_latencies_ms(
                collective="ep_dispatch_combine",
                active_ranks=self.model.num_gpus,
                message_counts=rank_messages,
                transferred_bytes=rank_bytes,
            )
        else:
            rank_communication_ms = (
                rank_messages * self.network.fixed_message_latency_ms
                + rank_bandwidth_ms * (1.0 + self.network.congestion_factor * pressure)
            )
        critical_path_ms = float(
            np.max(attention_ms + rank_expert_ms + rank_communication_ms, axis=1).sum()
        )
        mean_tokens = rank_expert_tokens.mean(axis=1)
        load_cvs = np.divide(
            rank_expert_tokens.std(axis=1),
            mean_tokens,
            out=np.zeros(self.model.num_layers, dtype=float),
            where=mean_tokens > 0.0,
        )

        kv_mean = float(source_tokens.mean())
        estimate = BatchEstimate(
            critical_path_ms=critical_path_ms,
            mean_rank_expert_load_cv=float(load_cvs.mean()),
            locality=_mean_pairwise_similarity(list(histogram.signatures)),
            kv_rank_load_cv=(
                float(source_tokens.std() / kv_mean) if kv_mean > 0.0 else 0.0
            ),
            progress_span=(
                max(histogram.iterations) - min(histogram.iterations)
            ),
            unique_expert_invocations=int(active_experts.sum()),
            network_messages=int((one_way_messages * 2).sum()),
        )
        return estimate

    def _profile(
        self,
        request: Request,
        workload: DecodeWorkload,
        router: RouterLike,
        knowledge: RouteKnowledge,
    ) -> _RequestRouteProfile:
        key = (knowledge, request.request_id, request.iteration, request.kv_location)
        if key in self._profile_cache:
            return self._profile_cache[key]
        predicted_router = _PredictedRouter(router, workload, [request], knowledge)
        work_items = workload.work_items(request)
        expert_counts = np.zeros(
            (self.model.num_layers, self.model.num_experts),
            dtype=np.int64,
        )
        if knowledge == "previous" and request.iteration == 0:
            for layer_id in range(self.model.num_layers):
                for expert_id in router.prior_route(request.request_id, layer_id):
                    expert_counts[layer_id, expert_id] = len(work_items)
        else:
            for layer_id in range(self.model.num_layers):
                for item in work_items:
                    for expert_id in predicted_router.route(
                        item.request_id,
                        item.iteration,
                        layer_id,
                        item.position_id,
                    ):
                        expert_counts[layer_id, expert_id] += 1
        profile = self._make_profile(
            request,
            expert_counts,
            positions=len(work_items),
        )
        self._profile_cache[key] = profile
        return profile

    def observe_previous_route(
        self,
        request: Request,
        observed_routes: np.ndarray,
        workload: DecodeWorkload,
    ) -> None:
        if request.iteration <= 0 or request.done:
            return
        next_positions = workload.active_positions(request)
        previous_positions = observed_routes.shape[1]
        if next_positions <= previous_positions:
            # Native refinement schedules normally shrink their active set.  In that
            # common path, the next prediction is simply the observed prefix and does
            # not need an index vector or advanced-indexing copy.
            expert_counts = observed_routes[:, :next_positions, :].sum(axis=1)
        else:
            mapped_positions = np.arange(next_positions, dtype=np.int64) % previous_positions
            expert_counts = observed_routes[:, mapped_positions, :].sum(axis=1)
        key = ("previous", request.request_id, request.iteration, request.kv_location)
        self._profile_cache[key] = self._make_profile(
            request,
            expert_counts,
            positions=next_positions,
        )

    def _make_profile(
        self,
        request: Request,
        expert_counts: np.ndarray,
        *,
        positions: int,
    ) -> _RequestRouteProfile:
        pair_counts = np.zeros(
            (self.model.num_layers, self.model.num_gpus, self.model.num_gpus),
            dtype=np.int64,
        )
        rank_counts = np.einsum(
            "le,leg->lg",
            expert_counts,
            self.placement_one_hot,
        )
        pair_counts[:, request.kv_location, :] = rank_counts
        pair_counts[:, request.kv_location, request.kv_location] = 0
        return _RequestRouteProfile(
            expert_counts=expert_counts,
            pair_counts=pair_counts,
            signature=expert_counts.sum(axis=0, dtype=np.int64).astype(float),
            positions=positions,
            source_gpu=request.kv_location,
        )


class CostAwareScheduler(Scheduler):
    """Greedily add the request with the lowest estimated batch cost."""

    def __init__(
        self,
        *,
        objective: CostObjective,
        route_knowledge: RouteKnowledge,
        model: ModelConfig,
        compute: ComputeConfig,
        network: NetworkConfig,
        scheduler: SchedulerConfig,
        calibration: CalibrationArtifact | None = None,
        placement: ExpertPlacement | None = None,
    ) -> None:
        super().__init__(
            base_overhead_ms=scheduler.base_overhead_ms,
            candidate_evaluation_overhead_ms=(
                scheduler.candidate_evaluation_overhead_ms
            ),
            proxy_evaluation_overhead_ms=scheduler.proxy_evaluation_overhead_ms,
        )
        self.objective = objective
        self.route_knowledge = route_knowledge
        self.config = scheduler
        self.network = network
        self.estimator = BatchCostEstimator(
            model,
            compute,
            network,
            calibration,
            placement,
        )

    def prepare_requests(
        self,
        requests: list[Request],
        router: RouterLike,
        workload: DecodeWorkload,
        observed_routes: dict[int, np.ndarray] | None = None,
    ) -> None:
        if self.route_knowledge != "previous":
            return
        if self.config.one_shot_proxy_batch and not self._proxy_is_active():
            return
        for request in requests:
            if not request.done:
                if observed_routes is not None and request.request_id in observed_routes:
                    self.estimator.observe_previous_route(
                        request,
                        observed_routes[request.request_id],
                        workload,
                    )
                else:
                    self.estimator.profile(
                        request,
                        workload,
                        router,
                        self.route_knowledge,
                    )

    def select(
        self,
        ready: list[Request],
        max_batch_size: int,
        now_ms: float,
        router: RouterLike,
        workload: DecodeWorkload,
    ) -> list[Request]:
        if not ready:
            self.record_step_overhead(0)
            return []
        if self.config.one_shot_proxy_batch:
            if not self._proxy_is_active():
                self.record_step_overhead(0)
                return sorted(
                    ready,
                    key=lambda request: (request.ready_since_ms, request.request_id),
                )[:max_batch_size]
            return self._select_one_shot_proxy(
                ready,
                max_batch_size,
                now_ms,
                router,
                workload,
            )
        remaining = sorted(ready, key=lambda req: (req.ready_since_ms, req.request_id))
        selected = [remaining.pop(0)]
        candidate_evaluations = 0
        proxy_evaluations = 0
        histogram: _BatchHistogram | None = None

        while remaining and len(selected) < max_batch_size:
            oldest = remaining[0]
            if now_ms - oldest.ready_since_ms >= self.config.max_wait_ms:
                choice = oldest
            else:
                scored: list[tuple[float, float, int, Request]] = []
                candidates = (
                    remaining[: self.config.candidate_pool_size]
                    if self.config.candidate_pool_size is not None
                    else remaining
                )
                if len(candidates) == 1:
                    choice = candidates[0]
                else:
                    if histogram is None:
                        histogram = self.estimator.histogram(
                            selected,
                            workload,
                            router,
                            self.route_knowledge,
                        )
                    profiles = {
                        candidate.request_id: self.estimator.profile(
                            candidate,
                            workload,
                            router,
                            self.route_knowledge,
                        )
                        for candidate in candidates
                    }
                    shortlist_size = self.config.full_evaluation_shortlist_size
                    exact_candidates = candidates
                    if shortlist_size is not None and shortlist_size < len(candidates):
                        proxy_evaluations += len(candidates)
                        proxy_scores = self.estimator.cheap_extension_scores(
                            histogram,
                            [profiles[candidate.request_id] for candidate in candidates],
                            [candidate.iteration for candidate in candidates],
                            self.config,
                        )
                        scores_by_request = {
                            candidate.request_id: float(score)
                            for candidate, score in zip(
                                candidates,
                                proxy_scores,
                                strict=True,
                            )
                        }
                        exact_candidates = sorted(
                            candidates,
                            key=lambda candidate: (
                                scores_by_request[candidate.request_id],
                                candidate.ready_since_ms,
                                candidate.request_id,
                            ),
                        )[:shortlist_size]
                    if len(exact_candidates) == 1:
                        choice = exact_candidates[0]
                    else:
                        for candidate in exact_candidates:
                            candidate_evaluations += 1
                            estimate = self.estimator.estimate_extension(
                                histogram,
                                profiles[candidate.request_id],
                                candidate.iteration,
                            )
                            score = self._score(
                                estimate,
                                candidate,
                                remaining,
                                now_ms,
                            )
                            scored.append(
                                (
                                    score,
                                    candidate.ready_since_ms,
                                    candidate.request_id,
                                    candidate,
                                )
                            )
                        choice = min(scored, key=lambda item: item[:3])[3]
            remaining.remove(choice)
            selected.append(choice)
            if histogram is not None:
                choice_profile = self.estimator.profile(
                    choice,
                    workload,
                    router,
                    self.route_knowledge,
                )
                histogram = self.estimator.extend(
                    histogram,
                    choice_profile,
                    choice.iteration,
                )

        self.record_step_overhead(candidate_evaluations, proxy_evaluations)
        return selected

    def _select_one_shot_proxy(
        self,
        ready: list[Request],
        max_batch_size: int,
        now_ms: float,
        router: RouterLike,
        workload: DecodeWorkload,
    ) -> list[Request]:
        remaining = sorted(ready, key=lambda req: (req.ready_since_ms, req.request_id))
        selected = [remaining.pop(0)]
        slots = max_batch_size - 1
        overdue = [
            request
            for request in remaining
            if now_ms - request.ready_since_ms >= self.config.max_wait_ms
        ][:slots]
        for request in overdue:
            remaining.remove(request)
        selected.extend(overdue)
        slots -= len(overdue)
        proxy_evaluations = 0

        if slots > 0 and remaining:
            candidates = (
                remaining[: self.config.candidate_pool_size]
                if self.config.candidate_pool_size is not None
                else remaining
            )
            if len(candidates) <= slots:
                selected.extend(candidates)
            else:
                histogram = self.estimator.histogram(
                    selected,
                    workload,
                    router,
                    self.route_knowledge,
                )
                profiles = [
                    self.estimator.profile(
                        candidate,
                        workload,
                        router,
                        self.route_knowledge,
                    )
                    for candidate in candidates
                ]
                proxy_scores = self.estimator.cheap_extension_scores(
                    histogram,
                    profiles,
                    [candidate.iteration for candidate in candidates],
                    self.config,
                )
                proxy_evaluations = len(candidates)
                ranked = sorted(
                    zip(candidates, proxy_scores, strict=True),
                    key=lambda item: (
                        float(item[1]),
                        item[0].ready_since_ms,
                        item[0].request_id,
                    ),
                )
                selected.extend(candidate for candidate, _ in ranked[:slots])

        self.record_step_overhead(0, proxy_evaluations)
        return selected

    def _proxy_is_active(self) -> bool:
        if self.config.proxy_activation_mode == "always":
            return True
        return (
            self.network.bandwidth_gb_per_s
            <= self.config.proxy_bandwidth_threshold_gb_per_s
            or self.network.fixed_message_latency_ms
            >= self.config.proxy_latency_threshold_ms
        )

    def _score(
        self,
        estimate: BatchEstimate,
        candidate: Request,
        remaining: list[Request],
        now_ms: float,
    ) -> float:
        if self.objective == "load_balance_only":
            return estimate.mean_rank_expert_load_cv
        if self.objective == "critical_path_only":
            return estimate.critical_path_ms
        if self.objective == "locality_plus_load":
            return (
                estimate.mean_rank_expert_load_cv
                + self.config.locality_weight * (1.0 - estimate.locality)
            )

        unselected = [request for request in remaining if request is not candidate]
        oldest_wait_ratio = max(
            (
                max(0.0, now_ms - request.ready_since_ms) / self.config.max_wait_ms
                for request in unselected
            ),
            default=0.0,
        )
        return (
            estimate.critical_path_ms
            + self.config.deadline_weight_ms * oldest_wait_ratio
            + self.config.kv_fragmentation_weight_ms * estimate.kv_rank_load_cv
            + self.config.progress_fragmentation_weight_ms * estimate.progress_span
            - self.config.locality_benefit_ms * estimate.locality
        )


def _mean_pairwise_similarity(signatures: list[np.ndarray]) -> float:
    if len(signatures) < 2:
        return 1.0
    similarities = [
        cosine_similarity(signatures[left], signatures[right])
        for left in range(len(signatures))
        for right in range(left + 1, len(signatures))
    ]
    return float(np.mean(similarities))
