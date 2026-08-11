from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from itertools import combinations

from .calibration import CalibrationArtifact
from .config import ExecutionMode, SimulationConfig
from .cost import LayerCostModel
from .models import Request
from .router import make_router
from .workloads import AutoregressiveWorkload, BlockRefinementWorkload, make_workload
from .workloads.base import DecodeWorkload

ProgressState = tuple[int, ...]
BatchAction = tuple[int, ...]


@dataclass(frozen=True)
class GlobalOracleResult:
    """Exact offline makespan optimum for a deliberately small workload."""

    mode: ExecutionMode
    optimal_makespan_ms: float
    batch_count: int
    explored_states: int
    actions: tuple[BatchAction, ...]


class ExactGlobalMakespanOracle:
    """Dijkstra search over all legal batch sequences.

    The search is exponential and therefore intentionally restricted to small
    validation configurations. It optimizes deterministic execution makespan, not
    latency percentiles or online decision overhead.
    """

    def __init__(
        self,
        config: SimulationConfig,
        mode: ExecutionMode,
        *,
        max_states: int = 200_000,
    ) -> None:
        config.validate()
        self.config = config
        self.mode = mode
        self.max_states = max_states
        self.workload = make_workload(config.workload, mode, config.model.num_gpus)
        self.router = make_router(config.model, config.router, config.seed)
        expected_checksum = (
            getattr(self.router, "bundle_sha256", None)
            if config.calibration.require_trace_checksum_match
            else None
        )
        calibration = (
            CalibrationArtifact.load(
                config.calibration.artifact_path,
                expected_bundle_sha256=expected_checksum,
            )
            if config.calibration.artifact_path is not None
            else None
        )
        if calibration is not None and (
            config.calibration.use_expert_kernel_curve
            or config.calibration.use_network_curves
        ):
            calibration = replace(
                calibration,
                expert_kernel_curve=(
                    calibration.expert_kernel_curve
                    if config.calibration.use_expert_kernel_curve
                    else None
                ),
                network_curves=(
                    calibration.network_curves
                    if config.calibration.use_network_curves
                    else ()
                ),
            )
        else:
            calibration = None
        self.cost = LayerCostModel(
            config.model,
            config.compute,
            config.network,
            calibration,
            self.router.placement,
        )
        self.total_steps = tuple(
            self._total_steps(self.workload, request) for request in self.workload.requests
        )
        self.goal: ProgressState = self.total_steps
        self._batch_cost_cache: dict[tuple[ProgressState, BatchAction], float] = {}

    def solve(self) -> GlobalOracleResult:
        initial: ProgressState = tuple(0 for _ in self.workload.requests)
        distances: dict[ProgressState, float] = {initial: 0.0}
        predecessors: dict[ProgressState, tuple[ProgressState, BatchAction]] = {}
        frontier: list[tuple[float, ProgressState]] = [(0.0, initial)]
        explored_states = 0

        while frontier:
            now_ms, state = heapq.heappop(frontier)
            if now_ms > distances[state] + 1e-12:
                continue
            explored_states += 1
            if explored_states > self.max_states:
                raise RuntimeError(
                    f"exact oracle exceeded max_states={self.max_states}; "
                    "use a smaller validation workload"
                )
            if state == self.goal:
                actions = self._reconstruct(predecessors, state)
                replayed = self.replay(actions)
                if abs(replayed - now_ms) > 1e-9:
                    raise RuntimeError(
                        f"oracle replay mismatch: search={now_ms}, replay={replayed}"
                    )
                return GlobalOracleResult(
                    mode=self.mode,
                    optimal_makespan_ms=now_ms,
                    batch_count=len(actions),
                    explored_states=explored_states,
                    actions=actions,
                )

            decision_time = self._next_decision_time(state, now_ms)
            for action in self._actions(state, decision_time):
                next_state = tuple(
                    progress + (1 if request_id in action else 0)
                    for request_id, progress in enumerate(state)
                )
                next_time = decision_time + self._batch_cost(state, action)
                if next_time + 1e-12 < distances.get(next_state, float("inf")):
                    distances[next_state] = next_time
                    predecessors[next_state] = (state, action)
                    heapq.heappush(frontier, (next_time, next_state))

        raise RuntimeError("exact oracle could not reach the completed state")

    def replay(self, actions: tuple[BatchAction, ...]) -> float:
        state: ProgressState = tuple(0 for _ in self.workload.requests)
        now_ms = 0.0
        for action in actions:
            decision_time = self._next_decision_time(state, now_ms)
            legal_actions = set(self._actions(state, decision_time))
            if action not in legal_actions:
                raise ValueError(f"illegal oracle replay action {action} at state {state}")
            now_ms = decision_time + self._batch_cost(state, action)
            state = tuple(
                progress + (1 if request_id in action else 0)
                for request_id, progress in enumerate(state)
            )
        if state != self.goal:
            raise ValueError(f"oracle replay ended at incomplete state {state}")
        return now_ms

    def _actions(self, state: ProgressState, now_ms: float) -> list[BatchAction]:
        ready = [
            request_id
            for request_id, progress in enumerate(state)
            if progress < self.total_steps[request_id]
            and self.workload.requests[request_id].arrival_time_ms <= now_ms
        ]
        max_size = min(self.config.workload.max_batch_size, len(ready))
        return [
            action
            for size in range(1, max_size + 1)
            for action in combinations(ready, size)
        ]

    def _next_decision_time(self, state: ProgressState, now_ms: float) -> float:
        if any(
            progress < self.total_steps[request_id]
            and self.workload.requests[request_id].arrival_time_ms <= now_ms
            for request_id, progress in enumerate(state)
        ):
            return now_ms
        next_arrival = min(
            request.arrival_time_ms
            for request, progress in zip(self.workload.requests, state, strict=True)
            if progress < self.total_steps[request.request_id]
        )
        return max(now_ms, next_arrival)

    def _batch_cost(self, state: ProgressState, action: BatchAction) -> float:
        key = (state, action)
        if key in self._batch_cost_cache:
            return self._batch_cost_cache[key]
        requests = [self._request_at(request_id, state[request_id]) for request_id in action]
        work_items = self.workload.ready_work_items(requests)
        elapsed_ms = self.config.scheduler.base_overhead_ms
        for layer_id in range(self.config.model.num_layers):
            elapsed_ms += self.cost.execute_layer(
                requests,
                work_items,
                layer_id,
                self.router,
            ).elapsed_ms
        self._batch_cost_cache[key] = elapsed_ms
        return elapsed_ms

    def _request_at(self, request_id: int, progress: int) -> Request:
        template = self.workload.requests[request_id]
        if isinstance(self.workload, AutoregressiveWorkload):
            finalized_tokens = progress
        elif isinstance(self.workload, BlockRefinementWorkload):
            rounds = len(self.config.workload.active_position_schedule)
            finalized_tokens = (
                progress // rounds * self.config.workload.diffusion_block_size
            )
        else:
            raise TypeError(f"unsupported exact-oracle workload: {type(self.workload).__name__}")
        return Request(
            request_id=request_id,
            arrival_time_ms=template.arrival_time_ms,
            output_tokens=template.output_tokens,
            kv_location=template.kv_location,
            iteration=progress,
            finalized_tokens=min(finalized_tokens, template.output_tokens),
            ready_since_ms=0.0,
        )

    def _total_steps(self, workload: DecodeWorkload, request: Request) -> int:
        if isinstance(workload, AutoregressiveWorkload):
            return request.output_tokens
        if isinstance(workload, BlockRefinementWorkload):
            blocks = request.output_tokens // self.config.workload.diffusion_block_size
            return blocks * len(self.config.workload.active_position_schedule)
        raise TypeError(f"unsupported exact-oracle workload: {type(workload).__name__}")

    @staticmethod
    def _reconstruct(
        predecessors: dict[ProgressState, tuple[ProgressState, BatchAction]],
        state: ProgressState,
    ) -> tuple[BatchAction, ...]:
        reversed_actions: list[BatchAction] = []
        while state in predecessors:
            state, action = predecessors[state]
            reversed_actions.append(action)
        return tuple(reversed(reversed_actions))
