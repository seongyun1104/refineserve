from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from .calibration import CalibrationArtifact
from .config import ExecutionMode, SimulationConfig
from .cost import LayerCostModel
from .metrics import RunResult, build_summary
from .models import (
    BatchExecution,
    BatchRequestExecution,
    RankExecution,
    RuntimeDiagnostics,
)
from .router import make_router
from .schedulers import CostAwareScheduler, ExpertLocalityScheduler, FIFOScheduler, Scheduler
from .schedulers.expert_locality import mean_pairwise_similarity
from .workloads import make_workload


class Simulator:
    def __init__(self, config: SimulationConfig, mode: ExecutionMode):
        config.validate()
        self.config = config
        self.mode = mode
        self.router = make_router(config.model, config.router, config.seed)
        expected_checksum = (
            getattr(self.router, "bundle_sha256", None)
            if config.calibration.require_trace_checksum_match
            else None
        )
        self.calibration = (
            CalibrationArtifact.load(
                config.calibration.artifact_path,
                expected_bundle_sha256=expected_checksum,
            )
            if config.calibration.artifact_path is not None
            else None
        )
        if (
            self.calibration is not None
            and config.calibration.use_expert_kernel_curve
            and self.calibration.expert_kernel_curve is None
        ):
            raise ValueError("calibration artifact does not contain an expert kernel curve")
        if (
            self.calibration is not None
            and config.calibration.use_network_curves
            and not self.calibration.network_curves
        ):
            raise ValueError("calibration artifact does not contain network curves")
        active_calibration = (
            replace(
                self.calibration,
                expert_kernel_curve=(
                    self.calibration.expert_kernel_curve
                    if config.calibration.use_expert_kernel_curve
                    else None
                ),
                network_curves=(
                    self.calibration.network_curves
                    if config.calibration.use_network_curves
                    else ()
                ),
            )
            if self.calibration is not None
            and (
                config.calibration.use_expert_kernel_curve
                or config.calibration.use_network_curves
            )
            else None
        )
        self.cost_model = LayerCostModel(
            config.model,
            config.compute,
            config.network,
            active_calibration,
            self.router.placement,
        )
        self.active_calibration = active_calibration
        self.scheduler = self._make_scheduler()

    def _make_scheduler(self) -> Scheduler:
        name = self.config.scheduler.name
        overhead = {
            "base_overhead_ms": self.config.scheduler.base_overhead_ms,
            "candidate_evaluation_overhead_ms": (
                self.config.scheduler.candidate_evaluation_overhead_ms
            ),
            "proxy_evaluation_overhead_ms": (
                self.config.scheduler.proxy_evaluation_overhead_ms
            ),
        }
        if name == "fifo":
            return FIFOScheduler(**overhead)
        if name in {"previous_route", "locality_only"}:
            return ExpertLocalityScheduler(
                oracle=False,
                max_wait_ms=self.config.scheduler.max_wait_ms,
                candidate_pool_size=self.config.scheduler.candidate_pool_size,
                **overhead,
            )
        if name == "oracle":
            return ExpertLocalityScheduler(
                oracle=True,
                max_wait_ms=self.config.scheduler.max_wait_ms,
                candidate_pool_size=self.config.scheduler.candidate_pool_size,
                **overhead,
            )
        objectives = {
            "load_balance_only": ("load_balance_only", "previous"),
            "critical_path_only": ("critical_path_only", "previous"),
            "locality_plus_load": ("locality_plus_load", "previous"),
            "joint": ("joint", "previous"),
            "routing_oracle": ("joint", "routing_oracle"),
            "runtime_oracle": ("joint", "runtime_oracle"),
        }
        if name in objectives:
            objective, route_knowledge = objectives[name]
            return CostAwareScheduler(
                objective=objective,
                route_knowledge=route_knowledge,
                model=self.config.model,
                compute=self.config.compute,
                network=self.config.network,
                scheduler=self.config.scheduler,
                calibration=self.active_calibration,
                placement=self.router.placement,
            )
        raise ValueError(f"unsupported scheduler: {name}")

    def run(self) -> RunResult:
        simulator_wall_start = perf_counter()
        workload = make_workload(self.config.workload, self.mode, self.config.model.num_gpus)
        requests = workload.requests
        now_ms = 0.0
        processed_positions = 0
        expert_batch_sizes: list[int] = []
        expert_invocations = 0
        expert_busy_gpu_ms = 0.0
        expert_token_counts = np.zeros(self.config.model.num_experts, dtype=np.int64)
        all_to_all_calls = 0
        network_messages = 0
        transferred_bytes = 0
        communication_time_ms = 0.0
        routing_stability_values: list[float] = []
        locality_values: list[float] = []
        rank_executions: list[RankExecution] = []
        batch_sizes: list[int] = []
        batch_executions: list[BatchExecution] = []
        batch_requests: list[BatchRequestExecution] = []
        scheduler_overhead_ms = 0.0
        scheduler_selection_wall_time_ms = 0.0
        scheduler_selection_wall_times_ms: list[float] = []
        scheduler_selection_calls = 0
        scheduler_profile_update_wall_times_ms: list[float] = []
        loop_count = 0

        while any(not request.done for request in requests):
            loop_count += 1
            if loop_count > 10_000_000:
                raise RuntimeError("simulation exceeded loop safety limit")
            ready = workload.ready_requests(now_ms)
            if not ready:
                now_ms = workload.next_arrival_time()
                continue

            model_states_before = {
                request.request_id: workload.refinement_state(request) for request in ready
            }
            scheduler_wall_start = perf_counter()
            batch = self.scheduler.select(
                ready,
                self.config.workload.max_batch_size,
                now_ms,
                self.router,
                workload,
            )
            selection_wall_time_ms = (perf_counter() - scheduler_wall_start) * 1_000.0
            scheduler_selection_wall_time_ms += selection_wall_time_ms
            scheduler_selection_wall_times_ms.append(selection_wall_time_ms)
            scheduler_selection_calls += 1
            model_states_after = {
                request.request_id: workload.refinement_state(request) for request in ready
            }
            if model_states_after != model_states_before:
                raise RuntimeError(
                    "runtime scheduler altered model-owned refinement semantics"
                )
            if not batch or len({request.request_id for request in batch}) != len(batch):
                raise RuntimeError("scheduler returned an empty or duplicate batch")
            batch_sizes.append(len(batch))
            step_scheduler_overhead_ms = self.scheduler.consume_step_overhead()
            scheduler_overhead_ms += step_scheduler_overhead_ms
            now_ms += step_scheduler_overhead_ms
            batch_start_ms = now_ms
            waits = [now_ms - request.ready_since_ms for request in batch]
            iterations = [request.iteration for request in batch]
            for request in batch:
                request.total_queue_delay_ms += now_ms - request.ready_since_ms
                routing_stability_values.extend(self.router.routing_stability(request, workload))

            actual_signatures = [
                self.router.signature(request, workload, oracle=True) for request in batch
            ]
            locality_values.append(mean_pairwise_similarity(actual_signatures))
            work_items = workload.ready_work_items(batch)
            active_positions_by_request = {
                request.request_id: workload.active_positions(request) for request in batch
            }
            refinement_states = {
                request.request_id: workload.refinement_state(request) for request in batch
            }
            batch_requests.extend(
                BatchRequestExecution(
                    batch_id=loop_count - 1,
                    request_id=request.request_id,
                    iteration=request.iteration,
                    active_positions=active_positions_by_request[request.request_id],
                    block_width=refinement_states[request.request_id].block_width,
                    finalized_positions_per_step=refinement_states[
                        request.request_id
                    ].finalized_positions_per_step,
                    order_policy=refinement_states[request.request_id].order_policy,
                    kv_location=request.kv_location,
                    wait_ms=now_ms - request.ready_since_ms,
                )
                for request in batch
            )
            processed_positions += len(work_items)

            step_elapsed_ms = 0.0
            batch_expert_invocations = 0
            batch_network_messages = 0
            batch_transferred_bytes = 0
            batch_communication_ms = 0.0
            observed_routes = {
                request.request_id: np.zeros(
                    (
                        self.config.model.num_layers,
                        active_positions_by_request[request.request_id],
                        self.config.model.num_experts,
                    ),
                    dtype=np.int16,
                )
                for request in batch
            }
            for layer_id in range(self.config.model.num_layers):
                layer = self.cost_model.execute_layer(
                    batch,
                    work_items,
                    layer_id,
                    self.router,
                    batch_id=loop_count - 1,
                )
                expected = len(work_items) * self.config.model.top_k
                if layer.assignments != expected:
                    raise RuntimeError(
                        f"assignment conservation failed at layer {layer_id}: "
                        f"expected {expected}, got {layer.assignments}"
                    )
                step_elapsed_ms += layer.elapsed_ms
                expert_batch_sizes.extend(layer.expert_batch_sizes)
                expert_invocations += layer.expert_invocations
                batch_expert_invocations += layer.expert_invocations
                expert_busy_gpu_ms += sum(layer.per_gpu_expert_ms)
                expert_token_counts += np.asarray(layer.expert_token_counts)
                all_to_all_calls += layer.all_to_all_calls
                network_messages += layer.network_messages
                batch_network_messages += layer.network_messages
                transferred_bytes += layer.transferred_bytes
                batch_transferred_bytes += layer.transferred_bytes
                communication_time_ms += layer.communication_ms
                batch_communication_ms += layer.communication_ms
                rank_executions.extend(layer.rank_executions)
                for assignment in layer.token_assignments:
                    observed_routes[assignment.request_id][
                        layer_id,
                        assignment.position_id,
                        assignment.expert_id,
                    ] += 1

            now_ms += step_elapsed_ms
            finalized_before = sum(request.finalized_tokens for request in batch)
            workload.finalize(work_items, now_ms)
            profile_update_wall_start = perf_counter()
            self.scheduler.prepare_requests(
                batch,
                self.router,
                workload,
                observed_routes,
            )
            scheduler_profile_update_wall_times_ms.append(
                (perf_counter() - profile_update_wall_start) * 1_000.0
            )
            finalized_after = sum(request.finalized_tokens for request in batch)
            batch_executions.append(
                BatchExecution(
                    batch_id=loop_count - 1,
                    start_ms=batch_start_ms,
                    end_ms=now_ms,
                    scheduler_overhead_ms=step_scheduler_overhead_ms,
                    request_count=len(batch),
                    processed_positions=len(work_items),
                    finalized_tokens=finalized_after - finalized_before,
                    progress_span=max(iterations) - min(iterations),
                    mean_wait_ms=float(np.mean(waits)),
                    max_wait_ms=max(waits),
                    expert_invocations=batch_expert_invocations,
                    network_messages=batch_network_messages,
                    transferred_bytes=batch_transferred_bytes,
                    communication_ms=batch_communication_ms,
                    layer_execution_ms=step_elapsed_ms,
                    underfilled=len(batch) < self.config.workload.max_batch_size,
                )
            )

        summary = build_summary(
            config=self.config,
            mode=self.mode,
            requests=requests,
            makespan_ms=now_ms,
            processed_positions=processed_positions,
            expert_batch_sizes=expert_batch_sizes,
            expert_invocations=expert_invocations,
            expert_busy_gpu_ms=expert_busy_gpu_ms,
            expert_token_counts=expert_token_counts,
            all_to_all_calls=all_to_all_calls,
            network_messages=network_messages,
            transferred_bytes=transferred_bytes,
            communication_time_ms=communication_time_ms,
            routing_stability_values=routing_stability_values,
            locality_values=locality_values,
            rank_executions=rank_executions,
            batch_sizes=batch_sizes,
            scheduler_overhead_ms=scheduler_overhead_ms,
            scheduler_candidate_evaluations=(
                self.scheduler.total_candidate_evaluations
            ),
            scheduler_proxy_evaluations=self.scheduler.total_proxy_evaluations,
            trace_bundle_sha256=getattr(self.router, "bundle_sha256", None),
            calibration_source_bundle_sha256=(
                self.active_calibration.source_bundle_sha256
                if self.active_calibration is not None
                else None
            ),
        )
        return RunResult(
            config=self.config,
            summary=summary,
            requests=requests,
            expert_batch_sizes=expert_batch_sizes,
            rank_executions=rank_executions,
            batch_executions=batch_executions,
            batch_requests=batch_requests,
            runtime_diagnostics=RuntimeDiagnostics(
                simulator_wall_time_ms=(perf_counter() - simulator_wall_start) * 1_000.0,
                scheduler_selection_wall_time_ms=scheduler_selection_wall_time_ms,
                scheduler_selection_calls=scheduler_selection_calls,
                scheduler_selection_mean_ms=float(
                    np.mean(scheduler_selection_wall_times_ms)
                ),
                scheduler_selection_p50_ms=float(
                    np.percentile(scheduler_selection_wall_times_ms, 50)
                ),
                scheduler_selection_p95_ms=float(
                    np.percentile(scheduler_selection_wall_times_ms, 95)
                ),
                scheduler_selection_p99_ms=float(
                    np.percentile(scheduler_selection_wall_times_ms, 99)
                ),
                scheduler_selection_max_ms=max(scheduler_selection_wall_times_ms),
                scheduler_profile_update_wall_time_ms=sum(
                    scheduler_profile_update_wall_times_ms
                ),
                scheduler_profile_update_mean_ms=float(
                    np.mean(scheduler_profile_update_wall_times_ms)
                ),
                scheduler_profile_update_p95_ms=float(
                    np.percentile(scheduler_profile_update_wall_times_ms, 95)
                ),
                scheduler_profile_update_max_ms=max(
                    scheduler_profile_update_wall_times_ms
                ),
                scheduler_total_wall_time_ms=(
                    scheduler_selection_wall_time_ms
                    + sum(scheduler_profile_update_wall_times_ms)
                ),
            ),
        )
