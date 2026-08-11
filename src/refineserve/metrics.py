from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExecutionMode, SimulationConfig
from .models import (
    BatchExecution,
    BatchRequestExecution,
    RankExecution,
    Request,
    RuntimeDiagnostics,
)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


@dataclass(frozen=True)
class RunSummary:
    mode: ExecutionMode
    scheduler: str
    seed: int
    makespan_ms: float
    completed_requests: int
    finalized_tokens: int
    processed_positions: int
    useful_work_ratio: float
    batch_count: int
    mean_requests_per_batch: float
    underfilled_batch_ratio: float
    requests_per_second: float
    finalized_tokens_per_second: float
    processed_positions_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    mean_queue_delay_ms: float
    mean_expert_batch_size: float
    expert_batch_p50: float
    expert_batch_p95: float
    expert_invocations: int
    kernel_launch_count: int
    total_expert_token_executions: int
    mean_unique_experts_per_layer: float
    mean_unique_experts_per_rank_layer: float
    expert_weight_bytes_read: int
    expert_weight_read_work_ms: float
    expert_utilization: float
    load_imbalance_cv: float
    modeled_rank_critical_path_ms: float
    modeled_rank_critical_path_fraction: float
    mean_rank_layer_load_cv: float
    max_straggler_gpu_share: float
    runtime_uses_rank_critical_path: bool
    all_to_all_calls: int
    network_messages: int
    transferred_bytes: int
    communication_time_ms: float
    communication_time_fraction: float
    routing_stability: float
    expert_locality_hit_rate: float
    scheduler_modeled_overhead_ms: float
    scheduler_modeled_overhead_fraction: float
    scheduler_candidate_evaluations: int
    scheduler_proxy_evaluations: int
    trace_bundle_sha256: str | None
    calibration_source_bundle_sha256: str | None


@dataclass(frozen=True)
class RunResult:
    config: SimulationConfig
    summary: RunSummary
    requests: list[Request]
    expert_batch_sizes: list[int]
    rank_executions: list[RankExecution]
    batch_executions: list[BatchExecution]
    batch_requests: list[BatchRequestExecution]
    runtime_diagnostics: RuntimeDiagnostics

    def write(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(self.summary)]).to_csv(output_dir / "summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "request_id": request.request_id,
                    "arrival_time_ms": request.arrival_time_ms,
                    "completion_time_ms": request.completion_time_ms,
                    "latency_ms": (request.completion_time_ms or 0.0) - request.arrival_time_ms,
                    "queue_delay_ms": request.total_queue_delay_ms,
                    "finalized_tokens": request.finalized_tokens,
                }
                for request in self.requests
            ]
        ).to_csv(output_dir / "request_latencies.csv", index=False)
        pd.DataFrame({"expert_batch_size": self.expert_batch_sizes}).to_csv(
            output_dir / "expert_batches.csv", index=False
        )
        pd.DataFrame([asdict(execution) for execution in self.rank_executions]).to_csv(
            output_dir / "rank_layers.csv", index=False
        )
        pd.DataFrame([asdict(execution) for execution in self.batch_executions]).to_csv(
            output_dir / "batches.csv", index=False
        )
        pd.DataFrame([asdict(execution) for execution in self.batch_requests]).to_csv(
            output_dir / "batch_requests.csv", index=False
        )
        metadata = {
            "config": asdict(self.config),
            "summary": asdict(self.summary),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "runtime_diagnostics.json").write_text(
            json.dumps(asdict(self.runtime_diagnostics), indent=2, sort_keys=True) + "\n"
        )
        return output_dir


def build_summary(
    *,
    config: SimulationConfig,
    mode: ExecutionMode,
    requests: list[Request],
    makespan_ms: float,
    processed_positions: int,
    expert_batch_sizes: list[int],
    expert_invocations: int,
    expert_busy_gpu_ms: float,
    expert_token_counts: np.ndarray,
    all_to_all_calls: int,
    network_messages: int,
    transferred_bytes: int,
    communication_time_ms: float,
    routing_stability_values: list[float],
    locality_values: list[float],
    rank_executions: list[RankExecution],
    batch_sizes: list[int],
    scheduler_overhead_ms: float,
    scheduler_candidate_evaluations: int,
    scheduler_proxy_evaluations: int,
    trace_bundle_sha256: str | None = None,
    calibration_source_bundle_sha256: str | None = None,
) -> RunSummary:
    finalized_tokens = sum(request.finalized_tokens for request in requests)
    completed = sum(request.done for request in requests)
    latencies = [
        (request.completion_time_ms or makespan_ms) - request.arrival_time_ms
        for request in requests
    ]
    queue_delays = [request.total_queue_delay_ms for request in requests]
    makespan_s = makespan_ms / 1_000.0
    expert_mean = float(np.mean(expert_batch_sizes)) if expert_batch_sizes else 0.0
    token_mean = float(expert_token_counts.mean())
    load_cv = float(expert_token_counts.std() / token_mean) if token_mean > 0.0 else 0.0
    layer_execution_count = len(rank_executions) / config.model.num_gpus
    rank_unique_experts = [execution.unique_experts for execution in rank_executions]
    expert_weight_bytes_read = sum(execution.expert_weight_bytes for execution in rank_executions)
    expert_weight_read_work_ms = sum(
        execution.expert_weight_read_ms for execution in rank_executions
    )
    critical_executions = [execution for execution in rank_executions if execution.is_critical]
    modeled_rank_critical_path_ms = sum(
        execution.layer_time_ms for execution in critical_executions
    )
    rank_layer_load_cvs: list[float] = []
    for start in range(0, len(rank_executions), config.model.num_gpus):
        layer_times = np.asarray(
            [
                execution.layer_time_ms
                for execution in rank_executions[start : start + config.model.num_gpus]
            ],
            dtype=float,
        )
        layer_mean = float(layer_times.mean())
        rank_layer_load_cvs.append(
            float(layer_times.std() / layer_mean) if layer_mean > 0.0 else 0.0
        )
    straggler_counts = np.bincount(
        [execution.gpu_id for execution in critical_executions],
        minlength=config.model.num_gpus,
    )
    max_straggler_gpu_share = (
        float(straggler_counts.max() / len(critical_executions)) if critical_executions else 0.0
    )
    return RunSummary(
        mode=mode,
        scheduler=config.scheduler.name,
        seed=config.seed,
        makespan_ms=makespan_ms,
        completed_requests=completed,
        finalized_tokens=finalized_tokens,
        processed_positions=processed_positions,
        useful_work_ratio=finalized_tokens / processed_positions,
        batch_count=len(batch_sizes),
        mean_requests_per_batch=(float(np.mean(batch_sizes)) if batch_sizes else 0.0),
        underfilled_batch_ratio=(
            sum(size < config.workload.max_batch_size for size in batch_sizes)
            / len(batch_sizes)
            if batch_sizes
            else 0.0
        ),
        requests_per_second=completed / makespan_s,
        finalized_tokens_per_second=finalized_tokens / makespan_s,
        processed_positions_per_second=processed_positions / makespan_s,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        mean_queue_delay_ms=float(np.mean(queue_delays)),
        mean_expert_batch_size=expert_mean,
        expert_batch_p50=_percentile([float(v) for v in expert_batch_sizes], 50),
        expert_batch_p95=_percentile([float(v) for v in expert_batch_sizes], 95),
        expert_invocations=expert_invocations,
        kernel_launch_count=expert_invocations,
        total_expert_token_executions=int(expert_token_counts.sum()),
        mean_unique_experts_per_layer=(
            expert_invocations / layer_execution_count if layer_execution_count > 0.0 else 0.0
        ),
        mean_unique_experts_per_rank_layer=(
            float(np.mean(rank_unique_experts)) if rank_unique_experts else 0.0
        ),
        expert_weight_bytes_read=expert_weight_bytes_read,
        expert_weight_read_work_ms=expert_weight_read_work_ms,
        expert_utilization=expert_busy_gpu_ms / (makespan_ms * config.model.num_gpus),
        load_imbalance_cv=load_cv,
        modeled_rank_critical_path_ms=modeled_rank_critical_path_ms,
        modeled_rank_critical_path_fraction=(modeled_rank_critical_path_ms / makespan_ms),
        mean_rank_layer_load_cv=(
            float(np.mean(rank_layer_load_cvs)) if rank_layer_load_cvs else 0.0
        ),
        max_straggler_gpu_share=max_straggler_gpu_share,
        runtime_uses_rank_critical_path=config.network.use_rank_critical_path,
        all_to_all_calls=all_to_all_calls,
        network_messages=network_messages,
        transferred_bytes=transferred_bytes,
        communication_time_ms=communication_time_ms,
        communication_time_fraction=communication_time_ms / makespan_ms,
        routing_stability=float(np.mean(routing_stability_values))
        if routing_stability_values
        else 0.0,
        expert_locality_hit_rate=float(np.mean(locality_values)) if locality_values else 0.0,
        scheduler_modeled_overhead_ms=scheduler_overhead_ms,
        scheduler_modeled_overhead_fraction=scheduler_overhead_ms / makespan_ms,
        scheduler_candidate_evaluations=scheduler_candidate_evaluations,
        scheduler_proxy_evaluations=scheduler_proxy_evaluations,
        trace_bundle_sha256=trace_bundle_sha256,
        calibration_source_bundle_sha256=calibration_source_bundle_sha256,
    )
