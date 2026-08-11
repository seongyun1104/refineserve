from __future__ import annotations

import pytest

from refineserve.calibration import (
    CalibrationArtifact,
    MonotoneLatencyCurve,
    NetworkCurve,
)
from refineserve.config import (
    ComputeConfig,
    ModelConfig,
    NetworkConfig,
    RouterConfig,
    WorkloadConfig,
)
from refineserve.cost import LayerCostModel
from refineserve.placement import ExpertPlacement
from refineserve.router import SyntheticRouter
from refineserve.workloads import make_workload


class CrossRankRouter:
    def route(
        self,
        request_id: int,
        iteration: int,
        layer_id: int,
        position_id: int,
    ) -> tuple[int, ...]:
        del request_id, iteration, layer_id, position_id
        return (1,)


def test_layer_preserves_top_k_assignment_count() -> None:
    model = ModelConfig(num_layers=2, num_experts=8, top_k=2, num_gpus=2)
    workload = WorkloadConfig(
        num_requests=4,
        output_tokens=4,
        max_batch_size=4,
        diffusion_block_size=2,
        active_position_schedule=(2, 1),
    )
    execution_workload = make_workload(workload, "diffusion", model.num_gpus)
    requests = execution_workload.requests
    work_items = execution_workload.ready_work_items(requests)
    router = SyntheticRouter(model, RouterConfig(), seed=9)
    cost = LayerCostModel(
        model,
        ComputeConfig(
            expert_weight_bytes=1_024,
            expert_memory_bandwidth_gb_per_s=1.0,
        ),
        NetworkConfig(use_rank_critical_path=True),
    )

    execution = cost.execute_layer(requests, work_items, layer_id=0, router=router)

    assert execution.assignments == 4 * 2 * model.top_k
    assert sum(execution.expert_batch_sizes) == execution.assignments
    assert sum(execution.expert_token_counts) == execution.assignments
    assert execution.elapsed_ms > 0.0
    assert len(execution.rank_executions) == model.num_gpus
    assert sum(rank.unique_experts for rank in execution.rank_executions) == (
        execution.expert_invocations
    )
    assert sum(rank.expert_weight_bytes for rank in execution.rank_executions) == (
        execution.expert_invocations * 1_024
    )
    assert sum(rank.is_critical for rank in execution.rank_executions) == 1
    assert execution.rank_critical_path_ms == max(
        rank.layer_time_ms for rank in execution.rank_executions
    )
    assert execution.elapsed_ms == execution.rank_critical_path_ms


def test_rank_local_network_curve_drives_layer_critical_path() -> None:
    model = ModelConfig(
        num_layers=1,
        num_experts=2,
        top_k=1,
        num_gpus=2,
        hidden_size=64,
    )
    workload = make_workload(
        WorkloadConfig(
            num_requests=1,
            output_tokens=1,
            max_batch_size=1,
            diffusion_block_size=1,
            active_position_schedule=(1,),
        ),
        "autoregressive",
        model.num_gpus,
    )
    byte_curve = MonotoneLatencyCurve.fit(
        [
            {"transferred_bytes": "1", "latency_ms": "0.5", "warmup": "0"},
            {"transferred_bytes": "1024", "latency_ms": "0.5", "warmup": "0"},
        ],
        input_name="transferred_bytes",
    )
    calibration = CalibrationArtifact(
        schema_version=1,
        source_bundle_sha256="b" * 64,
        expert_kernel_curve=None,
        network_curves=(
            NetworkCurve(
                collective="ep_dispatch_combine",
                active_ranks=2,
                message_count=2,
                latency_by_bytes=byte_curve,
            ),
        ),
    )
    cost = LayerCostModel(
        model,
        ComputeConfig(),
        NetworkConfig(use_rank_critical_path=True),
        calibration,
    )

    execution = cost.execute_layer(
        workload.requests,
        workload.ready_work_items(workload.requests),
        layer_id=0,
        router=CrossRankRouter(),
    )

    assert execution.aggregate_communication_ms == pytest.approx(0.5)
    assert [rank.communication_ms for rank in execution.rank_executions] == pytest.approx(
        [0.5, 0.5]
    )
    assert execution.communication_ms == pytest.approx(0.5)


def test_layer_execution_uses_explicit_expert_placement() -> None:
    model = ModelConfig(
        num_layers=1,
        num_experts=2,
        top_k=1,
        num_gpus=2,
        hidden_size=64,
    )
    workload = make_workload(
        WorkloadConfig(
            num_requests=1,
            output_tokens=1,
            max_batch_size=1,
            diffusion_block_size=1,
            active_position_schedule=(1,),
        ),
        "autoregressive",
        model.num_gpus,
    )

    local = LayerCostModel(model, ComputeConfig(), NetworkConfig()).execute_layer(
        workload.requests,
        workload.ready_work_items(workload.requests),
        layer_id=0,
        router=type("LocalRouter", (), {"route": lambda *args: (0,)})(),
    )
    remote = LayerCostModel(
        model,
        ComputeConfig(),
        NetworkConfig(),
        placement=ExpertPlacement.from_rows([[1, 0]], model),
    ).execute_layer(
        workload.requests,
        workload.ready_work_items(workload.requests),
        layer_id=0,
        router=type("LocalRouter", (), {"route": lambda *args: (0,)})(),
    )

    assert local.transferred_bytes == 0
    assert remote.transferred_bytes > 0
