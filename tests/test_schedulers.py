from __future__ import annotations

from dataclasses import asdict

import numpy as np
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
    SchedulerConfig,
    WorkloadConfig,
)
from refineserve.models import Request
from refineserve.router import SyntheticRouter
from refineserve.schedulers.cost_aware import (
    BatchCostEstimator,
    CostAwareScheduler,
    ReferenceBatchCostEstimator,
)
from refineserve.schedulers.expert_locality import ExpertLocalityScheduler
from refineserve.workloads import DecodeWorkload, make_workload


class SignatureRouter:
    def signature(self, request: Request, workload: DecodeWorkload, *, oracle: bool) -> np.ndarray:
        del oracle, workload
        return {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([1.0, 0.0]),
        }[request.request_id]


class RouteFixture:
    def route(
        self,
        request_id: int,
        iteration: int,
        layer_id: int,
        position_id: int,
    ) -> tuple[int, ...]:
        del iteration, layer_id, position_id
        return (0,) if request_id in {0, 1} else (1,)


def test_locality_scheduler_keeps_oldest_and_adds_similar_request() -> None:
    workload = WorkloadConfig(num_requests=3, output_tokens=32, max_batch_size=2)
    execution = make_workload(workload, "autoregressive", num_gpus=1)
    requests = execution.requests
    scheduler = ExpertLocalityScheduler(oracle=True, max_wait_ms=100.0)

    selected = scheduler.select(
        requests,
        2,
        now_ms=0.2,
        router=SignatureRouter(),
        workload=execution,
    )

    assert [request.request_id for request in selected] == [0, 2]


def test_locality_scheduler_honors_wait_bound() -> None:
    workload = WorkloadConfig(num_requests=3, output_tokens=32, max_batch_size=2)
    execution = make_workload(workload, "autoregressive", num_gpus=1)
    requests = execution.requests
    scheduler = ExpertLocalityScheduler(oracle=True, max_wait_ms=0.05)

    selected = scheduler.select(
        requests,
        2,
        now_ms=1.0,
        router=SignatureRouter(),
        workload=execution,
    )

    assert [request.request_id for request in selected] == [0, 1]


def test_critical_path_scheduler_avoids_locality_induced_rank_straggler() -> None:
    model = ModelConfig(num_layers=2, num_experts=2, top_k=1, num_gpus=2)
    workload_config = WorkloadConfig(
        num_requests=3,
        output_tokens=32,
        max_batch_size=2,
    )
    workload = make_workload(workload_config, "autoregressive", num_gpus=2)
    scheduler = CostAwareScheduler(
        objective="critical_path_only",
        route_knowledge="routing_oracle",
        model=model,
        compute=ComputeConfig(),
        network=NetworkConfig(fixed_message_latency_ms=0.0),
        scheduler=SchedulerConfig(max_wait_ms=100.0),
    )

    selected = scheduler.select(
        workload.requests,
        2,
        now_ms=0.2,
        router=RouteFixture(),
        workload=workload,
    )

    assert [request.request_id for request in selected] == [0, 2]
    assert scheduler.total_candidate_evaluations == 2


def test_candidate_pool_bounds_cost_evaluations() -> None:
    model = ModelConfig(num_layers=2, num_experts=2, top_k=1, num_gpus=2)
    workload = make_workload(
        WorkloadConfig(num_requests=3, output_tokens=32, max_batch_size=2),
        "autoregressive",
        num_gpus=2,
    )
    scheduler = CostAwareScheduler(
        objective="critical_path_only",
        route_knowledge="routing_oracle",
        model=model,
        compute=ComputeConfig(),
        network=NetworkConfig(fixed_message_latency_ms=0.0),
        scheduler=SchedulerConfig(max_wait_ms=100.0, candidate_pool_size=1),
    )

    selected = scheduler.select(
        workload.requests,
        2,
        now_ms=0.2,
        router=RouteFixture(),
        workload=workload,
    )

    assert [request.request_id for request in selected] == [0, 1]
    assert scheduler.total_candidate_evaluations == 0


@pytest.mark.parametrize("knowledge", ["previous", "routing_oracle"])
def test_histogram_estimator_matches_reference_layer_replay(knowledge: str) -> None:
    model = ModelConfig(
        num_layers=3,
        num_experts=8,
        top_k=2,
        num_gpus=2,
        hidden_size=128,
    )
    workload = make_workload(
        WorkloadConfig(
            num_requests=3,
            output_tokens=4,
            max_batch_size=3,
            diffusion_block_size=2,
            active_position_schedule=(2, 1),
        ),
        "diffusion",
        num_gpus=2,
    )
    workload.requests[0].iteration = 1
    workload.requests[1].iteration = 2
    router = SyntheticRouter(model, RouterConfig(), seed=29)
    compute = ComputeConfig(
        expert_weight_bytes=4_096,
        expert_memory_bandwidth_gb_per_s=10.0,
    )
    network = NetworkConfig(
        bandwidth_gb_per_s=25.0,
        fixed_message_latency_ms=0.02,
        use_rank_critical_path=True,
    )
    fast = BatchCostEstimator(model, compute, network).estimate(
        workload.requests,
        workload,
        router,
        knowledge,
    )
    reference = ReferenceBatchCostEstimator(model, compute, network).estimate(
        workload.requests,
        workload,
        router,
        knowledge,
    )

    assert asdict(fast) == pytest.approx(asdict(reference))


def test_incremental_histogram_matches_full_batch_estimate() -> None:
    model = ModelConfig(num_layers=2, num_experts=4, top_k=2, num_gpus=2)
    workload = make_workload(
        WorkloadConfig(
            num_requests=3,
            output_tokens=4,
            max_batch_size=3,
            diffusion_block_size=2,
            active_position_schedule=(2, 1),
        ),
        "diffusion",
        num_gpus=2,
    )
    router = SyntheticRouter(model, RouterConfig(), seed=31)
    estimator = BatchCostEstimator(model, ComputeConfig(), NetworkConfig())
    histogram = estimator.histogram(
        workload.requests[:2],
        workload,
        router,
        "previous",
    )
    candidate = workload.requests[2]
    profile = estimator.profile(candidate, workload, router, "previous")

    incremental = estimator.estimate_extension(histogram, profile, candidate.iteration)
    full = estimator.estimate(workload.requests, workload, router, "previous")

    assert asdict(incremental) == pytest.approx(asdict(full))


def test_calibrated_histogram_estimator_matches_reference_replay() -> None:
    model = ModelConfig(
        num_layers=2,
        num_experts=4,
        top_k=2,
        num_gpus=2,
        hidden_size=128,
    )
    workload = make_workload(
        WorkloadConfig(
            num_requests=3,
            output_tokens=4,
            max_batch_size=3,
            diffusion_block_size=2,
            active_position_schedule=(2, 1),
        ),
        "diffusion",
        num_gpus=2,
    )
    router = SyntheticRouter(model, RouterConfig(), seed=37)
    curve = MonotoneLatencyCurve.fit(
        [
            {"token_count": "1", "latency_ms": "0.1", "warmup": "0"},
            {"token_count": "64", "latency_ms": "0.8", "warmup": "0"},
        ],
        input_name="token_count",
    )
    byte_curve = MonotoneLatencyCurve.fit(
        [
            {"transferred_bytes": "1", "latency_ms": "0.02", "warmup": "0"},
            {"transferred_bytes": "1000000", "latency_ms": "1.0", "warmup": "0"},
        ],
        input_name="transferred_bytes",
    )
    calibration = CalibrationArtifact(
        schema_version=1,
        source_bundle_sha256="a" * 64,
        expert_kernel_curve=curve,
        network_curves=tuple(
            NetworkCurve(
                collective="ep_dispatch_combine",
                active_ranks=2,
                message_count=message_count,
                latency_by_bytes=byte_curve,
            )
            for message_count in (2, 4)
        ),
    )
    compute = ComputeConfig()
    network = NetworkConfig(use_rank_critical_path=True)

    fast = BatchCostEstimator(model, compute, network, calibration).estimate(
        workload.requests,
        workload,
        router,
        "previous",
    )
    reference = ReferenceBatchCostEstimator(model, compute, network, calibration).estimate(
        workload.requests,
        workload,
        router,
        "previous",
    )

    assert asdict(fast) == pytest.approx(asdict(reference))


def test_proxy_shortlist_reduces_full_candidate_evaluations() -> None:
    model = ModelConfig(num_layers=2, num_experts=2, top_k=1, num_gpus=2)
    workload = make_workload(
        WorkloadConfig(num_requests=4, output_tokens=32, max_batch_size=2),
        "autoregressive",
        num_gpus=2,
    )
    scheduler = CostAwareScheduler(
        objective="joint",
        route_knowledge="routing_oracle",
        model=model,
        compute=ComputeConfig(),
        network=NetworkConfig(),
        scheduler=SchedulerConfig(
            max_wait_ms=100.0,
            full_evaluation_shortlist_size=1,
        ),
    )

    selected = scheduler.select(
        workload.requests,
        2,
        now_ms=0.2,
        router=RouteFixture(),
        workload=workload,
    )

    assert len(selected) == 2
    assert scheduler.total_candidate_evaluations == 0
    assert scheduler.total_proxy_evaluations == 3


def test_one_shot_proxy_forms_batch_with_one_proxy_pass() -> None:
    model = ModelConfig(num_layers=2, num_experts=2, top_k=1, num_gpus=2)
    workload = make_workload(
        WorkloadConfig(num_requests=4, output_tokens=32, max_batch_size=3),
        "autoregressive",
        num_gpus=2,
    )
    scheduler = CostAwareScheduler(
        objective="joint",
        route_knowledge="routing_oracle",
        model=model,
        compute=ComputeConfig(),
        network=NetworkConfig(),
        scheduler=SchedulerConfig(
            max_wait_ms=100.0,
            full_evaluation_shortlist_size=1,
            one_shot_proxy_batch=True,
        ),
    )

    selected = scheduler.select(
        workload.requests,
        3,
        now_ms=0.2,
        router=RouteFixture(),
        workload=workload,
    )

    assert len(selected) == 3
    assert selected[0].request_id == 0
    assert scheduler.total_candidate_evaluations == 0
    assert scheduler.total_proxy_evaluations == 3


def test_adaptive_proxy_falls_back_on_fast_fabric() -> None:
    model = ModelConfig(num_layers=2, num_experts=2, top_k=1, num_gpus=2)
    workload = make_workload(
        WorkloadConfig(num_requests=4, output_tokens=32, max_batch_size=3),
        "autoregressive",
        num_gpus=2,
    )
    scheduler = CostAwareScheduler(
        objective="joint",
        route_knowledge="routing_oracle",
        model=model,
        compute=ComputeConfig(),
        network=NetworkConfig(
            bandwidth_gb_per_s=900.0,
            fixed_message_latency_ms=0.002,
        ),
        scheduler=SchedulerConfig(
            max_wait_ms=100.0,
            full_evaluation_shortlist_size=1,
            one_shot_proxy_batch=True,
            proxy_activation_mode="communication_bound",
        ),
    )

    selected = scheduler.select(
        workload.requests,
        3,
        now_ms=0.2,
        router=RouteFixture(),
        workload=workload,
    )

    assert [request.request_id for request in selected] == [0, 1, 2]
    assert scheduler.total_proxy_evaluations == 0
