from __future__ import annotations

from dataclasses import replace

import pytest

from refineserve.config import SimulationConfig, WorkloadConfig, load_config


def test_baseline_config_is_the_eight_layer_contract() -> None:
    config = load_config("configs/baseline.yaml")

    assert config.model.num_layers == 8
    assert config.model.num_experts == 16
    assert config.model.top_k == 2
    assert config.model.num_gpus == 4
    assert config.workload.active_position_schedule == (32, 24, 16, 12, 8, 4, 2, 1)
    assert config.compute.expert_weight_bytes == 0
    assert not config.network.use_rank_critical_path


def test_m1_profile_enables_rank_critical_path_and_weight_traffic() -> None:
    config = load_config("configs/m1_critical_path.yaml")

    assert config.compute.expert_weight_bytes == 134_217_728
    assert config.compute.expert_memory_bandwidth_gb_per_s == 3_000.0
    assert config.network.use_rank_critical_path


def test_m1_online_config_is_the_selected_policy_contract() -> None:
    config = load_config("configs/m1_online.yaml")

    assert config.model.num_layers == 8
    assert config.scheduler.name == "joint"
    assert config.scheduler.one_shot_proxy_batch
    assert config.scheduler.full_evaluation_shortlist_size == 1
    assert config.scheduler.proxy_activation_mode == "communication_bound"
    assert config.scheduler.proxy_bandwidth_threshold_gb_per_s == 100.0
    assert config.scheduler.proxy_latency_threshold_ms == 0.02


def test_validation_rejects_nonfinalizing_diffusion_schedule() -> None:
    config = SimulationConfig(
        workload=replace(WorkloadConfig(), active_position_schedule=(8, 4, 2))
    )

    with pytest.raises(ValueError, match="end at 1"):
        config.validate()


def test_validation_rejects_unaligned_staggered_minimum() -> None:
    config = SimulationConfig(
        workload=replace(
            WorkloadConfig(),
            output_length_pattern="staggered",
            minimum_output_tokens=17,
        )
    )

    with pytest.raises(ValueError, match="divisible by diffusion_block_size"):
        config.validate()


def test_validation_rejects_empty_candidate_pool() -> None:
    config = SimulationConfig(
        scheduler=replace(SimulationConfig().scheduler, candidate_pool_size=0)
    )

    with pytest.raises(ValueError, match="candidate_pool_size"):
        config.validate()


def test_validation_rejects_empty_full_evaluation_shortlist() -> None:
    config = SimulationConfig(
        scheduler=replace(
            SimulationConfig().scheduler,
            full_evaluation_shortlist_size=0,
        )
    )

    with pytest.raises(ValueError, match="full_evaluation_shortlist_size"):
        config.validate()


def test_one_shot_proxy_requires_proxy_only_shortlist() -> None:
    config = SimulationConfig(
        scheduler=replace(
            SimulationConfig().scheduler,
            one_shot_proxy_batch=True,
        )
    )

    with pytest.raises(ValueError, match="requires full_evaluation_shortlist_size=1"):
        config.validate()


def test_proxy_activation_thresholds_must_be_positive() -> None:
    config = SimulationConfig(
        scheduler=replace(
            SimulationConfig().scheduler,
            proxy_latency_threshold_ms=0.0,
        )
    )

    with pytest.raises(ValueError, match="proxy activation thresholds"):
        config.validate()


def test_trace_router_requires_a_path() -> None:
    config = SimulationConfig(
        router=replace(SimulationConfig().router, source="trace")
    )

    with pytest.raises(ValueError, match="trace_path is required"):
        config.validate()


def test_calibration_checksum_matching_requires_an_artifact() -> None:
    config = SimulationConfig(
        calibration=replace(
            SimulationConfig().calibration,
            require_trace_checksum_match=True,
        )
    )

    with pytest.raises(ValueError, match="requires calibration.artifact_path"):
        config.validate()
