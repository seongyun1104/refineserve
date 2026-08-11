from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .config import SimulationConfig

ScenarioName = Literal["compute_bound", "communication_bound", "deadline_bound"]
SCENARIOS: tuple[ScenarioName, ...] = (
    "compute_bound",
    "communication_bound",
    "deadline_bound",
)


def scenario_config(base: SimulationConfig, scenario: ScenarioName) -> SimulationConfig:
    if scenario == "compute_bound":
        config = replace(
            base,
            router=replace(
                base.router,
                distribution="request_correlated",
                request_cluster_probability=0.92,
            ),
            compute=replace(
                base.compute,
                expert_peak_tokens_per_ms=24.0,
                expert_memory_bandwidth_gb_per_s=1_500.0,
            ),
            network=replace(
                base.network,
                profile="compute_bound_fast_fabric",
                fixed_message_latency_ms=0.002,
                bandwidth_gb_per_s=900.0,
                congestion_factor=0.005,
            ),
            scheduler=replace(base.scheduler, max_wait_ms=100.0),
        )
    elif scenario == "communication_bound":
        config = replace(
            base,
            router=replace(
                base.router,
                distribution="uniform",
                temporal_stability=0.5,
            ),
            compute=replace(
                base.compute,
                expert_peak_tokens_per_ms=100.0,
                expert_weight_bytes=16_777_216,
            ),
            network=replace(
                base.network,
                profile="communication_bound_slow_fabric",
                fixed_message_latency_ms=0.03,
                bandwidth_gb_per_s=25.0,
                congestion_factor=0.10,
            ),
            scheduler=replace(base.scheduler, max_wait_ms=100.0),
        )
    elif scenario == "deadline_bound":
        config = replace(
            base,
            workload=replace(
                base.workload,
                arrival_interval_ms=0.01,
                output_length_pattern="staggered",
                minimum_output_tokens=base.workload.diffusion_block_size,
            ),
            scheduler=replace(base.scheduler, max_wait_ms=5.0),
        )
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    config.validate()
    return config
