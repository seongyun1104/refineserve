from __future__ import annotations

import pytest

from refineserve.config import (
    ModelConfig,
    NetworkConfig,
    SimulationConfig,
    WorkloadConfig,
)
from refineserve.global_oracle import ExactGlobalMakespanOracle
from refineserve.simulator import Simulator


def oracle_config() -> SimulationConfig:
    return SimulationConfig(
        seed=13,
        model=ModelConfig(
            num_layers=2,
            num_experts=4,
            top_k=1,
            num_gpus=2,
            hidden_size=64,
        ),
        workload=WorkloadConfig(
            num_requests=3,
            output_tokens=2,
            max_batch_size=2,
            arrival_interval_ms=0.01,
            diffusion_block_size=2,
            active_position_schedule=(2, 1),
        ),
        network=NetworkConfig(use_rank_critical_path=True),
    )


@pytest.mark.parametrize("mode", ["autoregressive", "diffusion"])
def test_exact_oracle_is_replayable_and_no_slower_than_fifo(mode: str) -> None:
    config = oracle_config()
    oracle = ExactGlobalMakespanOracle(config, mode).solve()
    fifo = Simulator(config, mode).run().summary

    assert oracle.optimal_makespan_ms <= fifo.makespan_ms + 1e-9
    assert oracle.batch_count == len(oracle.actions)
    assert oracle.explored_states > 0
    assert ExactGlobalMakespanOracle(config, mode).replay(oracle.actions) == pytest.approx(
        oracle.optimal_makespan_ms
    )


def test_exact_oracle_rejects_too_small_state_budget() -> None:
    with pytest.raises(RuntimeError, match="exceeded max_states"):
        ExactGlobalMakespanOracle(
            oracle_config(),
            "diffusion",
            max_states=1,
        ).solve()
