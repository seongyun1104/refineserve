from __future__ import annotations

import csv
from dataclasses import asdict, replace

import pytest

from refineserve.config import (
    ModelConfig,
    SimulationConfig,
    WorkloadConfig,
)
from refineserve.simulator import Simulator


class RefinementMutatingScheduler:
    """Deliberately violates the model/runtime ownership boundary."""

    def select(self, ready, max_batch_size, now_ms, router, workload):
        del now_ms, router, workload
        ready[0].iteration += 1
        return ready[:max_batch_size]


def small_config() -> SimulationConfig:
    return SimulationConfig(
        seed=41,
        model=ModelConfig(
            num_layers=2,
            num_experts=4,
            top_k=2,
            num_gpus=2,
            hidden_size=128,
        ),
        workload=WorkloadConfig(
            num_requests=4,
            output_tokens=4,
            max_batch_size=2,
            arrival_interval_ms=0.01,
            diffusion_block_size=2,
            active_position_schedule=(2, 1),
        ),
    )


def test_simulator_is_reproducible() -> None:
    config = small_config()

    first = Simulator(config, "diffusion").run()
    second = Simulator(config, "diffusion").run()

    assert asdict(first.summary) == asdict(second.summary)
    assert first.expert_batch_sizes == second.expert_batch_sizes


def test_ar_and_diffusion_finalize_the_same_number_of_tokens() -> None:
    config = small_config()
    autoregressive = Simulator(config, "autoregressive").run().summary
    diffusion = Simulator(config, "diffusion").run().summary

    assert autoregressive.finalized_tokens == diffusion.finalized_tokens == 16
    assert autoregressive.processed_positions == 16
    assert diffusion.processed_positions == 24
    assert autoregressive.useful_work_ratio == 1.0
    assert diffusion.useful_work_ratio == 2 / 3
    assert autoregressive.total_expert_token_executions == (
        autoregressive.processed_positions * config.model.num_layers * config.model.top_k
    )
    assert diffusion.total_expert_token_executions == (
        diffusion.processed_positions * config.model.num_layers * config.model.top_k
    )
    assert autoregressive.batch_count > 0
    assert diffusion.batch_count > 0
    assert 0.0 <= diffusion.underfilled_batch_ratio <= 1.0


def test_runtime_scheduler_cannot_change_model_refinement_state() -> None:
    simulator = Simulator(small_config(), "diffusion")
    simulator.scheduler = RefinementMutatingScheduler()

    with pytest.raises(
        RuntimeError, match="runtime scheduler altered model-owned refinement semantics"
    ):
        simulator.run()


def test_result_export_contains_machine_readable_outputs(tmp_path) -> None:
    config = replace(
        small_config(),
        scheduler=replace(small_config().scheduler, name="previous_route"),
    )
    result = Simulator(config, "autoregressive").run()

    output = result.write(tmp_path / "run")

    assert (output / "summary.csv").is_file()
    assert (output / "request_latencies.csv").is_file()
    assert (output / "expert_batches.csv").is_file()
    assert (output / "rank_layers.csv").is_file()
    assert (output / "batches.csv").is_file()
    assert (output / "batch_requests.csv").is_file()
    assert (output / "metadata.json").is_file()
    assert (output / "runtime_diagnostics.json").is_file()
    with (output / "batch_requests.csv").open(newline="") as handle:
        columns = set(next(csv.DictReader(handle)).keys())
    assert {
        "block_width",
        "active_positions",
        "finalized_positions_per_step",
        "order_policy",
    } <= columns


def test_rank_critical_path_profile_conserves_expert_weight_work() -> None:
    base = small_config()
    config = replace(
        base,
        compute=replace(
            base.compute,
            expert_weight_bytes=2_048,
            expert_memory_bandwidth_gb_per_s=1.0,
        ),
        network=replace(base.network, use_rank_critical_path=True),
    )

    result = Simulator(config, "diffusion").run()

    assert result.summary.runtime_uses_rank_critical_path
    assert result.summary.expert_weight_bytes_read == (result.summary.expert_invocations * 2_048)
    assert result.summary.modeled_rank_critical_path_ms == result.summary.makespan_ms
    assert len(result.rank_executions) % config.model.num_gpus == 0


def test_batch_trace_conserves_work_and_finalization() -> None:
    result = Simulator(small_config(), "diffusion").run()

    assert len(result.batch_executions) == result.summary.batch_count
    assert sum(batch.processed_positions for batch in result.batch_executions) == (
        result.summary.processed_positions
    )
    assert sum(batch.finalized_tokens for batch in result.batch_executions) == (
        result.summary.finalized_tokens
    )
    assert len(result.batch_requests) == sum(
        batch.request_count for batch in result.batch_executions
    )
    assert result.runtime_diagnostics.scheduler_selection_calls == result.summary.batch_count
    assert result.runtime_diagnostics.scheduler_selection_wall_time_ms >= 0.0
    assert result.runtime_diagnostics.scheduler_selection_p50_ms >= 0.0
    assert result.runtime_diagnostics.scheduler_selection_p95_ms >= (
        result.runtime_diagnostics.scheduler_selection_p50_ms
    )
    assert result.runtime_diagnostics.scheduler_selection_p99_ms >= (
        result.runtime_diagnostics.scheduler_selection_p95_ms
    )
    assert result.runtime_diagnostics.scheduler_selection_max_ms >= (
        result.runtime_diagnostics.scheduler_selection_p99_ms
    )
    assert result.runtime_diagnostics.scheduler_profile_update_wall_time_ms >= 0.0
    assert result.runtime_diagnostics.scheduler_total_wall_time_ms == (
        result.runtime_diagnostics.scheduler_selection_wall_time_ms
        + result.runtime_diagnostics.scheduler_profile_update_wall_time_ms
    )


def test_scheduler_overhead_is_included_in_makespan() -> None:
    base = small_config()
    baseline = Simulator(base, "autoregressive").run().summary
    config = replace(
        base,
        scheduler=replace(base.scheduler, base_overhead_ms=0.01),
    )

    result = Simulator(config, "autoregressive").run().summary

    assert result.scheduler_modeled_overhead_ms > 0.0
    assert result.scheduler_candidate_evaluations == 0
    assert result.makespan_ms > baseline.makespan_ms
    assert result.scheduler_modeled_overhead_fraction == (
        result.scheduler_modeled_overhead_ms / result.makespan_ms
    )


def test_routing_and_runtime_oracles_match_in_deterministic_model() -> None:
    base = small_config()
    routing_config = replace(
        base,
        scheduler=replace(base.scheduler, name="routing_oracle"),
    )
    runtime_config = replace(
        base,
        scheduler=replace(base.scheduler, name="runtime_oracle"),
    )

    routing = asdict(Simulator(routing_config, "diffusion").run().summary)
    runtime = asdict(Simulator(runtime_config, "diffusion").run().summary)
    routing.pop("scheduler")
    runtime.pop("scheduler")

    assert routing == runtime
