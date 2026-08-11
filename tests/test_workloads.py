from __future__ import annotations

from refineserve.config import WorkloadConfig
from refineserve.workloads import (
    AutoregressiveWorkload,
    BlockRefinementWorkload,
    make_workload,
)


def workload_config() -> WorkloadConfig:
    return WorkloadConfig(
        num_requests=1,
        output_tokens=4,
        max_batch_size=1,
        diffusion_block_size=2,
        active_position_schedule=(2, 1),
    )


def test_factory_keeps_native_workload_types_separate_from_runtime() -> None:
    autoregressive = make_workload(workload_config(), "autoregressive", num_gpus=1)
    refinement = make_workload(workload_config(), "diffusion", num_gpus=1)

    assert isinstance(autoregressive, AutoregressiveWorkload)
    assert isinstance(refinement, BlockRefinementWorkload)


def test_autoregressive_work_item_directly_finalizes_one_token() -> None:
    workload = make_workload(workload_config(), "autoregressive", num_gpus=1)
    items = workload.ready_work_items(workload.requests)

    assert len(items) == 1
    assert items[0].is_finalization_eligible

    workload.finalize(items, now_ms=1.0)

    assert workload.requests[0].finalized_tokens == 1


def test_block_refinement_only_finalizes_on_last_native_round() -> None:
    workload = make_workload(workload_config(), "diffusion", num_gpus=1)

    first_round = workload.ready_work_items(workload.requests)
    first_state = workload.refinement_state(workload.requests[0])
    assert first_state.block_width == 2
    assert first_state.active_position_count == 2
    assert first_state.finalized_positions_per_step == 0
    assert first_state.order_policy == "model_defined"
    assert len(first_round) == 2
    assert not any(item.is_finalization_eligible for item in first_round)
    workload.finalize(first_round, now_ms=1.0)
    assert workload.requests[0].finalized_tokens == 0

    final_round = workload.ready_work_items(workload.requests)
    final_state = workload.refinement_state(workload.requests[0])
    assert final_state.block_width == 2
    assert final_state.active_position_count == 1
    assert final_state.finalized_positions_per_step == 2
    assert len(final_round) == 1
    assert final_round[0].is_finalization_eligible
    workload.finalize(final_round, now_ms=2.0)
    assert workload.requests[0].finalized_tokens == 2


def test_staggered_output_lengths_are_deterministic_block_multiples() -> None:
    config = WorkloadConfig(
        num_requests=6,
        output_tokens=96,
        max_batch_size=3,
        diffusion_block_size=32,
        active_position_schedule=(32, 1),
        output_length_pattern="staggered",
        minimum_output_tokens=32,
    )

    workload = make_workload(config, "diffusion", num_gpus=2)

    assert [request.output_tokens for request in workload.requests] == [
        32,
        64,
        96,
        32,
        64,
        96,
    ]
