from __future__ import annotations

import numpy as np

from hardware.coordinated_scheduling import plan_cost
from hardware.proxy_validation_contract import (
    ARMS,
    PREFERENCES,
    constructed_split_contract,
    global_plans,
    validate_split_contract,
)


def test_constructed_proxy_split_contract_is_payload_symmetric() -> None:
    positions = 64
    contract = constructed_split_contract(positions=positions)

    validate_split_contract(contract, positions=positions)

    destination_totals = {}
    for arm in ARMS:
        totals = [0, 0, 0, 0]
        for source_batches in contract[arm]:
            for counts in source_batches:
                totals = [
                    left + right
                    for left, right in zip(totals, counts, strict=True)
                ]
        destination_totals[arm] = totals
    assert len({tuple(values) for values in destination_totals.values()}) == 1


def test_low_dose_requires_source_specific_receive_splits() -> None:
    contract = constructed_split_contract(positions=64)

    first_batch_to_rank_one = [
        contract["dose_083_constructed"][source][0][1]
        for source in range(4)
    ]
    assert len(set(first_batch_to_rank_one)) > 1


def test_constructed_proxy_objective_doses_are_exact() -> None:
    positions = 64
    layers = 8
    experts = 16
    experts_per_rank = 4
    counts = np.zeros((4, 8, layers, experts), dtype=np.int64)
    for source in range(4):
        for request, preferred_rank in enumerate(PREFERENCES):
            counts[source, request, :, preferred_rank * experts_per_rank] = positions
            counts[
                source,
                request,
                :,
                ((preferred_rank + 1) % 4) * experts_per_rank + 1,
            ] = positions
    objectives = {
        arm: plan_cost(counts, plan, experts_per_rank)
        for arm, plan in global_plans().items()
    }
    fifo = objectives["fifo_constructed"]

    assert (fifo - objectives["dose_083_constructed"]) / fifo == 1.0 / 12.0
    assert (fifo - objectives["balanced_constructed"]) / fifo == 1.0 / 3.0
