"""Pure constructed-plan contract shared by the Gate 2B benchmark and tests."""

from __future__ import annotations

PREFERENCES = (0, 0, 1, 2, 1, 2, 3, 3)
FIFO_PLAN = ((0, 1, 2, 3), (4, 5, 6, 7))
BALANCED_PLAN = ((0, 2, 3, 6), (1, 4, 5, 7))
ARMS = (
    "fifo_constructed",
    "dose_083_constructed",
    "balanced_constructed",
)


def global_plans() -> dict[str, tuple[tuple[tuple[int, ...], ...], ...]]:
    return {
        "fifo_constructed": (FIFO_PLAN,) * 4,
        "dose_083_constructed": (
            BALANCED_PLAN,
            FIFO_PLAN,
            FIFO_PLAN,
            FIFO_PLAN,
        ),
        "balanced_constructed": (BALANCED_PLAN,) * 4,
    }


def constructed_split_contract(
    *, positions: int, world_size: int = 4
) -> dict[str, list[list[list[int]]]]:
    """Return ``[arm][source][batch][destination]`` assignment counts."""
    if positions <= 0:
        raise ValueError("positions must be positive")
    if world_size != 4:
        raise ValueError("the constructed Gate 2B contract requires four ranks")
    contract: dict[str, list[list[list[int]]]] = {}
    for arm, source_plans in global_plans().items():
        source_splits: list[list[list[int]]] = []
        for plan in source_plans:
            batch_splits: list[list[int]] = []
            for selected_requests in plan:
                counts = [0] * world_size
                for request in selected_requests:
                    preferred_rank = PREFERENCES[request]
                    counts[preferred_rank] += positions
                    counts[(preferred_rank + 1) % world_size] += positions
                batch_splits.append(counts)
            source_splits.append(batch_splits)
        contract[arm] = source_splits
    return contract


def validate_split_contract(
    contract: dict[str, list[list[list[int]]]], *, positions: int
) -> None:
    expected_per_batch = 4 * positions * 2
    destination_totals_by_arm: dict[str, list[int]] = {}
    for arm in ARMS:
        source_splits = contract[arm]
        if len(source_splits) != 4:
            raise ValueError(f"{arm} must define exactly four source ranks")
        totals = [0, 0, 0, 0]
        for batches in source_splits:
            if len(batches) != 2:
                raise ValueError(f"{arm} must define two batches per source")
            for counts in batches:
                if len(counts) != 4 or sum(counts) != expected_per_batch:
                    raise ValueError(f"invalid split vector in {arm}: {counts}")
                if any(count <= 0 for count in counts):
                    raise ValueError(f"every destination must be non-empty: {counts}")
                totals = [
                    left + right
                    for left, right in zip(totals, counts, strict=True)
                ]
        destination_totals_by_arm[arm] = totals
    reference = destination_totals_by_arm[ARMS[0]]
    if any(totals != reference for totals in destination_totals_by_arm.values()):
        raise ValueError(
            "constructed arms must preserve global destination assignment totals"
        )
