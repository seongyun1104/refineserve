from __future__ import annotations

import numpy as np

from hardware.synthetic_routes import make_routes


def routes(mode: str, seed: int, strength: float = 0.75) -> np.ndarray:
    return make_routes(
        seed=seed,
        mode=mode,
        global_request_ids=np.arange(16),
        layers=4,
        positions=16,
        experts=16,
        request_correlation_strength=strength,
    )


def test_uniform_is_seeded_random_not_round_robin() -> None:
    first = routes("uniform", 17)
    repeated = routes("uniform", 17)
    second = routes("uniform", 29)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)


def test_balanced_round_robin_is_explicit_legacy_control() -> None:
    assert np.array_equal(
        routes("balanced_round_robin", 17),
        routes("balanced_round_robin", 29),
    )


def test_request_correlation_is_seeded_and_controllable() -> None:
    low = routes("request_correlated", 17, strength=0.0)
    high = routes("request_correlated", 17, strength=1.0)
    other_seed = routes("request_correlated", 29, strength=1.0)
    low_destinations = low // 4
    high_destinations = high // 4

    low_collision = np.mean(low_destinations[..., 0] == low_destinations[..., 1])
    high_collision = np.mean(high_destinations[..., 0] == high_destinations[..., 1])
    assert high_collision > low_collision
    assert not np.array_equal(high, other_seed)
