from __future__ import annotations

from refineserve.config import ModelConfig, RouterConfig
from refineserve.router import SyntheticRouter


def test_router_is_deterministic_independent_of_call_order() -> None:
    model = ModelConfig(num_layers=2, num_experts=8, top_k=2, num_gpus=2)
    config = RouterConfig(distribution="request_correlated", temporal_stability=0.7)
    first = SyntheticRouter(model, config, seed=123)
    second = SyntheticRouter(model, config, seed=123)

    expected = first.route(3, 4, 1, 7)
    second.route(99, 8, 0, 2)

    assert second.route(3, 4, 1, 7) == expected


def test_stability_one_reuses_the_previous_route() -> None:
    model = ModelConfig(num_layers=2, num_experts=8, top_k=2, num_gpus=2)
    config = RouterConfig(distribution="uniform", temporal_stability=1.0)
    router = SyntheticRouter(model, config, seed=123)

    assert router.route(2, 5, 1, 3) == router.route(2, 0, 1, 3)


def test_uniform_prior_route_is_deterministic_without_history() -> None:
    model = ModelConfig(num_layers=2, num_experts=8, top_k=2, num_gpus=2)
    router = SyntheticRouter(model, RouterConfig(distribution="uniform"), seed=123)

    assert router.prior_route(request_id=7, layer_id=1) == (0, 1)
