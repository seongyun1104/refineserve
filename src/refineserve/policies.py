from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .config import SchedulerConfig

PolicyName = Literal[
    "fifo",
    "full_joint",
    "greedy_proxy",
    "online_proxy",
    "adaptive_online",
]
POLICIES: tuple[PolicyName, ...] = (
    "fifo",
    "full_joint",
    "greedy_proxy",
    "online_proxy",
    "adaptive_online",
)


def policy_config(scheduler: SchedulerConfig, policy: PolicyName) -> SchedulerConfig:
    common = {
        "candidate_pool_size": None,
        "full_evaluation_shortlist_size": None,
        "one_shot_proxy_batch": False,
        "proxy_activation_mode": "always",
    }
    if policy == "fifo":
        return replace(scheduler, name="fifo", **common)
    if policy == "full_joint":
        return replace(scheduler, name="joint", **common)
    if policy == "greedy_proxy":
        return replace(
            scheduler,
            name="joint",
            candidate_pool_size=None,
            full_evaluation_shortlist_size=1,
            one_shot_proxy_batch=False,
            proxy_activation_mode="always",
        )
    if policy == "online_proxy":
        return replace(
            scheduler,
            name="joint",
            candidate_pool_size=None,
            full_evaluation_shortlist_size=1,
            one_shot_proxy_batch=True,
            proxy_activation_mode="always",
        )
    if policy == "adaptive_online":
        return replace(
            scheduler,
            name="joint",
            candidate_pool_size=None,
            full_evaluation_shortlist_size=1,
            one_shot_proxy_batch=True,
            proxy_activation_mode="communication_bound",
        )
    raise ValueError(f"unsupported policy: {policy}")
