from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import Request
from ..router import RouterLike
from ..workloads.base import DecodeWorkload


class Scheduler(ABC):
    def __init__(
        self,
        *,
        base_overhead_ms: float = 0.0,
        candidate_evaluation_overhead_ms: float = 0.0,
        proxy_evaluation_overhead_ms: float = 0.0,
    ) -> None:
        self.base_overhead_ms = base_overhead_ms
        self.candidate_evaluation_overhead_ms = candidate_evaluation_overhead_ms
        self.proxy_evaluation_overhead_ms = proxy_evaluation_overhead_ms
        self.last_overhead_ms = 0.0
        self.total_candidate_evaluations = 0
        self.total_proxy_evaluations = 0

    def record_step_overhead(
        self,
        candidate_evaluations: int,
        proxy_evaluations: int = 0,
    ) -> None:
        self.total_candidate_evaluations += candidate_evaluations
        self.total_proxy_evaluations += proxy_evaluations
        self.last_overhead_ms = (
            self.base_overhead_ms
            + candidate_evaluations * self.candidate_evaluation_overhead_ms
            + proxy_evaluations * self.proxy_evaluation_overhead_ms
        )

    def consume_step_overhead(self) -> float:
        overhead = self.last_overhead_ms
        self.last_overhead_ms = 0.0
        return overhead

    def prepare_requests(
        self,
        requests: list[Request],
        router: RouterLike,
        workload: DecodeWorkload,
        observed_routes: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Update scheduler-side request features after native iteration completion."""
        del requests, router, workload, observed_routes

    @abstractmethod
    def select(
        self,
        ready: list[Request],
        max_batch_size: int,
        now_ms: float,
        router: RouterLike,
        workload: DecodeWorkload,
    ) -> list[Request]:
        """Select at most max_batch_size requests without mutating ready."""
