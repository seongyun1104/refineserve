from __future__ import annotations

import numpy as np

from ..models import Request
from ..router import RouterLike, cosine_similarity
from ..workloads.base import DecodeWorkload
from .base import Scheduler


class ExpertLocalityScheduler(Scheduler):
    def __init__(
        self,
        *,
        oracle: bool,
        max_wait_ms: float,
        base_overhead_ms: float = 0.0,
        candidate_evaluation_overhead_ms: float = 0.0,
        proxy_evaluation_overhead_ms: float = 0.0,
        candidate_pool_size: int | None = None,
    ):
        super().__init__(
            base_overhead_ms=base_overhead_ms,
            candidate_evaluation_overhead_ms=candidate_evaluation_overhead_ms,
            proxy_evaluation_overhead_ms=proxy_evaluation_overhead_ms,
        )
        self.oracle = oracle
        self.max_wait_ms = max_wait_ms
        self.candidate_pool_size = candidate_pool_size

    def select(
        self,
        ready: list[Request],
        max_batch_size: int,
        now_ms: float,
        router: RouterLike,
        workload: DecodeWorkload,
    ) -> list[Request]:
        if not ready:
            self.record_step_overhead(0)
            return []
        remaining = sorted(ready, key=lambda req: (req.ready_since_ms, req.request_id))
        selected = [remaining.pop(0)]
        aggregate = router.signature(selected[0], workload, oracle=self.oracle).copy()
        candidate_evaluations = 1

        while remaining and len(selected) < max_batch_size:
            oldest = remaining[0]
            if now_ms - oldest.ready_since_ms >= self.max_wait_ms:
                choice = oldest
            else:
                candidates = (
                    remaining[: self.candidate_pool_size]
                    if self.candidate_pool_size is not None
                    else remaining
                )
                candidate_evaluations += len(candidates)
                choice = max(
                    candidates,
                    key=lambda req: (
                        cosine_similarity(
                            aggregate,
                            router.signature(req, workload, oracle=self.oracle),
                        ),
                        -req.ready_since_ms,
                        -req.request_id,
                    ),
                )
            remaining.remove(choice)
            selected.append(choice)
            aggregate += router.signature(choice, workload, oracle=self.oracle)
        self.record_step_overhead(candidate_evaluations)
        return selected


def mean_pairwise_similarity(signatures: list[np.ndarray]) -> float:
    if len(signatures) < 2:
        return 1.0
    values = [
        cosine_similarity(signatures[i], signatures[j])
        for i in range(len(signatures))
        for j in range(i + 1, len(signatures))
    ]
    return float(np.mean(values))
