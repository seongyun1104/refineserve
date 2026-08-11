from __future__ import annotations

from ..models import Request
from ..router import RouterLike
from ..workloads.base import DecodeWorkload
from .base import Scheduler


class FIFOScheduler(Scheduler):
    def select(
        self,
        ready: list[Request],
        max_batch_size: int,
        now_ms: float,
        router: RouterLike,
        workload: DecodeWorkload,
    ) -> list[Request]:
        del now_ms, router, workload
        self.record_step_overhead(0)
        return sorted(ready, key=lambda req: (req.ready_since_ms, req.request_id))[:max_batch_size]
