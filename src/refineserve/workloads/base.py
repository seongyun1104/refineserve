from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import ExecutionMode, WorkloadConfig
from ..models import RefinementState, Request


@dataclass(frozen=True)
class WorkItem:
    request_id: int
    position_id: int
    iteration: int
    is_finalization_eligible: bool
    route_signature: tuple[int, ...] | None = None


class DecodeWorkload(ABC):
    """Native decode work producer consumed by the common MoE runtime."""

    mode: ExecutionMode

    def __init__(self, config: WorkloadConfig, num_gpus: int):
        self.config = config
        self.requests = [
            Request(
                request_id=request_id,
                arrival_time_ms=request_id * config.arrival_interval_ms,
                output_tokens=self._output_tokens(request_id),
                kv_location=request_id % num_gpus,
                ready_since_ms=request_id * config.arrival_interval_ms,
            )
            for request_id in range(config.num_requests)
        ]

    def _output_tokens(self, request_id: int) -> int:
        if self.config.output_length_pattern == "fixed":
            return self.config.output_tokens
        minimum = self.config.minimum_output_tokens
        if minimum is None:
            raise RuntimeError("staggered output lengths require minimum_output_tokens")
        block = self.config.diffusion_block_size
        minimum_blocks = minimum // block
        maximum_blocks = self.config.output_tokens // block
        level = request_id % 3
        blocks = round(minimum_blocks + (maximum_blocks - minimum_blocks) * level / 2)
        return blocks * block

    def ready_requests(self, now_ms: float) -> list[Request]:
        return [
            request
            for request in self.requests
            if not request.done and request.arrival_time_ms <= now_ms
        ]

    def next_arrival_time(self) -> float:
        return min(request.arrival_time_ms for request in self.requests if not request.done)

    def ready_work_items(self, requests: list[Request]) -> list[WorkItem]:
        return [item for request in requests for item in self.work_items(request)]

    @abstractmethod
    def work_items(self, request: Request) -> list[WorkItem]:
        """Return positions processed in the request's current native iteration."""

    def active_positions(self, request: Request) -> int:
        return len(self.work_items(request))

    @abstractmethod
    def refinement_state(self, request: Request) -> RefinementState:
        """Return model-owned position selection and finalization semantics."""

    @abstractmethod
    def previous_active_positions(self, request: Request) -> int:
        """Return the preceding native iteration width for route prediction."""

    @abstractmethod
    def finalize(self, completed_items: list[WorkItem], now_ms: float) -> None:
        """Commit completed native work into request progress."""

    @staticmethod
    def _requests_in(items: list[WorkItem]) -> list[int]:
        return sorted({item.request_id for item in items})

    def _request(self, request_id: int) -> Request:
        return self.requests[request_id]

    @staticmethod
    def _advance(request: Request, finalized: int, now_ms: float) -> None:
        if request.done:
            raise RuntimeError(f"request {request.request_id} advanced after completion")
        request.iteration += 1
        request.finalized_tokens += finalized
        request.ready_since_ms = now_ms
        if request.done:
            request.completion_time_ms = now_ms
