from __future__ import annotations

from ..models import RefinementState, Request
from .base import DecodeWorkload, WorkItem


class AutoregressiveWorkload(DecodeWorkload):
    mode = "autoregressive"

    def work_items(self, request: Request) -> list[WorkItem]:
        return [
            WorkItem(
                request_id=request.request_id,
                position_id=0,
                iteration=request.iteration,
                is_finalization_eligible=True,
            )
        ]

    def previous_active_positions(self, request: Request) -> int:
        del request
        return 1

    def refinement_state(self, request: Request) -> RefinementState:
        return RefinementState(
            block_width=1,
            active_position_count=1,
            finalized_positions_per_step=min(
                1, request.output_tokens - request.finalized_tokens
            ),
            order_policy="left_to_right",
        )

    def finalize(self, completed_items: list[WorkItem], now_ms: float) -> None:
        for request_id in self._requests_in(completed_items):
            request = self._request(request_id)
            finalized = min(1, request.output_tokens - request.finalized_tokens)
            self._advance(request, finalized, now_ms)
