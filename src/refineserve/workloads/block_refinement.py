from __future__ import annotations

from ..models import RefinementState, Request
from .base import DecodeWorkload, WorkItem


class BlockRefinementWorkload(DecodeWorkload):
    """Native position-parallel block refinement workload."""

    mode = "diffusion"

    def _round(self, request: Request) -> int:
        return request.iteration % len(self.config.active_position_schedule)

    def work_items(self, request: Request) -> list[WorkItem]:
        round_index = self._round(request)
        active_positions = self.config.active_position_schedule[round_index]
        final_round = round_index == len(self.config.active_position_schedule) - 1
        return [
            WorkItem(
                request_id=request.request_id,
                position_id=position_id,
                iteration=request.iteration,
                is_finalization_eligible=final_round,
            )
            for position_id in range(active_positions)
        ]

    def previous_active_positions(self, request: Request) -> int:
        if request.iteration == 0:
            return self.active_positions(request)
        previous_round = (request.iteration - 1) % len(self.config.active_position_schedule)
        return self.config.active_position_schedule[previous_round]

    def refinement_state(self, request: Request) -> RefinementState:
        final_round = self._round(request) == len(self.config.active_position_schedule) - 1
        finalized = (
            min(
                self.config.diffusion_block_size,
                request.output_tokens - request.finalized_tokens,
            )
            if final_round
            else 0
        )
        return RefinementState(
            block_width=self.config.diffusion_block_size,
            active_position_count=self.active_positions(request),
            finalized_positions_per_step=finalized,
            order_policy=self.config.order_policy,
        )

    def finalize(self, completed_items: list[WorkItem], now_ms: float) -> None:
        items_by_request = {
            request_id: [item for item in completed_items if item.request_id == request_id]
            for request_id in self._requests_in(completed_items)
        }
        for request_id, items in items_by_request.items():
            request = self._request(request_id)
            finalized = 0
            if all(item.is_finalization_eligible for item in items):
                finalized = min(
                    self.config.diffusion_block_size,
                    request.output_tokens - request.finalized_tokens,
                )
            self._advance(request, finalized, now_ms)
