from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class ExpertPlacement:
    """Layer-local expert-to-EP-rank mapping used by execution and estimation."""

    ranks: tuple[tuple[int, ...], ...]
    num_gpus: int

    @classmethod
    def round_robin(cls, model: ModelConfig) -> ExpertPlacement:
        row = tuple(expert_id % model.num_gpus for expert_id in range(model.num_experts))
        return cls(ranks=tuple(row for _ in range(model.num_layers)), num_gpus=model.num_gpus)

    @classmethod
    def from_rows(
        cls,
        rows: list[list[int]] | tuple[tuple[int, ...], ...],
        model: ModelConfig,
    ) -> ExpertPlacement:
        ranks = tuple(tuple(int(rank) for rank in layer) for layer in rows)
        if len(ranks) != model.num_layers:
            raise ValueError("expert placement must contain one row per model layer")
        if any(len(layer) != model.num_experts for layer in ranks):
            raise ValueError("expert placement must contain one rank per layer expert")
        if any(
            rank < 0 or rank >= model.num_gpus
            for layer in ranks
            for rank in layer
        ):
            raise ValueError("expert placement contains an out-of-range EP rank")
        return cls(ranks=ranks, num_gpus=model.num_gpus)

    def rank(self, layer_id: int, expert_id: int) -> int:
        return self.ranks[layer_id][expert_id]

    def one_hot(self) -> np.ndarray:
        result = np.zeros(
            (len(self.ranks), len(self.ranks[0]), self.num_gpus),
            dtype=np.int64,
        )
        for layer_id, layer in enumerate(self.ranks):
            result[layer_id, np.arange(len(layer)), np.asarray(layer)] = 1
        return result
