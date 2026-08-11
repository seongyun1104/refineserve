from __future__ import annotations

from ..config import ExecutionMode, WorkloadConfig
from .autoregressive import AutoregressiveWorkload
from .base import DecodeWorkload
from .block_refinement import BlockRefinementWorkload


def make_workload(config: WorkloadConfig, mode: ExecutionMode, num_gpus: int) -> DecodeWorkload:
    if mode == "autoregressive":
        return AutoregressiveWorkload(config, num_gpus)
    if mode == "diffusion":
        return BlockRefinementWorkload(config, num_gpus)
    raise ValueError(f"unsupported execution mode: {mode}")
