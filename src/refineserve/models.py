from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenAssignment:
    request_id: int
    position_id: int
    expert_id: int
    source_gpu: int


@dataclass
class Request:
    request_id: int
    arrival_time_ms: float
    output_tokens: int
    kv_location: int
    iteration: int = 0
    finalized_tokens: int = 0
    ready_since_ms: float = 0.0
    completion_time_ms: float | None = None
    total_queue_delay_ms: float = 0.0

    @property
    def done(self) -> bool:
        return self.finalized_tokens >= self.output_tokens


@dataclass(frozen=True)
class RefinementState:
    """Model-owned decode semantics that runtime scheduling must not alter."""

    block_width: int
    active_position_count: int
    finalized_positions_per_step: int
    order_policy: str

    def __post_init__(self) -> None:
        if self.block_width <= 0:
            raise ValueError("block_width must be positive")
        if not 0 < self.active_position_count <= self.block_width:
            raise ValueError("active_position_count must be in [1, block_width]")
        if not 0 <= self.finalized_positions_per_step <= self.block_width:
            raise ValueError("finalized_positions_per_step must be in [0, block_width]")
        if not self.order_policy:
            raise ValueError("order_policy must be non-empty")


@dataclass(frozen=True)
class Expert:
    expert_id: int
    gpu_id: int


@dataclass(frozen=True)
class RankExecution:
    batch_id: int
    layer_id: int
    gpu_id: int
    expert_compute_ms: float
    communication_ms: float
    layer_time_ms: float
    unique_experts: int
    expert_tokens: int
    expert_weight_bytes: int
    expert_weight_read_ms: float
    network_messages: int
    network_endpoint_bytes: int
    is_critical: bool


@dataclass(frozen=True)
class BatchRequestExecution:
    batch_id: int
    request_id: int
    iteration: int
    active_positions: int
    block_width: int
    finalized_positions_per_step: int
    order_policy: str
    kv_location: int
    wait_ms: float


@dataclass(frozen=True)
class BatchExecution:
    batch_id: int
    start_ms: float
    end_ms: float
    scheduler_overhead_ms: float
    request_count: int
    processed_positions: int
    finalized_tokens: int
    progress_span: int
    mean_wait_ms: float
    max_wait_ms: float
    expert_invocations: int
    network_messages: int
    transferred_bytes: int
    communication_ms: float
    layer_execution_ms: float
    underfilled: bool


@dataclass(frozen=True)
class RuntimeDiagnostics:
    simulator_wall_time_ms: float
    scheduler_selection_wall_time_ms: float
    scheduler_selection_calls: int
    scheduler_selection_mean_ms: float
    scheduler_selection_p50_ms: float
    scheduler_selection_p95_ms: float
    scheduler_selection_p99_ms: float
    scheduler_selection_max_ms: float
    scheduler_profile_update_wall_time_ms: float
    scheduler_profile_update_mean_ms: float
    scheduler_profile_update_p95_ms: float
    scheduler_profile_update_max_ms: float
    scheduler_total_wall_time_ms: float


@dataclass
class LayerExecution:
    layer_id: int
    elapsed_ms: float
    attention_ms: float
    expert_compute_ms: float
    communication_ms: float
    aggregate_communication_ms: float
    rank_critical_path_ms: float
    critical_gpu_id: int
    assignments: int
    expert_batch_sizes: list[int] = field(default_factory=list)
    expert_invocations: int = 0
    network_messages: int = 0
    transferred_bytes: int = 0
    all_to_all_calls: int = 0
    per_gpu_expert_ms: list[float] = field(default_factory=list)
    expert_token_counts: list[int] = field(default_factory=list)
    rank_executions: list[RankExecution] = field(default_factory=list)
    token_assignments: list[TokenAssignment] = field(default_factory=list)
