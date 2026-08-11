from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

ExecutionMode = Literal["autoregressive", "diffusion"]
OutputLengthPattern = Literal["fixed", "staggered"]
OrderPolicy = Literal["left_to_right", "confidence", "entropy_bounded", "model_defined"]
ProxyActivationMode = Literal["always", "communication_bound"]
RouterSource = Literal["synthetic", "trace"]
RouterDistribution = Literal[
    "uniform",
    "zipf",
    "hot_expert",
    "request_correlated",
    "temporally_stable",
    "temporally_unstable",
]
SchedulerName = Literal[
    "fifo",
    "previous_route",
    "oracle",
    "locality_only",
    "load_balance_only",
    "critical_path_only",
    "locality_plus_load",
    "joint",
    "routing_oracle",
    "runtime_oracle",
]


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int = 8
    num_experts: int = 16
    top_k: int = 2
    num_gpus: int = 4
    hidden_size: int = 2048
    bytes_per_element: int = 2


@dataclass(frozen=True)
class WorkloadConfig:
    num_requests: int = 32
    output_tokens: int = 64
    max_batch_size: int = 8
    arrival_interval_ms: float = 0.05
    diffusion_block_size: int = 32
    active_position_schedule: tuple[int, ...] = (32, 24, 16, 12, 8, 4, 2, 1)
    order_policy: OrderPolicy = "model_defined"
    output_length_pattern: OutputLengthPattern = "fixed"
    minimum_output_tokens: int | None = None


@dataclass(frozen=True)
class RouterConfig:
    source: RouterSource = "synthetic"
    trace_path: str | None = None
    distribution: RouterDistribution = "request_correlated"
    zipf_alpha: float = 1.2
    hot_expert_probability: float = 0.7
    request_cluster_probability: float = 0.85
    temporal_stability: float = 0.75


@dataclass(frozen=True)
class ComputeConfig:
    attention_base_ms: float = 0.015
    attention_token_ms: float = 0.0008
    expert_launch_ms: float = 0.012
    expert_peak_tokens_per_ms: float = 50.0
    expert_saturation_tokens: int = 32
    expert_weight_bytes: int = 0
    expert_memory_bandwidth_gb_per_s: float = 3_000.0


@dataclass(frozen=True)
class NetworkConfig:
    profile: str = "nvlink_like"
    fixed_message_latency_ms: float = 0.006
    bandwidth_gb_per_s: float = 450.0
    congestion_factor: float = 0.02
    aggregate_messages: bool = True
    use_rank_critical_path: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    name: SchedulerName = "fifo"
    max_wait_ms: float = 2.0
    locality_weight: float = 0.25
    locality_benefit_ms: float = 0.25
    deadline_weight_ms: float = 0.25
    kv_fragmentation_weight_ms: float = 0.10
    progress_fragmentation_weight_ms: float = 0.50
    base_overhead_ms: float = 0.0
    candidate_evaluation_overhead_ms: float = 0.0
    proxy_evaluation_overhead_ms: float = 0.0
    candidate_pool_size: int | None = None
    full_evaluation_shortlist_size: int | None = None
    one_shot_proxy_batch: bool = False
    proxy_activation_mode: ProxyActivationMode = "always"
    proxy_bandwidth_threshold_gb_per_s: float = 100.0
    proxy_latency_threshold_ms: float = 0.02


@dataclass(frozen=True)
class CalibrationConfig:
    artifact_path: str | None = None
    use_expert_kernel_curve: bool = True
    use_network_curves: bool = False
    require_trace_checksum_match: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 17
    model: ModelConfig = field(default_factory=ModelConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def validate(self) -> None:
        m, w, r, c, n, s = (
            self.model,
            self.workload,
            self.router,
            self.compute,
            self.network,
            self.scheduler,
        )
        positive_ints = {
            "model.num_layers": m.num_layers,
            "model.num_experts": m.num_experts,
            "model.top_k": m.top_k,
            "model.num_gpus": m.num_gpus,
            "model.hidden_size": m.hidden_size,
            "workload.num_requests": w.num_requests,
            "workload.output_tokens": w.output_tokens,
            "workload.max_batch_size": w.max_batch_size,
            "workload.diffusion_block_size": w.diffusion_block_size,
            "compute.expert_saturation_tokens": c.expert_saturation_tokens,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if m.top_k > m.num_experts:
            raise ValueError("model.top_k cannot exceed model.num_experts")
        if m.num_experts % m.num_gpus != 0:
            raise ValueError("model.num_experts must be divisible by model.num_gpus")
        if w.output_tokens % w.diffusion_block_size != 0:
            raise ValueError("output_tokens must be divisible by diffusion_block_size in MVP")
        if w.output_length_pattern == "staggered":
            if w.minimum_output_tokens is None:
                raise ValueError(
                    "minimum_output_tokens is required for staggered output lengths"
                )
            if not 0 < w.minimum_output_tokens <= w.output_tokens:
                raise ValueError("minimum_output_tokens must be in (0, output_tokens]")
            if w.minimum_output_tokens % w.diffusion_block_size != 0:
                raise ValueError(
                    "minimum_output_tokens must be divisible by diffusion_block_size"
                )
        if not w.active_position_schedule or w.active_position_schedule[-1] != 1:
            raise ValueError("active_position_schedule must be non-empty and end at 1")
        if any(p <= 0 or p > w.diffusion_block_size for p in w.active_position_schedule):
            raise ValueError("active positions must be in [1, diffusion_block_size]")
        if any(
            a < b
            for a, b in zip(
                w.active_position_schedule,
                w.active_position_schedule[1:],
                strict=False,
            )
        ):
            raise ValueError("active_position_schedule must be monotonically non-increasing")
        if w.order_policy not in {
            "left_to_right",
            "confidence",
            "entropy_bounded",
            "model_defined",
        }:
            raise ValueError(f"unsupported workload.order_policy: {w.order_policy}")
        for name, value in {
            "router.temporal_stability": r.temporal_stability,
            "router.hot_expert_probability": r.hot_expert_probability,
            "router.request_cluster_probability": r.request_cluster_probability,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if r.source == "trace" and not r.trace_path:
            raise ValueError("router.trace_path is required when router.source='trace'")
        if r.source == "synthetic" and r.trace_path is not None:
            raise ValueError("router.trace_path requires router.source='trace'")
        if c.expert_weight_bytes < 0:
            raise ValueError("compute.expert_weight_bytes cannot be negative")
        if (
            c.expert_peak_tokens_per_ms <= 0
            or c.expert_memory_bandwidth_gb_per_s <= 0
            or n.bandwidth_gb_per_s <= 0
        ):
            raise ValueError("compute throughput and network bandwidth must be positive")
        if s.max_wait_ms <= 0:
            raise ValueError("scheduler.max_wait_ms must be positive")
        for name, value in {
            "scheduler.locality_weight": s.locality_weight,
            "scheduler.locality_benefit_ms": s.locality_benefit_ms,
            "scheduler.deadline_weight_ms": s.deadline_weight_ms,
            "scheduler.kv_fragmentation_weight_ms": s.kv_fragmentation_weight_ms,
            "scheduler.progress_fragmentation_weight_ms": (
                s.progress_fragmentation_weight_ms
            ),
            "scheduler.base_overhead_ms": s.base_overhead_ms,
            "scheduler.candidate_evaluation_overhead_ms": (
                s.candidate_evaluation_overhead_ms
            ),
            "scheduler.proxy_evaluation_overhead_ms": s.proxy_evaluation_overhead_ms,
        }.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if s.candidate_pool_size is not None and s.candidate_pool_size <= 0:
            raise ValueError("scheduler.candidate_pool_size must be positive when set")
        if (
            s.full_evaluation_shortlist_size is not None
            and s.full_evaluation_shortlist_size <= 0
        ):
            raise ValueError(
                "scheduler.full_evaluation_shortlist_size must be positive when set"
            )
        if s.one_shot_proxy_batch and s.full_evaluation_shortlist_size != 1:
            raise ValueError(
                "one_shot_proxy_batch requires full_evaluation_shortlist_size=1"
            )
        if (
            s.proxy_bandwidth_threshold_gb_per_s <= 0
            or s.proxy_latency_threshold_ms <= 0
        ):
            raise ValueError("proxy activation thresholds must be positive")
        if self.calibration.require_trace_checksum_match and not self.calibration.artifact_path:
            raise ValueError("calibration checksum matching requires calibration.artifact_path")

    def with_overrides(self, *, mode_scheduler: SchedulerName | None = None) -> SimulationConfig:
        if mode_scheduler is None:
            return self
        return replace(self, scheduler=replace(self.scheduler, name=mode_scheduler))


def _construct[T](cls: type[T], raw: dict[str, Any]) -> T:
    fields = cls.__dataclass_fields__
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    if cls is WorkloadConfig and "active_position_schedule" in raw:
        raw = {**raw, "active_position_schedule": tuple(raw["active_position_schedule"])}
    return cls(**raw)


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    allowed = {
        "seed",
        "model",
        "workload",
        "router",
        "compute",
        "network",
        "scheduler",
        "calibration",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config fields: {sorted(unknown)}")
    cfg = SimulationConfig(
        seed=int(raw.get("seed", 17)),
        model=_construct(ModelConfig, raw.get("model", {})),
        workload=_construct(WorkloadConfig, raw.get("workload", {})),
        router=_construct(RouterConfig, raw.get("router", {})),
        compute=_construct(ComputeConfig, raw.get("compute", {})),
        network=_construct(NetworkConfig, raw.get("network", {})),
        scheduler=_construct(SchedulerConfig, raw.get("scheduler", {})),
        calibration=_construct(CalibrationConfig, raw.get("calibration", {})),
    )
    cfg.validate()
    return cfg
