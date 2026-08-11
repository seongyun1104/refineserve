from __future__ import annotations

from collections import Counter

import numpy as np

from .calibration import CalibrationArtifact
from .config import ComputeConfig, ModelConfig, NetworkConfig
from .models import LayerExecution, RankExecution, Request, TokenAssignment
from .placement import ExpertPlacement
from .router import RouterLike
from .workloads import WorkItem


class LayerCostModel:
    def __init__(
        self,
        model: ModelConfig,
        compute: ComputeConfig,
        network: NetworkConfig,
        calibration: CalibrationArtifact | None = None,
        placement: ExpertPlacement | None = None,
    ):
        self.model = model
        self.compute = compute
        self.network = network
        self.calibration = calibration
        self.placement = placement or ExpertPlacement.round_robin(model)

    def execute_layer(
        self,
        requests: list[Request],
        work_items: list[WorkItem],
        layer_id: int,
        router: RouterLike,
        *,
        batch_id: int = 0,
    ) -> LayerExecution:
        assignments: list[TokenAssignment] = []
        source_token_counts = Counter[int]()
        requests_by_id = {request.request_id: request for request in requests}
        for item in work_items:
            request = requests_by_id[item.request_id]
            source_token_counts[request.kv_location] += 1
            for expert_id in router.route(
                item.request_id,
                item.iteration,
                layer_id,
                item.position_id,
            ):
                assignments.append(
                    TokenAssignment(
                        request_id=item.request_id,
                        position_id=item.position_id,
                        expert_id=expert_id,
                        source_gpu=request.kv_location,
                    )
                )

        attention_tokens = max(source_token_counts.values(), default=0)
        attention_ms = (
            self.compute.attention_base_ms + attention_tokens * self.compute.attention_token_ms
        )

        expert_counts = Counter(assignment.expert_id for assignment in assignments)
        per_gpu_expert_ms = [0.0] * self.model.num_gpus
        per_gpu_weight_read_ms = [0.0] * self.model.num_gpus
        per_gpu_weight_bytes = [0] * self.model.num_gpus
        per_gpu_unique_experts = [0] * self.model.num_gpus
        per_gpu_expert_tokens = [0] * self.model.num_gpus
        expert_batch_sizes: list[int] = []
        for expert_id, token_count in sorted(expert_counts.items()):
            gpu_id = self.placement.rank(layer_id, expert_id)
            utilization = max(
                0.05,
                min(1.0, token_count / self.compute.expert_saturation_tokens),
            )
            effective_tps = self.compute.expert_peak_tokens_per_ms * utilization
            token_compute_ms = token_count / effective_tps
            weight_read_ms = self.compute.expert_weight_bytes / (
                self.compute.expert_memory_bandwidth_gb_per_s * 1_000_000.0
            )
            if self.calibration is not None and self.calibration.expert_kernel_curve is not None:
                elapsed = self.calibration.expert_kernel_curve.latency_ms(token_count)
            else:
                elapsed = self.compute.expert_launch_ms + max(token_compute_ms, weight_read_ms)
            per_gpu_expert_ms[gpu_id] += elapsed
            per_gpu_weight_read_ms[gpu_id] += weight_read_ms
            per_gpu_weight_bytes[gpu_id] += self.compute.expert_weight_bytes
            per_gpu_unique_experts[gpu_id] += 1
            per_gpu_expert_tokens[gpu_id] += token_count
            expert_batch_sizes.append(token_count)
        expert_compute_ms = max(per_gpu_expert_ms, default=0.0)

        cross_gpu = [
            assignment
            for assignment in assignments
            if assignment.source_gpu
            != self.placement.rank(layer_id, assignment.expert_id)
        ]
        pair_counts: Counter[tuple[int, int]] = Counter(
            (
                assignment.source_gpu,
                self.placement.rank(layer_id, assignment.expert_id),
            )
            for assignment in cross_gpu
        )
        one_way_messages = len(pair_counts) if self.network.aggregate_messages else len(cross_gpu)
        network_messages = one_way_messages * 2
        bytes_per_activation = self.model.hidden_size * self.model.bytes_per_element
        transferred_bytes = len(cross_gpu) * bytes_per_activation * 2
        bandwidth_bytes_per_ms = self.network.bandwidth_gb_per_s * 1_000_000.0
        bandwidth_ms = transferred_bytes / bandwidth_bytes_per_ms
        if pair_counts:
            loads = np.array(list(pair_counts.values()), dtype=float)
            imbalance = float(loads.max() / loads.mean() - 1.0)
        else:
            imbalance = 0.0
        aggregate_communication_ms = (
            network_messages * self.network.fixed_message_latency_ms
            + bandwidth_ms
            + self.network.congestion_factor * imbalance * bandwidth_ms
        )
        all_to_all_calls = 2 if cross_gpu else 0

        per_gpu_messages = [0] * self.model.num_gpus
        per_gpu_network_bytes = [0] * self.model.num_gpus
        for (source_gpu, destination_gpu), token_count in pair_counts.items():
            endpoint_messages = 2 if self.network.aggregate_messages else 2 * token_count
            endpoint_bytes = token_count * bytes_per_activation * 2
            per_gpu_messages[source_gpu] += endpoint_messages
            per_gpu_messages[destination_gpu] += endpoint_messages
            per_gpu_network_bytes[source_gpu] += endpoint_bytes
            per_gpu_network_bytes[destination_gpu] += endpoint_bytes

        mean_endpoint_bytes = float(np.mean(per_gpu_network_bytes))
        if self.calibration is not None and self.calibration.network_curves:
            per_gpu_communication_ms = self.calibration.network_latencies_ms(
                collective="ep_dispatch_combine",
                active_ranks=self.model.num_gpus,
                message_counts=np.asarray(per_gpu_messages),
                transferred_bytes=np.asarray(per_gpu_network_bytes),
            ).tolist()
            aggregate_communication_ms = max(per_gpu_communication_ms, default=0.0)
        else:
            per_gpu_communication_ms = []
            for gpu_id in range(self.model.num_gpus):
                rank_bandwidth_ms = per_gpu_network_bytes[gpu_id] / bandwidth_bytes_per_ms
                rank_pressure = (
                    max(0.0, per_gpu_network_bytes[gpu_id] / mean_endpoint_bytes - 1.0)
                    if mean_endpoint_bytes > 0.0
                    else 0.0
                )
                per_gpu_communication_ms.append(
                    per_gpu_messages[gpu_id] * self.network.fixed_message_latency_ms
                    + rank_bandwidth_ms * (1.0 + self.network.congestion_factor * rank_pressure)
                )

        per_gpu_layer_ms = [
            attention_ms + per_gpu_expert_ms[gpu_id] + per_gpu_communication_ms[gpu_id]
            for gpu_id in range(self.model.num_gpus)
        ]
        rank_critical_path_ms = max(per_gpu_layer_ms, default=attention_ms)
        critical_gpu_id = int(np.argmax(per_gpu_layer_ms)) if per_gpu_layer_ms else 0
        communication_ms = (
            per_gpu_communication_ms[critical_gpu_id]
            if self.network.use_rank_critical_path
            else aggregate_communication_ms
        )
        elapsed_ms = (
            rank_critical_path_ms
            if self.network.use_rank_critical_path
            else attention_ms + aggregate_communication_ms + expert_compute_ms
        )
        rank_executions = [
            RankExecution(
                batch_id=batch_id,
                layer_id=layer_id,
                gpu_id=gpu_id,
                expert_compute_ms=per_gpu_expert_ms[gpu_id],
                communication_ms=per_gpu_communication_ms[gpu_id],
                layer_time_ms=per_gpu_layer_ms[gpu_id],
                unique_experts=per_gpu_unique_experts[gpu_id],
                expert_tokens=per_gpu_expert_tokens[gpu_id],
                expert_weight_bytes=per_gpu_weight_bytes[gpu_id],
                expert_weight_read_ms=per_gpu_weight_read_ms[gpu_id],
                network_messages=per_gpu_messages[gpu_id],
                network_endpoint_bytes=per_gpu_network_bytes[gpu_id],
                is_critical=gpu_id == critical_gpu_id,
            )
            for gpu_id in range(self.model.num_gpus)
        ]

        return LayerExecution(
            layer_id=layer_id,
            elapsed_ms=elapsed_ms,
            attention_ms=attention_ms,
            expert_compute_ms=expert_compute_ms,
            communication_ms=communication_ms,
            aggregate_communication_ms=aggregate_communication_ms,
            rank_critical_path_ms=rank_critical_path_ms,
            critical_gpu_id=critical_gpu_id,
            assignments=len(assignments),
            expert_batch_sizes=expert_batch_sizes,
            expert_invocations=len(expert_counts),
            network_messages=network_messages,
            transferred_bytes=transferred_bytes,
            all_to_all_calls=all_to_all_calls,
            per_gpu_expert_ms=per_gpu_expert_ms,
            expert_token_counts=[expert_counts.get(i, 0) for i in range(self.model.num_experts)],
            rank_executions=rank_executions,
            token_assignments=assignments,
        )
