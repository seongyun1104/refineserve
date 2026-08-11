#!/usr/bin/env python3
"""Audit provisional EP accessibility without hiding accounting choices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

MLP_MATMULS = 3
FLOPS_PER_MATMUL_ELEMENT = 2
MLP_FLOP_FACTOR = MLP_MATMULS * FLOPS_PER_MATMUL_ELEMENT

PROFILES = {
    "controlled_toy": {
        "sparse_layers": 8,
        "experts": 16,
        "top_k": 2,
        "hidden": 2048,
        "moe_intermediate": 8192,
        "shared_experts": 0,
    },
    "llada2_mini": {
        "sparse_layers": 19,
        "experts": 256,
        "top_k": 8,
        "hidden": 2048,
        "moe_intermediate": 512,
        "shared_experts": 1,
        "attention_layers": 20,
        "dense_layers": 1,
        "dense_intermediate": 5120,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 157184,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-positions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--effective-tflops", type=float, default=500.0)
    parser.add_argument("--effective-bandwidth-gbps", type=float, default=400.0)
    parser.add_argument("--collective-fixed-us", type=float, default=8.0)
    parser.add_argument("--target-percent", type=float, default=4.0)
    return parser.parse_args()


def expected_distinct_destination_ranks(
    *, experts: int, top_k: int, world_size: int
) -> float:
    experts_per_rank = experts // world_size
    probability_rank_absent = math.comb(
        experts - experts_per_rank, top_k
    ) / math.comb(experts, top_k)
    return world_size * (1.0 - probability_rank_absent)


def compute_ms(flops_global: float, world_size: int, tflops: float) -> float:
    return flops_global / world_size / (tflops * 1e12) * 1000.0


def expert_flops(
    *,
    global_tokens_per_batch: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    shared_experts: int,
    layers: int,
    batches: int,
) -> float:
    routed = (
        global_tokens_per_batch
        * top_k
        * MLP_FLOP_FACTOR
        * hidden
        * intermediate
        * batches
        * layers
    )
    shared = (
        global_tokens_per_batch
        * shared_experts
        * MLP_FLOP_FACTOR
        * hidden
        * intermediate
        * batches
        * layers
    )
    return float(routed + shared)


def assignment_bytes_per_token(profile: dict[str, int]) -> float:
    hidden = profile["hidden"]
    return float(profile["top_k"] * (2 * hidden * 2 + 4))


def coalesced_bytes_per_token(
    profile: dict[str, int], expected_destinations: float
) -> float:
    # Hidden dispatch and partial-result combine occur once per token/destination.
    # Every assignment still carries one int32 local expert ID and one FP32 weight.
    return float(
        expected_destinations * (2 * profile["hidden"] * 2)
        + profile["top_k"] * (4 + 4)
    )


def full_iteration_non_ep_flops(
    *, profile: dict[str, int], sequence_length: int, global_requests: int, batches: int
) -> dict[str, float]:
    hidden = profile["hidden"]
    heads = profile["num_attention_heads"]
    kv_heads = profile["num_key_value_heads"]
    head_dim = profile["head_dim"]
    qkv_width = (heads + 2 * kv_heads) * head_dim
    attention_projection = (
        2
        * sequence_length
        * hidden
        * (qkv_width + hidden)
        * profile["attention_layers"]
        * global_requests
        * batches
    )
    attention_scores = (
        4
        * sequence_length
        * sequence_length
        * heads
        * head_dim
        * profile["attention_layers"]
        * global_requests
        * batches
    )
    dense_mlp = (
        MLP_FLOP_FACTOR
        * sequence_length
        * hidden
        * profile["dense_intermediate"]
        * profile["dense_layers"]
        * global_requests
        * batches
    )
    router = (
        2
        * sequence_length
        * hidden
        * profile["experts"]
        * profile["sparse_layers"]
        * global_requests
        * batches
    )
    lm_head = (
        2
        * sequence_length
        * hidden
        * profile["vocab_size"]
        * global_requests
        * batches
    )
    return {
        "attention_projection_flops_global": float(attention_projection),
        "attention_score_value_flops_global": float(attention_scores),
        "dense_first_layer_flops_global": float(dense_mlp),
        "router_flops_global": float(router),
        "lm_head_flops_global": float(lm_head),
    }


def main() -> None:
    args = parse_args()
    if args.world_size != 4:
        raise ValueError("current profile comparison is defined for EP=4")
    records: list[dict[str, object]] = []
    global_requests = args.world_size * args.batch_size
    for profile_name, profile in PROFILES.items():
        if profile["experts"] % args.world_size:
            raise ValueError("profile experts must divide evenly across ranks")
        expected_destinations = expected_distinct_destination_ranks(
            experts=profile["experts"],
            top_k=profile["top_k"],
            world_size=args.world_size,
        )
        communication_paths = (
            ("assignment_granular", assignment_bytes_per_token(profile)),
            (
                "destination_coalesced",
                coalesced_bytes_per_token(profile, expected_destinations),
            ),
        )
        denominator_scopes = [("ep_only", False)]
        if profile_name == "llada2_mini":
            denominator_scopes.append(("full_iteration", True))
        for positions in args.active_positions:
            for communication_path, bytes_per_token in communication_paths:
                for denominator_scope, full_iteration in denominator_scopes:
                    compute_positions = (
                        args.prefix_length + positions if full_iteration else positions
                    )
                    global_tokens_per_batch = global_requests * compute_positions
                    expert_flops_global = expert_flops(
                        global_tokens_per_batch=global_tokens_per_batch,
                        top_k=profile["top_k"],
                        hidden=profile["hidden"],
                        intermediate=profile["moe_intermediate"],
                        shared_experts=profile["shared_experts"],
                        layers=profile["sparse_layers"],
                        batches=args.batches,
                    )
                    expert_compute_ms = compute_ms(
                        expert_flops_global,
                        args.world_size,
                        args.effective_tflops,
                    )
                    non_ep_flops: dict[str, float] = {}
                    if full_iteration:
                        non_ep_flops = full_iteration_non_ep_flops(
                            profile=profile,
                            sequence_length=compute_positions,
                            global_requests=global_requests,
                            batches=args.batches,
                        )
                    non_ep_compute_ms = sum(
                        compute_ms(value, args.world_size, args.effective_tflops)
                        for value in non_ep_flops.values()
                    )
                    logical_bytes = (
                        global_tokens_per_batch
                        * bytes_per_token
                        * args.batches
                        * profile["sparse_layers"]
                    )
                    cross_bytes = logical_bytes * (args.world_size - 1) / args.world_size
                    payload_ms = (
                        cross_bytes
                        / (args.effective_bandwidth_gbps * 1e9)
                        * 1000.0
                    )
                    fixed_ms = (
                        3
                        * args.batches
                        * profile["sparse_layers"]
                        * args.collective_fixed_us
                        / 1000.0
                    )
                    total_ms = (
                        expert_compute_ms + non_ep_compute_ms + payload_ms + fixed_ms
                    )
                    accessible_fraction = payload_ms / total_ms
                    records.append(
                        {
                            "profile": profile_name,
                            "communication_path": communication_path,
                            "denominator_scope": denominator_scope,
                            "active_positions": positions,
                            "model_forward_positions": compute_positions,
                            "prefix_length": args.prefix_length,
                            **profile,
                            "expert_mlp_matmuls": MLP_MATMULS,
                            "expert_mlp_flop_factor": MLP_FLOP_FACTOR,
                            "assumed_effective_tflops": args.effective_tflops,
                            "assumed_effective_bandwidth_gbps": (
                                args.effective_bandwidth_gbps
                            ),
                            "assumed_collective_fixed_us": (
                                args.collective_fixed_us
                            ),
                            "expert_compute_ms": expert_compute_ms,
                            "non_ep_compute_ms": non_ep_compute_ms,
                            "payload_ms": payload_ms,
                            "data_collective_fixed_ms": fixed_ms,
                            "estimated_total_ms": total_ms,
                            "provisional_accessible_fraction": accessible_fraction,
                            "required_realized_reduction_fraction": (
                                args.target_percent / 100.0 / accessible_fraction
                            ),
                            "expected_distinct_destination_ranks_uniform": (
                                expected_destinations
                            ),
                            "expected_assignment_to_destination_coalescing_fraction": (
                                1.0 - expected_destinations / profile["top_k"]
                            ),
                            **non_ep_flops,
                        }
                    )
    result = pd.DataFrame.from_records(records)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "shape_accessibility.csv", index=False)
    metadata = {
        "accounting_revision": "shape_accessibility_v2_explicit_paths_and_denominators",
        "evidence_class": "PROVISIONAL_ROOFLINE_SENSITIVITY",
        "timing_gate_eligible": False,
        "target_percent": args.target_percent,
        "expert_mlp_definition": {
            "matmuls": MLP_MATMULS,
            "names": ["gate_proj", "up_proj", "down_proj"],
            "flops_per_matmul_element": FLOPS_PER_MATMUL_ELEMENT,
            "flop_factor": MLP_FLOP_FACTOR,
        },
        "denominator_definitions": {
            "ep_only": (
                "routed/shared sparse-expert MLP compute + EP payload + three "
                "data-collective fixed costs for the declared position width"
            ),
            "full_iteration": (
                "all prefix+block positions through sparse experts/EP plus 20-layer "
                "attention projections and score/value matmuls, first dense SwiGLU, "
                "FP32 routers, and LM head; norms/softmax/activations are omitted"
            ),
        },
        "communication_path_definitions": {
            "assignment_granular": (
                "one hidden dispatch and one hidden return per expert assignment; "
                "one int32 local expert ID per assignment"
            ),
            "destination_coalesced": (
                "one hidden dispatch and one partial return per token/destination; "
                "one int32 local expert ID and one FP32 weight per assignment"
            ),
        },
        "self_traffic_policy": "cross-rank bytes = logical bytes * (R-1)/R",
        "collective_fixed_cost_policy": (
            "three data collectives per sparse layer for both communication paths"
        ),
        "changelog": [
            (
                "v2 makes the three-matmul SwiGLU factor explicit. Earlier reviewer "
                "recalculations that used a two-matmul factor understated toy compute "
                "by 1.5x; the generated v1 artifact already used factor six but did "
                "not declare it in metadata."
            ),
            (
                "v2 adds assignment/coalesced and EP-only/full-iteration rows instead "
                "of using assignment-granular EP-only accessibility as a native-model "
                "timing threshold."
            ),
        ],
        "warning": (
            "These are not measured native accessibility values. Full-iteration FLOPs "
            "use one common effective-TFLOPS assumption and omit non-matmul kernels; "
            "actual traces, kernels, packing, and grouped-P2P behavior must be measured."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        result[
            [
                "profile",
                "communication_path",
                "denominator_scope",
                "active_positions",
                "model_forward_positions",
                "expert_compute_ms",
                "non_ep_compute_ms",
                "payload_ms",
                "data_collective_fixed_ms",
                "provisional_accessible_fraction",
                "required_realized_reduction_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
