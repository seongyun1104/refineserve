#!/usr/bin/env python3
"""Collect native block-position router IDs from stock LLaDA2.0-mini on one GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

GENERAL_TOPICS = (
    "distributed systems",
    "GPU kernels",
    "network congestion",
    "mathematical proofs",
    "Python debugging",
    "database indexing",
    "climate science",
    "molecular biology",
    "economic history",
    "robot motion planning",
    "compiler optimization",
    "cryptographic protocols",
    "medical imaging",
    "music composition",
    "formal logic",
    "space exploration",
    "energy storage",
    "language acquisition",
    "graph algorithms",
    "probability theory",
    "operating systems",
    "computer architecture",
    "numerical analysis",
    "software testing",
    "parallel programming",
    "information retrieval",
    "control theory",
    "computer vision",
    "natural language processing",
    "scientific reproducibility",
    "queueing theory",
    "memory hierarchy",
    "fault tolerance",
    "statistics",
    "game theory",
    "geometry",
    "education policy",
    "supply chains",
    "signal processing",
    "astronomy",
)
GENERAL_TASKS = (
    "Explain the central trade-off in {topic} with a concrete example.",
    "Compare two common approaches to {topic} and state when each is preferable.",
    "Write a concise technical checklist for evaluating a claim about {topic}.",
    "Identify a subtle failure mode in {topic} and propose a controlled experiment.",
)
PROMPT_SUFFIX = (
    " Give a technically precise answer. State the assumptions, identify the main "
    "mechanism, distinguish correlation from causation, and include a controlled "
    "comparison with a plausible counterexample. Explain which quantities should be "
    "measured, which variables must remain fixed, what outcome would falsify the "
    "claim, and how the conclusion changes under a different workload or hardware "
    "constraint. End with a short reproducibility checklist covering inputs, random "
    "seeds, versions, raw observations, and uncertainty."
)

REASONING_TOPICS = (
    "conditional probability",
    "invariants",
    "graph connectivity",
    "number theory",
    "combinatorics",
    "geometric construction",
    "recurrence relations",
    "optimization under constraints",
)
REASONING_TASKS = (
    "Solve a multi-step problem about {topic} and verify every inference.",
    "Construct a counterexample involving {topic}, then explain why it works.",
    "Compare two proof strategies for {topic} and complete the stronger one.",
    "Derive a quantitative bound for {topic} and test it on an edge case.",
)
CODE_TOPICS = (
    "concurrent queues",
    "graph traversal",
    "dynamic programming",
    "database transactions",
    "stream parsing",
    "memory allocation",
    "distributed retries",
    "numerical kernels",
)
CODE_TASKS = (
    "Write Python code for {topic}, state its complexity, and test edge cases.",
    "Debug a plausible implementation of {topic} and explain the root cause.",
    "Design an API for {topic} and provide deterministic unit tests.",
    "Implement two approaches to {topic} and compare their failure modes.",
)

POSITION_ROLE_PREFIX = 0
POSITION_ROLE_CURRENT_BLOCK_FINALIZED = 1
POSITION_ROLE_CURRENT_BLOCK_MASKED = 2
POSITION_ROLE_NAMES = {
    POSITION_ROLE_PREFIX: "prefix",
    POSITION_ROLE_CURRENT_BLOCK_FINALIZED: "current_block_finalized",
    POSITION_ROLE_CURRENT_BLOCK_MASKED: "current_block_masked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="inclusionAI/LLaDA2.0-mini")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--active-positions", type=int, nargs="+", default=[1, 16, 32, 64])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["reasoning", "code", "general"],
        default=["reasoning", "code", "general"],
    )
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--mask-id", type=int, default=156895)
    parser.add_argument("--denoising-block-width", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=32)
    parser.add_argument("--denoising-threshold", type=float, default=0.95)
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument(
        "--include-initial-width-ablation",
        action="store_true",
        help="Also capture the controlled initial K sweep; native M2.1 does not require it.",
    )
    return parser.parse_args()


def prompts(workload: str) -> list[str]:
    if workload == "reasoning":
        topics, tasks = REASONING_TOPICS, REASONING_TASKS
    elif workload == "code":
        topics, tasks = CODE_TOPICS, CODE_TASKS
    elif workload == "general":
        topics, tasks = GENERAL_TOPICS, GENERAL_TASKS
    else:
        raise ValueError(f"unsupported workload: {workload}")
    return [
        template.format(topic=topic) + PROMPT_SUFFIX
        for topic in topics
        for template in tasks
    ]


def block_attention_mask(
    prefix_length: int,
    block_width: int,
    device: object,
    batch_size: int = 1,
) -> object:
    """Reproduce the checkpoint's block-diagonal causal mask for one new block."""
    import torch

    if prefix_length % block_width:
        raise ValueError("prefix length must be divisible by every block width")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    num_blocks = prefix_length // block_width + 1
    total = prefix_length + block_width
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    return (
        block_mask.repeat_interleave(block_width, dim=0)
        .repeat_interleave(block_width, dim=1)[:total, :total]
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .log()
        .to(torch.bfloat16)
    )


def extract_routes(entry: object, routed_scaling_factor: float) -> tuple[object, object]:
    import torch

    if not isinstance(entry, (tuple, list)) or len(entry) < 2:
        raise RuntimeError("unexpected LLaDA2 router output; expected (logits, topk_ids)")
    topk = entry[1]
    if not isinstance(topk, torch.Tensor) or topk.ndim != 3:
        raise RuntimeError(f"unexpected top-k route tensor: {type(topk)}")
    logits = entry[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError(f"unexpected router-logit tensor: {type(logits)}")
    scores = torch.sigmoid(logits.float())
    selected = torch.gather(scores, dim=-1, index=topk)
    weights = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return topk, weights


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLaDA2 router collection requires a CUDA GPU")
    if args.requests != 32:
        raise ValueError("screening contract currently requires exactly 32 requests")
    if args.microbatch_size <= 0 or args.requests % args.microbatch_size:
        raise ValueError("microbatch size must be positive and divide requests")
    if any(args.prefix_length % width for width in args.active_positions):
        raise ValueError(
            "prefix length must be divisible by every active-position/block width"
        )
    if args.prefix_length % args.denoising_block_width:
        raise ValueError("prefix length must align with the denoising block width")
    if args.denoising_steps <= 0:
        raise ValueError("denoising steps must be positive")
    if args.generation_length <= 0 or (
        args.generation_length % args.denoising_block_width
    ):
        raise ValueError("generation length must be a positive multiple of block width")
    device = torch.device("cuda:0")
    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    prepared_segments: list[tuple[int, int, str, list[str], object]] = []
    segment_id = 0
    for workload in args.workloads:
        prompt_pool = prompts(workload)
        if len(prompt_pool) < args.requests:
            raise ValueError(f"{workload} prompt pool is smaller than --requests")
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            selected_prompts = [
                prompt_pool[index]
                for index in rng.choice(
                    len(prompt_pool), size=args.requests, replace=False
                ).tolist()
            ]
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in selected_prompts
            ]
            encoded = tokenizer(
                formatted_prompts,
                padding="max_length",
                truncation=True,
                max_length=args.prefix_length,
                return_tensors="pt",
            )
            if not bool(encoded.attention_mask.all()):
                raise ValueError(
                    "prompt suite did not fill the aligned prefix; increase prompt text "
                    "instead of treating padding as native clean-prefix content"
                )
            prepared_segments.append(
                (segment_id, seed, workload, formatted_prompts, encoded)
            )
            segment_id += 1
    started = time.perf_counter()
    causal_lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    base_model = causal_lm.model
    sparse_start = int(config.first_k_dense_replace)
    sparse_layers = int(config.num_hidden_layers) - sparse_start
    top_k_routes = int(config.num_experts_per_tok)
    routed_scaling_factor = float(config.routed_scaling_factor)
    route_arrays: dict[str, np.ndarray] = {}
    route_weight_arrays: dict[str, np.ndarray] = {}
    position_role_arrays: dict[str, np.ndarray] = {}
    observation_rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []
    all_denoising_results: list[dict[str, object]] = []
    with torch.inference_mode():
        for segment_id, seed, workload, formatted_prompts, encoded in prepared_segments:
            prefix_ids = encoded.input_ids.to(device)
            for request_id, (prompt, token_ids) in enumerate(
                zip(formatted_prompts, encoded.input_ids.tolist(), strict=True)
            ):
                prompt_rows.append(
                    {
                        "segment_id": segment_id,
                        "seed": seed,
                        "workload_class": workload,
                        "request_id": request_id,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "input_ids_sha256": hashlib.sha256(
                            np.asarray(token_ids, dtype=np.int64).tobytes()
                        ).hexdigest(),
                    }
                )
            for block_width in (
                args.active_positions if args.include_initial_width_ablation else []
            ):
                active_positions = block_width
                model_forward_positions = args.prefix_length + active_positions
                initial_routes = np.empty(
                    (
                        args.requests,
                        sparse_layers,
                        model_forward_positions,
                        top_k_routes,
                    ),
                    dtype=np.uint16,
                )
                initial_weights = np.empty(initial_routes.shape, dtype=np.float16)
                for request_start in range(0, args.requests, args.microbatch_size):
                    request_end = request_start + args.microbatch_size
                    microbatch_prefix = prefix_ids[request_start:request_end]
                    masked = torch.full(
                        (args.microbatch_size, active_positions),
                        args.mask_id,
                        dtype=torch.long,
                        device=device,
                    )
                    input_ids = torch.cat([microbatch_prefix, masked], dim=1)
                    position_ids = (
                        torch.arange(input_ids.shape[1], device=device)
                        .unsqueeze(0)
                        .expand(args.microbatch_size, -1)
                    )
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=block_attention_mask(
                            args.prefix_length,
                            block_width,
                            device,
                            batch_size=args.microbatch_size,
                        ),
                        position_ids=position_ids,
                        use_cache=False,
                        output_router_logits=True,
                        return_dict=True,
                    )
                    router_outputs = outputs.router_logits
                    if router_outputs is None:
                        raise RuntimeError("stock model returned no router outputs")
                    if len(router_outputs) != sparse_layers:
                        raise RuntimeError(
                            f"unexpected sparse layer count: {len(router_outputs)}"
                        )
                    for sparse_offset, entry in enumerate(router_outputs):
                        topk, weights = extract_routes(entry, routed_scaling_factor)
                        topk = topk.cpu().numpy()
                        initial_routes[
                            request_start:request_end,
                            sparse_offset,
                        ] = topk.astype(np.uint16, copy=False)
                        initial_weights[
                            request_start:request_end,
                            sparse_offset,
                        ] = weights.cpu().numpy().astype(np.float16, copy=False)
                    del outputs
                array_key = f"initial_s{segment_id}_k{active_positions}"
                route_arrays[array_key] = initial_routes
                route_weight_arrays[array_key] = initial_weights
                initial_roles = np.full(
                    (args.requests, model_forward_positions),
                    POSITION_ROLE_PREFIX,
                    dtype=np.uint8,
                )
                initial_roles[:, -active_positions:] = POSITION_ROLE_CURRENT_BLOCK_MASKED
                position_role_arrays[array_key] = initial_roles
                observation_rows.append(
                    {
                        "trace_phase": "initial_width_ablation",
                        "segment_id": segment_id,
                        "seed": seed,
                        "workload_class": workload,
                        "active_positions": active_positions,
                        "block_width": block_width,
                        "block_id": 0,
                        "denoise_step": 0,
                        "position_width_source": "initial_width_ablation_controlled",
                        "model_forward_positions": model_forward_positions,
                        "iteration": 0,
                        "array_key": array_key,
                        "request_ids": json.dumps(list(range(args.requests))),
                        "masked_positions_before_step": json.dumps(
                            [active_positions] * args.requests
                        ),
                        "masked_positions_after_step": "",
                        "finalized_positions_this_step": "",
                    }
                )
            denoising_results: list[dict[str, object]] = []
            block_width = args.denoising_block_width
            transfer_schedule = causal_lm._get_num_transfer_tokens(
                block_width,
                args.denoising_steps,
            )
            generation_blocks = args.generation_length // block_width
            for request_start in range(0, args.requests, args.microbatch_size):
                request_end = request_start + args.microbatch_size
                microbatch_prefix = prefix_ids[request_start:request_end]
                generated_prefix = torch.empty(
                    (args.microbatch_size, 0), dtype=torch.long, device=device
                )
                torch.manual_seed(seed * 1000 + request_start)
                for block_id in range(generation_blocks):
                    block_tokens = torch.full(
                        (args.microbatch_size, block_width),
                        args.mask_id,
                        dtype=torch.long,
                        device=device,
                    )
                    clean_prefix = torch.cat([microbatch_prefix, generated_prefix], dim=1)
                    for step in range(args.denoising_steps):
                        unfinished = (block_tokens == args.mask_id).any(dim=1)
                        active_local = torch.nonzero(
                            unfinished, as_tuple=False
                        ).flatten()
                        if active_local.numel() == 0:
                            break
                        active_prefix = clean_prefix[active_local]
                        active_block = block_tokens[active_local]
                        input_ids = torch.cat([active_prefix, active_block], dim=1)
                        position_ids = (
                            torch.arange(input_ids.shape[1], device=device)
                            .unsqueeze(0)
                            .expand(len(active_local), -1)
                        )
                        outputs = causal_lm(
                            input_ids=input_ids,
                            attention_mask=block_attention_mask(
                                active_prefix.shape[1],
                                block_width,
                                device,
                                batch_size=len(active_local),
                            ),
                            position_ids=position_ids,
                            use_cache=False,
                            output_router_logits=True,
                            return_dict=True,
                        )
                        router_outputs = outputs.router_logits
                        if router_outputs is None:
                            raise RuntimeError("stock model returned no router outputs")
                        active_logits = outputs.logits[:, -block_width:, :]
                        sampled, sampled_probability = (
                            causal_lm._sample_with_temperature_topk_topp(
                                active_logits,
                                temperature=0.0,
                                top_k=None,
                                top_p=None,
                            )
                        )
                        active_mask = active_block == args.mask_id
                        transfer_index = torch.zeros_like(active_mask)
                        num_to_transfer = int(transfer_schedule[step].item())
                        masked_before = active_mask.sum(dim=1)
                        for row_index in range(len(active_local)):
                            confidence = torch.where(
                                active_mask[row_index],
                                sampled_probability[row_index],
                                -torch.inf,
                            )
                            high_confidence = confidence > args.denoising_threshold
                            if int(high_confidence.sum().item()) >= num_to_transfer:
                                transfer_index[row_index] = high_confidence
                            else:
                                transfer_count = min(
                                    num_to_transfer,
                                    int(active_mask[row_index].sum().item()),
                                )
                                if transfer_count:
                                    _, selected = torch.topk(
                                        confidence,
                                        k=transfer_count,
                                    )
                                    transfer_index[row_index, selected] = True
                        updated_block = active_block.clone()
                        updated_block[transfer_index] = sampled[transfer_index]
                        masked_after = (updated_block == args.mask_id).sum(dim=1)
                        finalized = masked_before - masked_after
                        block_tokens[active_local] = updated_block
                        if len(router_outputs) != sparse_layers:
                            raise RuntimeError(
                                f"unexpected sparse layer count: {len(router_outputs)}"
                            )
                        step_routes = np.empty(
                            (
                                len(active_local),
                                sparse_layers,
                                input_ids.shape[1],
                                top_k_routes,
                            ),
                            dtype=np.uint16,
                        )
                        step_weights = np.empty(step_routes.shape, dtype=np.float16)
                        for sparse_offset, entry in enumerate(router_outputs):
                            topk, weights = extract_routes(
                                entry, routed_scaling_factor
                            )
                            step_routes[:, sparse_offset] = topk.cpu().numpy().astype(
                                np.uint16, copy=False
                            )
                            step_weights[:, sparse_offset] = (
                                weights.cpu().numpy().astype(np.float16, copy=False)
                            )
                        array_key = (
                            f"denoise_s{segment_id}_mb{request_start}_"
                            f"block{block_id}_step{step}"
                        )
                        route_arrays[array_key] = step_routes
                        route_weight_arrays[array_key] = step_weights
                        roles = np.full(
                            (len(active_local), input_ids.shape[1]),
                            POSITION_ROLE_PREFIX,
                            dtype=np.uint8,
                        )
                        before_mask_np = active_mask.cpu().numpy()
                        current_roles = np.where(
                            before_mask_np,
                            POSITION_ROLE_CURRENT_BLOCK_MASKED,
                            POSITION_ROLE_CURRENT_BLOCK_FINALIZED,
                        ).astype(np.uint8)
                        roles[:, -block_width:] = current_roles
                        position_role_arrays[array_key] = roles
                        request_ids = [
                            request_start + int(value.item()) for value in active_local
                        ]
                        global_iteration = block_id * args.denoising_steps + step
                        observation_rows.append(
                            {
                                "trace_phase": "native_denoising",
                                "segment_id": segment_id,
                                "seed": seed,
                                "workload_class": workload,
                                "active_positions": input_ids.shape[1],
                                "block_width": block_width,
                                "block_id": block_id,
                                "denoise_step": step,
                                "position_width_source": (
                                    "native_trajectory_full_window_compute"
                                ),
                                "model_forward_positions": input_ids.shape[1],
                                "iteration": global_iteration,
                                "array_key": array_key,
                                "request_ids": json.dumps(request_ids),
                                "masked_positions_before_step": json.dumps(
                                    [int(value) for value in masked_before.tolist()]
                                ),
                                "masked_positions_after_step": json.dumps(
                                    [int(value) for value in masked_after.tolist()]
                                ),
                                "finalized_positions_this_step": json.dumps(
                                    [int(value) for value in finalized.tolist()]
                                ),
                            }
                        )
                        del outputs
                    if bool((block_tokens == args.mask_id).any()):
                        raise RuntimeError(
                            f"native denoising left masks in block {block_id}"
                        )
                    generated_prefix = torch.cat(
                        [generated_prefix, block_tokens], dim=1
                    )
                for local_request in range(args.microbatch_size):
                    request_id = request_start + local_request
                    output_ids = generated_prefix[local_request].cpu().numpy()
                    denoising_results.append(
                        {
                            "segment_id": segment_id,
                            "seed": seed,
                            "workload_class": workload,
                            "request_id": request_id,
                            "block_width": block_width,
                            "generation_length": args.generation_length,
                            "remaining_masks": int(
                                (output_ids == args.mask_id).sum()
                            ),
                            "output_ids_sha256": hashlib.sha256(
                                output_ids.astype(np.int64).tobytes()
                            ).hexdigest(),
                        }
                    )
            if any(result["remaining_masks"] for result in denoising_results):
                raise RuntimeError("native denoising trace left unfinished mask tokens")
            all_denoising_results.extend(denoising_results)
    args.output.mkdir(parents=True, exist_ok=True)
    routes_path = args.output / "routes_dense.npz"
    route_weights_path = args.output / "route_weights_dense.npz"
    position_roles_path = args.output / "position_roles_dense.npz"
    observations_path = args.output / "route_observations.csv"
    prompts_path = args.output / "prompt_manifest.csv"
    denoising_results_path = args.output / "denoising_results.csv"
    np.savez_compressed(routes_path, **route_arrays)
    np.savez_compressed(route_weights_path, **route_weight_arrays)
    np.savez_compressed(position_roles_path, **position_role_arrays)
    with observations_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observation_rows[0]))
        writer.writeheader()
        writer.writerows(observation_rows)
    with prompts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_rows[0]))
        writer.writeheader()
        writer.writerows(prompt_rows)
    with denoising_results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_denoising_results[0]))
        writer.writeheader()
        writer.writerows(all_denoising_results)
    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "trace_kind": "native_llada2_dense_router_observational",
        "evidence_class": "MEASURED_ROUTE_SUPPORTING_CALIBRATION",
        "eligible_for_scheduler_opportunity_screening": True,
        "eligible_for_gate3_timing_authorization": False,
        "eligible_for_full_native_replay": False,
        "model_identifier": args.model,
        "model_revision": args.revision,
        "seeds": args.seeds,
        "workloads": args.workloads,
        "segments_per_workload": len(args.seeds),
        "requests_per_segment": args.requests,
        "inference_microbatch_size": args.microbatch_size,
        "prefix_length": args.prefix_length,
        "active_positions": args.active_positions,
        "block_widths": args.active_positions,
        "block_width_equals_active_positions": False,
        "native_semantics_schema_version": 1,
        "route_storage": {
            "format": "compressed_dense_npz_plus_observation_manifest",
            "array_shape": (
                "[requests_in_observation, sparse_layers, model_forward_positions, "
                "top_k]"
            ),
            "dtype": "uint16",
        },
        "route_weight_storage": {
            "format": "compressed_dense_npz",
            "array_shape": (
                "[requests_in_observation, sparse_layers, model_forward_positions, "
                "top_k]"
            ),
            "dtype": "float16",
            "derivation": (
                "sigmoid(router_logits), gather selected IDs, normalize, then apply "
                "routed_scaling_factor"
            ),
        },
        "position_role_storage": {
            "format": "compressed_dense_npz",
            "array_shape": "[requests_in_observation, model_forward_positions]",
            "dtype": "uint8",
            "codes": {str(key): value for key, value in POSITION_ROLE_NAMES.items()},
            "state_time": "immediately_before the routed model forward",
        },
        "finalized_positions_per_step": "derivable_from_observation_manifest",
        "position_width_semantics": {
            "initial_width_ablation": (
                "active_positions is a controlled new-block width"
            ),
            "native_denoising": (
                "block width is fixed at 32 in the stock loop; masked positions vary "
                "with confidence, while every unfinished request recomputes the full "
                "prefix+block model_forward_positions"
            ),
        },
        "order_policy": "phase_specific_initial_or_stock_confidence_threshold",
        "native_denoising_capture": {
            "block_width": args.denoising_block_width,
            "generation_length": args.generation_length,
            "generation_blocks": args.generation_length // args.denoising_block_width,
            "maximum_steps": args.denoising_steps,
            "threshold": args.denoising_threshold,
            "temperature": 0.0,
            "top_k_sampling": None,
            "top_p_sampling": None,
            "compute_width_semantics": (
                "the full block executes every unfinished request step; remaining "
                "masked positions are recorded separately"
            ),
        },
        "num_layers": int(config.num_hidden_layers),
        "first_sparse_layer": int(config.first_k_dense_replace),
        "num_experts": int(config.num_experts),
        "top_k": int(config.num_experts_per_tok),
        "mask_id": args.mask_id,
        "attention_mask_semantics": (
            "official checkpoint block-diagonal causal mask; each observation "
            "contains an aligned clean prefix and one fully masked block"
        ),
        "routing_weights_captured": True,
        "routed_scaling_factor": routed_scaling_factor,
        "hypothetical_ep_mapping_required_for_screening": True,
        "artifact_sha256": {
            "routes_dense.npz": file_sha256(routes_path),
            "route_weights_dense.npz": file_sha256(route_weights_path),
            "position_roles_dense.npz": file_sha256(position_roles_path),
            "route_observations.csv": file_sha256(observations_path),
            "prompt_manifest.csv": file_sha256(prompts_path),
            "denoising_results.csv": file_sha256(denoising_results_path),
        },
        "gpu_model": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "torch_cuda_version": torch.version.cuda,
        "elapsed_s_including_load": time.perf_counter() - started,
        "semantic_limit": (
            "captures optional initial width ablations and stock-semantics multi-block "
            "denoising routing; "
            "it does not measure EP timing or establish task quality"
        ),
    }
    (args.output / "route_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "observations": len(observation_rows),
                "route_arrays": len(route_arrays),
                "metadata": metadata,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
