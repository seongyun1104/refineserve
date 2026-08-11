#!/usr/bin/env python3
"""Collect observational AR routing traces from an actual vLLM MoE model.

These traces calibrate routing skew and temporal stability. They are deliberately
not emitted as an M2 ``native_position_parallel`` bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams

PROMPTS = (
    "Explain why load imbalance matters in a mixture-of-experts inference system.",
    "Write a concise description of GPU communication overhead during model serving.",
    "Compare latency and throughput when batching language model requests.",
    "Describe how a scheduler can reduce a distributed critical path.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--backend", choices=("fa3", "flashinfer"), default="fa3")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--additional-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--all2all-backend", default=None)
    return parser.parse_args()


def attention_config(backend: str) -> dict[str, object]:
    if backend == "fa3":
        return {"backend": "FLASH_ATTN", "flash_attn_version": 3}
    return {"backend": "FLASHINFER"}


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("token limits must be positive")
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    num_layers = int(config.num_hidden_layers)
    num_experts = int(config.num_experts)
    top_k = int(config.num_experts_per_tok)
    if args.data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    if args.all2all_backend is not None and not args.enable_expert_parallel:
        raise ValueError("all2all_backend requires expert parallelism")

    started = time.perf_counter()
    llm_kwargs = dict(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=1,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_seqs=len(PROMPTS),
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        enable_return_routed_experts=True,
        attention_config=attention_config(args.backend),
    )
    if args.all2all_backend is not None:
        llm_kwargs["all2all_backend"] = args.all2all_backend
    llm = LLM(**llm_kwargs)

    segments: list[tuple[int, list[object]]] = []
    for seed in [args.seed, *args.additional_seeds]:
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            seed=seed,
            routed_experts_prompt_start=0,
        )
        segments.append((seed, llm.generate(list(PROMPTS), sampling)))
    elapsed_s = time.perf_counter() - started

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    shapes: list[list[int]] = []
    shape_segments: list[dict[str, object]] = []
    for segment_id, (seed, outputs) in enumerate(segments):
        segment_shapes: list[list[int]] = []
        for request_index, request_output in enumerate(outputs):
            completion = request_output.outputs[0]
            routes = completion.routed_experts
            if routes is None:
                raise RuntimeError("vLLM returned no routed experts")
            routes = np.asarray(routes)
            expected_tail = (num_layers, top_k)
            if routes.ndim != 3 or tuple(routes.shape[1:]) != expected_tail:
                raise RuntimeError(
                    f"unexpected routed_experts shape {routes.shape}; expected (*, {expected_tail})"
                )
            segment_shapes.append(list(routes.shape))
            global_request_id = segment_id * len(PROMPTS) + request_index
            for position_id in range(routes.shape[0]):
                for layer_id in range(num_layers):
                    experts = routes[position_id, layer_id]
                    for route_slot, expert_id in enumerate(experts.tolist()):
                        rows.append(
                            {
                                "segment_id": segment_id,
                                "seed": seed,
                                "request_id": global_request_id,
                                "position_id": position_id,
                                "layer_id": layer_id,
                                "route_slot": route_slot,
                                "expert_id": int(expert_id),
                            }
                        )
        shapes.extend(segment_shapes)
        shape_segments.append({"segment_id": segment_id, "seed": seed, "shapes": segment_shapes})

    csv_path = args.output / "routes_observational.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "trace_kind": "ar_decode_observational",
        "eligible_for_m2_native_replay": False,
        "purpose": "routing_skew_and_temporal_stability_calibration",
        "created_at_unix": time.time(),
        "model_identifier": args.model,
        "model_revision": args.revision or "repository_default",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.moe_intermediate_size),
        "backend": args.backend,
        "attention_config": attention_config(args.backend),
        "tensor_parallel_size": 1,
        "data_parallel_size": args.data_parallel_size,
        "expert_parallel_size": (
            args.data_parallel_size if args.enable_expert_parallel else 1
        ),
        "enable_expert_parallel": args.enable_expert_parallel,
        "all2all_backend": args.all2all_backend,
        "seed": args.seed,
        "seeds": [args.seed, *args.additional_seeds],
        "prompt_count": len(PROMPTS),
        "max_tokens": args.max_tokens,
        "route_shapes": shapes,
        "route_shape_segments": shape_segments,
        "elapsed_s_including_load": elapsed_s,
        "gpu_model": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": version("vllm"),
    }
    metadata_path = args.output / "route_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"csv": str(csv_path), "metadata": metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
