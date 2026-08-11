#!/usr/bin/env python3
"""Collect actual routed-expert traces with vLLM external-launcher DP/EP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from importlib.metadata import version
from multiprocessing import Process
from pathlib import Path

import numpy as np
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
    parser.add_argument("--revision", required=True)
    parser.add_argument("--backend", choices=("fa3", "flashinfer"), required=True)
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    parser.add_argument("--data-parallel-size", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def attention_config(backend: str) -> dict[str, object]:
    if backend == "fa3":
        return {"backend": "FLASH_ATTN", "flash_attn_version": 3}
    return {"backend": "FLASHINFER"}


def open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def worker(
    *,
    rank: int,
    dp_size: int,
    master_port: int,
    output: Path,
    model: str,
    revision: str,
    backend: str,
    all2all_backend: str,
    seeds: list[int],
    max_tokens: int,
    max_model_len: int,
) -> None:
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    config = AutoConfig.from_pretrained(model, revision=revision)
    num_layers = int(config.num_hidden_layers)
    top_k = int(config.num_experts_per_tok)
    llm = LLM(
        model=model,
        revision=revision,
        tensor_parallel_size=1,
        data_parallel_size=dp_size,
        distributed_executor_backend="external_launcher",
        enable_expert_parallel=True,
        all2all_backend=all2all_backend,
        dtype="bfloat16",
        seed=seeds[0],
        max_model_len=max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_return_routed_experts=True,
        attention_config=attention_config(backend),
    )

    rows: list[dict[str, object]] = []
    shapes: list[dict[str, object]] = []
    for segment_id, seed in enumerate(seeds):
        sampling = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_tokens,
            ignore_eos=True,
            seed=seed,
            routed_experts_prompt_start=0,
        )
        request_output = llm.generate([PROMPTS[rank % len(PROMPTS)]], sampling)[0]
        routes = request_output.outputs[0].routed_experts
        if routes is None:
            raise RuntimeError("vLLM returned no routed experts")
        routes = np.asarray(routes)
        if routes.ndim != 3 or tuple(routes.shape[1:]) != (num_layers, top_k):
            raise RuntimeError(f"unexpected routed-expert shape: {routes.shape}")
        shapes.append({"segment_id": segment_id, "seed": seed, "shape": list(routes.shape)})
        request_id = segment_id * dp_size + rank
        for position_id in range(routes.shape[0]):
            for layer_id in range(num_layers):
                for route_slot, expert_id in enumerate(
                    routes[position_id, layer_id].tolist()
                ):
                    rows.append(
                        {
                            "segment_id": segment_id,
                            "seed": seed,
                            "dp_rank": rank,
                            "request_id": request_id,
                            "position_id": position_id,
                            "layer_id": layer_id,
                            "route_slot": route_slot,
                            "expert_id": int(expert_id),
                        }
                    )

    output.mkdir(parents=True, exist_ok=True)
    with (output / f"routes_rank{rank}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / f"rank{rank}_metadata.json").write_text(
        json.dumps({"rank": rank, "shapes": shapes}, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.data_parallel_size != 4:
        raise ValueError("this collector is frozen for DP=EP=4")
    args.output.mkdir(parents=True, exist_ok=True)
    master_port = open_port()
    started = time.perf_counter()
    processes = [
        Process(
            target=worker,
            kwargs={
                "rank": rank,
                "dp_size": args.data_parallel_size,
                "master_port": master_port,
                "output": args.output,
                "model": args.model,
                "revision": args.revision,
                "backend": args.backend,
                "all2all_backend": args.all2all_backend,
                "seeds": args.seeds,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
            },
        )
        for rank in range(args.data_parallel_size)
    ]
    for process in processes:
        process.start()
    failures: list[dict[str, int | None]] = []
    for rank, process in enumerate(processes):
        process.join(timeout=args.timeout)
        if process.exitcode is None:
            process.kill()
            failures.append({"rank": rank, "exitcode": None})
        elif process.exitcode != 0:
            failures.append({"rank": rank, "exitcode": process.exitcode})
    if failures:
        raise RuntimeError(f"DP workers failed: {failures}")

    combined: list[dict[str, str]] = []
    for rank in range(args.data_parallel_size):
        with (args.output / f"routes_rank{rank}.csv").open(newline="") as handle:
            combined.extend(csv.DictReader(handle))
    with (args.output / "routes_observational.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    metadata = {
        "trace_kind": "ar_decode_observational_ep4",
        "eligible_for_m2_native_replay": False,
        "model_identifier": args.model,
        "model_revision": args.revision,
        "num_layers": int(config.num_hidden_layers),
        "num_experts": int(config.num_experts),
        "top_k": int(config.num_experts_per_tok),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.moe_intermediate_size),
        "tensor_parallel_size": 1,
        "data_parallel_size": args.data_parallel_size,
        "expert_parallel_size": args.data_parallel_size,
        "backend": args.backend,
        "attention_config": attention_config(args.backend),
        "all2all_backend": args.all2all_backend,
        "seeds": args.seeds,
        "max_tokens": args.max_tokens,
        "elapsed_s_including_load": time.perf_counter() - started,
        "vllm_version": version("vllm"),
    }
    (args.output / "route_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"rows": len(combined), "metadata": metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
