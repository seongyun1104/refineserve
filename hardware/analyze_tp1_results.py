#!/usr/bin/env python3
"""Validate and summarize the H100 TP=1 hardware artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def route_sets(frame: pd.DataFrame) -> dict[tuple[int, int, int], frozenset[int]]:
    return {
        tuple(int(value) for value in key): frozenset(group["expert_id"].astype(int))
        for key, group in frame.groupby(
            ["request_id", "position_id", "layer_id"], sort=True
        )
    }


def validate_routes(frame: pd.DataFrame, metadata: dict[str, object]) -> None:
    required = {"request_id", "position_id", "layer_id", "route_slot", "expert_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"route CSV missing columns: {sorted(missing)}")
    top_k = int(metadata["top_k"])
    num_layers = int(metadata["num_layers"])
    num_experts = int(metadata["num_experts"])
    if frame.duplicated(list(required - {"expert_id"})).any():
        raise ValueError("route CSV contains duplicate route slots")
    sizes = frame.groupby(["request_id", "position_id", "layer_id"]).size()
    if not (sizes == top_k).all():
        raise ValueError("every position/layer must contain exactly top_k rows")
    unique_sizes = frame.groupby(["request_id", "position_id", "layer_id"])[
        "expert_id"
    ].nunique()
    if not (unique_sizes == top_k).all():
        raise ValueError("a position/layer contains duplicate expert IDs")
    if frame["layer_id"].min() < 0 or frame["layer_id"].max() >= num_layers:
        raise ValueError("layer ID is outside metadata dimensions")
    if frame["expert_id"].min() < 0 or frame["expert_id"].max() >= num_experts:
        raise ValueError("expert ID is outside metadata dimensions")


def route_summary(frame: pd.DataFrame, metadata: dict[str, object]) -> dict[str, object]:
    sets = route_sets(frame)
    temporal: list[float] = []
    by_request_layer: dict[tuple[int, int], list[tuple[int, frozenset[int]]]] = {}
    for (request_id, position_id, layer_id), experts in sets.items():
        by_request_layer.setdefault((request_id, layer_id), []).append(
            (position_id, experts)
        )
    for observations in by_request_layer.values():
        observations.sort()
        for (_, left), (_, right) in zip(observations, observations[1:], strict=False):
            temporal.append(len(left & right) / len(left | right))

    counts = frame["expert_id"].value_counts().sort_index()
    probabilities = counts.to_numpy(dtype=float) / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalized_entropy = entropy / np.log(int(metadata["num_experts"]))

    union_rows: list[dict[str, float | int]] = []
    top_k = int(metadata["top_k"])
    for width in (1, 2, 4, 8, 16, 32):
        unions: list[int] = []
        for observations in by_request_layer.values():
            observations.sort()
            expert_sets = [experts for _, experts in observations]
            if len(expert_sets) < width:
                continue
            for start in range(len(expert_sets) - width + 1):
                unions.append(len(frozenset().union(*expert_sets[start : start + width])))
        if unions:
            mean_union = float(np.mean(unions))
            union_rows.append(
                {
                    "width": width,
                    "mean_unique_experts": mean_union,
                    "p95_unique_experts": float(np.percentile(unions, 95)),
                    "mean_assignments_per_unique_expert": width * top_k / mean_union,
                    "window_count": len(unions),
                }
            )
    return {
        "row_count": len(frame),
        "request_count": int(frame["request_id"].nunique()),
        "position_count": int(
            frame[["request_id", "position_id"]].drop_duplicates().shape[0]
        ),
        "mean_consecutive_route_jaccard": float(np.mean(temporal)),
        "p05_consecutive_route_jaccard": float(np.percentile(temporal, 5)),
        "max_expert_assignment_share": float(probabilities.max()),
        "normalized_expert_entropy": normalized_entropy,
        "union_by_width": union_rows,
    }


def main() -> None:
    args = parse_args()
    kernel = pd.read_csv(args.root / "kernel" / "expert_kernel_samples.csv")
    measured = kernel[kernel["warmup"].astype(int) == 0]
    kernel_summary = (
        measured.groupby("token_count")["latency_ms"]
        .agg(
            median_ms="median",
            p05_ms=lambda values: values.quantile(0.05),
            p95_ms=lambda values: values.quantile(0.95),
            samples="count",
        )
        .reset_index()
    )
    kernel_summary["tokens_per_ms_at_median"] = (
        kernel_summary["token_count"] / kernel_summary["median_ms"]
    )
    kernel_summary.to_csv(args.root / "expert_kernel_summary.csv", index=False)

    backend_frames: dict[str, pd.DataFrame] = {}
    backend_metadata: dict[str, dict[str, object]] = {}
    summaries: dict[str, object] = {}
    for directory in ("routes-fa3", "routes-flashinfer"):
        metadata = json.loads((args.root / directory / "route_metadata.json").read_text())
        frame = pd.read_csv(args.root / directory / "routes_observational.csv")
        validate_routes(frame, metadata)
        backend = str(metadata["backend"])
        backend_frames[backend] = frame
        backend_metadata[backend] = metadata
        summary = route_summary(frame, metadata)
        pd.DataFrame(summary.pop("union_by_width")).to_csv(
            args.root / f"route_union_{backend}.csv", index=False
        )
        summaries[backend] = summary

    fa3_sets = route_sets(backend_frames["fa3"])
    flashinfer_sets = route_sets(backend_frames["flashinfer"])
    if fa3_sets.keys() != flashinfer_sets.keys():
        raise ValueError("backend traces do not cover the same route keys")
    mismatch_count = sum(
        fa3_sets[key] != flashinfer_sets[key] for key in fa3_sets
    )
    comparison = {
        "route_group_count": len(fa3_sets),
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_count / len(fa3_sets),
    }

    output = {
        "status": "HARDWARE_MEASUREMENT_TP1",
        "scope": {
            "tensor_parallel_size": 1,
            "network_ep_calibrated": False,
            "native_position_parallel_trace": False,
        },
        "kernel": {
            "minimum_tokens": int(kernel_summary["token_count"].min()),
            "maximum_tokens": int(kernel_summary["token_count"].max()),
            "sample_count": int(len(measured)),
            "minimum_median_ms": float(kernel_summary["median_ms"].min()),
            "maximum_median_ms": float(kernel_summary["median_ms"].max()),
        },
        "routes": summaries,
        "backend_comparison": comparison,
        "environment": {
            backend: {
                key: metadata[key]
                for key in (
                    "gpu_model",
                    "compute_capability",
                    "pytorch_version",
                    "torch_cuda_version",
                    "vllm_version",
                    "attention_config",
                )
            }
            for backend, metadata in backend_metadata.items()
        },
    }
    output_path = args.root / "summary.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
