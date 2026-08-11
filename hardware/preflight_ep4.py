#!/usr/bin/env python3
"""Fail-fast admission check for a single-node 4xH100 EP rental."""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command_output(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--require-nvlink", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    gpu_count = torch.cuda.device_count()
    if gpu_count != args.expected_gpus:
        failures.append(f"expected {args.expected_gpus} GPUs, found {gpu_count}")

    devices: list[dict[str, object]] = []
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
            }
        )
        if args.require_h100 and "H100" not in properties.name:
            failures.append(f"GPU {index} is not H100: {properties.name}")
        if properties.major < 9:
            failures.append(f"GPU {index} does not support Hopper kernels")

    topology = command_output(["nvidia-smi", "topo", "-m"])
    if args.require_nvlink:
        gpu_rows = [line for line in topology.splitlines() if line.startswith("GPU")]
        if len(gpu_rows) != args.expected_gpus:
            failures.append("nvidia-smi topology does not contain all GPU rows")
        for row in gpu_rows:
            cells = row.split()[1 : 1 + args.expected_gpus]
            remote_links = [cell for cell in cells if cell != "X"]
            if any(not cell.startswith("NV") for cell in remote_links):
                failures.append(f"non-NVLink GPU path found: {row}")

    peer_access: dict[str, bool] = {}
    for source in range(gpu_count):
        for target in range(gpu_count):
            if source == target:
                continue
            key = f"{source}->{target}"
            accessible = bool(torch.cuda.can_device_access_peer(source, target))
            peer_access[key] = accessible
            if not accessible:
                failures.append(f"CUDA peer access unavailable for {key}")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "expected_parallelism": {
            "tensor_parallel_size": 1,
            "data_parallel_size": 4,
            "expert_parallel_size": 4,
        },
        "devices": devices,
        "peer_access": peer_access,
        "topology": topology,
        "driver_version": command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ).splitlines(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "vllm_version": package_version("vllm"),
        "flashinfer_version": package_version("flashinfer-python"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
