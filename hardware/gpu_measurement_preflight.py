#!/usr/bin/env python3
"""Fail-closed GPU clock/telemetry preflight for percent-level measurements."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

QUERY_FIELDS = (
    "index,uuid,name,persistence_mode,clocks.current.graphics,"
    "clocks.current.memory,clocks.applications.graphics,"
    "clocks.applications.memory,temperature.gpu,power.draw,power.limit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphics-clock", type=int)
    parser.add_argument("--memory-clock", type=int)
    parser.add_argument("--require-lock", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    args = parse_args()
    if args.require_lock and (
        args.graphics_clock is None or args.memory_clock is None
    ):
        raise ValueError(
            "--require-lock requires both --graphics-clock and --memory-clock"
        )
    records: list[dict[str, object]] = []
    records.append(run(["nvidia-smi", "-pm", "1"]))
    if args.graphics_clock is not None:
        records.append(run(["nvidia-smi", "-lgc", str(args.graphics_clock)]))
    if args.memory_clock is not None:
        records.append(run(["nvidia-smi", "-lmc", str(args.memory_clock)]))
    query = run(
        [
            "nvidia-smi",
            f"--query-gpu={QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
        ]
    )
    topology = run(["nvidia-smi", "topo", "-m"])
    supported_clocks = run(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"])
    records.extend([query, topology, supported_clocks])
    lock_commands = records[: 1 + int(args.graphics_clock is not None) + int(
        args.memory_clock is not None
    )]
    lock_ok = all(int(record["returncode"]) == 0 for record in lock_commands)
    status = "PASS" if lock_ok or not args.require_lock else "FAIL"
    artifact = {
        "timestamp_unix": time.time(),
        "status": status,
        "lock_required": args.require_lock,
        "lock_succeeded": lock_ok,
        "requested_graphics_clock_mhz": args.graphics_clock,
        "requested_memory_clock_mhz": args.memory_clock,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": status}))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
