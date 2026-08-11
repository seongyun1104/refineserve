#!/usr/bin/env python3
"""Sample GPU clocks, temperature, and power to CSV during a benchmark."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

FIELDS = (
    "index,uuid,clocks.current.graphics,clocks.current.memory,"
    "temperature.gpu,power.draw,power.limit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Stop early after the current sample when this file appears.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0 or args.duration_seconds <= 0:
        raise ValueError("interval and duration must be positive")
    if args.stop_file is not None and args.stop_file.exists():
        raise ValueError("stop file already exists before telemetry starts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration_seconds
    header_written = False
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        while time.monotonic() < deadline:
            sampled_at = time.time()
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            if not header_written:
                writer.writerow(
                    [
                        "timestamp_unix",
                        "index",
                        "uuid",
                        "graphics_clock_mhz",
                        "memory_clock_mhz",
                        "temperature_c",
                        "power_w",
                        "power_limit_w",
                    ]
                )
                header_written = True
            for line in completed.stdout.splitlines():
                writer.writerow([sampled_at, *[value.strip() for value in line.split(",")]])
            handle.flush()
            if args.stop_file is not None and args.stop_file.exists():
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
