#!/usr/bin/env python3
"""Hash the exact Gate 2/2B code and preregistration documents before rental."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_PATHS = (
    "hardware/build_gate2_bundle_manifest.py",
    "hardware/preflight_ep4.py",
    "hardware/gpu_measurement_preflight.py",
    "hardware/gpu_telemetry.py",
    "hardware/benchmark_timing_gate_ep4.py",
    "hardware/analyze_timing_gate_ep4.py",
    "hardware/benchmark_proxy_validation_ep4.py",
    "hardware/analyze_proxy_validation_ep4.py",
    "hardware/proxy_validation_contract.py",
    "hardware/build_scheduler_screening_profile.py",
    "hardware/coordinated_scheduling.py",
    "hardware/synthetic_routes.py",
    "docs/hardware_execution_contract.md",
    "docs/m2_followup_measurement_plan.md",
    "docs/m2_ep4_rental_runbook.md",
    "docs/gate2_internal_double_check.md",
    "docs/gate2_paid_boundary.md",
    "results/preflight_20260812/paid-run-candidate.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    missing = [relative for relative in REQUIRED_PATHS if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"Gate 2 bundle is missing required files: {missing}")
    if args.verify is not None:
        expected = json.loads(args.verify.read_text())
        failures = []
        for relative, recorded in expected.get("files", {}).items():
            path = root / relative
            if not path.is_file():
                failures.append(f"missing: {relative}")
                continue
            if path.stat().st_size != int(recorded["size_bytes"]):
                failures.append(f"size mismatch: {relative}")
            if sha256(path) != str(recorded["sha256"]):
                failures.append(f"sha256 mismatch: {relative}")
        if failures:
            print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
            raise SystemExit(2)
        print(json.dumps({"status": "PASS", "verified": str(args.verify)}))
        return
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "decision": {
            "gate2": "GO",
            "gate2b": "GO",
            "gate3": "NO-GO",
        },
        "files": {
            relative: {
                "size_bytes": (root / relative).stat().st_size,
                "sha256": sha256(root / relative),
            }
            for relative in REQUIRED_PATHS
        },
    }
    if args.output is None:
        raise AssertionError("output is required in build mode")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": "PASS"}))


if __name__ == "__main__":
    main()
