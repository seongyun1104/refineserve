from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from refineserve.calibration import fit_calibration
from refineserve.config import load_config
from refineserve.trace_bundle import RouteTraceBundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit bounded M2 timing curves.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = RouteTraceBundle.load(args.trace, expected_model=config.model)
    artifact = fit_calibration(bundle)
    artifact.write(args.output)
    print(json.dumps(asdict(artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
