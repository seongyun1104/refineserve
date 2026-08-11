from __future__ import annotations

import argparse
import json
from pathlib import Path

from refineserve.config import load_config
from refineserve.trace_bundle import RouteTraceBundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an M2 native-route trace bundle.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = RouteTraceBundle.load(args.trace, expected_model=config.model)
    report = {
        "status": "valid",
        "schema_version": bundle.metadata.schema_version,
        "source": bundle.metadata.source,
        "bundle_sha256": bundle.bundle_sha256,
        "route_groups": len(bundle.routes),
        "prior_groups": len(bundle.priors),
        "expert_kernel_samples": len(bundle.expert_kernel_samples),
        "network_samples": len(bundle.network_samples),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
