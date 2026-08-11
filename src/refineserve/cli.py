from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from .config import ExecutionMode, SchedulerName, load_config
from .simulator import Simulator

SCHEDULERS = (
    "fifo",
    "previous_route",
    "oracle",
    "locality_only",
    "load_balance_only",
    "critical_path_only",
    "locality_plus_load",
    "joint",
    "routing_oracle",
    "runtime_oracle",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("autoregressive", "diffusion"), required=True)
    parser.add_argument("--scheduler", choices=SCHEDULERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scheduler = cast(SchedulerName, args.scheduler)
    mode = cast(ExecutionMode, args.mode)
    config = load_config(args.config).with_overrides(mode_scheduler=scheduler)
    result = Simulator(config, mode).run()
    result.write(args.output)
    print(json.dumps(asdict(result.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
