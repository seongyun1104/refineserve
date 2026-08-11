from __future__ import annotations

from refineserve.config import load_config
from refineserve.scenarios import SCENARIOS, scenario_config


def test_named_scenarios_are_valid_and_distinct() -> None:
    base = load_config("configs/m1_scheduler_diagnostic.yaml")
    configs = {scenario: scenario_config(base, scenario) for scenario in SCENARIOS}

    assert configs["compute_bound"].network.bandwidth_gb_per_s == 900.0
    assert configs["communication_bound"].network.bandwidth_gb_per_s == 25.0
    assert configs["deadline_bound"].scheduler.max_wait_ms == 5.0
    assert configs["deadline_bound"].workload.output_length_pattern == "staggered"
