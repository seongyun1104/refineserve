from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hardware import gpu_measurement_preflight


def test_require_lock_requires_both_clock_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu_measurement_preflight.py",
            "--output",
            str(tmp_path / "preflight.json"),
            "--require-lock",
        ],
    )

    with pytest.raises(ValueError, match="requires both"):
        gpu_measurement_preflight.main()
