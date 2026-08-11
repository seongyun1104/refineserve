from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .trace_bundle import RouteTraceBundle, TraceValidationError

CALIBRATION_SCHEMA_VERSION = 1


class CalibrationRangeError(ValueError):
    """Raised when replay would extrapolate beyond the measured sample range."""

    def __init__(
        self,
        message: str,
        *,
        input_name: str,
        observed_min: float,
        observed_max: float,
        calibrated_min: float | None,
        calibrated_max: float | None,
    ) -> None:
        super().__init__(message)
        self.input_name = input_name
        self.observed_min = observed_min
        self.observed_max = observed_max
        self.calibrated_min = calibrated_min
        self.calibrated_max = calibrated_max

    @property
    def maximum_overflow(self) -> float:
        overflow = 0.0
        if self.calibrated_min is not None:
            overflow = max(overflow, self.calibrated_min - self.observed_min)
        if self.calibrated_max is not None:
            overflow = max(overflow, self.observed_max - self.calibrated_max)
        return overflow


@dataclass(frozen=True)
class LatencyPoint:
    input_value: int
    raw_median_ms: float
    fitted_latency_ms: float
    p05_ms: float
    p95_ms: float
    sample_count: int


@dataclass(frozen=True)
class MonotoneLatencyCurve:
    input_name: str
    points: tuple[LatencyPoint, ...]

    @classmethod
    def fit(
        cls,
        rows: Iterable[dict[str, str]],
        *,
        input_name: str,
    ) -> MonotoneLatencyCurve:
        grouped: dict[int, list[float]] = {}
        for row in rows:
            if _is_warmup(row["warmup"]):
                continue
            input_value = int(row[input_name])
            grouped.setdefault(input_value, []).append(float(row["latency_ms"]))
        if not grouped:
            raise TraceValidationError(
                f"no non-warmup samples available for {input_name!r} calibration"
            )
        raw_medians = np.asarray(
            [np.median(grouped[value]) for value in sorted(grouped)],
            dtype=float,
        )
        fitted = np.maximum.accumulate(raw_medians)
        points = tuple(
            LatencyPoint(
                input_value=value,
                raw_median_ms=float(raw_median),
                fitted_latency_ms=float(fitted_latency),
                p05_ms=float(np.percentile(grouped[value], 5)),
                p95_ms=float(np.percentile(grouped[value], 95)),
                sample_count=len(grouped[value]),
            )
            for value, raw_median, fitted_latency in zip(
                sorted(grouped),
                raw_medians,
                fitted,
                strict=True,
            )
        )
        return cls(input_name=input_name, points=points)

    @property
    def minimum_input(self) -> int:
        return self.points[0].input_value

    @property
    def maximum_input(self) -> int:
        return self.points[-1].input_value

    def latency_ms(self, input_value: int) -> float:
        if not self.minimum_input <= input_value <= self.maximum_input:
            raise CalibrationRangeError(
                f"{self.input_name}={input_value} is outside measured range "
                f"[{self.minimum_input}, {self.maximum_input}]",
                input_name=self.input_name,
                observed_min=float(input_value),
                observed_max=float(input_value),
                calibrated_min=float(self.minimum_input),
                calibrated_max=float(self.maximum_input),
            )
        inputs = np.asarray([point.input_value for point in self.points], dtype=float)
        latencies = np.asarray(
            [point.fitted_latency_ms for point in self.points],
            dtype=float,
        )
        return float(np.interp(input_value, inputs, latencies))

    def latencies_ms(self, input_values: np.ndarray) -> np.ndarray:
        values = np.asarray(input_values)
        active = values > 0
        if np.any(active):
            active_values = values[active]
            if (
                int(active_values.min()) < self.minimum_input
                or int(active_values.max()) > self.maximum_input
            ):
                raise CalibrationRangeError(
                    f"{self.input_name} values are outside measured range "
                    f"[{self.minimum_input}, {self.maximum_input}]",
                    input_name=self.input_name,
                    observed_min=float(active_values.min()),
                    observed_max=float(active_values.max()),
                    calibrated_min=float(self.minimum_input),
                    calibrated_max=float(self.maximum_input),
                )
        result = np.zeros(values.shape, dtype=float)
        if np.any(active):
            inputs = np.asarray([point.input_value for point in self.points], dtype=float)
            latencies = np.asarray(
                [point.fitted_latency_ms for point in self.points],
                dtype=float,
            )
            result[active] = np.interp(values[active], inputs, latencies)
        return result


@dataclass(frozen=True)
class NetworkCurve:
    collective: str
    active_ranks: int
    message_count: int
    latency_by_bytes: MonotoneLatencyCurve


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: int
    source_bundle_sha256: str
    expert_kernel_curve: MonotoneLatencyCurve | None
    network_curves: tuple[NetworkCurve, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_bundle_sha256: str | None = None,
    ) -> CalibrationArtifact:
        path = Path(path)
        try:
            raw: dict[str, Any] = json.loads(path.read_text())
            schema_version = int(raw["schema_version"])
            source_checksum = str(raw["source_bundle_sha256"])
            expert_raw = raw["expert_kernel_curve"]
            expert_curve = _curve_from_raw(expert_raw) if expert_raw is not None else None
            network_curves = tuple(
                NetworkCurve(
                    collective=str(item["collective"]),
                    active_ranks=int(item["active_ranks"]),
                    message_count=int(item["message_count"]),
                    latency_by_bytes=_curve_from_raw(item["latency_by_bytes"]),
                )
                for item in raw["network_curves"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise TraceValidationError(f"invalid calibration artifact: {error}") from error
        if schema_version != CALIBRATION_SCHEMA_VERSION:
            raise TraceValidationError(
                f"unsupported calibration schema_version={schema_version}"
            )
        if len(source_checksum) != 64:
            raise TraceValidationError("calibration source bundle checksum is malformed")
        if expected_bundle_sha256 is not None and source_checksum != expected_bundle_sha256:
            raise TraceValidationError("calibration artifact belongs to a different trace bundle")
        return cls(
            schema_version=schema_version,
            source_bundle_sha256=source_checksum,
            expert_kernel_curve=expert_curve,
            network_curves=network_curves,
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path

    def network_latencies_ms(
        self,
        *,
        collective: str,
        active_ranks: int,
        message_counts: np.ndarray,
        transferred_bytes: np.ndarray,
    ) -> np.ndarray:
        messages = np.asarray(message_counts)
        sizes = np.asarray(transferred_bytes)
        if messages.shape != sizes.shape:
            raise ValueError("network message and byte arrays must have the same shape")
        result = np.zeros(messages.shape, dtype=float)
        active = messages > 0
        families = {
            curve.message_count: curve
            for curve in self.network_curves
            if curve.collective == collective and curve.active_ranks == active_ranks
        }
        for message_count in np.unique(messages[active]):
            family = families.get(int(message_count))
            if family is None:
                raise CalibrationRangeError(
                    f"no measured {collective} network curve for "
                    f"active_ranks={active_ranks}, message_count={int(message_count)}",
                    input_name="message_count",
                    observed_min=float(message_count),
                    observed_max=float(message_count),
                    calibrated_min=(float(min(families)) if families else None),
                    calibrated_max=(float(max(families)) if families else None),
                )
            mask = active & (messages == message_count)
            result[mask] = family.latency_by_bytes.latencies_ms(sizes[mask])
        return result


def fit_calibration(bundle: RouteTraceBundle) -> CalibrationArtifact:
    expert_curve = (
        MonotoneLatencyCurve.fit(
            bundle.expert_kernel_samples,
            input_name="token_count",
        )
        if bundle.expert_kernel_samples
        else None
    )
    network_groups: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in bundle.network_samples:
        key = (row["collective"], int(row["active_ranks"]), int(row["message_count"]))
        network_groups.setdefault(key, []).append(row)
    network_curves = tuple(
        NetworkCurve(
            collective=key[0],
            active_ranks=key[1],
            message_count=key[2],
            latency_by_bytes=MonotoneLatencyCurve.fit(
                rows,
                input_name="transferred_bytes",
            ),
        )
        for key, rows in sorted(network_groups.items())
    )
    return CalibrationArtifact(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        source_bundle_sha256=bundle.bundle_sha256,
        expert_kernel_curve=expert_curve,
        network_curves=network_curves,
    )


def _is_warmup(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


def _curve_from_raw(raw: dict[str, Any]) -> MonotoneLatencyCurve:
    points = tuple(LatencyPoint(**point) for point in raw["points"])
    if not points:
        raise TraceValidationError("calibration curve contains no points")
    inputs = [point.input_value for point in points]
    fitted = [point.fitted_latency_ms for point in points]
    if inputs != sorted(set(inputs)):
        raise TraceValidationError("calibration curve inputs must be unique and sorted")
    if any(value < 0.0 for value in fitted) or any(
        left > right for left, right in zip(fitted, fitted[1:], strict=False)
    ):
        raise TraceValidationError("calibration curve must be non-negative and monotone")
    return MonotoneLatencyCurve(input_name=str(raw["input_name"]), points=points)
