from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .placement import ExpertPlacement

TRACE_SCHEMA_VERSION = 2
TRACE_KIND = "native_position_parallel"

RouteKey = tuple[int, int, int, int]
PriorKey = tuple[int, int]


class TraceValidationError(ValueError):
    """Raised when a trace bundle cannot be replayed without ambiguity."""


@dataclass(frozen=True)
class TraceMetadata:
    schema_version: int
    trace_kind: str
    created_at_utc: str
    source: str
    model_identifier: str
    model_revision: str
    random_seed: int
    model: ModelConfig
    placement: ExpertPlacement
    measurement_environment: dict[str, Any]
    latency_unit: str
    size_unit: str


@dataclass(frozen=True)
class RouteTraceBundle:
    root: Path
    metadata: TraceMetadata
    routes: dict[RouteKey, tuple[int, ...]]
    priors: dict[PriorKey, tuple[int, ...]]
    route_weights: dict[RouteKey, tuple[float, ...]]
    bundle_sha256: str
    expert_kernel_samples: tuple[dict[str, str], ...] = ()
    network_samples: tuple[dict[str, str], ...] = ()

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_model: ModelConfig | None = None,
    ) -> RouteTraceBundle:
        root = Path(root)
        metadata_path = root / "metadata.json"
        routes_path = root / "routes.csv"
        priors_path = root / "route_priors.csv"
        for path in (metadata_path, routes_path, priors_path):
            if not path.is_file():
                raise TraceValidationError(f"required trace file is missing: {path}")

        metadata = _load_metadata(metadata_path)
        if expected_model is not None and metadata.model != expected_model:
            raise TraceValidationError(
                "trace model does not match replay config: "
                f"trace={asdict(metadata.model)}, config={asdict(expected_model)}"
            )
        routes, route_weights = _load_routes(routes_path, metadata.model)
        priors = _load_priors(priors_path, metadata.model)
        expert_samples = _load_optional_samples(
            root / "expert_kernel_samples.csv",
            required={
                "sample_id",
                "gpu_id",
                "expert_id",
                "token_count",
                "latency_ms",
                "warmup",
                "repetition",
            },
            non_negative={"gpu_id", "expert_id", "token_count", "latency_ms", "repetition"},
            upper_bounds={
                "gpu_id": metadata.model.num_gpus,
                "expert_id": metadata.model.num_experts,
            },
        )
        network_samples = _load_optional_samples(
            root / "network_samples.csv",
            required={
                "sample_id",
                "collective",
                "active_ranks",
                "message_count",
                "transferred_bytes",
                "latency_ms",
                "warmup",
                "repetition",
            },
            non_negative={
                "active_ranks",
                "message_count",
                "transferred_bytes",
                "latency_ms",
                "repetition",
            },
            upper_bounds={"active_ranks": metadata.model.num_gpus + 1},
        )
        return cls(
            root=root,
            metadata=metadata,
            routes=routes,
            priors=priors,
            route_weights=route_weights,
            bundle_sha256=_bundle_checksum(root),
            expert_kernel_samples=expert_samples,
            network_samples=network_samples,
        )


def _load_metadata(path: Path) -> TraceMetadata:
    try:
        raw: dict[str, Any] = json.loads(path.read_text())
        model_raw = raw["model"]
        units = raw["units"]
        metadata = TraceMetadata(
            schema_version=int(raw["schema_version"]),
            trace_kind=str(raw["trace_kind"]),
            created_at_utc=str(raw["created_at_utc"]),
            source=str(raw["source"]),
            model_identifier=str(raw["model_identifier"]),
            model_revision=str(raw["model_revision"]),
            random_seed=int(raw["random_seed"]),
            model=ModelConfig(**model_raw),
            placement=ExpertPlacement.from_rows(
                raw["expert_to_rank_mapping"],
                ModelConfig(**model_raw),
            ),
            measurement_environment=dict(raw["measurement_environment"]),
            latency_unit=str(units["latency"]),
            size_unit=str(units["size"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"invalid trace metadata: {error}") from error
    if metadata.schema_version != TRACE_SCHEMA_VERSION:
        raise TraceValidationError(
            f"unsupported trace schema_version={metadata.schema_version}; "
            f"expected {TRACE_SCHEMA_VERSION}"
        )
    if metadata.trace_kind != TRACE_KIND:
        raise TraceValidationError(f"unsupported trace_kind={metadata.trace_kind!r}")
    if not all(
        (
            metadata.created_at_utc,
            metadata.source,
            metadata.model_identifier,
            metadata.model_revision,
        )
    ):
        raise TraceValidationError("trace identity and provenance fields must be non-empty")
    if metadata.latency_unit != "ms" or metadata.size_unit != "bytes":
        raise TraceValidationError("trace units must be latency='ms' and size='bytes'")
    required_environment = {
        "gpu_model",
        "gpu_count",
        "topology",
        "node_scope",
        "cuda_version",
        "nccl_version",
        "pytorch_version",
        "kernel_backend",
        "dtype",
        "intermediate_size",
        "concurrent_streams",
        "warmup_count",
        "measurement_iterations",
    }
    missing_environment = required_environment - set(metadata.measurement_environment)
    if missing_environment:
        raise TraceValidationError(
            "trace measurement_environment missing fields: "
            f"{sorted(missing_environment)}"
        )
    return metadata


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise TraceValidationError(f"{path.name} missing columns: {sorted(missing)}")
        return list(reader)


def _integer(row: dict[str, str], column: str, path: Path) -> int:
    try:
        value = int(row[column])
    except (KeyError, ValueError) as error:
        raise TraceValidationError(f"{path.name} has invalid integer {column!r}") from error
    if value < 0:
        raise TraceValidationError(f"{path.name} has negative {column!r}")
    return value


def _load_routes(
    path: Path,
    model: ModelConfig,
) -> tuple[dict[RouteKey, tuple[int, ...]], dict[RouteKey, tuple[float, ...]]]:
    required = {
        "request_id",
        "iteration",
        "layer_id",
        "position_id",
        "route_slot",
        "expert_id",
        "routing_weight",
        "batch_size",
        "active_position_count",
        "context_length",
    }
    groups: defaultdict[RouteKey, dict[int, int]] = defaultdict(dict)
    weight_groups: defaultdict[RouteKey, dict[int, float]] = defaultdict(dict)
    contexts: dict[RouteKey, tuple[int, int, int]] = {}
    for row in _read_rows(path, required):
        request_id = _integer(row, "request_id", path)
        iteration = _integer(row, "iteration", path)
        layer_id = _integer(row, "layer_id", path)
        position_id = _integer(row, "position_id", path)
        slot = _integer(row, "route_slot", path)
        expert_id = _integer(row, "expert_id", path)
        batch_size = _integer(row, "batch_size", path)
        active_positions = _integer(row, "active_position_count", path)
        context_length = _integer(row, "context_length", path)
        try:
            routing_weight = float(row["routing_weight"])
        except ValueError as error:
            raise TraceValidationError(f"{path.name} has invalid routing_weight") from error
        if routing_weight < 0.0:
            raise TraceValidationError(f"{path.name} has negative routing_weight")
        if batch_size <= 0 or active_positions <= 0:
            raise TraceValidationError(
                f"{path.name} batch_size and active_position_count must be positive"
            )
        if layer_id >= model.num_layers or expert_id >= model.num_experts:
            raise TraceValidationError(f"{path.name} contains an out-of-range layer/expert ID")
        key = (request_id, iteration, layer_id, position_id)
        if slot in groups[key]:
            raise TraceValidationError(f"duplicate route key/slot in {path.name}: {key + (slot,)}")
        groups[key][slot] = expert_id
        weight_groups[key][slot] = routing_weight
        context = (batch_size, active_positions, context_length)
        if key in contexts and contexts[key] != context:
            raise TraceValidationError(f"{path.name} route slots disagree on context: {key}")
        contexts[key] = context
    routes = _finalize_route_groups(groups, model.top_k, path)
    weights = _finalize_weight_groups(weight_groups, model.top_k, path)
    return routes, weights


def _load_priors(path: Path, model: ModelConfig) -> dict[PriorKey, tuple[int, ...]]:
    required = {"request_id", "layer_id", "route_slot", "expert_id"}
    groups: defaultdict[PriorKey, dict[int, int]] = defaultdict(dict)
    for row in _read_rows(path, required):
        request_id = _integer(row, "request_id", path)
        layer_id = _integer(row, "layer_id", path)
        slot = _integer(row, "route_slot", path)
        expert_id = _integer(row, "expert_id", path)
        if layer_id >= model.num_layers or expert_id >= model.num_experts:
            raise TraceValidationError(f"{path.name} contains an out-of-range layer/expert ID")
        key = (request_id, layer_id)
        if slot in groups[key]:
            raise TraceValidationError(f"duplicate prior key/slot in {path.name}: {key + (slot,)}")
        groups[key][slot] = expert_id
    return _finalize_route_groups(groups, model.top_k, path)


def _finalize_route_groups[K: tuple[int, ...]](
    groups: dict[K, dict[int, int]],
    top_k: int,
    path: Path,
) -> dict[K, tuple[int, ...]]:
    expected_slots = set(range(top_k))
    result: dict[K, tuple[int, ...]] = {}
    for key, by_slot in groups.items():
        if set(by_slot) != expected_slots:
            raise TraceValidationError(
                f"{path.name} route group {key} must contain slots 0..{top_k - 1}"
            )
        experts = tuple(by_slot[slot] for slot in range(top_k))
        if len(set(experts)) != top_k:
            raise TraceValidationError(f"{path.name} route group {key} repeats an expert")
        result[key] = experts
    if not result:
        raise TraceValidationError(f"{path.name} contains no route groups")
    return result


def _finalize_weight_groups[K: tuple[int, ...]](
    groups: dict[K, dict[int, float]],
    top_k: int,
    path: Path,
) -> dict[K, tuple[float, ...]]:
    expected_slots = set(range(top_k))
    result: dict[K, tuple[float, ...]] = {}
    for key, by_slot in groups.items():
        if set(by_slot) != expected_slots:
            raise TraceValidationError(
                f"{path.name} weight group {key} must contain slots 0..{top_k - 1}"
            )
        result[key] = tuple(by_slot[slot] for slot in range(top_k))
    return result


def _load_optional_samples(
    path: Path,
    *,
    required: set[str],
    non_negative: set[str],
    upper_bounds: dict[str, int],
) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        return ()
    rows = _read_rows(path, required)
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in sample_ids:
            raise TraceValidationError(f"{path.name} has an empty or duplicate sample_id")
        sample_ids.add(sample_id)
        if row["warmup"].strip().lower() not in {"0", "1", "false", "true"}:
            raise TraceValidationError(f"{path.name} has invalid warmup flag")
        for column in non_negative:
            try:
                value = float(row[column])
            except (KeyError, ValueError) as error:
                raise TraceValidationError(
                    f"{path.name} has invalid numeric value in {column!r}"
                ) from error
            if value < 0.0:
                raise TraceValidationError(f"{path.name} has negative {column!r}")
        for column, upper_bound in upper_bounds.items():
            if float(row[column]) >= upper_bound:
                raise TraceValidationError(f"{path.name} has out-of-range {column!r}")
    return tuple(rows)


def _bundle_checksum(root: Path) -> str:
    digest = hashlib.sha256()
    names = (
        "metadata.json",
        "routes.csv",
        "route_priors.csv",
        "expert_kernel_samples.csv",
        "network_samples.csv",
    )
    for name in names:
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()
