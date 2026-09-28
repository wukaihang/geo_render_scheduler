"""Immutable cross-module data contracts for innovation one."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from .errors import ValidationError

Vector3 = Tuple[float, float, float]
Dimensions3 = Tuple[int, int, int]
Bounds6 = Tuple[float, float, float, float, float, float]


def _require_text(path: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a non-empty string; got {value!r}")


def _require_finite(path: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValidationError(f"{path} must be finite; got {value!r}")


def _require_positive(path: str, value: float) -> None:
    _require_finite(path, value)
    if value <= 0:
        raise ValidationError(f"{path} must be positive; got {value!r}")


def _require_non_negative(path: str, value: float) -> None:
    _require_finite(path, value)
    if value < 0:
        raise ValidationError(f"{path} must be non-negative; got {value!r}")


def _validate_vector(path: str, value: Vector3) -> None:
    if len(value) != 3:
        raise ValidationError(f"{path} must contain three values; got {value!r}")
    for index, coordinate in enumerate(value):
        _require_finite(f"{path}[{index}]", coordinate)


def _freeze_float_mapping(path: str, values: Mapping[str, float]) -> Mapping[str, float]:
    frozen = {}
    for key, value in values.items():
        _require_text(f"{path}.key", key)
        _require_non_negative(f"{path}[{key!r}]", value)
        frozen[key] = float(value)
    if not frozen:
        raise ValidationError(f"{path} must not be empty")
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class Camera:
    position: Vector3
    focal_point: Vector3
    view_up: Vector3 = (0.0, 1.0, 0.0)
    projection: str = "perspective"
    view_angle_deg: float = 30.0

    def __post_init__(self) -> None:
        _validate_vector("camera.position", self.position)
        _validate_vector("camera.focal_point", self.focal_point)
        _validate_vector("camera.view_up", self.view_up)
        if self.position == self.focal_point:
            raise ValidationError("camera.position must differ from camera.focal_point")
        if math.sqrt(sum(value * value for value in self.view_up)) == 0:
            raise ValidationError("camera.view_up must be non-zero")
        if self.projection not in {"perspective", "parallel"}:
            raise ValidationError(
                f"camera.projection must be 'perspective' or 'parallel'; got {self.projection!r}"
            )
        _require_positive("camera.view_angle_deg", self.view_angle_deg)
        if self.view_angle_deg >= 180:
            raise ValidationError("camera.view_angle_deg must be less than 180")


@dataclass(frozen=True)
class RenderRequest:
    request_id: str
    user_id: str
    session_id: str
    trajectory_id: str
    model_id: str
    camera: Camera
    clip_fraction: float
    output_width: int
    output_height: int
    sample_step: float
    shadows: bool
    transfer_function_id: str
    arrival_ms: float
    slo_ms: Optional[float] = None

    def __post_init__(self) -> None:
        for path in (
            "request_id",
            "user_id",
            "session_id",
            "trajectory_id",
            "model_id",
            "transfer_function_id",
        ):
            _require_text(path, getattr(self, path))
        _require_finite("clip_fraction", self.clip_fraction)
        if not 0 <= self.clip_fraction <= 1:
            raise ValidationError(
                f"clip_fraction must be in [0, 1]; got {self.clip_fraction!r}"
            )
        if self.output_width <= 0:
            raise ValidationError(
                f"output_width must be positive; got {self.output_width!r}"
            )
        if self.output_height <= 0:
            raise ValidationError(
                f"output_height must be positive; got {self.output_height!r}"
            )
        _require_positive("sample_step", self.sample_step)
        _require_non_negative("arrival_ms", self.arrival_ms)
        if self.slo_ms is not None:
            _require_positive("slo_ms", self.slo_ms)


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    dimensions: Dimensions3
    active_voxel_fraction: float
    data_bytes: int
    layer_count: int
    bounds: Bounds6
    sha256: str

    def __post_init__(self) -> None:
        _require_text("model_id", self.model_id)
        if len(self.dimensions) != 3 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.dimensions
        ):
            raise ValidationError(
                f"dimensions must contain three positive integers; got {self.dimensions!r}"
            )
        _require_finite("active_voxel_fraction", self.active_voxel_fraction)
        if not 0 <= self.active_voxel_fraction <= 1:
            raise ValidationError(
                "active_voxel_fraction must be in [0, 1]; "
                f"got {self.active_voxel_fraction!r}"
            )
        if self.data_bytes <= 0:
            raise ValidationError(f"data_bytes must be positive; got {self.data_bytes!r}")
        if self.layer_count <= 0:
            raise ValidationError(f"layer_count must be positive; got {self.layer_count!r}")
        if len(self.bounds) != 6:
            raise ValidationError(f"bounds must contain six values; got {self.bounds!r}")
        for index, value in enumerate(self.bounds):
            _require_finite(f"bounds[{index}]", value)
        for lower, upper, axis in zip(self.bounds[::2], self.bounds[1::2], "xyz"):
            if lower >= upper:
                raise ValidationError(
                    f"bounds for axis {axis} must have lower < upper; got {(lower, upper)!r}"
                )
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.sha256
        ):
            raise ValidationError("sha256 must contain exactly 64 hexadecimal characters")


@dataclass(frozen=True)
class DeviceState:
    gpu_id: str
    model: str
    memory_total_bytes: int
    utilization_pct: Optional[float]
    memory_utilization_pct: Optional[float]
    temperature_c: Optional[float]
    power_w: Optional[float]
    historical_speed: float
    sampled_at_ms: float

    def __post_init__(self) -> None:
        _require_text("gpu_id", self.gpu_id)
        _require_text("device.model", self.model)
        if self.memory_total_bytes <= 0:
            raise ValidationError("memory_total_bytes must be positive")
        for path in ("utilization_pct", "memory_utilization_pct"):
            value = getattr(self, path)
            if value is not None:
                _require_finite(path, value)
                if not 0 <= value <= 100:
                    raise ValidationError(f"{path} must be in [0, 100]; got {value!r}")
        for path in ("temperature_c", "power_w"):
            value = getattr(self, path)
            if value is not None:
                _require_non_negative(path, value)
        _require_positive("historical_speed", self.historical_speed)
        _require_non_negative("sampled_at_ms", self.sampled_at_ms)


@dataclass(frozen=True)
class DurationPrediction:
    p50_ms: float
    p95_ms: float
    model_version: str

    def __post_init__(self) -> None:
        _require_positive("p50_ms", self.p50_ms)
        _require_positive("p95_ms", self.p95_ms)
        if self.p95_ms < self.p50_ms:
            raise ValidationError(
                f"p95_ms must be greater than or equal to p50_ms; got {self.p95_ms!r}"
            )
        _require_text("model_version", self.model_version)


@dataclass(frozen=True)
class QueuedRequest:
    request: RenderRequest
    predicted_p95_ms: float
    predicted_readback_ms: float
    predicted_encode_ms: float
    assigned_ms: float

    def __post_init__(self) -> None:
        _require_positive("queued.predicted_p95_ms", self.predicted_p95_ms)
        _require_non_negative("queued.predicted_readback_ms", self.predicted_readback_ms)
        _require_non_negative("queued.predicted_encode_ms", self.predicted_encode_ms)
        _require_non_negative("queued.assigned_ms", self.assigned_ms)

    @property
    def predicted_service_ms(self) -> float:
        return (
            self.predicted_p95_ms
            + self.predicted_readback_ms
            + self.predicted_encode_ms
        )


@dataclass(frozen=True)
class WorkerSnapshot:
    gpu_id: str
    device: DeviceState
    current_request_id: Optional[str] = None
    running_predicted_finish_ms: Optional[float] = None
    queued: Tuple[QueuedRequest, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_text("worker.gpu_id", self.gpu_id)
        if self.device.gpu_id != self.gpu_id:
            raise ValidationError("worker.gpu_id must match worker.device.gpu_id")
        if (self.current_request_id is None) != (
            self.running_predicted_finish_ms is None
        ):
            raise ValidationError(
                "current_request_id and running_predicted_finish_ms must be set together"
            )
        if self.current_request_id is not None:
            _require_text("current_request_id", self.current_request_id)
            _require_non_negative(
                "running_predicted_finish_ms", self.running_predicted_finish_ms or 0.0
            )


@dataclass(frozen=True)
class CostEstimate:
    gpu_id: str
    running_remaining_ms: float
    queued_work_ms: float
    predicted_render_p50_ms: float
    predicted_render_p95_ms: float
    predicted_readback_ms: float
    predicted_encode_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        _require_text("cost.gpu_id", self.gpu_id)
        for path in (
            "running_remaining_ms",
            "queued_work_ms",
            "predicted_readback_ms",
            "predicted_encode_ms",
        ):
            _require_non_negative(path, getattr(self, path))
        _require_positive("predicted_render_p50_ms", self.predicted_render_p50_ms)
        _require_positive("predicted_render_p95_ms", self.predicted_render_p95_ms)
        if self.predicted_render_p95_ms < self.predicted_render_p50_ms:
            raise ValidationError("predicted_render_p95_ms must be >= predicted_render_p50_ms")
        expected = (
            self.running_remaining_ms
            + self.queued_work_ms
            + self.predicted_render_p95_ms
            + self.predicted_readback_ms
            + self.predicted_encode_ms
        )
        if not math.isclose(self.total_ms, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValidationError(
                f"total_ms must equal the EFT component sum; expected {expected!r}"
            )


@dataclass(frozen=True)
class ScheduleDecision:
    request_id: str
    gpu_id: str
    policy: str
    reason: str
    costs: Mapping[str, CostEstimate]

    def __post_init__(self) -> None:
        _require_text("decision.request_id", self.request_id)
        _require_text("decision.gpu_id", self.gpu_id)
        _require_text("decision.policy", self.policy)
        _require_text("decision.reason", self.reason)
        copied = dict(self.costs)
        if self.gpu_id not in copied:
            raise ValidationError("decision.costs must contain the selected gpu_id")
        object.__setattr__(self, "costs", MappingProxyType(copied))


@dataclass(frozen=True)
class RenderResult:
    request_id: str
    gpu_id: str
    start_ms: float
    finish_ms: float
    render_ms: float
    readback_ms: float
    encode_ms: float

    def __post_init__(self) -> None:
        _require_text("result.request_id", self.request_id)
        _require_text("result.gpu_id", self.gpu_id)
        _require_non_negative("result.start_ms", self.start_ms)
        _require_non_negative("result.finish_ms", self.finish_ms)
        if self.finish_ms < self.start_ms:
            raise ValidationError("result.finish_ms must be >= result.start_ms")
        _require_positive("result.render_ms", self.render_ms)
        _require_non_negative("result.readback_ms", self.readback_ms)
        _require_non_negative("result.encode_ms", self.encode_ms)


@dataclass(frozen=True)
class TraceRecord:
    request: RenderRequest
    actual_render_ms_by_gpu: Mapping[str, float]
    actual_readback_ms_by_gpu: Mapping[str, float]
    actual_encode_ms: float
    isolated_p50_ms: Optional[float]
    source: str

    def __post_init__(self) -> None:
        renders = _freeze_float_mapping(
            "actual_render_ms_by_gpu", self.actual_render_ms_by_gpu
        )
        readbacks = _freeze_float_mapping(
            "actual_readback_ms_by_gpu", self.actual_readback_ms_by_gpu
        )
        if set(renders) != set(readbacks):
            raise ValidationError(
                "actual render and readback mappings must contain identical GPU IDs"
            )
        if any(value <= 0 for value in renders.values()):
            raise ValidationError("actual render durations must be positive")
        _require_non_negative("actual_encode_ms", self.actual_encode_ms)
        if self.isolated_p50_ms is not None:
            _require_positive("isolated_p50_ms", self.isolated_p50_ms)
        if self.source not in {"synthetic", "measured"}:
            raise ValidationError(
                f"source must be 'synthetic' or 'measured'; got {self.source!r}"
            )
        object.__setattr__(self, "actual_render_ms_by_gpu", renders)
        object.__setattr__(self, "actual_readback_ms_by_gpu", readbacks)
