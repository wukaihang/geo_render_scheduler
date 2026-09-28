"""Pure external feature extraction for geological volume render requests."""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

from geo_render.common.errors import ValidationError
from geo_render.common.types import (
    DeviceState,
    DurationPrediction,
    ModelManifest,
    RenderRequest,
)


FeatureValue = Union[float, str]


def _telemetry(
    values: Dict[str, FeatureValue],
    name: str,
    missing_name: str,
    value: Optional[float],
) -> None:
    values[missing_name] = 1.0 if value is None else 0.0
    values[name] = 0.0 if value is None else float(value)


def extract_features(
    request: RenderRequest,
    manifest: ModelManifest,
    device: DeviceState,
    history: DurationPrediction,
) -> Dict[str, FeatureValue]:
    """Build only externally observable features for one request/GPU pair."""
    if request.model_id != manifest.model_id:
        raise ValidationError(
            "request.model_id must match manifest.model_id; "
            f"got {request.model_id!r} and {manifest.model_id!r}"
        )

    center = tuple(
        (lower + upper) / 2.0
        for lower, upper in zip(manifest.bounds[::2], manifest.bounds[1::2])
    )
    position = request.camera.position
    focal = request.camera.focal_point
    view = tuple(focal[index] - position[index] for index in range(3))
    view_norm = math.sqrt(sum(value * value for value in view))
    view_unit = tuple(value / view_norm for value in view)
    distance_to_center = math.sqrt(
        sum((position[index] - center[index]) ** 2 for index in range(3))
    )

    extent_x = manifest.bounds[1] - manifest.bounds[0]
    extent_y = manifest.bounds[3] - manifest.bounds[2]
    extent_z = manifest.bounds[5] - manifest.bounds[4]
    face_areas = (extent_y * extent_z, extent_x * extent_z, extent_x * extent_y)
    projected_area = sum(
        abs(view_unit[index]) * face_areas[index] for index in range(3)
    )
    projected_area_ratio = min(1.0, projected_area / sum(face_areas))

    features: Dict[str, FeatureValue] = {
        "model_id": request.model_id,
        "gpu_id": device.gpu_id,
        "gpu_model": device.model,
        "dim_x": float(manifest.dimensions[0]),
        "dim_y": float(manifest.dimensions[1]),
        "dim_z": float(manifest.dimensions[2]),
        "voxel_count": float(math.prod(manifest.dimensions)),
        "active_voxel_fraction": manifest.active_voxel_fraction,
        "model_bytes": float(manifest.data_bytes),
        "layer_count": float(manifest.layer_count),
        "camera_distance": distance_to_center,
        "view_x": view_unit[0],
        "view_y": view_unit[1],
        "view_z": view_unit[2],
        "projected_area_ratio": projected_area_ratio,
        "projection": request.camera.projection,
        "view_angle_deg": request.camera.view_angle_deg,
        "clip_fraction": request.clip_fraction,
        "output_width": float(request.output_width),
        "output_height": float(request.output_height),
        "output_pixels": float(request.output_width * request.output_height),
        "sample_step": request.sample_step,
        "shadows": "on" if request.shadows else "off",
        "transfer_function_id": request.transfer_function_id,
        "gpu_memory_total_bytes": float(device.memory_total_bytes),
        "gpu_historical_speed": device.historical_speed,
        "history_p50_ms": history.p50_ms,
        "history_p95_ms": history.p95_ms,
        "history_version": history.model_version,
    }
    _telemetry(
        features,
        "gpu_utilization_pct",
        "gpu_utilization_missing",
        device.utilization_pct,
    )
    _telemetry(
        features,
        "gpu_memory_utilization_pct",
        "gpu_memory_utilization_missing",
        device.memory_utilization_pct,
    )
    _telemetry(
        features,
        "gpu_temperature",
        "gpu_temperature_missing",
        device.temperature_c,
    )
    _telemetry(features, "gpu_power_w", "gpu_power_missing", device.power_w)
    return features
