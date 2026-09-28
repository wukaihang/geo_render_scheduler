"""Strict JSON configuration conversion for synthetic verification runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

from geo_render.common.errors import ValidationError
from geo_render.common.types import DeviceState, ModelManifest
from geo_render.workload.replay import ReplayEngine
from geo_render.workload.synthetic import SyntheticTraceConfig


def load_config(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError("top-level config must be a JSON object")
    return value


def synthetic_trace_config(
    config: Mapping[str, object],
    request_count: Optional[int] = None,
    seed: Optional[int] = None,
) -> SyntheticTraceConfig:
    raw = config.get("trace")
    if not isinstance(raw, Mapping):
        raise ValidationError("config.trace must be an object")
    return SyntheticTraceConfig(
        seed=int(config["seed"] if seed is None else seed),
        request_count=int(
            raw["request_count"] if request_count is None else request_count
        ),
        arrival_rate_per_second=float(raw["arrival_rate_per_second"]),
        user_count=int(raw["user_count"]),
        trajectory_count=int(raw["trajectory_count"]),
        model_base_render_ms={
            str(key): float(value)
            for key, value in dict(raw["model_base_render_ms"]).items()
        },
        gpu_speed={
            str(key): float(value) for key, value in dict(raw["gpu_speed"]).items()
        },
        output_sizes=tuple(
            (int(pair[0]), int(pair[1])) for pair in raw["output_sizes"]
        ),
        sample_steps=tuple(float(value) for value in raw["sample_steps"]),
        slo_ms=float(raw["slo_ms"]),
    )


def devices_from_config(config: Mapping[str, object]) -> Tuple[DeviceState, ...]:
    raw_devices = config.get("devices")
    if not isinstance(raw_devices, list) or not raw_devices:
        raise ValidationError("config.devices must be a non-empty array")
    return tuple(
        DeviceState(
            gpu_id=str(raw["gpu_id"]),
            model=str(raw["model"]),
            memory_total_bytes=int(raw["memory_total_bytes"]),
            utilization_pct=(
                None
                if raw.get("utilization_pct") is None
                else float(raw["utilization_pct"])
            ),
            memory_utilization_pct=(
                None
                if raw.get("memory_utilization_pct") is None
                else float(raw["memory_utilization_pct"])
            ),
            temperature_c=(
                None if raw.get("temperature_c") is None else float(raw["temperature_c"])
            ),
            power_w=None if raw.get("power_w") is None else float(raw["power_w"]),
            historical_speed=float(raw["historical_speed"]),
            sampled_at_ms=float(raw.get("sampled_at_ms", 0.0)),
        )
        for raw in raw_devices
    )


def manifests_from_config(
    config: Mapping[str, object]
) -> Mapping[str, ModelManifest]:
    raw_manifests = config.get("manifests")
    if not isinstance(raw_manifests, list) or not raw_manifests:
        raise ValidationError("config.manifests must be a non-empty array")
    manifests = {}
    for raw in raw_manifests:
        manifest = ModelManifest(
            model_id=str(raw["model_id"]),
            dimensions=tuple(int(value) for value in raw["dimensions"]),
            active_voxel_fraction=float(raw["active_voxel_fraction"]),
            data_bytes=int(raw["data_bytes"]),
            layer_count=int(raw["layer_count"]),
            bounds=tuple(float(value) for value in raw["bounds"]),
            sha256=str(raw["sha256"]),
        )
        if manifest.model_id in manifests:
            raise ValidationError(f"duplicate manifest {manifest.model_id!r}")
        manifests[manifest.model_id] = manifest
    return manifests


def replay_engine(config: Mapping[str, object]) -> ReplayEngine:
    raw = config.get("prediction")
    if not isinstance(raw, Mapping):
        raise ValidationError("config.prediction must be an object")
    return ReplayEngine(
        devices=devices_from_config(config),
        manifests=manifests_from_config(config),
        predicted_readback_ms_by_gpu={
            str(key): float(value)
            for key, value in dict(raw["readback_ms_by_gpu"]).items()
        },
        predicted_encode_ms=float(raw["encode_ms"]),
        default_history_ms=float(raw["history_default_ms"]),
        history_alpha=float(raw.get("history_alpha", 0.2)),
        history_window_size=int(raw.get("history_window_size", 64)),
    )
