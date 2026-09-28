import math

from geo_render.common.types import DeviceState, DurationPrediction, ModelManifest
from geo_render.prediction.features import extract_features

from test_types import make_request


def make_manifest() -> ModelManifest:
    return ModelManifest(
        model_id="model-a",
        dimensions=(128, 256, 64),
        active_voxel_fraction=0.6,
        data_bytes=128 * 256 * 64 * 2,
        layer_count=5,
        bounds=(-2.0, 2.0, -1.0, 1.0, -0.5, 0.5),
        sha256="b" * 64,
    )


def make_device() -> DeviceState:
    return DeviceState(
        gpu_id="gpu-0",
        model="RTX 5090",
        memory_total_bytes=32 * 1024**3,
        utilization_pct=42.0,
        memory_utilization_pct=25.0,
        temperature_c=60.0,
        power_w=300.0,
        historical_speed=1.0,
        sampled_at_ms=0.0,
    )


def test_feature_extraction_normalizes_view_direction() -> None:
    values = extract_features(
        make_request(),
        make_manifest(),
        make_device(),
        DurationPrediction(20.0, 30.0, "history-v1"),
    )
    norm = math.sqrt(sum(values[key] ** 2 for key in ("view_x", "view_y", "view_z")))
    assert norm == 1.0
    assert 0.0 <= values["projected_area_ratio"] <= 1.0
    assert values["history_p50_ms"] == 20.0
    assert values["gpu_id"] == "gpu-0"


def test_feature_extraction_marks_missing_telemetry() -> None:
    device = DeviceState(
        gpu_id="gpu-0",
        model="unknown",
        memory_total_bytes=32 * 1024**3,
        utilization_pct=None,
        memory_utilization_pct=None,
        temperature_c=None,
        power_w=None,
        historical_speed=1.0,
        sampled_at_ms=0.0,
    )
    values = extract_features(
        make_request(),
        make_manifest(),
        device,
        DurationPrediction(20.0, 30.0, "history-v1"),
    )
    assert values["gpu_utilization_missing"] == 1.0
    assert values["gpu_temperature_missing"] == 1.0
    assert values["gpu_utilization_pct"] == 0.0
