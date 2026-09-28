from dataclasses import FrozenInstanceError, replace

import pytest

from geo_render.common.errors import ValidationError
from geo_render.common.types import (
    Camera,
    DeviceState,
    DurationPrediction,
    ModelManifest,
    RenderRequest,
    TraceRecord,
)


def make_camera() -> Camera:
    return Camera(
        position=(0.0, 0.0, 5.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
        projection="perspective",
        view_angle_deg=30.0,
    )


def make_request() -> RenderRequest:
    return RenderRequest(
        request_id="request-1",
        user_id="user-1",
        session_id="session-1",
        trajectory_id="trajectory-1",
        model_id="model-a",
        camera=make_camera(),
        clip_fraction=1.0,
        output_width=800,
        output_height=600,
        sample_step=0.5,
        shadows=False,
        transfer_function_id="default",
        arrival_ms=10.0,
        slo_ms=100.0,
    )


def test_request_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValidationError, match="output_width"):
        replace(make_request(), output_width=0)


def test_request_rejects_out_of_range_clip_fraction() -> None:
    with pytest.raises(ValidationError, match="clip_fraction"):
        replace(make_request(), clip_fraction=1.1)


def test_camera_rejects_coincident_position_and_focal_point() -> None:
    with pytest.raises(ValidationError, match="camera.position"):
        Camera(
            position=(1.0, 1.0, 1.0),
            focal_point=(1.0, 1.0, 1.0),
            view_up=(0.0, 1.0, 0.0),
        )


def test_manifest_validates_shape_bounds_and_checksum() -> None:
    with pytest.raises(ValidationError, match="dimensions"):
        ModelManifest(
            model_id="model-a",
            dimensions=(128, 0, 128),
            active_voxel_fraction=0.5,
            data_bytes=1024,
            layer_count=3,
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            sha256="a" * 64,
        )


def test_unknown_device_telemetry_is_none_not_zero() -> None:
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
    assert device.utilization_pct is None
    assert device.temperature_c is None


@pytest.mark.parametrize(
    ("p50", "p95", "message"),
    [(0.0, 1.0, "p50_ms"), (2.0, 1.0, "p95_ms")],
)
def test_duration_prediction_requires_ordered_positive_quantiles(
    p50: float, p95: float, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        DurationPrediction(p50_ms=p50, p95_ms=p95, model_version="v1")


def test_trace_record_freezes_oracle_measurements() -> None:
    actual = {"gpu-0": 20.0, "gpu-1": 25.0}
    trace = TraceRecord(
        request=make_request(),
        actual_render_ms_by_gpu=actual,
        actual_readback_ms_by_gpu={"gpu-0": 2.0, "gpu-1": 2.5},
        actual_encode_ms=3.0,
        isolated_p50_ms=25.0,
        source="synthetic",
    )
    actual["gpu-0"] = 999.0
    assert trace.actual_render_ms_by_gpu["gpu-0"] == 20.0
    with pytest.raises(TypeError):
        trace.actual_render_ms_by_gpu["gpu-0"] = 999.0
    with pytest.raises(FrozenInstanceError):
        trace.source = "measured"
