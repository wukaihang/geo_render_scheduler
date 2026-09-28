import pytest

from geo_render.common.errors import HardwareBackendUnavailable
from geo_render.rendering.unavailable import (
    UnavailableDeviceStateProvider,
    UnavailableRenderer,
)

from test_types import make_request


def test_unavailable_renderer_never_fabricates_measurements() -> None:
    with pytest.raises(HardwareBackendUnavailable, match=r"Linux.*NVIDIA.*VTK EGL"):
        UnavailableRenderer().render(make_request(), "gpu-0")


def test_unavailable_device_provider_never_fabricates_telemetry() -> None:
    with pytest.raises(HardwareBackendUnavailable, match=r"NVML.*GPU binding"):
        UnavailableDeviceStateProvider().snapshot()
