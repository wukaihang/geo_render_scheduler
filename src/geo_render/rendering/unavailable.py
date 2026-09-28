"""Explicit failure adapters used until Linux RTX hardware is connected."""

from typing import Tuple

from geo_render.common.errors import HardwareBackendUnavailable
from geo_render.common.types import DeviceState, RenderRequest, RenderResult


class UnavailableRenderer:
    def render(self, request: RenderRequest, gpu_id: str) -> RenderResult:
        raise HardwareBackendUnavailable(
            "Real rendering requires Linux, NVIDIA drivers, verified per-process "
            "GPU binding, and VTK EGL; implement geo_render.rendering.Renderer "
            "on the dual-RTX-5090 host."
        )


class UnavailableDeviceStateProvider:
    def snapshot(self) -> Tuple[DeviceState, ...]:
        raise HardwareBackendUnavailable(
            "Real telemetry requires NVML on Linux plus verified EGL GPU binding; "
            "implement geo_render.rendering.DeviceStateProvider on the target host."
        )
