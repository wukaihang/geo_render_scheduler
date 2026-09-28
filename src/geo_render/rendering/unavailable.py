"""Linux RTX 硬件接入前使用的明确失败适配器。"""

from typing import Tuple

from geo_render.common.errors import HardwareBackendUnavailable
from geo_render.common.types import DeviceState, RenderRequest, RenderResult


class UnavailableRenderer:
    def render(self, request: RenderRequest, gpu_id: str) -> RenderResult:
        raise HardwareBackendUnavailable(
            "真实渲染需要 Linux、NVIDIA 驱动、经验证的逐进程 GPU 绑定和 "
            "VTK EGL；请在双 RTX 5090 主机上实现 geo_render.rendering.Renderer。"
        )


class UnavailableDeviceStateProvider:
    def snapshot(self) -> Tuple[DeviceState, ...]:
        raise HardwareBackendUnavailable(
            "真实遥测需要 Linux 上的 NVML 以及经验证的 EGL GPU 绑定；请在目标主机上"
            "实现 geo_render.rendering.DeviceStateProvider。"
        )
