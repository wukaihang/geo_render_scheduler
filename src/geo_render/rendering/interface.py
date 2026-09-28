"""真实或受控渲染后端需要实现的协议。"""

from typing import Protocol, Tuple

from geo_render.common.types import DeviceState, RenderRequest, RenderResult


class Renderer(Protocol):
    def render(self, request: RenderRequest, gpu_id: str) -> RenderResult:
        """在选定 GPU 上渲染一个模型已驻留的请求。"""
        ...


class DeviceStateProvider(Protocol):
    def snapshot(self) -> Tuple[DeviceState, ...]:
        """为每张可调度 GPU 返回一份当前状态。"""
        ...
