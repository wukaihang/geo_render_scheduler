"""Protocols implemented by real or controlled rendering backends."""

from typing import Protocol, Tuple

from geo_render.common.types import DeviceState, RenderRequest, RenderResult


class Renderer(Protocol):
    def render(self, request: RenderRequest, gpu_id: str) -> RenderResult:
        """Render one already-resident model request on the selected GPU."""
        ...


class DeviceStateProvider(Protocol):
    def snapshot(self) -> Tuple[DeviceState, ...]:
        """Return one current state per schedulable GPU."""
        ...
