"""硬件适配器契约。"""

from .interface import DeviceStateProvider, Renderer
from .unavailable import UnavailableDeviceStateProvider, UnavailableRenderer

__all__ = [
    "DeviceStateProvider",
    "Renderer",
    "UnavailableDeviceStateProvider",
    "UnavailableRenderer",
]
