"""含义稳定的项目专用异常。"""


class GeoRenderError(Exception):
    """项目异常基类。"""


class ValidationError(GeoRenderError, ValueError):
    """领域对象包含非法字段值时抛出。"""


class HardwareBackendUnavailable(GeoRenderError, RuntimeError):
    """真实 GPU 渲染或遥测适配器不可用时抛出。"""


class ModelNotFittedError(GeoRenderError, RuntimeError):
    """在模型拟合前请求预测时抛出。"""


class StateTransitionError(GeoRenderError, RuntimeError):
    """Worker 状态机发生非法转换时抛出。"""


class OracleAccessError(GeoRenderError, RuntimeError):
    """在离线回放之外请求 Oracle 信息时抛出。"""
