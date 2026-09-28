"""Project-specific exceptions with stable meanings."""


class GeoRenderError(Exception):
    """Base class for project errors."""


class ValidationError(GeoRenderError, ValueError):
    """Raised when a domain object contains an invalid field value."""


class HardwareBackendUnavailable(GeoRenderError, RuntimeError):
    """Raised when the real GPU rendering or telemetry adapter is unavailable."""


class ModelNotFittedError(GeoRenderError, RuntimeError):
    """Raised when prediction is requested before fitting."""


class StateTransitionError(GeoRenderError, RuntimeError):
    """Raised for an invalid worker state-machine transition."""


class OracleAccessError(GeoRenderError, RuntimeError):
    """Raised when oracle information is requested outside offline replay."""
