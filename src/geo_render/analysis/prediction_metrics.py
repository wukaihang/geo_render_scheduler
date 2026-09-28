"""Prediction accuracy and P95 calibration metrics."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from geo_render.common.errors import ValidationError


def prediction_metrics(
    actual_ms: Sequence[float],
    predicted_p50_ms: Sequence[float],
    predicted_p95_ms: Sequence[float],
) -> Dict[str, float]:
    if not actual_ms or not (
        len(actual_ms) == len(predicted_p50_ms) == len(predicted_p95_ms)
    ):
        raise ValidationError("prediction arrays must have the same non-zero length")
    actual = np.asarray(actual_ms, dtype=float)
    p50 = np.asarray(predicted_p50_ms, dtype=float)
    p95 = np.asarray(predicted_p95_ms, dtype=float)
    if not np.all(np.isfinite(actual)) or np.any(actual <= 0):
        raise ValidationError("actual_ms values must be finite and positive")
    if not np.all(np.isfinite(p50)) or np.any(p50 <= 0):
        raise ValidationError("predicted_p50_ms values must be finite and positive")
    if not np.all(np.isfinite(p95)) or np.any(p95 <= 0):
        raise ValidationError("predicted_p95_ms values must be finite and positive")
    if np.any(p95 < p50):
        raise ValidationError("each P95 prediction must be >= its P50 prediction")
    absolute_error = np.abs(actual - p50)
    return {
        "mae_ms": float(np.mean(absolute_error)),
        "mape": float(np.mean(absolute_error / actual)),
        "p95_absolute_error_ms": float(np.percentile(absolute_error, 95)),
        "underestimate_rate": float(np.mean(p50 < actual)),
        "p95_coverage": float(np.mean(actual <= p95)),
    }
