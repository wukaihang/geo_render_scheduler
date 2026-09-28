"""Online completed-request duration statistics with hierarchical fallback."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from geo_render.common.errors import ValidationError
from geo_render.common.types import DurationPrediction


@dataclass
class _Series:
    window_size: int
    ewma_ms: Optional[float] = None
    recent_ms: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.recent_ms = deque(maxlen=self.window_size)

    def observe(self, duration_ms: float, alpha: float) -> None:
        if self.ewma_ms is None:
            self.ewma_ms = duration_ms
        else:
            self.ewma_ms = alpha * duration_ms + (1.0 - alpha) * self.ewma_ms
        self.recent_ms.append(duration_ms)

    def prediction(self, suffix: str) -> DurationPrediction:
        if self.ewma_ms is None:
            raise RuntimeError("empty series cannot produce a prediction")
        empirical_p95 = float(np.percentile(tuple(self.recent_ms), 95))
        return DurationPrediction(
            p50_ms=self.ewma_ms,
            p95_ms=max(self.ewma_ms, empirical_p95),
            model_version=f"ewma:{suffix}",
        )


class OnlineDurationStats:
    """Tracks only observations explicitly reported through :meth:`observe`."""

    def __init__(self, alpha: float, window_size: int, default_ms: float) -> None:
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValidationError(f"alpha must be in (0, 1]; got {alpha!r}")
        if window_size < 2:
            raise ValidationError("window_size must be at least 2")
        if not math.isfinite(default_ms) or default_ms <= 0:
            raise ValidationError("default_ms must be finite and positive")
        self.alpha = alpha
        self.window_size = window_size
        self.default_ms = default_ms
        self._by_gpu_model: Dict[Tuple[str, str], _Series] = {}
        self._by_gpu: Dict[str, _Series] = {}
        self._global = _Series(window_size)
        self._observation_count = 0

    @property
    def observation_count(self) -> int:
        return self._observation_count

    def _series(self, mapping: Dict, key: object) -> _Series:
        if key not in mapping:
            mapping[key] = _Series(self.window_size)
        return mapping[key]

    def observe(self, gpu_id: str, model_id: str, duration_ms: float) -> None:
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValidationError(
                f"duration_ms must be finite and positive; got {duration_ms!r}"
            )
        if not gpu_id or not model_id:
            raise ValidationError("gpu_id and model_id must be non-empty")
        self._series(self._by_gpu_model, (gpu_id, model_id)).observe(
            duration_ms, self.alpha
        )
        self._series(self._by_gpu, gpu_id).observe(duration_ms, self.alpha)
        self._global.observe(duration_ms, self.alpha)
        self._observation_count += 1

    def estimate(self, gpu_id: str, model_id: str) -> DurationPrediction:
        exact = self._by_gpu_model.get((gpu_id, model_id))
        if exact is not None:
            return exact.prediction("gpu-model")
        gpu = self._by_gpu.get(gpu_id)
        if gpu is not None:
            return gpu.prediction("gpu")
        if self._global.ewma_ms is not None:
            return self._global.prediction("global")
        return DurationPrediction(
            p50_ms=self.default_ms,
            p95_ms=self.default_ms,
            model_version="ewma:default",
        )
