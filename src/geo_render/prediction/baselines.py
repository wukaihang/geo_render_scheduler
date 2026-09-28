"""仅使用历史信息的耗时预测基线。"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Mapping, Sequence, Tuple

import numpy as np

from geo_render.common.errors import ModelNotFittedError, ValidationError
from geo_render.common.types import DurationPrediction

from .dataset import LabeledSample
from .features import FeatureValue
from .online_stats import OnlineDurationStats


def _identity(features: Mapping[str, FeatureValue]) -> Tuple[str, str]:
    gpu_id = features.get("gpu_id")
    model_id = features.get("model_id")
    if not isinstance(gpu_id, str) or not gpu_id:
        raise ValidationError("features['gpu_id'] 必须是非空字符串")
    if not isinstance(model_id, str) or not model_id:
        raise ValidationError("features['model_id'] 必须是非空字符串")
    return gpu_id, model_id


def _summary(values: Sequence[float], suffix: str) -> DurationPrediction:
    p50 = float(np.mean(values))
    p95 = max(p50, float(np.percentile(values, 95)))
    return DurationPrediction(p50_ms=p50, p95_ms=p95, model_version=suffix)


class GlobalMeanPredictor:
    def __init__(self) -> None:
        self._by_gpu_model: Dict[Tuple[str, str], DurationPrediction] = {}
        self._by_gpu: Dict[str, DurationPrediction] = {}
        self._global: DurationPrediction | None = None

    def fit(self, samples: Sequence[LabeledSample]) -> "GlobalMeanPredictor":
        if not samples:
            raise ValidationError("samples 不得为空")
        by_gpu_model: DefaultDict[Tuple[str, str], list[float]] = defaultdict(list)
        by_gpu: DefaultDict[str, list[float]] = defaultdict(list)
        global_values = []
        for sample in samples:
            by_gpu_model[(sample.gpu_id, sample.model_id)].append(sample.target_ms)
            by_gpu[sample.gpu_id].append(sample.target_ms)
            global_values.append(sample.target_ms)
        self._by_gpu_model = {
            key: _summary(values, "global-mean:gpu-model")
            for key, values in by_gpu_model.items()
        }
        self._by_gpu = {
            key: _summary(values, "global-mean:gpu")
            for key, values in by_gpu.items()
        }
        self._global = _summary(global_values, "global-mean:global")
        return self

    def predict(self, features: Mapping[str, FeatureValue]) -> DurationPrediction:
        if self._global is None:
            raise ModelNotFittedError("GlobalMeanPredictor 必须先拟合再执行预测")
        gpu_id, model_id = _identity(features)
        return self._by_gpu_model.get(
            (gpu_id, model_id), self._by_gpu.get(gpu_id, self._global)
        )

    def artifact_parameters(self) -> dict:
        return {"aggregation": "mean", "p95": "empirical"}


class EWMAPredictor:
    def __init__(self, alpha: float, window_size: int, default_ms: float) -> None:
        self.stats = OnlineDurationStats(alpha, window_size, default_ms)
        self._fitted = False

    def fit(self, samples: Sequence[LabeledSample]) -> "EWMAPredictor":
        if not samples:
            raise ValidationError("samples 不得为空")
        for sample in samples:
            self.observe(sample.gpu_id, sample.model_id, sample.target_ms)
        self._fitted = True
        return self

    def observe(self, gpu_id: str, model_id: str, duration_ms: float) -> None:
        self.stats.observe(gpu_id, model_id, duration_ms)

    def predict(self, features: Mapping[str, FeatureValue]) -> DurationPrediction:
        if not self._fitted:
            raise ModelNotFittedError("EWMAPredictor 必须先拟合再执行预测")
        gpu_id, model_id = _identity(features)
        return self.stats.estimate(gpu_id, model_id)

    def artifact_parameters(self) -> dict:
        return {
            "alpha": self.stats.alpha,
            "window_size": self.stats.window_size,
            "default_ms": self.stats.default_ms,
            "p95": "empirical_recent_window",
        }
