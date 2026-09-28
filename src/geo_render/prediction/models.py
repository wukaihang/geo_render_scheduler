"""Content- and device-aware scikit-learn duration predictors."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from geo_render.common.errors import ModelNotFittedError, ValidationError
from geo_render.common.types import DurationPrediction

from .dataset import LabeledSample
from .features import FeatureValue


def _rows(samples: Sequence[LabeledSample]):
    if not samples:
        raise ValidationError("samples must not be empty")
    return [dict(sample.features) for sample in samples], np.asarray(
        [sample.target_ms for sample in samples], dtype=float
    )


class RidgeDurationPredictor:
    def __init__(self, alpha: float = 1.0, floor_ms: float = 0.001) -> None:
        if alpha < 0 or not math.isfinite(alpha):
            raise ValidationError("alpha must be finite and non-negative")
        if floor_ms <= 0 or not math.isfinite(floor_ms):
            raise ValidationError("floor_ms must be finite and positive")
        self.alpha = alpha
        self.floor_ms = floor_ms
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = Ridge(alpha=alpha, solver="lsqr")
        self.residual_p95_ms: Optional[float] = None

    def fit(
        self,
        samples: Sequence[LabeledSample],
        validation_samples: Optional[Sequence[LabeledSample]] = None,
    ) -> "RidgeDurationPredictor":
        features, targets = _rows(samples)
        matrix = self.vectorizer.fit_transform(features)
        self.model.fit(matrix, targets)
        calibration = validation_samples if validation_samples else samples
        calibration_features, calibration_targets = _rows(calibration)
        predictions = self.model.predict(self.vectorizer.transform(calibration_features))
        upper_residual = float(np.percentile(calibration_targets - predictions, 95))
        self.residual_p95_ms = max(0.0, upper_residual)
        return self

    def predict(self, features: Mapping[str, FeatureValue]) -> DurationPrediction:
        if self.residual_p95_ms is None:
            raise ModelNotFittedError("RidgeDurationPredictor must be fitted before predict")
        raw = float(self.model.predict(self.vectorizer.transform([dict(features)]))[0])
        p50 = max(self.floor_ms, raw)
        return DurationPrediction(
            p50_ms=p50,
            p95_ms=max(p50, p50 + self.residual_p95_ms),
            model_version="ridge-v1",
        )

    def artifact_parameters(self) -> dict:
        return {
            "alpha": self.alpha,
            "floor_ms": self.floor_ms,
            "residual_p95_ms": self.residual_p95_ms,
            "solver": "lsqr",
        }


class MeanGBDTPredictor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        seed: int = 0,
        floor_ms: float = 0.001,
    ) -> None:
        if n_estimators <= 0 or max_depth <= 0:
            raise ValidationError("n_estimators and max_depth must be positive")
        if learning_rate <= 0 or not math.isfinite(learning_rate):
            raise ValidationError("learning_rate must be finite and positive")
        if floor_ms <= 0 or not math.isfinite(floor_ms):
            raise ValidationError("floor_ms must be finite and positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.floor_ms = floor_ms
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        self.residual_p95_ms: Optional[float] = None

    def fit(
        self,
        samples: Sequence[LabeledSample],
        validation_samples: Optional[Sequence[LabeledSample]] = None,
    ) -> "MeanGBDTPredictor":
        features, targets = _rows(samples)
        matrix = self.vectorizer.fit_transform(features)
        self.model.fit(matrix, targets)
        calibration = validation_samples if validation_samples else samples
        calibration_features, calibration_targets = _rows(calibration)
        predictions = self.model.predict(self.vectorizer.transform(calibration_features))
        self.residual_p95_ms = max(
            0.0, float(np.percentile(calibration_targets - predictions, 95))
        )
        return self

    def predict(self, features: Mapping[str, FeatureValue]) -> DurationPrediction:
        if self.residual_p95_ms is None:
            raise ModelNotFittedError("MeanGBDTPredictor must be fitted before predict")
        raw = float(self.model.predict(self.vectorizer.transform([dict(features)]))[0])
        p50 = max(self.floor_ms, raw)
        return DurationPrediction(
            p50_ms=p50,
            p95_ms=max(p50, p50 + self.residual_p95_ms),
            model_version="mean-gbdt-v1",
        )

    def artifact_parameters(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "floor_ms": self.floor_ms,
            "residual_p95_ms": self.residual_p95_ms,
            "loss": "squared_error",
        }


class QuantileGBDTPredictor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        seed: int = 0,
        floor_ms: float = 0.001,
    ) -> None:
        if n_estimators <= 0 or max_depth <= 0:
            raise ValidationError("n_estimators and max_depth must be positive")
        if learning_rate <= 0 or not math.isfinite(learning_rate):
            raise ValidationError("learning_rate must be finite and positive")
        if floor_ms <= 0 or not math.isfinite(floor_ms):
            raise ValidationError("floor_ms must be finite and positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.floor_ms = floor_ms
        self.vectorizer = DictVectorizer(sparse=False)
        common = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": seed,
            "loss": "quantile",
        }
        self.p50_model = GradientBoostingRegressor(alpha=0.50, **common)
        self.p95_model = GradientBoostingRegressor(alpha=0.95, **common)
        self._fitted = False

    def fit(self, samples: Sequence[LabeledSample]) -> "QuantileGBDTPredictor":
        features, targets = _rows(samples)
        matrix = self.vectorizer.fit_transform(features)
        self.p50_model.fit(matrix, targets)
        self.p95_model.fit(matrix, targets)
        self._fitted = True
        return self

    def predict(self, features: Mapping[str, FeatureValue]) -> DurationPrediction:
        if not self._fitted:
            raise ModelNotFittedError(
                "QuantileGBDTPredictor must be fitted before predict"
            )
        matrix = self.vectorizer.transform([dict(features)])
        p50 = max(self.floor_ms, float(self.p50_model.predict(matrix)[0]))
        raw_p95 = max(self.floor_ms, float(self.p95_model.predict(matrix)[0]))
        return DurationPrediction(
            p50_ms=p50,
            p95_ms=max(p50, raw_p95),
            model_version="quantile-gbdt-v1",
        )

    def artifact_parameters(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "floor_ms": self.floor_ms,
            "quantiles": [0.5, 0.95],
            "loss": "quantile",
        }
