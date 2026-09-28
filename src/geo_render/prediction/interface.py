"""Predictor contracts shared by schedulers and training commands."""

from typing import Mapping, Protocol, Sequence

from geo_render.common.types import DurationPrediction

from .dataset import LabeledSample
from .features import FeatureValue


class DurationPredictor(Protocol):
    def fit(self, samples: Sequence[LabeledSample]) -> "DurationPredictor":
        """Fit the predictor from completed-request samples."""
        ...

    def predict(self, features: Mapping[str, FeatureValue]) -> DurationPrediction:
        """Predict resident-model render duration quantiles."""
        ...
