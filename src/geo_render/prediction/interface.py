"""调度器与训练命令共享的预测器契约。"""

from typing import Mapping, Protocol, Sequence

from geo_render.common.types import DurationPrediction

from .dataset import LabeledSample
from .features import FeatureValue


class DurationPredictor(Protocol):
    def fit(self, samples: Sequence[LabeledSample]) -> "DurationPredictor":
        """使用已完成请求样本拟合预测器。"""
        ...

    def predict(self, features: Mapping[str, FeatureValue]) -> DurationPrediction:
        """预测驻留模型的渲染耗时分位数。"""
        ...
