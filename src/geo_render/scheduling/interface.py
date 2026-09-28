"""调度契约与不可变决策上下文。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, Tuple

from geo_render.common.errors import ValidationError
from geo_render.common.types import (
    ModelManifest,
    RenderRequest,
    ScheduleDecision,
    WorkerSnapshot,
)
from geo_render.prediction.online_stats import OnlineDurationStats


@dataclass(frozen=True)
class SchedulerContext:
    now_ms: float
    workers: Tuple[WorkerSnapshot, ...]
    manifests: Mapping[str, ModelManifest]
    history: OnlineDurationStats
    predicted_readback_ms_by_gpu: Mapping[str, float]
    predicted_encode_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.now_ms) or self.now_ms < 0:
            raise ValidationError("调度器 now_ms 必须有限且非负")
        if not self.workers:
            raise ValidationError("调度器至少需要一个 worker")
        ordered = tuple(sorted(self.workers, key=lambda worker: worker.gpu_id))
        gpu_ids = [worker.gpu_id for worker in ordered]
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValidationError("调度器的 worker GPU ID 必须唯一")
        manifests = dict(self.manifests)
        if not manifests:
            raise ValidationError("调度器的 manifests 不得为空")
        readback = dict(self.predicted_readback_ms_by_gpu)
        if set(readback) != set(gpu_ids):
            raise ValidationError("回读预测键必须与 worker GPU ID 一致")
        if any(
            not math.isfinite(value) or value < 0 for value in readback.values()
        ):
            raise ValidationError("回读预测值必须有限且非负")
        if not math.isfinite(self.predicted_encode_ms) or self.predicted_encode_ms < 0:
            raise ValidationError("predicted_encode_ms 必须有限且非负")
        object.__setattr__(self, "workers", ordered)
        object.__setattr__(self, "manifests", MappingProxyType(manifests))
        object.__setattr__(
            self, "predicted_readback_ms_by_gpu", MappingProxyType(readback)
        )

    def worker(self, gpu_id: str) -> WorkerSnapshot:
        for worker in self.workers:
            if worker.gpu_id == gpu_id:
                return worker
        raise ValidationError(f"未知 gpu_id {gpu_id!r}")


class SchedulerPolicy(Protocol):
    @property
    def name(self) -> str:
        """写入实验输出的稳定策略标识符。"""
        ...

    def choose(
        self, request: RenderRequest, context: SchedulerContext
    ) -> ScheduleDecision:
        """在不修改集群状态的前提下选出一个 worker。"""
        ...
