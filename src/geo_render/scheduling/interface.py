"""Scheduling contracts and immutable decision context."""

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
            raise ValidationError("scheduler now_ms must be finite and non-negative")
        if not self.workers:
            raise ValidationError("scheduler requires at least one worker")
        ordered = tuple(sorted(self.workers, key=lambda worker: worker.gpu_id))
        gpu_ids = [worker.gpu_id for worker in ordered]
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValidationError("scheduler worker GPU IDs must be unique")
        manifests = dict(self.manifests)
        if not manifests:
            raise ValidationError("scheduler manifests must not be empty")
        readback = dict(self.predicted_readback_ms_by_gpu)
        if set(readback) != set(gpu_ids):
            raise ValidationError("readback prediction keys must match worker GPU IDs")
        if any(
            not math.isfinite(value) or value < 0 for value in readback.values()
        ):
            raise ValidationError("readback predictions must be finite and non-negative")
        if not math.isfinite(self.predicted_encode_ms) or self.predicted_encode_ms < 0:
            raise ValidationError("predicted_encode_ms must be finite and non-negative")
        object.__setattr__(self, "workers", ordered)
        object.__setattr__(self, "manifests", MappingProxyType(manifests))
        object.__setattr__(
            self, "predicted_readback_ms_by_gpu", MappingProxyType(readback)
        )

    def worker(self, gpu_id: str) -> WorkerSnapshot:
        for worker in self.workers:
            if worker.gpu_id == gpu_id:
                return worker
        raise ValidationError(f"unknown gpu_id {gpu_id!r}")


class SchedulerPolicy(Protocol):
    @property
    def name(self) -> str:
        """Stable policy identifier written into experiment outputs."""
        ...

    def choose(
        self, request: RenderRequest, context: SchedulerContext
    ) -> ScheduleDecision:
        """Choose exactly one worker without mutating cluster state."""
        ...
