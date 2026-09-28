"""Leakage-safe labeled samples and group-based dataset splitting."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Mapping, Sequence, Tuple, TypeVar

from geo_render.common.errors import ValidationError

from .features import FeatureValue


@dataclass(frozen=True)
class LabeledSample:
    request_id: str
    group_id: str
    gpu_id: str
    model_id: str
    features: Mapping[str, FeatureValue]
    target_ms: float

    def __post_init__(self) -> None:
        for path in ("request_id", "group_id", "gpu_id", "model_id"):
            if not getattr(self, path):
                raise ValidationError(f"{path} must be non-empty")
        if not self.features:
            raise ValidationError("features must not be empty")
        if not math.isfinite(self.target_ms) or self.target_ms <= 0:
            raise ValidationError("target_ms must be finite and positive")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


T = TypeVar("T")


def _group_id(record: object) -> str:
    explicit = getattr(record, "group_id", None)
    if explicit:
        return str(explicit)
    request = getattr(record, "request", None)
    trajectory_id = getattr(request, "trajectory_id", None)
    if trajectory_id:
        return str(trajectory_id)
    raise ValidationError(
        "each record must expose group_id or request.trajectory_id for leakage-safe splitting"
    )


def group_split(
    records: Sequence[T],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> Tuple[Tuple[T, ...], Tuple[T, ...], Tuple[T, ...]]:
    """Split complete groups while preserving original row order in each partition."""
    if not records:
        raise ValidationError("records must not be empty")
    if not 0 < train_fraction < 1:
        raise ValidationError("train_fraction must be in (0, 1)")
    if not 0 < validation_fraction < 1:
        raise ValidationError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValidationError("train and validation fractions must sum to less than 1")

    group_ids = sorted({_group_id(record) for record in records})
    if len(group_ids) < 3:
        raise ValidationError("at least three groups are required for train/validation/test")
    random.Random(seed).shuffle(group_ids)
    train_count = max(1, int(len(group_ids) * train_fraction))
    validation_count = max(1, int(len(group_ids) * validation_fraction))
    if train_count + validation_count >= len(group_ids):
        validation_count = max(1, len(group_ids) - train_count - 1)
    if train_count + validation_count >= len(group_ids):
        train_count = len(group_ids) - validation_count - 1

    train_groups = set(group_ids[:train_count])
    validation_groups = set(
        group_ids[train_count : train_count + validation_count]
    )
    test_groups = set(group_ids[train_count + validation_count :])
    partitions = (
        tuple(record for record in records if _group_id(record) in train_groups),
        tuple(record for record in records if _group_id(record) in validation_groups),
        tuple(record for record in records if _group_id(record) in test_groups),
    )
    if any(not part for part in partitions):
        raise ValidationError("group split produced an empty partition")
    return partitions
