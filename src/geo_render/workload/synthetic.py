"""Seeded open-arrival synthetic traces for implementation verification."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

from geo_render.common.errors import ValidationError
from geo_render.common.types import Camera, RenderRequest, TraceRecord


@dataclass(frozen=True)
class SyntheticTraceConfig:
    seed: int
    request_count: int
    arrival_rate_per_second: float
    user_count: int
    trajectory_count: int
    model_base_render_ms: Mapping[str, float]
    gpu_speed: Mapping[str, float]
    output_sizes: Tuple[Tuple[int, int], ...]
    sample_steps: Tuple[float, ...]
    slo_ms: float

    def __post_init__(self) -> None:
        if self.request_count <= 0:
            raise ValidationError("request_count must be positive")
        if self.user_count <= 0 or self.trajectory_count <= 0:
            raise ValidationError("user_count and trajectory_count must be positive")
        if not math.isfinite(self.arrival_rate_per_second) or self.arrival_rate_per_second <= 0:
            raise ValidationError("arrival_rate_per_second must be finite and positive")
        models = dict(self.model_base_render_ms)
        speeds = dict(self.gpu_speed)
        if not models or any(
            not math.isfinite(value) or value <= 0 for value in models.values()
        ):
            raise ValidationError(
                "model_base_render_ms must contain finite positive values"
            )
        if not speeds or any(
            not math.isfinite(value) or value <= 0 for value in speeds.values()
        ):
            raise ValidationError("gpu_speed must contain finite positive values")
        if not self.output_sizes or any(
            width <= 0 or height <= 0 for width, height in self.output_sizes
        ):
            raise ValidationError("output_sizes must contain positive dimensions")
        if not self.sample_steps or any(value <= 0 for value in self.sample_steps):
            raise ValidationError("sample_steps must contain positive values")
        if not math.isfinite(self.slo_ms) or self.slo_ms <= 0:
            raise ValidationError("slo_ms must be finite and positive")
        object.__setattr__(self, "model_base_render_ms", MappingProxyType(models))
        object.__setattr__(self, "gpu_speed", MappingProxyType(speeds))


def generate_synthetic_trace(config: SyntheticTraceConfig) -> Tuple[TraceRecord, ...]:
    """Generate synthetic measurements that are always labeled as synthetic."""
    rng = random.Random(config.seed)
    arrival_ms = 0.0
    model_ids = tuple(sorted(config.model_base_render_ms))
    gpu_ids = tuple(sorted(config.gpu_speed))
    records = []
    rate_per_ms = config.arrival_rate_per_second / 1000.0

    for index in range(config.request_count):
        arrival_ms += rng.expovariate(rate_per_ms)
        model_id = rng.choice(model_ids)
        width, height = rng.choice(config.output_sizes)
        sample_step = rng.choice(config.sample_steps)
        user_index = rng.randrange(config.user_count)
        trajectory_index = rng.randrange(config.trajectory_count)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = rng.uniform(-0.35 * math.pi, 0.35 * math.pi)
        distance = rng.uniform(2.5, 7.5)
        position = (
            distance * math.cos(phi) * math.cos(theta),
            distance * math.cos(phi) * math.sin(theta),
            distance * math.sin(phi),
        )
        clip_fraction = rng.choice((0.5, 0.75, 1.0))
        shadows = rng.random() < 0.2
        request = RenderRequest(
            request_id=f"request-{index:06d}",
            user_id=f"user-{user_index:03d}",
            session_id=f"session-{user_index:03d}-{trajectory_index:03d}",
            trajectory_id=f"trajectory-{trajectory_index:03d}",
            model_id=model_id,
            camera=Camera(
                position=position,
                focal_point=(0.0, 0.0, 0.0),
                view_up=(0.0, 0.0, 1.0),
                projection="perspective",
                view_angle_deg=30.0,
            ),
            clip_fraction=clip_fraction,
            output_width=width,
            output_height=height,
            sample_step=sample_step,
            shadows=shadows,
            transfer_function_id="synthetic-default",
            arrival_ms=arrival_ms,
            slo_ms=config.slo_ms,
        )
        pixel_factor = (width * height) / (640.0 * 480.0)
        sampling_factor = 1.0 / sample_step
        content_factor = (0.5 + 0.5 * clip_fraction) * (
            1.2 if shadows else 1.0
        )
        view_factor = 0.9 + 0.2 * abs(math.sin(theta) * math.cos(phi))
        render_by_gpu = {}
        readback_by_gpu = {}
        for gpu_id in gpu_ids:
            speed = config.gpu_speed[gpu_id]
            jitter = math.exp(rng.normalvariate(0.0, 0.03))
            render_by_gpu[gpu_id] = (
                config.model_base_render_ms[model_id]
                * pixel_factor
                * sampling_factor
                * content_factor
                * view_factor
                * jitter
                / speed
            )
            readback_by_gpu[gpu_id] = (1.0 + 1.5 * pixel_factor) / speed
        encode_ms = 1.5 + 1.2 * pixel_factor
        isolated = statistics.median(
            render_by_gpu[gpu_id] + readback_by_gpu[gpu_id] + encode_ms
            for gpu_id in gpu_ids
        )
        records.append(
            TraceRecord(
                request=request,
                actual_render_ms_by_gpu=render_by_gpu,
                actual_readback_ms_by_gpu=readback_by_gpu,
                actual_encode_ms=encode_ms,
                isolated_p50_ms=isolated,
                source="synthetic",
            )
        )
    return tuple(records)
