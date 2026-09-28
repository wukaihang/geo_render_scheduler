import math
from pathlib import Path

import pytest

from geo_render.common.errors import ValidationError
from geo_render.workload.synthetic import SyntheticTraceConfig, generate_synthetic_trace
from geo_render.workload.trace import read_trace_csv, trace_sha256, write_trace_csv


def make_config(seed: int = 17, request_count: int = 24) -> SyntheticTraceConfig:
    return SyntheticTraceConfig(
        seed=seed,
        request_count=request_count,
        arrival_rate_per_second=20.0,
        user_count=4,
        trajectory_count=6,
        model_base_render_ms={"model-a": 20.0, "model-b": 50.0},
        gpu_speed={"gpu-0": 1.0, "gpu-1": 0.8},
        output_sizes=((640, 480), (1280, 720)),
        sample_steps=(0.5, 1.0),
        slo_ms=200.0,
    )


def test_trace_is_identical_for_same_seed() -> None:
    config = make_config()
    first = generate_synthetic_trace(config)
    second = generate_synthetic_trace(config)
    assert first == second
    assert all(record.source == "synthetic" for record in first)


def test_trace_changes_when_seed_changes() -> None:
    assert generate_synthetic_trace(make_config(seed=1)) != generate_synthetic_trace(
        make_config(seed=2)
    )


def test_trace_csv_round_trip_is_lossless_and_hash_stable(tmp_path: Path) -> None:
    trace = generate_synthetic_trace(make_config())
    path = tmp_path / "trace.csv"
    write_trace_csv(path, trace)
    restored = read_trace_csv(path)
    assert restored == trace
    assert trace_sha256(restored) == trace_sha256(trace)


def test_generated_arrivals_are_non_decreasing_and_ids_are_unique() -> None:
    trace = generate_synthetic_trace(make_config())
    arrivals = [record.request.arrival_ms for record in trace]
    request_ids = [record.request.request_id for record in trace]
    assert arrivals == sorted(arrivals)
    assert len(request_ids) == len(set(request_ids))


def test_synthetic_config_rejects_non_finite_model_or_gpu_values() -> None:
    with pytest.raises(ValidationError, match="finite positive"):
        SyntheticTraceConfig(
            seed=1,
            request_count=2,
            arrival_rate_per_second=1.0,
            user_count=1,
            trajectory_count=1,
            model_base_render_ms={"model-a": math.nan},
            gpu_speed={"gpu-0": 1.0},
            output_sizes=((10, 10),),
            sample_steps=(1.0,),
            slo_ms=10.0,
        )
