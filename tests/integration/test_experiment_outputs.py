import json
from pathlib import Path

import pytest

from geo_render.common.errors import ValidationError
from geo_render.experiments.compare import compare_policies
from geo_render.experiments.io import write_run_directory
from geo_render.scheduling.policies import LeastQueuePolicy, RoundRobinPolicy
from geo_render.workload.synthetic import generate_synthetic_trace

from test_replay import make_engine


def make_config(seed: int = 23, request_count: int = 10):
    from geo_render.workload.synthetic import SyntheticTraceConfig

    return SyntheticTraceConfig(
        seed=seed,
        request_count=request_count,
        arrival_rate_per_second=15.0,
        user_count=3,
        trajectory_count=4,
        model_base_render_ms={"model-a": 20.0, "model-b": 50.0},
        gpu_speed={"gpu-0": 1.0, "gpu-1": 0.8},
        output_sizes=((640, 480), (1280, 720)),
        sample_steps=(0.5, 1.0),
        slo_ms=200.0,
    )


def test_run_directory_is_self_contained(tmp_path: Path) -> None:
    config = make_config()
    trace = generate_synthetic_trace(config)
    result = make_engine().run(trace, RoundRobinPolicy())
    output = write_run_directory(
        tmp_path / "run",
        config={"seed": config.seed, "policy": result.policy},
        trace=trace,
        result=result,
    )
    assert {path.name for path in output.iterdir()} == {
        "config.json",
        "trace.csv",
        "requests.csv",
        "decisions.jsonl",
        "summary.json",
        "metadata.json",
    }
    metadata = json.loads((output / "metadata.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert metadata["source"] == "synthetic"
    assert len(metadata["trace_sha256"]) == 64
    assert summary["completed_requests"] == config.request_count


def test_run_directory_rejects_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    config = make_config(request_count=2)
    trace = generate_synthetic_trace(config)
    result = make_engine().run(trace, RoundRobinPolicy())
    with pytest.raises(ValidationError, match="already exists"):
        write_run_directory(output, {"seed": config.seed}, trace, result)


def test_policy_comparison_writes_each_run_and_top_level_summary(tmp_path: Path) -> None:
    config = make_config(request_count=8)
    trace = generate_synthetic_trace(config)
    output = tmp_path / "comparison"
    comparison = compare_policies(
        output_dir=output,
        config={"seed": config.seed},
        trace=trace,
        engine_factory=make_engine,
        policies=(RoundRobinPolicy(), LeastQueuePolicy()),
    )
    assert set(comparison["summaries"]) == {"round-robin", "least-queue"}
    assert comparison["failures"] == {}
    assert (output / "comparison.json").exists()
    assert (output / "round-robin" / "summary.json").exists()
    assert (output / "least-queue" / "summary.json").exists()
