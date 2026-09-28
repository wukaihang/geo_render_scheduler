from dataclasses import replace

import pytest
from test_replay import make_engine
from test_trace import make_config

from geo_render.analysis.scheduling_metrics import jain_index, scheduling_metrics
from geo_render.scheduling.policies import RoundRobinPolicy
from geo_render.workload.replay import ReplayResult
from geo_render.workload.synthetic import generate_synthetic_trace


def test_jain_index_is_one_for_equal_user_slowdown() -> None:
    assert jain_index([2.0, 2.0]) == pytest.approx(1.0)


def test_scheduling_metrics_cover_latency_slo_utilization_and_fairness() -> None:
    trace = generate_synthetic_trace(make_config(request_count=20))
    result = make_engine().run(trace, RoundRobinPolicy())
    metrics = scheduling_metrics(result)
    assert metrics["completed_requests"] == 20
    assert metrics["throughput_requests_per_second"] > 0
    assert set(metrics["end_to_end_ms"]) == {"p50", "p95", "p99"}
    assert set(metrics["queue_ms"]) == {"p50", "p95", "p99"}
    assert 0.0 <= metrics["slo_violation_rate"] <= 1.0
    assert 0.0 < metrics["fairness"]["jain_index"] <= 1.0
    assert set(metrics["gpu_busy_ratio"]) == {"gpu-0", "gpu-1"}
    assert metrics["workload_gap_ms"]["max"] >= metrics["workload_gap_ms"]["mean"]


def test_missing_isolated_labels_produce_null_slowdown_metrics() -> None:
    trace = generate_synthetic_trace(make_config(request_count=4))
    result = make_engine().run(trace, RoundRobinPolicy())
    completed = tuple(replace(row, isolated_p50_ms=None) for row in result.completed)
    without_labels = ReplayResult(
        policy=result.policy,
        source=result.source,
        completed=completed,
        decisions=result.decisions,
        worker_busy_ms=result.worker_busy_ms,
        first_arrival_ms=result.first_arrival_ms,
        last_finish_ms=result.last_finish_ms,
    )
    assert scheduling_metrics(without_labels)["fairness"] is None
