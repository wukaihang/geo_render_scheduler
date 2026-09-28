"""Tail latency, SLO, utilization, balance, and fairness metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from geo_render.common.errors import ValidationError
from geo_render.workload.replay import ReplayResult


def jain_index(values: Sequence[float]) -> float:
    if not values:
        raise ValidationError("Jain index requires at least one value")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValidationError("Jain index values must be finite and non-negative")
    squared_sum = sum(value * value for value in values)
    if squared_sum == 0:
        return 1.0
    return sum(values) ** 2 / (len(values) * squared_sum)


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _fairness(result: ReplayResult) -> Optional[Dict[str, object]]:
    if any(row.isolated_p50_ms is None for row in result.completed):
        return None
    by_user: Dict[str, List[float]] = defaultdict(list)
    for row in result.completed:
        assert row.isolated_p50_ms is not None
        by_user[row.user_id].append(row.end_to_end_ms / row.isolated_p50_ms)
    user_means = {
        user_id: float(np.mean(values)) for user_id, values in sorted(by_user.items())
    }
    means = list(user_means.values())
    return {
        "user_mean_slowdown": user_means,
        "max_user_mean_slowdown": max(means),
        "p95_user_mean_slowdown": float(np.percentile(means, 95)),
        "jain_index": jain_index(means),
    }


def scheduling_metrics(result: ReplayResult) -> Dict[str, object]:
    if not result.completed:
        raise ValidationError("scheduling metrics require completed requests")
    observation_ms = result.last_finish_ms - result.first_arrival_ms
    if observation_ms <= 0:
        raise ValidationError("replay observation window must be positive")
    slo_rows = [row for row in result.completed if row.slo_ms is not None]
    slo_violation_rate = (
        None
        if not slo_rows
        else sum(row.end_to_end_ms > row.slo_ms for row in slo_rows) / len(slo_rows)
    )
    gaps = [decision.workload_gap_ms for decision in result.decisions]
    return {
        "policy": result.policy,
        "source": result.source,
        "completed_requests": len(result.completed),
        "throughput_requests_per_second": len(result.completed)
        / (observation_ms / 1000.0),
        "end_to_end_ms": _percentiles(
            [row.end_to_end_ms for row in result.completed]
        ),
        "queue_ms": _percentiles([row.queue_ms for row in result.completed]),
        "slo_violation_rate": slo_violation_rate,
        "fairness": _fairness(result),
        "gpu_busy_ratio": {
            gpu_id: busy_ms / observation_ms
            for gpu_id, busy_ms in sorted(result.worker_busy_ms.items())
        },
        "workload_gap_ms": {
            "mean": float(np.mean(gaps)) if gaps else 0.0,
            "max": max(gaps) if gaps else 0.0,
        },
    }
