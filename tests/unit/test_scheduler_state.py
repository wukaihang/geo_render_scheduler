from dataclasses import replace

import pytest
from test_types import make_request

from geo_render.common.errors import StateTransitionError
from geo_render.common.types import DeviceState, QueuedRequest
from geo_render.scheduling.state import ClusterState, WorkerState


def make_device(gpu_id: str) -> DeviceState:
    return DeviceState(
        gpu_id=gpu_id,
        model="RTX 5090",
        memory_total_bytes=32 * 1024**3,
        utilization_pct=0.0,
        memory_utilization_pct=0.0,
        temperature_c=40.0,
        power_w=100.0,
        historical_speed=1.0,
        sampled_at_ms=0.0,
    )


def make_queued(request_id: str, arrival_ms: float = 0.0) -> QueuedRequest:
    request = replace(make_request(), request_id=request_id, arrival_ms=arrival_ms)
    return QueuedRequest(
        request=request,
        predicted_p95_ms=20.0,
        predicted_readback_ms=2.0,
        predicted_encode_ms=3.0,
        assigned_ms=arrival_ms,
    )


def test_worker_state_rejects_invalid_transitions() -> None:
    worker = WorkerState(make_device("gpu-0"))
    with pytest.raises(StateTransitionError, match="no running request"):
        worker.complete("missing", 10.0)
    worker.enqueue(make_queued("r1"))
    worker.start_next(0.0)
    with pytest.raises(StateTransitionError, match="already running"):
        worker.start_next(1.0)
    with pytest.raises(StateTransitionError, match="does not match"):
        worker.complete("wrong", 25.0)


def test_worker_snapshot_tracks_running_remainder_and_queued_predictions() -> None:
    worker = WorkerState(make_device("gpu-0"))
    worker.enqueue(make_queued("r1"))
    worker.enqueue(make_queued("r2"))
    started = worker.start_next(10.0)
    snapshot = worker.snapshot()
    assert started.request.request_id == "r1"
    assert snapshot.current_request_id == "r1"
    assert snapshot.running_predicted_finish_ms == 35.0
    assert [item.request.request_id for item in snapshot.queued] == ["r2"]


def test_cluster_rejects_duplicate_request_ids_across_workers() -> None:
    cluster = ClusterState((make_device("gpu-0"), make_device("gpu-1")))
    cluster.worker("gpu-0").enqueue(make_queued("r1"))
    with pytest.raises(StateTransitionError, match="duplicate request_id"):
        cluster.enqueue("gpu-1", make_queued("r1"))
