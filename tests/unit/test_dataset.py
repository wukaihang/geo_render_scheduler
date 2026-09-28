from dataclasses import replace

from geo_render.common.types import TraceRecord
from geo_render.prediction.dataset import group_split

from test_types import make_request


def make_records() -> tuple[TraceRecord, ...]:
    rows = []
    for group_index in range(10):
        for frame_index in range(2):
            request = replace(
                make_request(),
                request_id=f"request-{group_index}-{frame_index}",
                session_id=f"session-{group_index}",
                trajectory_id=f"trajectory-{group_index}",
                arrival_ms=float(group_index * 10 + frame_index),
            )
            rows.append(
                TraceRecord(
                    request=request,
                    actual_render_ms_by_gpu={"gpu-0": 20.0, "gpu-1": 25.0},
                    actual_readback_ms_by_gpu={"gpu-0": 2.0, "gpu-1": 2.5},
                    actual_encode_ms=3.0,
                    isolated_p50_ms=25.0,
                    source="synthetic",
                )
            )
    return tuple(rows)


def test_group_split_never_splits_a_trajectory() -> None:
    train, validation, test = group_split(
        make_records(), train_fraction=0.6, validation_fraction=0.2, seed=7
    )
    groups = [
        {record.request.trajectory_id for record in part}
        for part in (train, validation, test)
    ]
    assert groups[0].isdisjoint(groups[1] | groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert all(part for part in (train, validation, test))


def test_group_split_is_deterministic_and_preserves_input_order_within_parts() -> None:
    records = make_records()
    first = group_split(records, 0.6, 0.2, seed=11)
    second = group_split(records, 0.6, 0.2, seed=11)
    assert first == second
    for part in first:
        arrivals = [record.request.arrival_ms for record in part]
        assert arrivals == sorted(arrivals)
