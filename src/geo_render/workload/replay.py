"""确定性的请求级离散事件回放。"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

from geo_render.common.errors import ValidationError
from geo_render.common.types import (
    DeviceState,
    ModelManifest,
    QueuedRequest,
    ScheduleDecision,
    TraceRecord,
)
from geo_render.prediction.online_stats import OnlineDurationStats
from geo_render.scheduling.interface import SchedulerContext, SchedulerPolicy
from geo_render.scheduling.state import ClusterState


@dataclass(frozen=True)
class ReplayDecision:
    at_ms: float
    decision: ScheduleDecision
    workload_gap_ms: float


@dataclass(frozen=True)
class CompletedRequest:
    request_id: str
    user_id: str
    session_id: str
    trajectory_id: str
    model_id: str
    gpu_id: str
    arrival_ms: float
    dispatch_ms: float
    start_ms: float
    finish_ms: float
    queue_ms: float
    render_ms: float
    readback_ms: float
    encode_ms: float
    end_to_end_ms: float
    predicted_render_p50_ms: float
    predicted_render_p95_ms: float
    predicted_finish_ms: float
    slo_ms: Optional[float]
    isolated_p50_ms: Optional[float]
    source: str


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    source: str
    completed: Tuple[CompletedRequest, ...]
    decisions: Tuple[ReplayDecision, ...]
    worker_busy_ms: Mapping[str, float]
    first_arrival_ms: float
    last_finish_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worker_busy_ms", MappingProxyType(dict(self.worker_busy_ms))
        )


class ReplayEngine:
    def __init__(
        self,
        devices: Iterable[DeviceState],
        manifests: Mapping[str, ModelManifest],
        predicted_readback_ms_by_gpu: Mapping[str, float],
        predicted_encode_ms: float,
        default_history_ms: float,
        history_alpha: float = 0.2,
        history_window_size: int = 64,
    ) -> None:
        self.devices = tuple(sorted(devices, key=lambda device: device.gpu_id))
        if not self.devices:
            raise ValidationError("回放至少需要一个设备")
        self.manifests = MappingProxyType(dict(manifests))
        self.predicted_readback_ms_by_gpu = MappingProxyType(
            dict(predicted_readback_ms_by_gpu)
        )
        if set(self.predicted_readback_ms_by_gpu) != {
            device.gpu_id for device in self.devices
        }:
            raise ValidationError("回读预测键必须与回放设备一致")
        if not math.isfinite(predicted_encode_ms) or predicted_encode_ms < 0:
            raise ValidationError("predicted_encode_ms 必须有限且非负")
        self.predicted_encode_ms = predicted_encode_ms
        self.default_history_ms = default_history_ms
        self.history_alpha = history_alpha
        self.history_window_size = history_window_size

    def _validate_trace(self, trace: Tuple[TraceRecord, ...]) -> None:
        if not trace:
            raise ValidationError("回放轨迹不得为空")
        request_ids = [record.request.request_id for record in trace]
        if len(request_ids) != len(set(request_ids)):
            raise ValidationError("回放轨迹中的请求 ID 必须唯一")
        arrivals = [record.request.arrival_ms for record in trace]
        if arrivals != sorted(arrivals):
            raise ValidationError("回放轨迹必须按 arrival_ms 排序")
        gpu_ids = {device.gpu_id for device in self.devices}
        for record in trace:
            if set(record.actual_render_ms_by_gpu) != gpu_ids:
                raise ValidationError("轨迹渲染 GPU ID 必须与回放设备一致")
            if set(record.actual_readback_ms_by_gpu) != gpu_ids:
                raise ValidationError("轨迹回读 GPU ID 必须与回放设备一致")
            if record.request.model_id not in self.manifests:
                raise ValidationError(
                    f"缺少 model_id {record.request.model_id!r} 对应的 manifest"
                )

    def run(
        self, trace: Iterable[TraceRecord], policy: SchedulerPolicy
    ) -> ReplayResult:
        records = tuple(trace)
        self._validate_trace(records)
        sources = {record.source for record in records}
        if len(sources) != 1:
            raise ValidationError("一次回放不能混用实测与模拟来源")
        source = next(iter(sources))
        by_id = {record.request.request_id: record for record in records}
        cluster = ClusterState(self.devices)
        history = OnlineDurationStats(
            alpha=self.history_alpha,
            window_size=self.history_window_size,
            default_ms=self.default_history_ms,
        )
        busy = {device.gpu_id: 0.0 for device in self.devices}
        decisions = []
        completed = []
        decision_by_request: Dict[str, ScheduleDecision] = {}
        start_by_request: Dict[str, float] = {}
        event_sequence = itertools.count()
        events = []
        for record in records:
            heapq.heappush(
                events,
                (
                    record.request.arrival_ms,
                    1,
                    next(event_sequence),
                    "arrival",
                    record.request.request_id,
                    None,
                ),
            )

        def start_next(gpu_id: str, now_ms: float) -> None:
            worker = cluster.worker(gpu_id)
            queued = worker.start_next(now_ms)
            if queued is None:
                return
            request_id = queued.request.request_id
            record = by_id[request_id]
            render_ms = record.actual_render_ms_by_gpu[gpu_id]
            readback_ms = record.actual_readback_ms_by_gpu[gpu_id]
            service_ms = render_ms + readback_ms + record.actual_encode_ms
            finish_ms = now_ms + service_ms
            busy[gpu_id] += service_ms
            start_by_request[request_id] = now_ms
            heapq.heappush(
                events,
                (
                    finish_ms,
                    0,
                    next(event_sequence),
                    "completion",
                    request_id,
                    gpu_id,
                ),
            )

        while events:
            now_ms, _, _, event_type, request_id, event_gpu_id = heapq.heappop(events)
            record = by_id[request_id]
            request = record.request
            if event_type == "arrival":
                context = SchedulerContext(
                    now_ms=now_ms,
                    workers=cluster.snapshots(),
                    manifests=self.manifests,
                    history=history,
                    predicted_readback_ms_by_gpu=self.predicted_readback_ms_by_gpu,
                    predicted_encode_ms=self.predicted_encode_ms,
                )
                decision = policy.choose(request, context)
                if decision.request_id != request_id:
                    raise ValidationError("策略返回了其他请求的决策")
                selected_cost = decision.costs[decision.gpu_id]
                queued = QueuedRequest(
                    request=request,
                    predicted_p95_ms=selected_cost.predicted_render_p95_ms,
                    predicted_readback_ms=selected_cost.predicted_readback_ms,
                    predicted_encode_ms=selected_cost.predicted_encode_ms,
                    assigned_ms=now_ms,
                )
                cluster.enqueue(decision.gpu_id, queued)
                decision_by_request[request_id] = decision
                queued_work = [
                    cost.running_remaining_ms + cost.queued_work_ms
                    for cost in decision.costs.values()
                ]
                decisions.append(
                    ReplayDecision(
                        at_ms=now_ms,
                        decision=decision,
                        workload_gap_ms=max(queued_work) - min(queued_work),
                    )
                )
                worker = cluster.worker(decision.gpu_id)
                if worker.current is None:
                    start_next(decision.gpu_id, now_ms)
                continue

            if event_type != "completion" or event_gpu_id is None:
                raise ValidationError(f"未知回放事件 {event_type!r}")
            worker = cluster.worker(event_gpu_id)
            queued = worker.complete(request_id, now_ms)
            start_ms = start_by_request.pop(request_id)
            render_ms = record.actual_render_ms_by_gpu[event_gpu_id]
            readback_ms = record.actual_readback_ms_by_gpu[event_gpu_id]
            decision = decision_by_request[request_id]
            selected_cost = decision.costs[event_gpu_id]
            completed.append(
                CompletedRequest(
                    request_id=request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    trajectory_id=request.trajectory_id,
                    model_id=request.model_id,
                    gpu_id=event_gpu_id,
                    arrival_ms=request.arrival_ms,
                    dispatch_ms=queued.assigned_ms,
                    start_ms=start_ms,
                    finish_ms=now_ms,
                    queue_ms=start_ms - request.arrival_ms,
                    render_ms=render_ms,
                    readback_ms=readback_ms,
                    encode_ms=record.actual_encode_ms,
                    end_to_end_ms=now_ms - request.arrival_ms,
                    predicted_render_p50_ms=selected_cost.predicted_render_p50_ms,
                    predicted_render_p95_ms=selected_cost.predicted_render_p95_ms,
                    predicted_finish_ms=request.arrival_ms + selected_cost.total_ms,
                    slo_ms=request.slo_ms,
                    isolated_p50_ms=record.isolated_p50_ms,
                    source=record.source,
                )
            )
            history.observe(event_gpu_id, request.model_id, render_ms)
            observer = getattr(policy, "observe_completed", None)
            if callable(observer):
                observer(event_gpu_id, request.model_id, render_ms)
            start_next(event_gpu_id, now_ms)

        if len(completed) != len(records):
            raise RuntimeError("回放结束时仍有请求未完成")
        completed.sort(key=lambda row: (row.finish_ms, row.request_id))
        return ReplayResult(
            policy=policy.name,
            source=source,
            completed=tuple(completed),
            decisions=tuple(decisions),
            worker_busy_ms=busy,
            first_arrival_ms=records[0].request.arrival_ms,
            last_finish_ms=max(row.finish_ms for row in completed),
        )
