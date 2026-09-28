"""经过校验的 worker 与集群队列状态机。"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Tuple

from geo_render.common.errors import StateTransitionError, ValidationError
from geo_render.common.types import DeviceState, QueuedRequest, WorkerSnapshot


class WorkerState:
    def __init__(self, device: DeviceState) -> None:
        self.device = device
        self.current: Optional[QueuedRequest] = None
        self.current_started_ms: Optional[float] = None
        self.current_predicted_finish_ms: Optional[float] = None
        self.queue: Deque[QueuedRequest] = deque()

    def _contains(self, request_id: str) -> bool:
        return bool(
            (self.current and self.current.request.request_id == request_id)
            or any(item.request.request_id == request_id for item in self.queue)
        )

    def enqueue(self, queued: QueuedRequest) -> None:
        request_id = queued.request.request_id
        if self._contains(request_id):
            raise StateTransitionError(
                f"worker {self.device.gpu_id!r} 上的 request_id 重复：{request_id!r}"
            )
        self.queue.append(queued)

    def start_next(self, now_ms: float) -> Optional[QueuedRequest]:
        if not math.isfinite(now_ms) or now_ms < 0:
            raise ValidationError("now_ms 必须有限且非负")
        if self.current is not None:
            raise StateTransitionError(
                f"worker {self.device.gpu_id!r} 已在运行请求 "
                f"{self.current.request.request_id!r}"
            )
        if not self.queue:
            return None
        queued = self.queue.popleft()
        self.current = queued
        self.current_started_ms = now_ms
        self.current_predicted_finish_ms = now_ms + queued.predicted_service_ms
        return queued

    def complete(self, request_id: str, now_ms: float) -> QueuedRequest:
        if self.current is None:
            raise StateTransitionError(
                f"worker {self.device.gpu_id!r} 没有可完成的运行中请求"
            )
        if self.current.request.request_id != request_id:
            raise StateTransitionError(
                f"完成事件 request_id {request_id!r} 与运行中请求 "
                f"{self.current.request.request_id!r} 不一致"
            )
        if self.current_started_ms is None or now_ms < self.current_started_ms:
            raise StateTransitionError("完成时间早于请求开始时间")
        completed = self.current
        self.current = None
        self.current_started_ms = None
        self.current_predicted_finish_ms = None
        return completed

    def snapshot(self) -> WorkerSnapshot:
        return WorkerSnapshot(
            gpu_id=self.device.gpu_id,
            device=self.device,
            current_request_id=(
                self.current.request.request_id if self.current is not None else None
            ),
            running_predicted_finish_ms=self.current_predicted_finish_ms,
            queued=tuple(self.queue),
        )


class ClusterState:
    def __init__(self, devices: Iterable[DeviceState]) -> None:
        self._workers: Dict[str, WorkerState] = {}
        for device in devices:
            if device.gpu_id in self._workers:
                raise ValidationError(f"gpu_id 重复：{device.gpu_id!r}")
            self._workers[device.gpu_id] = WorkerState(device)
        if not self._workers:
            raise ValidationError("集群至少需要一个设备")

    @property
    def gpu_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._workers))

    def worker(self, gpu_id: str) -> WorkerState:
        try:
            return self._workers[gpu_id]
        except KeyError as error:
            raise ValidationError(f"未知 gpu_id {gpu_id!r}") from error

    def _request_exists(self, request_id: str) -> bool:
        return any(worker._contains(request_id) for worker in self._workers.values())

    def enqueue(self, gpu_id: str, queued: QueuedRequest) -> None:
        if self._request_exists(queued.request.request_id):
            raise StateTransitionError(
                f"集群中的 request_id 重复：{queued.request.request_id!r}"
            )
        self.worker(gpu_id).enqueue(queued)

    def snapshots(self) -> Tuple[WorkerSnapshot, ...]:
        return tuple(self._workers[gpu_id].snapshot() for gpu_id in self.gpu_ids)
