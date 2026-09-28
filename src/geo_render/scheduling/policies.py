"""Round-robin, queue, weighted, predicted-EFT, and offline-oracle policies."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

from geo_render.common.errors import OracleAccessError, ValidationError
from geo_render.common.types import (
    CostEstimate,
    DurationPrediction,
    RenderRequest,
    ScheduleDecision,
    TraceRecord,
    WorkerSnapshot,
)
from geo_render.prediction.features import extract_features
from geo_render.prediction.interface import DurationPredictor

from .interface import SchedulerContext


def _queued_work(worker: WorkerSnapshot) -> float:
    return sum(item.predicted_service_ms for item in worker.queued)


def _cost(
    request: RenderRequest,
    worker: WorkerSnapshot,
    context: SchedulerContext,
    prediction: DurationPrediction,
    readback_override_ms: Optional[float] = None,
    encode_override_ms: Optional[float] = None,
) -> CostEstimate:
    running = (
        0.0
        if worker.running_predicted_finish_ms is None
        else max(0.0, worker.running_predicted_finish_ms - context.now_ms)
    )
    queued = _queued_work(worker)
    readback = (
        context.predicted_readback_ms_by_gpu[worker.gpu_id]
        if readback_override_ms is None
        else readback_override_ms
    )
    encode = context.predicted_encode_ms if encode_override_ms is None else encode_override_ms
    total = running + queued + prediction.p95_ms + readback + encode
    return CostEstimate(
        gpu_id=worker.gpu_id,
        running_remaining_ms=running,
        queued_work_ms=queued,
        predicted_render_p50_ms=prediction.p50_ms,
        predicted_render_p95_ms=prediction.p95_ms,
        predicted_readback_ms=readback,
        predicted_encode_ms=encode,
        total_ms=total,
    )


def _history_costs(
    request: RenderRequest, context: SchedulerContext
) -> Dict[str, CostEstimate]:
    return {
        worker.gpu_id: _cost(
            request,
            worker,
            context,
            context.history.estimate(worker.gpu_id, request.model_id),
        )
        for worker in context.workers
    }


class RoundRobinPolicy:
    name = "round-robin"

    def __init__(self) -> None:
        self._next_index = 0

    def choose(
        self, request: RenderRequest, context: SchedulerContext
    ) -> ScheduleDecision:
        gpu_ids = tuple(worker.gpu_id for worker in context.workers)
        selected = gpu_ids[self._next_index % len(gpu_ids)]
        self._next_index += 1
        return ScheduleDecision(
            request_id=request.request_id,
            gpu_id=selected,
            policy=self.name,
            reason=f"stable_cycle_index={self._next_index - 1}",
            costs=_history_costs(request, context),
        )


class LeastQueuePolicy:
    name = "least-queue"

    def choose(
        self, request: RenderRequest, context: SchedulerContext
    ) -> ScheduleDecision:
        selected_worker = min(
            context.workers,
            key=lambda worker: (
                len(worker.queued)
                + (1 if worker.current_request_id is not None else 0),
                worker.gpu_id,
            ),
        )
        selected_length = len(selected_worker.queued) + (
            1 if selected_worker.current_request_id is not None else 0
        )
        return ScheduleDecision(
            request_id=request.request_id,
            gpu_id=selected_worker.gpu_id,
            policy=self.name,
            reason=f"outstanding_requests={selected_length}",
            costs=_history_costs(request, context),
        )


class StaticWeightedPolicy:
    name = "static-weighted"

    def __init__(self, service_rates: Mapping[str, float]) -> None:
        copied = dict(service_rates)
        if not copied or any(
            not math.isfinite(value) or value <= 0 for value in copied.values()
        ):
            raise ValidationError("service_rates must contain positive finite values")
        self.service_rates = MappingProxyType(copied)

    def choose(
        self, request: RenderRequest, context: SchedulerContext
    ) -> ScheduleDecision:
        context_ids = {worker.gpu_id for worker in context.workers}
        if set(self.service_rates) != context_ids:
            raise ValidationError("service_rates keys must match scheduler workers")
        scores = {
            worker.gpu_id: (
                (1 if worker.current_request_id is not None else 0)
                + len(worker.queued)
                + 1
            )
            / self.service_rates[worker.gpu_id]
            for worker in context.workers
        }
        selected = min(scores, key=lambda gpu_id: (scores[gpu_id], gpu_id))
        return ScheduleDecision(
            request_id=request.request_id,
            gpu_id=selected,
            policy=self.name,
            reason=f"normalized_request_load={scores[selected]:.9g}",
            costs=_history_costs(request, context),
        )


class EFTPolicy:
    def __init__(
        self, predictor: DurationPredictor, policy_name: str = "feature-eft"
    ) -> None:
        if not policy_name:
            raise ValidationError("policy_name must be non-empty")
        self.predictor = predictor
        self.name = policy_name

    def choose(
        self, request: RenderRequest, context: SchedulerContext
    ) -> ScheduleDecision:
        try:
            manifest = context.manifests[request.model_id]
        except KeyError as error:
            raise ValidationError(
                f"missing manifest for model_id {request.model_id!r}"
            ) from error
        costs = {}
        for worker in context.workers:
            history = context.history.estimate(worker.gpu_id, request.model_id)
            features = extract_features(request, manifest, worker.device, history)
            prediction = self.predictor.predict(features)
            costs[worker.gpu_id] = _cost(request, worker, context, prediction)
        selected = min(costs, key=lambda gpu_id: (costs[gpu_id].total_ms, gpu_id))
        return ScheduleDecision(
            request_id=request.request_id,
            gpu_id=selected,
            policy=self.name,
            reason=f"minimum_ect_ms={costs[selected].total_ms:.9g}",
            costs=costs,
        )

    def observe_completed(
        self, gpu_id: str, model_id: str, duration_ms: float
    ) -> None:
        observer = getattr(self.predictor, "observe", None)
        if callable(observer):
            observer(gpu_id, model_id, duration_ms)


_ORACLE_TOKEN = object()


class OracleEFTPolicy:
    name = "oracle-eft"

    def __init__(
        self,
        token: Optional[object] = None,
        trace: Optional[Tuple[TraceRecord, ...]] = None,
    ) -> None:
        if token is not _ORACLE_TOKEN or trace is None:
            raise OracleAccessError(
                "OracleEFTPolicy is available only through for_offline_replay()"
            )
        render = {}
        readback = {}
        encode = {}
        for record in trace:
            request_id = record.request.request_id
            if request_id in render:
                raise OracleAccessError(f"duplicate oracle request {request_id!r}")
            render[request_id] = MappingProxyType(
                dict(record.actual_render_ms_by_gpu)
            )
            readback[request_id] = MappingProxyType(
                dict(record.actual_readback_ms_by_gpu)
            )
            encode[request_id] = record.actual_encode_ms
        if not render:
            raise OracleAccessError("offline oracle trace must not be empty")
        self._actual_render_by_request = MappingProxyType(render)
        self._actual_readback_by_request = MappingProxyType(readback)
        self._actual_encode_by_request = MappingProxyType(encode)

    @classmethod
    def for_offline_replay(
        cls, trace: Tuple[TraceRecord, ...]
    ) -> "OracleEFTPolicy":
        return cls(_ORACLE_TOKEN, tuple(trace))

    @classmethod
    def from_trace(cls, trace: Tuple[TraceRecord, ...]) -> "OracleEFTPolicy":
        return cls.for_offline_replay(trace)

    def choose(
        self, request: RenderRequest, context: SchedulerContext
    ) -> ScheduleDecision:
        try:
            actual_render = self._actual_render_by_request[request.request_id]
            actual_readback = self._actual_readback_by_request[request.request_id]
            actual_encode = self._actual_encode_by_request[request.request_id]
        except KeyError as error:
            raise OracleAccessError(
                f"oracle has no offline durations for request {request.request_id!r}"
            ) from error
        costs = {}
        for worker in context.workers:
            try:
                duration = actual_render[worker.gpu_id]
                readback = actual_readback[worker.gpu_id]
            except KeyError as error:
                raise OracleAccessError(
                    f"oracle request {request.request_id!r} has no duration for "
                    f"GPU {worker.gpu_id!r}"
                ) from error
            prediction = DurationPrediction(duration, duration, "offline-oracle")
            costs[worker.gpu_id] = _cost(
                request,
                worker,
                context,
                prediction,
                readback_override_ms=readback,
                encode_override_ms=actual_encode,
            )
        selected = min(costs, key=lambda gpu_id: (costs[gpu_id].total_ms, gpu_id))
        return ScheduleDecision(
            request_id=request.request_id,
            gpu_id=selected,
            policy=self.name,
            reason=f"minimum_oracle_ect_ms={costs[selected].total_ms:.9g}",
            costs=costs,
        )
