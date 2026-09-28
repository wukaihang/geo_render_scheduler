from dataclasses import replace

import pytest
from test_scheduler_state import make_device
from test_types import make_request

from geo_render.common.errors import OracleAccessError
from geo_render.common.types import (
    DurationPrediction,
    ModelManifest,
    QueuedRequest,
    TraceRecord,
    WorkerSnapshot,
)
from geo_render.prediction.baselines import EWMAPredictor
from geo_render.prediction.dataset import LabeledSample
from geo_render.prediction.online_stats import OnlineDurationStats
from geo_render.scheduling.interface import SchedulerContext
from geo_render.scheduling.policies import (
    EFTPolicy,
    LeastQueuePolicy,
    OracleEFTPolicy,
    RoundRobinPolicy,
    StaticWeightedPolicy,
)


class DeviceAwarePredictor:
    def predict(self, features):
        value = 10.0 if features["gpu_id"] == "gpu-0" else 20.0
        return DurationPrediction(value, value, "device-aware-test")


def make_manifest() -> ModelManifest:
    return ModelManifest(
        model_id="model-a",
        dimensions=(128, 128, 128),
        active_voxel_fraction=0.5,
        data_bytes=128**3,
        layer_count=4,
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        sha256="c" * 64,
    )


def make_context(
    gpu0_finish=None,
    gpu0_queue=(),
    gpu1_queue=(),
) -> SchedulerContext:
    history = OnlineDurationStats(alpha=0.5, window_size=8, default_ms=30.0)
    return SchedulerContext(
        now_ms=0.0,
        workers=(
            WorkerSnapshot(
                gpu_id="gpu-0",
                device=make_device("gpu-0"),
                current_request_id="running" if gpu0_finish is not None else None,
                running_predicted_finish_ms=gpu0_finish,
                queued=tuple(gpu0_queue),
            ),
            WorkerSnapshot(
                gpu_id="gpu-1",
                device=make_device("gpu-1"),
                queued=tuple(gpu1_queue),
            ),
        ),
        manifests={"model-a": make_manifest()},
        history=history,
        predicted_readback_ms_by_gpu={"gpu-0": 2.0, "gpu-1": 3.0},
        predicted_encode_ms=4.0,
    )


def make_queued(request_id: str) -> QueuedRequest:
    return QueuedRequest(
        request=replace(make_request(), request_id=request_id),
        predicted_p95_ms=30.0,
        predicted_readback_ms=2.0,
        predicted_encode_ms=4.0,
        assigned_ms=0.0,
    )


def test_feature_eft_selects_lowest_expected_completion() -> None:
    context = make_context(gpu0_finish=80.0)
    decision = EFTPolicy(DeviceAwarePredictor()).choose(make_request(), context)
    assert decision.gpu_id == "gpu-1"
    assert decision.costs["gpu-1"].total_ms < decision.costs["gpu-0"].total_ms
    assert decision.costs["gpu-0"].running_remaining_ms == 80.0


def test_eft_includes_all_queued_predicted_service_stages() -> None:
    context = make_context(gpu0_queue=(make_queued("queued"),))
    decision = EFTPolicy(DeviceAwarePredictor()).choose(make_request(), context)
    assert decision.costs["gpu-0"].queued_work_ms == 36.0
    assert decision.costs["gpu-0"].total_ms == 52.0


def test_round_robin_is_stable_and_cycles() -> None:
    policy = RoundRobinPolicy()
    context = make_context()
    choices = [policy.choose(make_request(), context).gpu_id for _ in range(3)]
    assert choices == ["gpu-0", "gpu-1", "gpu-0"]


def test_least_queue_and_static_weighted_use_their_declared_scores() -> None:
    request = make_request()
    least_context = make_context(gpu0_queue=(make_queued("q0"),))
    assert LeastQueuePolicy().choose(request, least_context).gpu_id == "gpu-1"
    weighted_context = make_context()
    weighted = StaticWeightedPolicy({"gpu-0": 1.0, "gpu-1": 2.0})
    assert weighted.choose(request, weighted_context).gpu_id == "gpu-1"


def test_least_queue_counts_the_running_request() -> None:
    context = make_context(gpu0_finish=100.0)
    assert LeastQueuePolicy().choose(make_request(), context).gpu_id == "gpu-1"


def test_ewma_eft_uses_same_scheduler_contract() -> None:
    row = LabeledSample(
        request_id="profile-1",
        group_id="profile-trajectory",
        gpu_id="gpu-0",
        model_id="model-a",
        features={"gpu_id": "gpu-0", "model_id": "model-a"},
        target_ms=15.0,
    )
    row2 = replace(
        row,
        request_id="profile-2",
        gpu_id="gpu-1",
        features={"gpu_id": "gpu-1", "model_id": "model-a"},
        target_ms=30.0,
    )
    predictor = EWMAPredictor(0.5, 8, 50.0).fit((row, row2))
    decision = EFTPolicy(predictor, policy_name="ewma-eft").choose(
        make_request(), make_context()
    )
    assert decision.policy == "ewma-eft"
    assert decision.gpu_id == "gpu-0"


def test_oracle_cannot_be_constructed_for_online_use() -> None:
    with pytest.raises(OracleAccessError):
        OracleEFTPolicy()


def test_oracle_uses_private_offline_duration_mapping() -> None:
    record = TraceRecord(
        request=make_request(),
        actual_render_ms_by_gpu={"gpu-0": 50.0, "gpu-1": 10.0},
        actual_readback_ms_by_gpu={"gpu-0": 2.0, "gpu-1": 3.0},
        actual_encode_ms=4.0,
        isolated_p50_ms=17.0,
        source="synthetic",
    )
    policy = OracleEFTPolicy.for_offline_replay((record,))
    decision = policy.choose(make_request(), make_context())
    assert decision.gpu_id == "gpu-1"
    assert decision.policy == "oracle-eft"


def test_oracle_from_trace_uses_actual_readback_and_encode_stages() -> None:
    record = TraceRecord(
        request=make_request(),
        actual_render_ms_by_gpu={"gpu-0": 10.0, "gpu-1": 20.0},
        actual_readback_ms_by_gpu={"gpu-0": 100.0, "gpu-1": 0.0},
        actual_encode_ms=1.0,
        isolated_p50_ms=21.0,
        source="synthetic",
    )
    decision = OracleEFTPolicy.from_trace((record,)).choose(
        make_request(), make_context()
    )
    assert decision.gpu_id == "gpu-1"
    assert decision.costs["gpu-0"].predicted_readback_ms == 100.0
    assert decision.costs["gpu-1"].predicted_encode_ms == 1.0
