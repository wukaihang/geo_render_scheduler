from typing import Mapping

from geo_render.common.types import (
    DeviceState,
    DurationPrediction,
    ModelManifest,
    RenderRequest,
)
from geo_render.scheduling.policies import (
    EFTPolicy,
    LeastQueuePolicy,
    OracleEFTPolicy,
    RoundRobinPolicy,
    StaticWeightedPolicy,
)
from geo_render.workload.replay import ReplayEngine
from geo_render.workload.synthetic import SyntheticTraceConfig, generate_synthetic_trace


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


class ConstantPredictor:
    def predict(self, features: Mapping[str, object]) -> DurationPrediction:
        gpu_id = features["gpu_id"]
        value = 25.0 if gpu_id == "gpu-0" else 31.25
        return DurationPrediction(value, value * 1.2, "constant-test")


def make_devices() -> tuple[DeviceState, ...]:
    return tuple(
        DeviceState(
            gpu_id=gpu_id,
            model="synthetic-device",
            memory_total_bytes=32 * 1024**3,
            utilization_pct=None,
            memory_utilization_pct=None,
            temperature_c=None,
            power_w=None,
            historical_speed=speed,
            sampled_at_ms=0.0,
        )
        for gpu_id, speed in (("gpu-0", 1.0), ("gpu-1", 0.8))
    )


def make_manifests() -> dict[str, ModelManifest]:
    return {
        model_id: ModelManifest(
            model_id=model_id,
            dimensions=(128, 128, 128) if model_id == "model-a" else (256, 128, 128),
            active_voxel_fraction=0.5,
            data_bytes=128**3 if model_id == "model-a" else 256 * 128**2,
            layer_count=4 if model_id == "model-a" else 7,
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            sha256=("a" if model_id == "model-a" else "b") * 64,
        )
        for model_id in ("model-a", "model-b")
    }


def make_engine() -> ReplayEngine:
    return ReplayEngine(
        devices=make_devices(),
        manifests=make_manifests(),
        predicted_readback_ms_by_gpu={"gpu-0": 2.0, "gpu-1": 2.5},
        predicted_encode_ms=3.0,
        default_history_ms=40.0,
    )


def all_policies(trace):
    return (
        RoundRobinPolicy(),
        LeastQueuePolicy(),
        StaticWeightedPolicy({"gpu-0": 1.0, "gpu-1": 0.8}),
        EFTPolicy(ConstantPredictor(), policy_name="ewma-eft"),
        EFTPolicy(ConstantPredictor(), policy_name="feature-eft"),
        OracleEFTPolicy.for_offline_replay(tuple(trace)),
    )


def test_every_policy_completes_the_same_request_set() -> None:
    trace = generate_synthetic_trace(make_config(request_count=30))
    expected = {record.request.request_id for record in trace}
    results = [make_engine().run(trace, policy) for policy in all_policies(trace)]
    assert {result.policy for result in results} == {
        "round-robin",
        "least-queue",
        "static-weighted",
        "ewma-eft",
        "feature-eft",
        "oracle-eft",
    }
    assert all({row.request_id for row in result.completed} == expected for result in results)


class SpyingPolicy:
    name = "spy"

    def __init__(self) -> None:
        self.delegate = RoundRobinPolicy()
        self.observed_types = set()
        self.saw_oracle_field = False

    def choose(self, request, context):
        self.observed_types.add(type(request))
        self.saw_oracle_field |= hasattr(request, "actual_render_ms_by_gpu")
        decision = self.delegate.choose(request, context)
        return decision.__class__(
            request_id=decision.request_id,
            gpu_id=decision.gpu_id,
            policy=self.name,
            reason=decision.reason,
            costs=decision.costs,
        )


def test_online_policy_never_receives_trace_record_or_future_duration() -> None:
    trace = generate_synthetic_trace(make_config(request_count=10))
    policy = SpyingPolicy()
    make_engine().run(trace, policy)
    assert policy.observed_types == {RenderRequest}
    assert not policy.saw_oracle_field


def test_replay_is_deterministic_and_records_stage_boundaries() -> None:
    trace = generate_synthetic_trace(make_config(request_count=12))
    first = make_engine().run(trace, RoundRobinPolicy())
    second = make_engine().run(trace, RoundRobinPolicy())
    assert first == second
    assert all(row.start_ms >= row.arrival_ms for row in first.completed)
    assert all(row.finish_ms > row.start_ms for row in first.completed)
    assert all(row.end_to_end_ms == row.finish_ms - row.arrival_ms for row in first.completed)
    assert set(first.worker_busy_ms) == {"gpu-0", "gpu-1"}
