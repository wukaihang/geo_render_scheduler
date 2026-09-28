import pytest

from geo_render.common.errors import ValidationError
from geo_render.prediction.online_stats import OnlineDurationStats


def test_online_history_changes_only_after_observe() -> None:
    stats = OnlineDurationStats(alpha=0.5, window_size=4, default_ms=100.0)
    before = stats.estimate("gpu-0", "model-a")
    stats.observe("gpu-0", "model-a", 20.0)
    after = stats.estimate("gpu-0", "model-a")
    assert before.p50_ms == 100.0
    assert after.p50_ms == 20.0
    assert stats.observation_count == 1


def test_online_history_uses_hierarchical_fallback() -> None:
    stats = OnlineDurationStats(alpha=0.5, window_size=4, default_ms=100.0)
    stats.observe("gpu-0", "model-a", 20.0)
    stats.observe("gpu-0", "model-a", 40.0)
    exact = stats.estimate("gpu-0", "model-a")
    gpu_fallback = stats.estimate("gpu-0", "unseen-model")
    global_fallback = stats.estimate("gpu-1", "unseen-model")
    assert exact.model_version.endswith("gpu-model")
    assert gpu_fallback.model_version.endswith("gpu")
    assert global_fallback.model_version.endswith("global")
    assert exact.p50_ms == pytest.approx(30.0)
    assert exact.p95_ms >= exact.p50_ms


def test_online_history_rejects_invalid_observations() -> None:
    stats = OnlineDurationStats(alpha=0.2, window_size=4, default_ms=100.0)
    with pytest.raises(ValidationError, match="duration_ms"):
        stats.observe("gpu-0", "model-a", 0.0)
