import pytest

from geo_render.analysis.prediction_metrics import prediction_metrics
from geo_render.common.errors import ValidationError


def test_prediction_metrics_cover_required_error_and_calibration_values() -> None:
    metrics = prediction_metrics(
        actual_ms=[10.0, 20.0],
        predicted_p50_ms=[9.0, 18.0],
        predicted_p95_ms=[12.0, 19.0],
    )
    assert metrics["mae_ms"] == pytest.approx(1.5)
    assert metrics["mape"] == pytest.approx(0.1)
    assert metrics["p95_absolute_error_ms"] == pytest.approx(1.95)
    assert metrics["underestimate_rate"] == pytest.approx(1.0)
    assert metrics["p95_coverage"] == pytest.approx(0.5)


def test_prediction_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValidationError, match="相同的非零长度"):
        prediction_metrics([1.0], [1.0, 2.0], [2.0])
