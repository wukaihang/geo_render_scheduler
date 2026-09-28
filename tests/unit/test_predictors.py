from pathlib import Path

import joblib
import pytest

from geo_render.common.errors import ModelNotFittedError, ValidationError
from geo_render.prediction.artifacts import load_artifact, save_artifact
from geo_render.prediction.baselines import EWMAPredictor, GlobalMeanPredictor
from geo_render.prediction.dataset import LabeledSample
from geo_render.prediction.models import (
    MeanGBDTPredictor,
    QuantileGBDTPredictor,
    RidgeDurationPredictor,
)


def training_rows() -> tuple[LabeledSample, ...]:
    rows = []
    for index in range(40):
        gpu_id = f"gpu-{index % 2}"
        model_id = f"model-{index % 3}"
        complexity = float(1 + index % 10)
        target = 8.0 + complexity * 2.0 + (5.0 if gpu_id == "gpu-1" else 0.0)
        rows.append(
            LabeledSample(
                request_id=f"request-{index}",
                group_id=f"trajectory-{index // 4}",
                gpu_id=gpu_id,
                model_id=model_id,
                features={
                    "gpu_id": gpu_id,
                    "model_id": model_id,
                    "complexity": complexity,
                    "projection": "perspective" if index % 2 else "parallel",
                },
                target_ms=target,
            )
        )
    return tuple(rows)


def predictor_factories():
    return (
        lambda: GlobalMeanPredictor(),
        lambda: EWMAPredictor(alpha=0.3, window_size=16, default_ms=50.0),
        lambda: RidgeDurationPredictor(),
        lambda: MeanGBDTPredictor(n_estimators=20, max_depth=2, seed=7),
        lambda: QuantileGBDTPredictor(n_estimators=20, max_depth=2, seed=7),
    )


@pytest.mark.parametrize("factory", predictor_factories())
def test_predictors_return_ordered_positive_quantiles(factory) -> None:
    rows = training_rows()
    predictor = factory().fit(rows)
    prediction = predictor.predict(rows[0].features)
    assert 0 < prediction.p50_ms <= prediction.p95_ms
    assert prediction.model_version


@pytest.mark.parametrize("factory", predictor_factories())
def test_predictors_reject_prediction_before_fit(factory) -> None:
    with pytest.raises(ModelNotFittedError):
        factory().predict(training_rows()[0].features)


def test_global_mean_uses_gpu_model_then_gpu_then_global_fallback() -> None:
    predictor = GlobalMeanPredictor().fit(training_rows())
    exact = predictor.predict({"gpu_id": "gpu-0", "model_id": "model-0"})
    gpu = predictor.predict({"gpu_id": "gpu-0", "model_id": "unseen"})
    global_value = predictor.predict({"gpu_id": "gpu-x", "model_id": "unseen"})
    assert exact.model_version.endswith("gpu-model")
    assert gpu.model_version.endswith("gpu")
    assert global_value.model_version.endswith("global")


@pytest.mark.parametrize("factory", predictor_factories())
def test_artifact_round_trip_preserves_predictions(tmp_path: Path, factory) -> None:
    rows = training_rows()
    predictor = factory().fit(rows)
    path = tmp_path / "model.joblib"
    save_artifact(path, predictor, training_sha256="a" * 64)
    restored = load_artifact(path)
    assert restored.predict(rows[0].features) == predictor.predict(rows[0].features)


def test_artifact_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.joblib"
    joblib.dump({"schema_version": "wrong", "predictor": object()}, path)
    with pytest.raises(ValidationError, match="schema_version"):
        load_artifact(path)


def test_artifact_records_predictor_class_and_training_parameters(tmp_path: Path) -> None:
    predictor = MeanGBDTPredictor(
        n_estimators=17, max_depth=2, learning_rate=0.07, seed=9
    ).fit(training_rows())
    path = tmp_path / "mean-gbdt.joblib"
    save_artifact(path, predictor, training_sha256="b" * 64)
    envelope = joblib.load(path)
    assert envelope["predictor_class"] == "MeanGBDTPredictor"
    assert envelope["parameters"]["n_estimators"] == 17
    assert envelope["parameters"]["seed"] == 9
