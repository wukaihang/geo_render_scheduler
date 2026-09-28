"""防泄漏的画像转换、模型训练、评价与持久化。"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

from geo_render.analysis.prediction_metrics import prediction_metrics
from geo_render.common.errors import ValidationError
from geo_render.common.types import DeviceState, ModelManifest, TraceRecord
from geo_render.workload.trace import trace_sha256

from .artifacts import save_artifact
from .baselines import EWMAPredictor, GlobalMeanPredictor
from .dataset import LabeledSample, group_split
from .features import drop_feature_groups, extract_features
from .models import MeanGBDTPredictor, QuantileGBDTPredictor, RidgeDurationPredictor
from .online_stats import OnlineDurationStats


def samples_from_profile_trace(
    trace: Iterable[TraceRecord],
    manifests: Mapping[str, ModelManifest],
    devices: Sequence[DeviceState],
    default_history_ms: float,
    history_alpha: float = 0.2,
    history_window_size: int = 64,
) -> Tuple[LabeledSample, ...]:
    records = tuple(trace)
    if not records:
        raise ValidationError("画像轨迹不得为空")
    history = OnlineDurationStats(
        history_alpha, history_window_size, default_history_ms
    )
    samples = []
    for record in records:
        request = record.request
        try:
            manifest = manifests[request.model_id]
        except KeyError as error:
            raise ValidationError(
                f"缺少画像模型 {request.model_id!r} 的 manifest"
            ) from error
        for device in devices:
            try:
                target_ms = record.actual_render_ms_by_gpu[device.gpu_id]
            except KeyError as error:
                raise ValidationError(
                    f"画像记录缺少 GPU {device.gpu_id!r}"
                ) from error
            history_prediction = history.estimate(device.gpu_id, request.model_id)
            features = extract_features(request, manifest, device, history_prediction)
            samples.append(
                LabeledSample(
                    request_id=f"{request.request_id}:{device.gpu_id}",
                    group_id=request.trajectory_id,
                    gpu_id=device.gpu_id,
                    model_id=request.model_id,
                    features=features,
                    target_ms=target_ms,
                )
            )
        for device in devices:
            history.observe(
                device.gpu_id,
                request.model_id,
                record.actual_render_ms_by_gpu[device.gpu_id],
            )
    return tuple(samples)


def _evaluate(predictor, samples: Sequence[LabeledSample]) -> dict:
    started = time.perf_counter_ns()
    predictions = [predictor.predict(sample.features) for sample in samples]
    elapsed_ns = time.perf_counter_ns() - started
    metrics = prediction_metrics(
        [sample.target_ms for sample in samples],
        [prediction.p50_ms for prediction in predictions],
        [prediction.p95_ms for prediction in predictions],
    )
    metrics["mean_prediction_us"] = elapsed_ns / len(samples) / 1000.0
    return metrics


def _ablate_samples(
    samples: Sequence[LabeledSample], feature_group: str
) -> Tuple[LabeledSample, ...]:
    return tuple(
        LabeledSample(
            request_id=sample.request_id,
            group_id=sample.group_id,
            gpu_id=sample.gpu_id,
            model_id=sample.model_id,
            features=drop_feature_groups(sample.features, (feature_group,)),
            target_ms=sample.target_ms,
        )
        for sample in samples
    )


def train_all_predictors(
    trace: Iterable[TraceRecord],
    manifests: Mapping[str, ModelManifest],
    devices: Sequence[DeviceState],
    output_dir: Path,
    seed: int,
    default_history_ms: float,
    history_alpha: float = 0.2,
    history_window_size: int = 64,
    gbdt_n_estimators: int = 50,
    gbdt_max_depth: int = 3,
) -> dict:
    records = tuple(trace)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValidationError(f"模型输出目录已存在：{output_dir}")
    samples = samples_from_profile_trace(
        records,
        manifests,
        devices,
        default_history_ms,
        history_alpha,
        history_window_size,
    )
    train, validation, test = group_split(samples, 0.6, 0.2, seed)
    predictors = {
        "global_mean": GlobalMeanPredictor().fit(train),
        "ewma": EWMAPredictor(
            history_alpha, history_window_size, default_history_ms
        ).fit(train),
        "ridge": RidgeDurationPredictor().fit(train, validation),
        "mean_gbdt": MeanGBDTPredictor(
            n_estimators=gbdt_n_estimators,
            max_depth=gbdt_max_depth,
            seed=seed,
        ).fit(train, validation),
        "quantile_gbdt": QuantileGBDTPredictor(
            n_estimators=gbdt_n_estimators,
            max_depth=gbdt_max_depth,
            seed=seed,
        ).fit(train),
    }
    ablation_predictors = {}
    ablation_tests = {}
    for group in ("model", "view", "device", "history"):
        name = f"without_{group}"
        ablated_train = _ablate_samples(train, group)
        ablated_validation = _ablate_samples(validation, group)
        ablated_test = _ablate_samples(test, group)
        ablation_predictors[name] = QuantileGBDTPredictor(
            n_estimators=gbdt_n_estimators,
            max_depth=gbdt_max_depth,
            seed=seed,
        ).fit(ablated_train + ablated_validation)
        ablation_tests[name] = ablated_test
    training_hash = trace_sha256(records)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent))
    )
    try:
        for name, predictor in predictors.items():
            save_artifact(
                temporary / f"{name}.joblib", predictor, training_sha256=training_hash
            )
        for name, predictor in ablation_predictors.items():
            save_artifact(
                temporary / f"quantile_gbdt_{name}.joblib",
                predictor,
                training_sha256=training_hash,
            )
        metrics = {
            "training_trace_sha256": training_hash,
            "split": {
                "train_samples": len(train),
                "validation_samples": len(validation),
                "test_samples": len(test),
                "train_groups": sorted({sample.group_id for sample in train}),
                "validation_groups": sorted(
                    {sample.group_id for sample in validation}
                ),
                "test_groups": sorted({sample.group_id for sample in test}),
            },
            "test_metrics": {
                name: _evaluate(predictor, test)
                for name, predictor in predictors.items()
            },
            "feature_ablations": {
                name: _evaluate(predictor, ablation_tests[name])
                for name, predictor in ablation_predictors.items()
            },
        }
        (temporary / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "predictors": predictors,
        "ablation_predictors": ablation_predictors,
        "metrics": metrics,
        "output_dir": output_dir,
    }
