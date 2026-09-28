"""带版本信息的预测器工件持久化。"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy
import sklearn

from geo_render.common.errors import ValidationError

ARTIFACT_SCHEMA_VERSION = "innovation-one-artifact-v1"
FEATURE_SCHEMA_VERSION = "innovation-one-features-v1"


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValidationError("training_sha256 必须包含 64 个十六进制字符")


def save_artifact(path: Path, predictor: Any, training_sha256: str) -> Path:
    _validate_sha256(training_sha256)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_provider = getattr(predictor, "artifact_parameters", None)
    parameters = parameter_provider() if callable(parameter_provider) else {}
    envelope = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_sha256": training_sha256.lower(),
        "predictor_class": type(predictor).__name__,
        "parameters": parameters,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "predictor": predictor,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(envelope, temporary)
    temporary.replace(path)
    return path


def load_artifact(path: Path):
    envelope = joblib.load(Path(path))
    if not isinstance(envelope, Mapping):
        raise ValidationError("预测器工件必须包含映射封装")
    if envelope.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValidationError(
            "预测器工件 schema_version 不匹配："
            f"期望 {ARTIFACT_SCHEMA_VERSION!r}，实际为 {envelope.get('schema_version')!r}"
        )
    if envelope.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValidationError("预测器工件的 feature_schema_version 不匹配")
    predictor = envelope.get("predictor")
    if predictor is None or not callable(getattr(predictor, "predict", None)):
        raise ValidationError("预测器工件缺少有效的 predictor")
    return predictor
