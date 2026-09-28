import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "synthetic_example.json"


def run_cli(*arguments: str):
    return subprocess.run(
        [sys.executable, "-m", "geo_render", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_generate_trace_and_train_cli_create_versioned_artifacts(tmp_path: Path) -> None:
    trace_path = tmp_path / "profile.csv"
    generated = run_cli(
        "generate-trace",
        "--config",
        str(EXAMPLE_CONFIG),
        "--output",
        str(trace_path),
        "--requests",
        "36",
        "--seed",
        "91",
    )
    assert generated.returncode == 0, generated.stderr
    assert trace_path.exists()
    model_dir = tmp_path / "models"
    trained = run_cli(
        "train",
        "--config",
        str(EXAMPLE_CONFIG),
        "--trace",
        str(trace_path),
        "--output",
        str(model_dir),
    )
    assert trained.returncode == 0, trained.stderr
    assert {path.name for path in model_dir.iterdir()} == {
        "global_mean.joblib",
        "ewma.joblib",
        "ridge.joblib",
        "mean_gbdt.joblib",
        "quantile_gbdt.joblib",
        "quantile_gbdt_without_device.joblib",
        "quantile_gbdt_without_history.joblib",
        "quantile_gbdt_without_model.joblib",
        "quantile_gbdt_without_view.joblib",
        "metrics.json",
    }
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert set(metrics["test_metrics"]) == {
        "global_mean",
        "ewma",
        "ridge",
        "mean_gbdt",
        "quantile_gbdt",
    }
    assert set(metrics["feature_ablations"]) == {
        "without_device",
        "without_history",
        "without_model",
        "without_view",
    }


def test_compare_cli_creates_all_six_policy_summaries(tmp_path: Path) -> None:
    output = tmp_path / "results"
    result = run_cli(
        "compare",
        "--config",
        str(EXAMPLE_CONFIG),
        "--output",
        str(output),
        "--requests",
        "20",
    )
    assert result.returncode == 0, result.stderr
    comparison = json.loads((output / "comparison.json").read_text())
    assert set(comparison["summaries"]) == {
        "round-robin",
        "least-queue",
        "static-weighted",
        "ewma-eft",
        "feature-eft",
        "oracle-eft",
    }
    assert comparison["failures"] == {}


def test_hardware_check_fails_actionably() -> None:
    result = run_cli("check-hardware")
    assert result.returncode == 2
    assert "Linux" in result.stderr
    assert "EGL" in result.stderr
    assert "NVML" in result.stderr


def test_cli_parse_errors_are_chinese() -> None:
    missing = run_cli("generate-trace")
    assert missing.returncode == 2
    assert "错误：" in missing.stderr
    assert "缺少以下必需参数" in missing.stderr
    assert "the following arguments are required" not in missing.stderr

    unknown = run_cli(
        "generate-trace",
        "--config",
        str(EXAMPLE_CONFIG),
        "--output",
        "trace.csv",
        "--unknown",
    )
    assert unknown.returncode == 2
    assert "无法识别的参数" in unknown.stderr
    assert "unrecognized arguments" not in unknown.stderr

    invalid_integer = run_cli(
        "generate-trace",
        "--config",
        str(EXAMPLE_CONFIG),
        "--output",
        "trace.csv",
        "--requests",
        "不是整数",
    )
    assert invalid_integer.returncode == 2
    assert "无效的整数值" in invalid_integer.stderr
    assert "invalid int value" not in invalid_integer.stderr
