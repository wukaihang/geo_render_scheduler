"""Atomic self-contained experiment result directories."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy
import sklearn

from geo_render import __version__
from geo_render.analysis.scheduling_metrics import scheduling_metrics
from geo_render.common.errors import ValidationError
from geo_render.common.types import CostEstimate, TraceRecord
from geo_render.workload.replay import ReplayResult
from geo_render.workload.trace import trace_sha256, write_trace_csv


REQUEST_FIELDS = (
    "request_id",
    "user_id",
    "session_id",
    "trajectory_id",
    "model_id",
    "gpu_id",
    "arrival_ms",
    "dispatch_ms",
    "start_ms",
    "finish_ms",
    "queue_ms",
    "render_ms",
    "readback_ms",
    "encode_ms",
    "end_to_end_ms",
    "predicted_render_p50_ms",
    "predicted_render_p95_ms",
    "predicted_finish_ms",
    "slo_ms",
    "isolated_p50_ms",
    "source",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cost_dict(cost: CostEstimate) -> dict:
    return {
        "gpu_id": cost.gpu_id,
        "running_remaining_ms": cost.running_remaining_ms,
        "queued_work_ms": cost.queued_work_ms,
        "predicted_render_p50_ms": cost.predicted_render_p50_ms,
        "predicted_render_p95_ms": cost.predicted_render_p95_ms,
        "predicted_readback_ms": cost.predicted_readback_ms,
        "predicted_encode_ms": cost.predicted_encode_ms,
        "total_ms": cost.total_ms,
    }


def _write_requests(path: Path, result: ReplayResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUEST_FIELDS)
        writer.writeheader()
        for row in result.completed:
            writer.writerow({field: getattr(row, field) for field in REQUEST_FIELDS})


def _write_decisions(path: Path, result: ReplayResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in result.decisions:
            decision = item.decision
            value = {
                "at_ms": item.at_ms,
                "request_id": decision.request_id,
                "gpu_id": decision.gpu_id,
                "policy": decision.policy,
                "reason": decision.reason,
                "workload_gap_ms": item.workload_gap_ms,
                "costs": {
                    gpu_id: _cost_dict(cost)
                    for gpu_id, cost in sorted(decision.costs.items())
                },
            }
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _metadata(
    config: Mapping[str, object], trace: Iterable[TraceRecord], result: ReplayResult
) -> dict:
    return {
        "status": "complete",
        "source": result.source,
        "policy": result.policy,
        "seed": config.get("seed"),
        "trace_sha256": trace_sha256(trace),
        "code_commit": _git_commit(),
        "versions": {
            "geo_render": __version__,
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def write_run_directory(
    output_dir: Path,
    config: Mapping[str, object],
    trace: Iterable[TraceRecord],
    result: ReplayResult,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValidationError(f"output directory already exists: {output_dir}")
    records = tuple(trace)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent))
    )
    try:
        _write_json(temporary / "config.json", dict(config))
        write_trace_csv(temporary / "trace.csv", records)
        _write_requests(temporary / "requests.csv", result)
        _write_decisions(temporary / "decisions.jsonl", result)
        _write_json(temporary / "summary.json", scheduling_metrics(result))
        _write_json(temporary / "metadata.json", _metadata(config, records, result))
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def write_failed_run_directory(
    output_dir: Path,
    config: Mapping[str, object],
    trace: Iterable[TraceRecord],
    policy_name: str,
    error: BaseException,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValidationError(f"output directory already exists: {output_dir}")
    records = tuple(trace)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "config.json", dict(config))
    write_trace_csv(output_dir / "trace.csv", records)
    with (output_dir / "requests.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=REQUEST_FIELDS).writeheader()
    (output_dir / "decisions.jsonl").write_text("", encoding="utf-8")
    _write_json(output_dir / "summary.json", {"status": "failed"})
    _write_json(
        output_dir / "metadata.json",
        {
            "status": "failed",
            "source": records[0].source,
            "policy": policy_name,
            "seed": config.get("seed"),
            "trace_sha256": trace_sha256(records),
            "code_commit": _git_commit(),
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return output_dir
