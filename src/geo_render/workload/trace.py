"""不可变轨迹的无损 CSV 持久化与规范哈希。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from geo_render.common.errors import ValidationError
from geo_render.common.types import Camera, RenderRequest, TraceRecord

TRACE_FIELDS = (
    "request_id",
    "user_id",
    "session_id",
    "trajectory_id",
    "model_id",
    "camera_position",
    "camera_focal_point",
    "camera_view_up",
    "camera_projection",
    "camera_view_angle_deg",
    "clip_fraction",
    "output_width",
    "output_height",
    "sample_step",
    "shadows",
    "transfer_function_id",
    "arrival_ms",
    "slo_ms",
    "actual_render_ms_by_gpu",
    "actual_readback_ms_by_gpu",
    "actual_encode_ms",
    "isolated_p50_ms",
    "source",
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row(record: TraceRecord) -> Dict[str, str]:
    request = record.request
    return {
        "request_id": request.request_id,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "trajectory_id": request.trajectory_id,
        "model_id": request.model_id,
        "camera_position": _json(request.camera.position),
        "camera_focal_point": _json(request.camera.focal_point),
        "camera_view_up": _json(request.camera.view_up),
        "camera_projection": request.camera.projection,
        "camera_view_angle_deg": repr(request.camera.view_angle_deg),
        "clip_fraction": repr(request.clip_fraction),
        "output_width": str(request.output_width),
        "output_height": str(request.output_height),
        "sample_step": repr(request.sample_step),
        "shadows": "true" if request.shadows else "false",
        "transfer_function_id": request.transfer_function_id,
        "arrival_ms": repr(request.arrival_ms),
        "slo_ms": "" if request.slo_ms is None else repr(request.slo_ms),
        "actual_render_ms_by_gpu": _json(dict(record.actual_render_ms_by_gpu)),
        "actual_readback_ms_by_gpu": _json(dict(record.actual_readback_ms_by_gpu)),
        "actual_encode_ms": repr(record.actual_encode_ms),
        "isolated_p50_ms": (
            "" if record.isolated_p50_ms is None else repr(record.isolated_p50_ms)
        ),
        "source": record.source,
    }


def write_trace_csv(path: Path, trace: Iterable[TraceRecord]) -> Path:
    path = Path(path)
    records = tuple(trace)
    if not records:
        raise ValidationError("轨迹不得为空")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(_row(record) for record in records)
    return path


def _vector(raw: str):
    values = json.loads(raw)
    return tuple(float(value) for value in values)


def _mapping(raw: str):
    values = json.loads(raw)
    return {str(key): float(value) for key, value in values.items()}


def read_trace_csv(path: Path) -> Tuple[TraceRecord, ...]:
    records = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRACE_FIELDS:
            raise ValidationError("轨迹 CSV 列与版本化模式不匹配")
        for row in reader:
            request = RenderRequest(
                request_id=row["request_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                trajectory_id=row["trajectory_id"],
                model_id=row["model_id"],
                camera=Camera(
                    position=_vector(row["camera_position"]),
                    focal_point=_vector(row["camera_focal_point"]),
                    view_up=_vector(row["camera_view_up"]),
                    projection=row["camera_projection"],
                    view_angle_deg=float(row["camera_view_angle_deg"]),
                ),
                clip_fraction=float(row["clip_fraction"]),
                output_width=int(row["output_width"]),
                output_height=int(row["output_height"]),
                sample_step=float(row["sample_step"]),
                shadows=row["shadows"] == "true",
                transfer_function_id=row["transfer_function_id"],
                arrival_ms=float(row["arrival_ms"]),
                slo_ms=float(row["slo_ms"]) if row["slo_ms"] else None,
            )
            records.append(
                TraceRecord(
                    request=request,
                    actual_render_ms_by_gpu=_mapping(
                        row["actual_render_ms_by_gpu"]
                    ),
                    actual_readback_ms_by_gpu=_mapping(
                        row["actual_readback_ms_by_gpu"]
                    ),
                    actual_encode_ms=float(row["actual_encode_ms"]),
                    isolated_p50_ms=(
                        float(row["isolated_p50_ms"])
                        if row["isolated_p50_ms"]
                        else None
                    ),
                    source=row["source"],
                )
            )
    if not records:
        raise ValidationError("轨迹 CSV 至少必须包含一条记录")
    return tuple(records)


def trace_sha256(trace: Iterable[TraceRecord]) -> str:
    rows = [_row(record) for record in trace]
    if not rows:
        raise ValidationError("轨迹不得为空")
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
