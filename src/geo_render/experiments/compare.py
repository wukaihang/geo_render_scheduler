"""在同一份冻结轨迹上运行多种策略。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

from geo_render.common.errors import ValidationError
from geo_render.common.types import TraceRecord
from geo_render.scheduling.interface import SchedulerPolicy
from geo_render.workload.replay import ReplayEngine
from geo_render.workload.trace import trace_sha256

from .io import write_failed_run_directory, write_run_directory


def compare_policies(
    output_dir: Path,
    config: Mapping[str, object],
    trace: Iterable[TraceRecord],
    engine_factory: Callable[[], ReplayEngine],
    policies: Iterable[SchedulerPolicy],
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValidationError(f"比较结果输出目录已存在：{output_dir}")
    records = tuple(trace)
    policy_list = tuple(policies)
    if not records or not policy_list:
        raise ValidationError("策略比较需要非空轨迹和策略集合")
    names = [policy.name for policy in policy_list]
    if len(names) != len(set(names)):
        raise ValidationError("参与比较的策略名称必须唯一")
    output_dir.mkdir(parents=True)
    summaries = {}
    failures = {}
    for policy in policy_list:
        policy_config = dict(config)
        policy_config["policy"] = policy.name
        try:
            result = engine_factory().run(records, policy)
            run_dir = write_run_directory(
                output_dir / policy.name, policy_config, records, result
            )
            summaries[policy.name] = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
        except Exception as error:
            failures[policy.name] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            write_failed_run_directory(
                output_dir / policy.name,
                policy_config,
                records,
                policy.name,
                error,
            )
    comparison = {
        "trace_sha256": trace_sha256(records),
        "summaries": summaries,
        "failures": failures,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison
