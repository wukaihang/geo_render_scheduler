"""Command-line interface for innovation-one data, training, and comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from geo_render.common.errors import GeoRenderError, HardwareBackendUnavailable
from geo_render.experiments.compare import compare_policies
from geo_render.experiments.config import (
    devices_from_config,
    load_config,
    manifests_from_config,
    replay_engine,
    synthetic_trace_config,
)
from geo_render.prediction.baselines import EWMAPredictor
from geo_render.prediction.dataset import group_split
from geo_render.prediction.models import QuantileGBDTPredictor
from geo_render.prediction.training import (
    samples_from_profile_trace,
    train_all_predictors,
)
from geo_render.rendering.unavailable import UnavailableDeviceStateProvider
from geo_render.scheduling.policies import (
    EFTPolicy,
    LeastQueuePolicy,
    OracleEFTPolicy,
    RoundRobinPolicy,
    StaticWeightedPolicy,
)
from geo_render.workload.synthetic import generate_synthetic_trace
from geo_render.workload.trace import read_trace_csv, trace_sha256, write_trace_csv


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geo-render",
        description="Innovation-one render duration prediction and EFT experiments",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-trace", help="generate a frozen trace")
    _add_config(generate)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--requests", type=int)
    generate.add_argument("--seed", type=int)

    train = commands.add_parser("train", help="train and evaluate all predictors")
    _add_config(train)
    train.add_argument("--trace", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare", help="compare all six schedulers")
    _add_config(compare)
    compare.add_argument("--trace", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--requests", type=int)
    compare.add_argument("--seed", type=int)

    commands.add_parser(
        "check-hardware", help="verify that a real EGL/NVML adapter is available"
    )
    return parser


def _prediction_settings(config: dict) -> dict:
    value = config["prediction"]
    return {
        "default_history_ms": float(value["history_default_ms"]),
        "history_alpha": float(value.get("history_alpha", 0.2)),
        "history_window_size": int(value.get("history_window_size", 64)),
        "gbdt_n_estimators": int(value.get("gbdt_n_estimators", 50)),
        "gbdt_max_depth": int(value.get("gbdt_max_depth", 3)),
    }


def _generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    trace_config = synthetic_trace_config(config, args.requests, args.seed)
    trace = generate_synthetic_trace(trace_config)
    write_trace_csv(args.output, trace)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requests": len(trace),
                "source": "synthetic",
                "trace_sha256": trace_sha256(trace),
            },
            sort_keys=True,
        )
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    trace = read_trace_csv(args.trace)
    settings = _prediction_settings(config)
    result = train_all_predictors(
        trace=trace,
        manifests=manifests_from_config(config),
        devices=devices_from_config(config),
        output_dir=args.output,
        seed=int(config["seed"]),
        **settings,
    )
    print(
        json.dumps(
            {
                "output": str(result["output_dir"]),
                "training_trace_sha256": result["metrics"]["training_trace_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _fit_comparison_predictors(config: dict):
    settings = _prediction_settings(config)
    primary = synthetic_trace_config(config)
    profile_count = int(config.get("profile_request_count", max(60, primary.request_count)))
    profile_config = replace(
        primary, seed=primary.seed + 1, request_count=profile_count
    )
    profile_trace = generate_synthetic_trace(profile_config)
    samples = samples_from_profile_trace(
        profile_trace,
        manifests_from_config(config),
        devices_from_config(config),
        settings["default_history_ms"],
        settings["history_alpha"],
        settings["history_window_size"],
    )
    train, validation, _ = group_split(samples, 0.6, 0.2, primary.seed)
    offline_training = train + validation
    ewma = EWMAPredictor(
        settings["history_alpha"],
        settings["history_window_size"],
        settings["default_history_ms"],
    ).fit(offline_training)
    feature = QuantileGBDTPredictor(
        n_estimators=settings["gbdt_n_estimators"],
        max_depth=settings["gbdt_max_depth"],
        seed=primary.seed,
    ).fit(offline_training)
    return ewma, feature, trace_sha256(profile_trace)


def _compare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    trace_config = synthetic_trace_config(config, args.requests, args.seed)
    trace = (
        read_trace_csv(args.trace)
        if args.trace is not None
        else generate_synthetic_trace(trace_config)
    )
    ewma, feature, profile_hash = _fit_comparison_predictors(config)
    policies = (
        RoundRobinPolicy(),
        LeastQueuePolicy(),
        StaticWeightedPolicy(trace_config.gpu_speed),
        EFTPolicy(ewma, policy_name="ewma-eft"),
        EFTPolicy(feature, policy_name="feature-eft"),
        OracleEFTPolicy.for_offline_replay(tuple(trace)),
    )
    comparison_config = {
        "seed": trace_config.seed,
        "source": "synthetic" if args.trace is None else trace[0].source,
        "profile_trace_sha256": profile_hash,
        "request_count": len(trace),
    }
    result = compare_policies(
        output_dir=args.output,
        config=comparison_config,
        trace=trace,
        engine_factory=lambda: replay_engine(config),
        policies=policies,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "successful_policies": sorted(result["summaries"]),
                "failures": result["failures"],
            },
            sort_keys=True,
        )
    )
    return 0 if not result["failures"] else 1


def _check_hardware() -> int:
    UnavailableDeviceStateProvider().snapshot()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-trace":
            return _generate(args)
        if args.command == "train":
            return _train(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "check-hardware":
            return _check_hardware()
        parser.error(f"unknown command {args.command!r}")
    except HardwareBackendUnavailable as error:
        print(f"hardware backend unavailable: {error}", file=sys.stderr)
        return 2
    except (GeoRenderError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
