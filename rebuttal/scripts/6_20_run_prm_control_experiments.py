#!/usr/bin/env python3
"""Run max-drop and absolute-threshold PRM control experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "rebuttal/configs/6_20_qwen_prm_control_experiments.json"
)
sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.src.prm_control_experiment import (  # noqa: E402
    all_datasets_complete,
    prepare_result_dir,
    read_json,
    resolve,
    validate_control_result,
    validate_shared_source,
    write_json,
)
from rebuttal.src.prm_control_strategies import (  # noqa: E402
    PRM_CONTROL_SIGNALS,
)
from rebuttal.src.wallclock import discover_timing_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def pipeline_command(
    config: dict,
    dataset: str,
    gpus: str,
    result_dir: Path,
    figure_dir: Path,
    shared_checkpoint: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(resolve(PROJECT_ROOT, config["pipeline"])),
        "--gpus",
        gpus,
        "--model",
        config["model"],
        "--prm-model",
        config["prm_model"],
        "--dataset",
        dataset,
        "--budget",
        str(config["budget"]),
        "--n-sample",
        str(config["datasets"][dataset]["n_sample"]),
        "--seed",
        str(config["seed"]),
        "--signals",
        *config["signals"],
        "--prm-absolute-threshold",
        str(config["absolute_prm_threshold"]),
        "--nd",
        *[str(value) for value in config["nd"]],
        "--skip-phase",
        "1",
        "2",
        "25",
        "3",
        "5",
        "--checkpoint-part",
        str(shared_checkpoint),
        "--out-dir",
        str(result_dir),
        "--fig-dir",
        str(figure_dir),
        "--timing-mode",
        "auto",
    ]
    timing_manifest = discover_timing_manifest(shared_checkpoint)
    if timing_manifest is not None:
        command.extend(
            ["--reuse-timing-manifest", str(timing_manifest)]
        )
    return command


def validate_config(config: dict) -> None:
    if config["signals"] != PRM_CONTROL_SIGNALS:
        raise ValueError(
            f"Expected signals {PRM_CONTROL_SIGNALS}, "
            f"found {config['signals']}"
        )
    if int(config["seed"]) != 42:
        raise ValueError("PRM control experiments require seed 42")
    if float(config["absolute_prm_threshold"]) != 0.8:
        raise ValueError("Absolute PRM threshold must be 0.8")
    if int(config["budget"]) != 32:
        raise ValueError("PRM control experiments require budget 32")
    if [int(value) for value in config["nd"]] != [4, 8, 16]:
        raise ValueError("Expected nd=[4, 8, 16]")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    datasets = args.datasets or list(config["datasets"])
    unsupported = [
        dataset
        for dataset in datasets
        if dataset not in config["datasets"]
    ]
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")
    gpus = args.gpus or config["gpus"]
    if not gpus:
        raise ValueError("At least one GPU is required")

    results_root = resolve(PROJECT_ROOT, config["results_root"])
    figures_root = resolve(PROJECT_ROOT, config["figures_root"])
    for dataset in datasets:
        result_dir = results_root / dataset
        figure_dir = figures_root / dataset
        if args.dry_run:
            shared = validate_shared_source(
                config,
                dataset,
                PROJECT_ROOT,
            )
        else:
            shared = prepare_result_dir(
                config,
                dataset,
                PROJECT_ROOT,
                result_dir,
            )
            figure_dir.mkdir(parents=True, exist_ok=True)

        command = pipeline_command(
            config,
            dataset,
            gpus,
            result_dir,
            figure_dir,
            shared["checkpoint"],
        )
        print(
            f"\n=== {dataset}: max PRM drop and PRM<0.8 controls ===",
            flush=True,
        )
        print("command=" + " ".join(command), flush=True)
        if args.dry_run:
            continue
        if not args.validate_only:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        report = validate_control_result(
            config,
            dataset,
            PROJECT_ROOT,
            result_dir,
            figure_dir,
        )
        print(
            f"validated {dataset}: "
            f"{sum(report['local_suffix_counts'].values())} suffixes, "
            f"{report['evaluation_rows']} control rows",
            flush=True,
        )
        (result_dir / "COMPLETE").touch()

    if not args.dry_run and all_datasets_complete(
        config,
        results_root,
        config["datasets"],
    ):
        completion = {
            "status": "passed",
            "experiment_name": config["experiment_name"],
            "datasets": list(config["datasets"]),
            "seed": int(config["seed"]),
            "budget": int(config["budget"]),
            "nd": [int(value) for value in config["nd"]],
            "signals": list(config["signals"]),
            "absolute_prm_threshold": float(
                config["absolute_prm_threshold"]
            ),
        }
        write_json(results_root / "completion_report.json", completion)
        (results_root / "COMPLETE").touch()


if __name__ == "__main__":
    main()
