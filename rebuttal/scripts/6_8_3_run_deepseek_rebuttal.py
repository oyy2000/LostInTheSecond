#!/usr/bin/env python3
"""Run one configured DeepSeek rebuttal dataset end to end."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = PROJECT_ROOT / "scripts" / "6_8_budget_controlled_multisignal.py"
RECOUNT = PROJECT_ROOT / "scripts" / "6_8_1_recount_tokens.py"
BUILD_COMPARISON = (
    PROJECT_ROOT
    / "rebuttal"
    / "scripts"
    / "6_8_4_build_multisignal_comparison.py"
)
VALIDATE = (
    PROJECT_ROOT
    / "rebuttal"
    / "scripts"
    / "6_8_5_validate_multisignal_results.py"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--gpus",
        default="",
        help="Override the comma-separated GPU list in the config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete command sequence without executing it",
    )
    return parser.parse_args()


def load_config(path):
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "model",
        "prm_model",
        "datasets",
        "signals",
        "budget",
        "nd",
        "seed",
        "gpus",
        "results_root",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    return config_path, config


def make_commands(config_path, config, dataset, gpus):
    if dataset not in config["datasets"]:
        choices = ", ".join(sorted(config["datasets"]))
        raise ValueError(
            f"Dataset {dataset!r} is not configured; choose from: {choices}"
        )

    n_sample = int(config["datasets"][dataset]["n_sample"])
    out_dir = PROJECT_ROOT / config["results_root"] / dataset
    common = [
        "--gpus",
        gpus,
        "--model",
        config["model"],
        "--dataset",
        dataset,
        "--budget",
        str(config["budget"]),
        "--n-sample",
        str(n_sample),
        "--signals",
        *config["signals"],
        "--prm-model",
        config["prm_model"],
        "--nd",
        *[str(value) for value in config["nd"]],
        "--seed",
        str(config["seed"]),
        "--out-dir",
        str(out_dir),
        "--fig-dir",
        str(out_dir),
    ]
    pipeline_command = [sys.executable, str(PIPELINE), *common]
    recount_command = [
        sys.executable,
        str(RECOUNT),
        "--checkpoint",
        str(out_dir / "checkpoint.jsonl"),
        "--model",
        config["model"],
    ]
    reevaluate_command = [
        sys.executable,
        str(PIPELINE),
        *common,
        "--skip-phase",
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    comparison_command = [
        sys.executable,
        str(BUILD_COMPARISON),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
    ]
    validation_command = [
        sys.executable,
        str(VALIDATE),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
    ]
    return out_dir, [
        pipeline_command,
        recount_command,
        reevaluate_command,
        comparison_command,
        validation_command,
    ]


def main():
    args = parse_args()
    config_path, config = load_config(args.config)
    gpus = args.gpus or config["gpus"]
    out_dir, commands = make_commands(
        config_path.resolve(), config, args.dataset, gpus
    )

    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {config['datasets'][args.dataset]['n_sample']}")
    print(f"Model: {config['model']}")
    print(f"PRM: {config['prm_model']}")
    print(f"Signals: {config['signals']}")
    print(f"Budget: {config['budget']}; nd: {config['nd']}")
    print(f"Seed: {config['seed']}; GPUs: {gpus}")
    print(f"Output: {out_dir}")

    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if args.dry_run:
        print("\nDry run complete; no commands were executed.")
    else:
        print(f"\nCompleted and validated: {out_dir}")


if __name__ == "__main__":
    main()
