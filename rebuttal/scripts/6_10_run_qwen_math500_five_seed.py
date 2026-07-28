#!/usr/bin/env python3
"""Run and aggregate the corrected five-seed Qwen MATH-500 sweep."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "rebuttal/configs/6_10_qwen_math500_five_seed.json"
)
PIPELINE = (
    PROJECT_ROOT
    / "rebuttal/scripts/6_8_budget_controlled_multisignal_seeded.py"
)
sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.src.seed_sweep_summary import (  # noqa: E402
    aggregate_sc_seed_runs,
    aggregate_seed_runs,
    validate_seed_run,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gpus", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--aggregate-sc-only", action="store_true")
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help=(
            "Generate the 16 drafts plus 16 SC supplements per question, "
            "skipping PRM, NLL, and suffix phases. A later full run resumes "
            "from the same checkpoints."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset") != "math500":
        raise ValueError("This entrypoint only supports MATH-500.")
    if config.get("budget") != 32:
        raise ValueError("The corrected rebuttal sweep requires SC@32.")
    return config


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_output_dir(config: dict, seed: int) -> Path:
    return (
        resolve_path(config["out_root"])
        / f"seed_{seed}"
        / config["dataset"]
    )


def run_seed(
    config: dict,
    seed: int,
    gpus: str,
    dry_run: bool,
    generation_only: bool,
):
    output_dir = seed_output_dir(config, seed)
    figure_dir = (
        resolve_path(config["fig_root"])
        / f"seed_{seed}"
        / config["dataset"]
    )
    command = [
        sys.executable,
        "-u",
        str(PIPELINE),
        "--gpus", gpus,
        "--model", config["model"],
        "--dataset", config["dataset"],
        "--budget", str(config["budget"]),
        "--n-sample", str(config["n_sample"]),
        "--seed", str(seed),
        "--signals", *config["signals"],
        "--nd", *[str(value) for value in config["nd"]],
        "--out-dir", str(output_dir),
        "--fig-dir", str(figure_dir),
    ]
    if config.get("prm_model"):
        command += ["--prm-model", config["prm_model"]]
    skip_phases = set(config.get("skip_phase", []))
    if generation_only:
        skip_phases.update({2, 3, 4})
    if skip_phases:
        command += [
            "--skip-phase",
            *[str(value) for value in sorted(skip_phases)],
        ]

    print(f"seed={seed}")
    print(f"output={output_dir}")
    print("command=" + " ".join(command))
    if dry_run:
        return output_dir

    started = time.time()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    elapsed = time.time() - started
    print(f"seed={seed} completed in {elapsed / 3600:.2f} hours")

    nd_max = max(config["nd"])
    suffixes_per_signal = sum(
        max(
            config["budget"] // nd
            for nd in config["nd"]
            if draft_idx < nd
        )
        for draft_idx in range(nd_max)
    )
    report = validate_seed_run(
        output_dir,
        run_seed=seed,
        n_questions=config["n_sample"],
        nd_max=nd_max,
        budget=config["budget"],
        n_signals=(
            0 if generation_only else len(config["signals"])
        ),
        suffixes_per_signal_per_question=suffixes_per_signal,
        report_filename=(
            "sc_seed_validation.json"
            if generation_only else "seed_validation.json"
        ),
    )
    print(
        f"validated seed={seed}: SC@32={report['sc32_acc']:.4f}, "
        f"unique_seeds={report['unique_sampling_seeds']}"
    )
    return output_dir


def print_aggregate(summary):
    print(
        f"{'method':<40} {'n':>3} {'mean':>8} "
        f"{'std':>8} {'min':>8} {'max':>8}"
    )
    for row in summary:
        print(
            f"{row['method']:<40} {row['n_seeds']:>3d} "
            f"{row['acc_mean']:>8.4f} {row['acc_std']:>8.4f} "
            f"{row['acc_min']:>8.4f} {row['acc_max']:>8.4f}"
        )


def main():
    args = parse_args()
    if args.aggregate_only and args.aggregate_sc_only:
        raise ValueError(
            "--aggregate-only and --aggregate-sc-only are mutually exclusive"
        )
    config = load_config(Path(args.config))
    configured_seeds = config["seeds"]
    selected_seeds = (
        args.seeds if args.seeds is not None else configured_seeds
    )
    unknown = set(selected_seeds) - set(configured_seeds)
    if unknown:
        raise ValueError(
            f"Seeds not declared by config: {sorted(unknown)}"
        )
    gpus = args.gpus or config["gpus"]

    if not args.aggregate_only and not args.aggregate_sc_only:
        for seed in selected_seeds:
            run_seed(
                config,
                seed,
                gpus,
                args.dry_run,
                args.generation_only,
            )
    if args.dry_run or args.generation_only:
        return

    configured_seed_dirs = [
        (seed, seed_output_dir(config, seed))
        for seed in configured_seeds
    ]
    if args.aggregate_sc_only:
        sc_aggregate_path = (
            resolve_path(config["out_root"])
            / "math500_five_seed_sc_summary.json"
        )
        sc_summary = aggregate_sc_seed_runs(
            configured_seed_dirs,
            sc_aggregate_path,
        )
        print(f"aggregate={sc_aggregate_path}")
        print_aggregate(sc_summary)
        if not sc_summary or sc_summary[0]["n_seeds"] != len(
            configured_seeds
        ):
            raise RuntimeError(
                "SC aggregation is incomplete: expected "
                f"{len(configured_seeds)} validated seeds."
            )
        return

    seed_dirs = [
        (seed, out_dir)
        for seed, out_dir in configured_seed_dirs
        if (out_dir / "seed_validation.json").exists()
    ]
    aggregate_path = (
        resolve_path(config["out_root"])
        / "math500_five_seed_summary.json"
    )
    summary = aggregate_seed_runs(seed_dirs, aggregate_path)
    print(f"aggregate={aggregate_path}")
    print_aggregate(summary)


if __name__ == "__main__":
    main()
