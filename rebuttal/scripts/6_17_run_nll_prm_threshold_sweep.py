#!/usr/bin/env python3
"""Run independent stable-seed NLL/PRM threshold checkpoint deltas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "rebuttal/configs/6_17_qwen_nll_prm_threshold_sweep.json"
)
sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.src.nll_prm_controls import (  # noqa: E402
    NLL_SIGNAL,
    PRM_SIGNAL,
)
from rebuttal.src.threshold_sweep import (  # noqa: E402
    SIGNAL_CHECKPOINT_LAYOUT,
    aggregate_signal_dataset,
    prepare_shared_checkpoint,
    prepare_signal_checkpoint,
    read_json,
    signal_result_dir,
    signal_threshold_name,
    thresholds_for_signal,
    validate_shared_checkpoint,
    validate_signal_checkpoint,
    write_json,
    write_fresh_shared_manifest,
)
from rebuttal.src.wallclock import (  # noqa: E402
    discover_timing_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
    )
    parser.add_argument("--gpus", default="")
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument(
        "--reevaluate-only",
        action="store_true",
        help=(
            "Recompute fair/full evaluation artifacts from complete "
            "checkpoints without running any generation or scoring phase"
        ),
    )
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pipeline_command(
    config: dict,
    dataset: str,
    output_dir: Path,
    figure_dir: Path,
    gpus: str,
    nll_threshold: float,
    prm_threshold: float,
    skip_phases: list[int],
    *,
    signals: list[str] | None = None,
    checkpoint_parts: list[Path] | None = None,
    timing_mode: str = "auto",
    reuse_timing_manifests: list[Path] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(resolve(config["pipeline"])),
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
        *(signals or config["signals"]),
        "--prm-drop-threshold",
        str(prm_threshold),
        "--nll-drop-threshold",
        str(nll_threshold),
        "--logprob-batch-size",
        str(config.get("logprob_batch_size", 4)),
        "--nd",
        *[str(value) for value in config["nd"]],
        "--skip-phase",
        *[str(value) for value in skip_phases],
        "--out-dir",
        str(output_dir),
        "--fig-dir",
        str(figure_dir),
        "--timing-mode",
        timing_mode,
    ]
    for checkpoint_part in checkpoint_parts or []:
        command.extend(
            ["--checkpoint-part", str(checkpoint_part)]
        )
    for manifest in reuse_timing_manifests or []:
        command.extend(["--reuse-timing-manifest", str(manifest)])
    return command


def available_timing_manifests(
    checkpoints_or_dirs: list[Path],
) -> list[Path]:
    manifests = []
    seen = set()
    for value in checkpoints_or_dirs:
        checkpoint = (
            value / "checkpoint.jsonl" if value.is_dir() else value
        )
        manifest = discover_timing_manifest(checkpoint)
        if manifest is None:
            continue
        resolved = manifest.resolve()
        if resolved not in seen:
            seen.add(resolved)
            manifests.append(resolved)
    return manifests


def run_command(command: list[str], dry_run: bool) -> None:
    print("command=" + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def prepare_shared(
    config: dict,
    dataset: str,
    dataset_root: Path,
    figure_root: Path,
    gpus: str,
    dry_run: bool,
) -> Path:
    shared_dir = dataset_root / "_shared_seed42"
    stable_source = config.get("stable_sources", {}).get(dataset)
    source_mode = "stable_reuse" if stable_source else "fresh_generation"
    if dry_run:
        print(f"shared={shared_dir} source_mode={source_mode}")
    elif stable_source:
        manifest = prepare_shared_checkpoint(
            config, dataset, PROJECT_ROOT, shared_dir
        )
        print(
            "shared reuse="
            + json.dumps(manifest["record_counts"], sort_keys=True)
        )

    if stable_source:
        skip_phases = [1, 2, 25, 3, 4]
        reuse_timing_manifests = available_timing_manifests(
            [
                PROJECT_ROOT / stable_source["draft_checkpoint"],
                PROJECT_ROOT / stable_source["scoring_checkpoint"],
            ]
        )
    else:
        skip_phases = [25, 4]
        reuse_timing_manifests = []

    shared_command = pipeline_command(
        config,
        dataset,
        shared_dir,
        figure_root / "_shared_seed42",
        gpus,
        nll_threshold=float(config["nll_thresholds"][0]),
        prm_threshold=float(config["prm_thresholds"][0]),
        skip_phases=skip_phases,
        reuse_timing_manifests=reuse_timing_manifests,
    )
    shared_command.append("--skip-eval")
    run_command(shared_command, dry_run)
    recount = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "rebuttal/scripts/6_8_1_recount_tokens.py"),
        "--checkpoint",
        str(shared_dir / "checkpoint.jsonl"),
        "--model",
        config["model"],
    ]
    if dry_run or not (shared_dir / "draft_step_tokens.json").is_file():
        run_command(recount, dry_run)
    if not dry_run:
        report = validate_shared_checkpoint(config, dataset, shared_dir)
        if not stable_source:
            manifest = write_fresh_shared_manifest(
                config, dataset, shared_dir
            )
            print(
                "shared fresh="
                + json.dumps(manifest["record_counts"], sort_keys=True)
            )
        print(
            f"validated shared {dataset}: "
            f"{report['unique_sampling_seeds']} generation seeds"
        )
    return shared_dir


def run_dataset(
    config: dict,
    dataset: str,
    gpus: str,
    dry_run: bool,
    aggregate_only: bool,
    reevaluate_only: bool,
) -> None:
    results_root = resolve(config["results_root"])
    figures_root = resolve(config["figures_root"])
    dataset_root = results_root / dataset
    figure_root = figures_root / dataset
    if not dry_run:
        dataset_root.mkdir(parents=True, exist_ok=True)
        figure_root.mkdir(parents=True, exist_ok=True)
    shared_dir = dataset_root / "_shared_seed42"
    shared_checkpoint = shared_dir / "checkpoint.jsonl"

    if aggregate_only:
        aggregate_signal_dataset(
            config, dataset, dataset_root, figure_root / "aggregate"
        )
        return
    if reevaluate_only:
        for signal in config["signals"]:
            for threshold in thresholds_for_signal(config, signal):
                name = signal_threshold_name(signal, threshold)
                result_dir = signal_result_dir(
                    dataset_root, signal, threshold
                )
                signal_figure_dir = (
                    figure_root / "signals" / name
                )
                nll_threshold = (
                    threshold
                    if signal == NLL_SIGNAL
                    else float(config["nll_thresholds"][0])
                )
                prm_threshold = (
                    threshold
                    if signal == PRM_SIGNAL
                    else float(config["prm_thresholds"][0])
                )
                run_command(
                    pipeline_command(
                        config,
                        dataset,
                        result_dir,
                        signal_figure_dir,
                        gpus,
                        float(nll_threshold),
                        float(prm_threshold),
                        skip_phases=[1, 2, 25, 3, 4, 5],
                        signals=[signal],
                        checkpoint_parts=[shared_checkpoint],
                        timing_mode="eval-only",
                    ),
                    dry_run,
                )
                if not dry_run:
                    report = validate_signal_checkpoint(
                        config,
                        dataset,
                        shared_dir,
                        signal,
                        threshold,
                        result_dir,
                        signal_figure_dir,
                    )
                    print(
                        f"reevaluated {dataset}/{name}: "
                        f"{report['evaluation_rows']} rollback rows"
                    )
        if not dry_run:
            rows = aggregate_signal_dataset(
                config, dataset, dataset_root, figure_root / "aggregate"
            )
            print(
                f"reaggregated {dataset}: "
                f"{len(rows)} unique result rows"
            )
            (dataset_root / "COMPLETE").touch()
        return

    shared_dir = prepare_shared(
        config, dataset, dataset_root, figure_root, gpus, dry_run
    )
    shared_checkpoint = shared_dir / "checkpoint.jsonl"
    for signal in config["signals"]:
        for threshold in thresholds_for_signal(config, signal):
            name = signal_threshold_name(signal, threshold)
            result_dir = signal_result_dir(
                dataset_root, signal, threshold
            )
            signal_figure_dir = figure_root / "signals" / name
            nll_threshold = (
                threshold
                if signal == NLL_SIGNAL
                else float(config["nll_thresholds"][0])
            )
            prm_threshold = (
                threshold
                if signal == PRM_SIGNAL
                else float(config["prm_thresholds"][0])
            )
            print(
                f"\n=== {dataset}: {signal} "
                f"threshold={threshold} ===",
                flush=True,
            )
            if not dry_run:
                reuse = prepare_signal_checkpoint(
                    config,
                    dataset,
                    PROJECT_ROOT,
                    dataset_root,
                    shared_dir,
                    signal,
                    threshold,
                )
                print(
                    f"reused suffix records="
                    f"{reuse.get('imported_suffix_records', 0)}"
                )
                reuse_timing_manifests = available_timing_manifests(
                    [shared_dir]
                    + [
                        Path(source["source"])
                        for source in reuse.get("sources", [])
                        if source.get("status") == "used"
                    ]
                )
            else:
                reuse_timing_manifests = []
            run_command(
                pipeline_command(
                    config,
                    dataset,
                    result_dir,
                    signal_figure_dir,
                    gpus,
                    float(nll_threshold),
                    float(prm_threshold),
                    skip_phases=[1, 2, 25, 3, 5],
                    signals=[signal],
                    checkpoint_parts=[shared_checkpoint],
                    reuse_timing_manifests=reuse_timing_manifests,
                ),
                dry_run,
            )
            if not dry_run:
                report = validate_signal_checkpoint(
                    config,
                    dataset,
                    shared_dir,
                    signal,
                    threshold,
                    result_dir,
                    signal_figure_dir,
                )
                print(
                    f"validated {name}: "
                    f"{report['local_record_counts']['suffix']} suffixes"
                )
    if not dry_run:
        rows = aggregate_signal_dataset(
            config, dataset, dataset_root, figure_root / "aggregate"
        )
        print(f"aggregated {dataset}: {len(rows)} unique result rows")
        (dataset_root / "COMPLETE").touch()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    required_signals = {NLL_SIGNAL, PRM_SIGNAL}
    if set(config["signals"]) != required_signals:
        raise ValueError(f"Expected exactly {sorted(required_signals)}")
    if int(config["seed"]) != 42:
        raise ValueError("This rebuttal threshold sweep requires seed 42")
    expected_sc_supplement = int(config["budget"]) - max(
        int(value) for value in config["nd"]
    )
    configured_sc_supplement = config.get(
        "sc32_supplement_per_question"
    )
    if (
        configured_sc_supplement is not None
        and int(configured_sc_supplement) != expected_sc_supplement
    ):
        raise ValueError(
            "sc32_supplement_per_question="
            f"{configured_sc_supplement}, expected "
            f"budget-max(nd)={expected_sc_supplement}"
        )
    if (
        config.get("checkpoint_layout")
        != SIGNAL_CHECKPOINT_LAYOUT
    ):
        raise ValueError(
            "Expected checkpoint_layout="
            f"{SIGNAL_CHECKPOINT_LAYOUT}"
        )
    datasets = args.datasets or list(config["datasets"])
    unsupported = [
        dataset for dataset in datasets
        if dataset not in config["datasets"]
    ]
    if unsupported:
        raise ValueError(
            f"Datasets not configured in {args.config}: {unsupported}"
        )
    gpus = args.gpus or config["gpus"]
    for dataset in datasets:
        run_dataset(
            config,
            dataset,
            gpus,
            args.dry_run,
            args.aggregate_only,
            args.reevaluate_only,
        )
    if not args.dry_run:
        root = resolve(config["results_root"])
        if all(
            (root / dataset / "COMPLETE").is_file()
            for dataset in config["datasets"]
        ):
            completion = {
                "status": "passed",
                "datasets": list(config["datasets"]),
                "seed": config["seed"],
                "nll_thresholds": config["nll_thresholds"],
                "prm_thresholds": config["prm_thresholds"],
                "nll_backend": "vllm",
                "layout": SIGNAL_CHECKPOINT_LAYOUT,
                "cartesian_trial_checkpoints": False,
                "legacy_cartesian_trials_retained": True,
                "sc32_checkpoint_policy": config.get(
                    "sc32_checkpoint_policy", "resume_allowed"
                ),
                "sc32_semantics": (
                    f"{max(int(value) for value in config['nd'])} "
                    "stable draft answers plus "
                    f"{expected_sc_supplement} fullsc supplements "
                    "per question"
                ),
                "sc32_supplement_per_question": (
                    expected_sc_supplement
                ),
                "checkpoint_layout": (
                    "<dataset>/checkpoint_layout.json"
                ),
                "timing_schema_version": 1,
                "timing_manifest": (
                    "<dataset>/signals/<signal_threshold>/"
                    "timing/timing_manifest.json"
                ),
                "variants": ["fair", "fair_new", "full"],
                "fair_semantics": (
                    "nd original draft answers plus nd*(ns-1) "
                    "rollback suffix answers"
                ),
                "fair_new_semantics": (
                    "nd*ns rollback suffix answers and no original "
                    "draft answers"
                ),
                "full_semantics": (
                    "nd original draft answers plus nd*ns "
                    "rollback suffix answers"
                ),
            }
            write_json(root / "completion_report.json", completion)
            (root / "COMPLETE").touch()


if __name__ == "__main__":
    main()
