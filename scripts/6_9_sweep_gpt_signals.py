#!/usr/bin/env python3
"""
Queue runner: sweep 6_8_budget_controlled_multisignal.py across
multiple models and datasets sequentially.

Usage:
   
    systemd-run --user --scope -p MemoryMax=32G  python -u scripts/6_9_sweep_gpt_signals.py \
    --gpus 0,1,2,3 \
    --datasets hotpotqa 2wikimultihopqa musique strategyqa gpqa_diamond aime2024 olympiadbench \
    --signals prm_drop_fb_last nll_drop_fb_last \
    --skip-phase 25 \
    --n-sample 200 \
    --nd 4 8 16 \
    2>&1 | tee logs/6_9_sweep_gpt_signals_$(date +%Y%m%d_%H%M%S).log


    systemd-run --user --scope -p MemoryMax=32G python -u scripts/6_9_sweep_gpt_signals.py \
    --gpus 0,1,2,3 \
    --signals prm_drop_fb_last nll_drop_fb_last \
    --models qwen3b llama3b \
    --skip-phase 25 \
    --nd 4 8 16 \
    2>&1 | tee logs/6_9_sweep_gpt_signals_$(date +%Y%m%d_%H%M%S).log


 python -u scripts/6_9_sweep_gpt_signals.py \
    --gpus 4,5,6,7 \
    --datasets 2wikimultihopqa_open hotpotqa_open \
    --models llama3b \
    --signals prm_drop_fb_last nll_drop_fb_last \
    --skip-phase 25 \
    --nd 4 8 16 
       
    python scripts/6_9_sweep_gpt_signals.py \
    --gpus 4,5,6,7 \
    --models llama3b \
    --datasets math500 gsm8k\
    --signals prm_drop_fb_last \
    --prm-model Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --draft-checkpoint-dir results/qwen2.5_3b_instruct_budget_multisignal \
    --skip-phase 3 25 5 \
    --nd 4 8 16

"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "6_8_budget_controlled_multisignal.py"
PYTHON = sys.executable
RESULTS_ROOT = Path("/mnt/beegfs/youyang7/projects/LostInSecond/results")

# ---- Model registry ----
MODEL_REGISTRY = {
    "deepseek7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
}

# ---- Dataset groups ----
MATH_DATASETS = [
    "gsm8k", "math500", "amc2023",
    "aime2024", "olympiadbench",
]
MULTIHOP_DATASETS = [
    "hotpotqa", "2wikimultihopqa",
]
MULTIHOP_OPEN_DATASETS = [
    "hotpotqa_open", "2wikimultihopqa_open",
]
ALL_DATASETS = MATH_DATASETS + MULTIHOP_OPEN_DATASETS

# Per-dataset sample caps (0 = use full test set).
# Seed is fixed (42) in sweep_datasets, so the subset is deterministic.
DATASET_SAMPLE_LIMITS = {
    "gsm8k": 1319,              # full: 1,319
    "math500": 500,             # full: 500
    "aime2024": 90,             # full: 90
    "amc2023": 83,              # full: 83
    "olympiadbench": 674,       # full: 674
    "hotpotqa": 500,           # full: 7,405
    "hotpotqa_open": 500,      # full: 7,405
    "2wikimultihopqa": 1000,    # full: 12,576
    "2wikimultihopqa_open": 1000,  # full: 12,576
    "strategyqa": 687,          # full: 687
    "gpqa_diamond": 448,        # full: 448
    "humaneval": 164,           # full: 164
    "csqa": 1221,               # full: 1,221
}

DEFAULT_MODELS = list(MODEL_REGISTRY.keys())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=list(MODEL_REGISTRY.keys()),
                    help="Model short names to sweep")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    help="Dataset names to sweep")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--nd", type=int, nargs="+", default=None,
                    help="Which nd values to evaluate (default: all)")
    ap.add_argument("--n-sample", "--n_samples", type=int, default=0,
                    dest="n_sample",
                    help="Limit dataset samples; 0=all")
    ap.add_argument("--signals", nargs="+",
                    default=["gpt_fb_last", "gpt_fb_start"])
    ap.add_argument("--skip-phase", type=int, nargs="*", default=[2, 3],
                    help="Phases to skip (default: 2=PRM, 3=logprob)")
    ap.add_argument("--gpt-model", default="gpt-5.1")
    ap.add_argument("--gpt-max-workers", type=int, default=32)
    ap.add_argument("--prm-model", default="",
                    help="PRM model ID to pass to 6_8 "
                         "(default: Qwen2.5-Math-PRM-7B; "
                         "use Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B "
                         "for 1.5B)")
    ap.add_argument("--checkpoint-dir", default="",
                    help="Base dir with existing checkpoints to copy from "
                         "(e.g. results/); expects {dataset}_budget_controlled/")
    ap.add_argument("--draft-checkpoint-dir", default="",
                    help="Base dir with existing draft checkpoints to "
                         "import from (e.g. results/qwen2.5_3b_instruct"
                         "_budget_multisignal/); expects {dataset}/"
                         "checkpoint.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    return ap.parse_args()


def copy_checkpoint_if_available(dataset, model_tag, out_dir, ckpt_base):
    """Copy checkpoint from a previous run if available."""
    if not ckpt_base:
        return
    src = Path(ckpt_base) / f"{dataset}_budget_controlled" / "checkpoint.jsonl"
    dst = Path(out_dir) / "checkpoint.jsonl"
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)
        n = sum(1 for _ in open(dst))
        print(f"  Copied checkpoint: {src} -> {dst} ({n} records)")


def run_one(model_key, dataset, args):
    model_id = MODEL_REGISTRY[model_key]
    tag = model_key
    model_short = Path(model_id).name.lower().replace("-", "_")
    prm_tag = ""
    if args.prm_model and "skywork" in args.prm_model.lower():
        prm_short = Path(args.prm_model).name.lower().replace("-", "_")
        prm_tag = f"_prm_{prm_short}"
    out_dir = (RESULTS_ROOT
               / f"{model_short}_budget_multisignal{prm_tag}"
               / dataset)

    copy_checkpoint_if_available(
        dataset, tag, out_dir, args.checkpoint_dir)

    if args.n_sample > 0:
        n_sample = args.n_sample
    else:
        n_sample = DATASET_SAMPLE_LIMITS.get(dataset, 0)

    cmd = [
        PYTHON, str(SCRIPT),
        "--gpus", args.gpus,
        "--model", model_id,
        "--dataset", dataset,
        "--budget", str(args.budget),
        "--n-sample", str(n_sample),
        "--signals", *args.signals,
        "--gpt-model", args.gpt_model,
        "--gpt-max-workers", str(args.gpt_max_workers),
        "--out-dir", str(out_dir),
    ]
    if args.prm_model:
        cmd += ["--prm-model", args.prm_model]
    if args.draft_checkpoint_dir:
        draft_ckpt = (Path(args.draft_checkpoint_dir)
                      / dataset / "checkpoint.jsonl")
        if draft_ckpt.exists():
            cmd += ["--draft-checkpoint", str(draft_ckpt)]
    if args.skip_phase:
        cmd += ["--skip-phase"] + [str(p) for p in args.skip_phase]
    if args.nd:
        cmd += ["--nd"] + [str(n) for n in args.nd]

    print(f"\n{'='*70}")
    print(f"  Model: {model_key} ({model_id})")
    print(f"  Dataset: {dataset}")
    print(f"  Out: {out_dir}")
    print(f"  Cmd: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")

    if args.dry_run:
        print("  [DRY RUN] skipped")
        return 0

    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
    print(f"\n  [{status}] {model_key} x {dataset} "
          f"in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return ret.returncode


def main():
    args = parse_args()

    combos = [(m, d) for m in args.models for d in args.datasets]
    total = len(combos)
    print(f"Sweep: {len(args.models)} models x {len(args.datasets)} datasets "
          f"= {total} runs")
    print(f"Models:   {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Signals:  {args.signals}")
    print(f"GPUs:     {args.gpus}")
    if args.n_sample > 0:
        print(f"N-sample: {args.n_sample}")

    results = []
    t_start = time.time()
    for i, (model_key, dataset) in enumerate(combos, 1):
        print(f"\n>>> Run {i}/{total}: {model_key} x {dataset}")
        rc = run_one(model_key, dataset, args)
        results.append((model_key, dataset, rc))

    elapsed_total = time.time() - t_start
    print(f"\n\n{'='*70}")
    print(f"SWEEP COMPLETE  ({elapsed_total:.0f}s = {elapsed_total/3600:.1f}h)")
    print(f"{'='*70}")
    for m, d, rc in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  {status:>8s}  {m:<12s}  {d}")

    n_fail = sum(1 for _, _, rc in results if rc != 0)
    if n_fail:
        print(f"\n{n_fail}/{total} runs failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
