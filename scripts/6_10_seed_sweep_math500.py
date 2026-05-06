#!/usr/bin/env python3
"""
Seed sweep: run 6_8_budget_controlled_multisignal.py on MATH500 with
Qwen 2.5 3B multiple times using different random seeds, then aggregate
results to measure variance.

Usage:
    python -u scripts/6_10_seed_sweep_math500.py \
        --gpus 0,1,2,3 \
        --seeds 42 123 456 789 1024 \
        --signals prm_drop_fb_last nll_drop_fb_last \
        --skip-phase 25 \
        --nd 4 8 16 \
        2>&1 | tee logs/6_10_seed_sweep_$(date +%Y%m%d_%H%M%S).log
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "6_8_budget_controlled_multisignal.py"
PYTHON = sys.executable

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET = "math500"
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Seed sweep for MATH500 with Qwen 2.5 3B")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=DEFAULT_SEEDS,
                    help="Random seeds to sweep")
    ap.add_argument("--signals", nargs="+",
                    default=["prm_drop_fb_last",
                             "nll_drop_fb_last"])
    ap.add_argument("--nd", type=int, nargs="+",
                    default=[4, 8, 16],
                    help="Which nd values to evaluate")
    ap.add_argument("--skip-phase", type=int, nargs="*",
                    default=[25],
                    help="Phases to skip (default: 25=GPT)")
    ap.add_argument("--prm-model", default="",
                    help="PRM model ID")
    ap.add_argument("--draft-checkpoint-dir", default="",
                    help="Base dir with existing draft checkpoints")
    ap.add_argument("--gpt-model", default="gpt-5.1")
    ap.add_argument("--gpt-max-workers", type=int, default=32)
    ap.add_argument("--n-sample", type=int, default=0,
                    help="Limit dataset samples; 0=all")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def run_one_seed(seed, args):
    """Run the full pipeline for one seed."""
    model_short = Path(MODEL_ID).name.lower().replace("-", "_")
    prm_tag = ""
    if args.prm_model and "skywork" in args.prm_model.lower():
        prm_short = (Path(args.prm_model).name.lower()
                     .replace("-", "_"))
        prm_tag = f"_prm_{prm_short}"
    out_dir = (PROJECT_ROOT / "results"
               / f"{model_short}_budget_multisignal{prm_tag}"
               / f"{DATASET}_seed{seed}")

    cmd = [
        PYTHON, str(SCRIPT),
        "--gpus", args.gpus,
        "--model", MODEL_ID,
        "--dataset", DATASET,
        "--seed", str(seed),
        "--n-sample", str(args.n_sample),
        "--signals", *args.signals,
        "--gpt-model", args.gpt_model,
        "--gpt-max-workers", str(args.gpt_max_workers),
        "--out-dir", str(out_dir),
    ]
    if args.prm_model:
        cmd += ["--prm-model", args.prm_model]
    if args.draft_checkpoint_dir:
        draft_ckpt = (Path(args.draft_checkpoint_dir)
                      / DATASET / "checkpoint.jsonl")
        if draft_ckpt.exists():
            cmd += ["--draft-checkpoint", str(draft_ckpt)]
    if args.skip_phase:
        cmd += ["--skip-phase"] + [str(p) for p in args.skip_phase]
    if args.nd:
        cmd += ["--nd"] + [str(n) for n in args.nd]

    print(f"\n{'='*70}")
    print(f"  Seed: {seed}")
    print(f"  Out:  {out_dir}")
    print(f"  Cmd:  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")

    if args.dry_run:
        print("  [DRY RUN] skipped")
        return 0, out_dir

    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    status = ("OK" if ret.returncode == 0
              else f"FAILED (exit {ret.returncode})")
    print(f"\n  [{status}] seed={seed} "
          f"in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return ret.returncode, out_dir


def aggregate_results(seed_dirs, seeds, args):
    """Collect eval_summary.json from each seed and compute stats."""
    all_results = {}
    for seed, out_dir in zip(seeds, seed_dirs):
        summary_path = out_dir / "eval_summary.json"
        if not summary_path.exists():
            print(f"  WARNING: missing {summary_path}")
            continue
        data = json.loads(summary_path.read_text())
        for r in data:
            method = r["method"]
            if method not in all_results:
                all_results[method] = []
            all_results[method].append({
                "seed": seed,
                "acc": r["acc"],
                "tokens_per_q": r["tokens_per_q"],
            })

    summary = []
    for method, entries in sorted(all_results.items()):
        accs = [e["acc"] for e in entries]
        tpqs = [e["tokens_per_q"] for e in entries]
        summary.append({
            "method": method,
            "n_seeds": len(entries),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "acc_min": float(np.min(accs)),
            "acc_max": float(np.max(accs)),
            "tokens_per_q_mean": float(np.mean(tpqs)),
            "tokens_per_q_std": float(np.std(tpqs)),
            "per_seed": entries,
        })

    model_short = Path(MODEL_ID).name.lower().replace("-", "_")
    prm_tag = ""
    if args.prm_model and "skywork" in args.prm_model.lower():
        prm_short = (Path(args.prm_model).name.lower()
                     .replace("-", "_"))
        prm_tag = f"_prm_{prm_short}"
    base_dir = (PROJECT_ROOT / "results"
                / f"{model_short}_budget_multisignal{prm_tag}")
    out_path = base_dir / f"{DATASET}_seed_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved aggregated summary: {out_path}")

    print(f"\n{'Method':<40} {'Mean':>7} {'Std':>7} "
          f"{'Min':>7} {'Max':>7} {'Tok/Q':>8}")
    print("-" * 80)
    for r in summary:
        print(f"{r['method']:<40} "
              f"{r['acc_mean']:>7.4f} {r['acc_std']:>7.4f} "
              f"{r['acc_min']:>7.4f} {r['acc_max']:>7.4f} "
              f"{r['tokens_per_q_mean']:>8.0f}")

    return summary


def main():
    args = parse_args()
    seeds = args.seeds

    print(f"Seed sweep: {DATASET} x {MODEL_ID}")
    print(f"Seeds: {seeds}")
    print(f"Signals: {args.signals}")
    print(f"GPUs: {args.gpus}")
    print(f"nd: {args.nd}")

    results = []
    seed_dirs = []
    t_start = time.time()

    for i, seed in enumerate(seeds, 1):
        print(f"\n>>> Run {i}/{len(seeds)}: seed={seed}")
        rc, out_dir = run_one_seed(seed, args)
        results.append((seed, rc))
        seed_dirs.append(out_dir)

    elapsed_total = time.time() - t_start
    print(f"\n\n{'='*70}")
    print(f"SWEEP COMPLETE  "
          f"({elapsed_total:.0f}s = {elapsed_total/3600:.1f}h)")
    print(f"{'='*70}")
    for seed, rc in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  {status:>8s}  seed={seed}")

    n_fail = sum(1 for _, rc in results if rc != 0)
    if n_fail:
        print(f"\n{n_fail}/{len(seeds)} runs failed.")

    successful_dirs = [d for (_, rc), d
                       in zip(results, seed_dirs) if rc == 0]
    successful_seeds = [s for (s, rc) in results if rc == 0]
    if successful_dirs:
        aggregate_results(successful_dirs, successful_seeds, args)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
