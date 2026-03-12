#!/usr/bin/env python3
"""
Systematic LoRA hyperparameter sweep for low-data SFT.

Runs 25 experiments across 7 hyperparameter axes:
  1. Learning rate:     {1e-5, 5e-5, 1e-4, 2e-4, 5e-4}
  2. LoRA rank:         {4, 8, 16, 32, 64}  (alpha = 2r)
  3. Alpha/r ratio:     {1x, 2x, 4x}
  4. Regularization:    dropout × weight_decay grid
  5. Epochs:            {3, 5, 10, 15, 20}
  6. Target modules:    all-proj / attn-only / qv-only
  7. Dataset variant:   prefill / changed_only

Experiments run in parallel batches (one per GPU).
After training, results are compiled into a ranked summary.
Top-K adapters are then evaluated on GSM8K test set.

Usage:
    python 07_lora_sweep.py --gpus 0,1,2,3
    python 07_lora_sweep.py --gpus 1,2,3 --eval-top-k 5
    python 07_lora_sweep.py --phase eval-only --eval-top-k 5 --eval-gpu 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
TRAIN_SCRIPT = SCRIPT_DIR / "05_finetune_lora_same_dataset.py"
EVAL_SCRIPT = SCRIPTS_ROOT / "eval" / "06_eval_lora_effect.py"

BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
DATASET_PREFILL = BEEGFS_ARTIFACTS / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json"
DATASET_CHANGED = BEEGFS_ARTIFACTS / "samples_gsm8k_train_ds2_fix_step2_CHANGED_ONLY.json"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
HARNESS_DIR = PROJECT_ROOT / "lm-evaluation-harness"
PYTHON_BIN = "/mnt/beegfs/youyang7/.conda/envs/fact/bin/python"

ALL_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
ATTN_MODULES = "q_proj,k_proj,v_proj,o_proj"
QV_MODULES = "q_proj,v_proj"


def build_experiments() -> List[Dict[str, Any]]:
    exps = []

    def add(name, lr=2e-4, r=16, alpha=32, dropout=0.05, epochs=5,
            wd=0.0, modules=ALL_MODULES, dataset="prefill"):
        exps.append(dict(name=name, lr=lr, r=r, alpha=alpha, dropout=dropout,
                         epochs=epochs, wd=wd, modules=modules, dataset=dataset))

    # --- Phase 1: Learning Rate (most impactful HP) ---
    add("lr_1e-5",  lr=1e-5)
    add("lr_5e-5",  lr=5e-5)
    add("lr_1e-4",  lr=1e-4)
    add("lr_2e-4",  lr=2e-4)   # baseline config
    add("lr_5e-4",  lr=5e-4)

    # --- Phase 2: LoRA Rank (capacity vs overfitting) ---
    add("rank_4",   r=4,  alpha=8)
    add("rank_8",   r=8,  alpha=16)
    # r=16 already covered as lr_2e-4
    add("rank_32",  r=32, alpha=64)
    add("rank_64",  r=64, alpha=128)

    # --- Phase 3: Alpha / r ratio (effective lr scaling) ---
    # ratio=2 already covered (alpha=32, r=16)
    add("alpha_1x",  alpha=16)       # ratio = 1
    add("alpha_4x",  alpha=64)       # ratio = 4

    # --- Phase 4: Regularization (critical for 115-sample regime) ---
    add("drop0_wd0",         dropout=0.0,  wd=0.0)
    add("drop0.1_wd0",       dropout=0.1,  wd=0.0)
    add("drop0.05_wd0.01",   dropout=0.05, wd=0.01)
    add("drop0.05_wd0.1",    dropout=0.05, wd=0.1)
    add("drop0.1_wd0.01",    dropout=0.1,  wd=0.01)

    # --- Phase 5: Epochs (overfitting trajectory) ---
    add("epochs_3",   epochs=3)
    # epochs=5 already covered
    add("epochs_10",  epochs=10)
    add("epochs_15",  epochs=15)
    add("epochs_20",  epochs=20)

    # --- Phase 6: Target Modules ---
    add("attn_only", modules=ATTN_MODULES)
    add("qv_only",   modules=QV_MODULES)

    # --- Phase 7: Dataset Variant ---
    add("changed_only_ds", dataset="changed_only")

    # --- Phase 8: Combined configs (informed guesses) ---
    add("combo_conservative", lr=5e-5,  r=8,  alpha=16, dropout=0.1, epochs=10, wd=0.01)
    add("combo_aggressive",   lr=5e-4,  r=32, alpha=64, dropout=0.0, epochs=3,  wd=0.0)

    return exps


def launch_experiment(exp: Dict, gpu_id: int, sweep_root: Path) -> subprocess.Popen:
    name = exp["name"]
    output_dir = sweep_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DATASET_CHANGED if exp["dataset"] == "changed_only" else DATASET_PREFILL

    cmd = [
        PYTHON_BIN, str(TRAIN_SCRIPT),
        "--model-id",    MODEL_ID,
        "--dataset-path", str(dataset),
        "--output-dir",  str(output_dir),
        "--lora-r",      str(exp["r"]),
        "--lora-alpha",  str(exp["alpha"]),
        "--lora-dropout", str(exp["dropout"]),
        "--target-modules", exp["modules"],
        "--num-train-epochs", str(exp["epochs"]),
        "--learning-rate",    str(exp["lr"]),
        "--weight-decay",     str(exp["wd"]),
        "--warmup-ratio",     "0.03",
        "--gradient-accumulation-steps", "16",
        "--per-device-train-batch-size", "1",
        "--logging-steps",    "1",
        "--eval-steps",       "7",
        "--save-steps",       "9999",
        "--save-total-limit", "1",
        "--seed",             "42",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = output_dir / "training.log"
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    proc._log_fh = fh
    return proc


def run_sweep(experiments: List[Dict], gpus: List[int], sweep_root: Path):
    batch_size = len(gpus)
    total_batches = (len(experiments) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch = experiments[batch_start: batch_start + batch_size]

        print(f"\n{'=' * 70}")
        print(f"BATCH {batch_idx + 1}/{total_batches}  |  "
              f"Experiments: {[e['name'] for e in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, exp in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            print(f"  Launching {exp['name']} on GPU {gpu} ...")
            proc = launch_experiment(exp, gpu, sweep_root)
            procs.append((exp["name"], proc))
            time.sleep(15)

        for name, proc in procs:
            proc.wait()
            proc._log_fh.close()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

    print(f"\nAll {len(experiments)} experiments completed.")


def compile_results(sweep_root: Path) -> List[Dict]:
    results = []
    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir():
            continue
        metrics_file = d / "sweep_metrics.json"
        if not metrics_file.exists():
            results.append({"name": d.name, "status": "FAILED"})
            continue

        m = json.loads(metrics_file.read_text())
        eval_loss = m.get("eval_metrics", {}).get("eval_loss")
        train_loss = m.get("train_metrics", {}).get("train_loss")
        train_runtime = m.get("train_metrics", {}).get("train_runtime")
        cfg = m.get("config", {})

        log_history = m.get("log_history", [])
        train_losses = [x["loss"] for x in log_history if "loss" in x]
        eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]

        results.append({
            "name": d.name,
            "status": "OK",
            "eval_loss": eval_loss,
            "train_loss": train_loss,
            "train_runtime_sec": round(train_runtime, 1) if train_runtime else None,
            "min_train_loss": min(train_losses) if train_losses else None,
            "train_loss_trajectory": train_losses[-5:] if train_losses else [],
            "eval_loss_trajectory": eval_losses if eval_losses else [],
            "lr": cfg.get("learning_rate"),
            "r": cfg.get("lora_r"),
            "alpha": cfg.get("lora_alpha"),
            "dropout": cfg.get("lora_dropout"),
            "epochs": cfg.get("num_train_epochs"),
            "wd": cfg.get("weight_decay"),
            "modules": cfg.get("target_modules"),
            "train_samples": cfg.get("train_samples"),
        })

    results.sort(key=lambda x: x.get("eval_loss") if x.get("eval_loss") is not None else float("inf"))
    return results


def print_summary(results: List[Dict]):
    print(f"\n{'=' * 90}")
    print(f"{'SWEEP RESULTS SUMMARY':^90}")
    print(f"{'=' * 90}")
    print(f"{'Rank':<5} {'Name':<25} {'Eval Loss':<12} {'Train Loss':<12} "
          f"{'LR':<10} {'r':<4} {'α':<5} {'Ep':<4} {'Drop':<6} {'WD':<6}")
    print("-" * 90)

    for i, r in enumerate(results):
        if r["status"] != "OK":
            print(f"{i+1:<5} {r['name']:<25} {'FAILED':<12}")
            continue
        el = f"{r['eval_loss']:.4f}" if r['eval_loss'] is not None else "N/A"
        tl = f"{r['train_loss']:.4f}" if r['train_loss'] is not None else "N/A"
        lr = f"{r['lr']:.0e}" if r['lr'] is not None else "?"
        print(f"{i+1:<5} {r['name']:<25} {el:<12} {tl:<12} "
              f"{lr:<10} {r.get('r','?'):<4} {r.get('alpha','?'):<5} "
              f"{r.get('epochs','?'):<4} {r.get('dropout','?'):<6} {r.get('wd','?'):<6}")

    print(f"\nTotal: {sum(1 for r in results if r['status']=='OK')} OK, "
          f"{sum(1 for r in results if r['status']!='OK')} failed")

    if results and results[0]["status"] == "OK":
        best = results[0]
        print(f"\nBest config: {best['name']}")
        print(f"  eval_loss={best['eval_loss']:.4f}  train_loss={best['train_loss']:.4f}")
        print(f"  lr={best['lr']}  r={best['r']}  alpha={best['alpha']}  "
              f"epochs={best['epochs']}  dropout={best['dropout']}  wd={best['wd']}")


def evaluate_on_gsm8k(adapter_dir: Path, eval_output: Path,
                      gpu_id: int, task: str = "gsm8k_cot_zeroshot_unified",
                      limit: int = 0):
    """Run GSM8K evaluation for a single adapter."""
    adapter_path = adapter_dir / "final_adapter"
    if not adapter_path.exists():
        print(f"  Adapter not found: {adapter_path}")
        return None

    eval_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN, str(EVAL_SCRIPT),
        "--model-id",    MODEL_ID,
        "--lora-path",   str(adapter_path),
        "--task",        task,
        "--harness-dir", str(HARNESS_DIR),
        "--output-root", str(eval_output),
        "--batch-size",  "16",
        "--dtype",       "bfloat16",
        "--skip-base",
        "--cuda-visible-devices", str(gpu_id),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    log_file = eval_output / "eval.log"
    print(f"  Evaluating on GSM8K (gpu={gpu_id}) ...")
    with open(log_file, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"  Eval failed (rc={proc.returncode})")
        return None

    summary_file = eval_output / "comparison_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        score = summary.get("lora", {}).get("score_exact_match")
        print(f"  GSM8K exact_match = {score}")
        return score
    return None


def run_gsm8k_eval_phase(results: List[Dict], sweep_root: Path,
                         top_k: int, eval_gpu: int, eval_limit: int):
    print(f"\n{'=' * 70}")
    print(f"GSM8K EVALUATION PHASE  (top {top_k} configs)")
    print(f"{'=' * 70}")

    # First run base evaluation once
    print("\n[1/2] Running BASE model evaluation ...")
    base_eval_dir = sweep_root / "_gsm8k_eval" / "base_eval"
    base_eval_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        PYTHON_BIN, str(EVAL_SCRIPT),
        "--model-id",    MODEL_ID,
        "--lora-path",   str(sweep_root / results[0]["name"] / "final_adapter"),
        "--task",        "gsm8k_cot_zeroshot_unified",
        "--harness-dir", str(HARNESS_DIR),
        "--output-root", str(base_eval_dir),
        "--batch-size",  "8",
        "--dtype",       "bfloat16",
        "--skip-lora",
        "--cuda-visible-devices", str(eval_gpu),
    ]
    if eval_limit > 0:
        base_cmd.extend(["--limit", str(eval_limit)])

    with open(base_eval_dir / "eval.log", "w") as fh:
        subprocess.run(base_cmd, stdout=fh, stderr=subprocess.STDOUT)

    base_summary = base_eval_dir / "comparison_summary.json"
    base_score = None
    if base_summary.exists():
        base_score = json.loads(base_summary.read_text()).get("base", {}).get("score_exact_match")
        print(f"  Base model GSM8K exact_match = {base_score}")

    # Evaluate top-K LoRA adapters
    print(f"\n[2/2] Evaluating top-{top_k} LoRA adapters ...")
    eval_results = []
    ok_results = [r for r in results if r["status"] == "OK"]

    for i, r in enumerate(ok_results[:top_k]):
        name = r["name"]
        print(f"\n  [{i+1}/{top_k}] {name} (eval_loss={r['eval_loss']:.4f})")
        adapter_dir = sweep_root / name
        eval_output = sweep_root / "_gsm8k_eval" / name

        score = evaluate_on_gsm8k(adapter_dir, eval_output, eval_gpu, limit=eval_limit)
        delta = (score - base_score) if (score is not None and base_score is not None) else None
        eval_results.append({
            "name": name,
            "eval_loss": r["eval_loss"],
            "gsm8k_exact_match": score,
            "base_exact_match": base_score,
            "delta": delta,
        })

    # Print GSM8K evaluation summary
    print(f"\n{'=' * 70}")
    print(f"{'GSM8K EVALUATION SUMMARY':^70}")
    print(f"{'=' * 70}")
    print(f"Base model exact_match: {base_score}")
    print(f"\n{'Rank':<5} {'Name':<25} {'GSM8K EM':<12} {'Delta':<10} {'Eval Loss':<12}")
    print("-" * 65)
    eval_results.sort(key=lambda x: -(x["gsm8k_exact_match"] or 0))
    for i, er in enumerate(eval_results):
        em = f"{er['gsm8k_exact_match']:.4f}" if er['gsm8k_exact_match'] is not None else "FAIL"
        d = f"{er['delta']:+.4f}" if er['delta'] is not None else "N/A"
        el = f"{er['eval_loss']:.4f}" if er['eval_loss'] is not None else "N/A"
        print(f"{i+1:<5} {er['name']:<25} {em:<12} {d:<10} {el:<12}")

    # Save full evaluation summary
    final_summary = {
        "base_score": base_score,
        "eval_results": eval_results,
        "eval_limit": eval_limit,
    }
    out_path = sweep_root / "_gsm8k_eval" / "gsm8k_summary.json"
    out_path.write_text(json.dumps(final_summary, indent=2))
    print(f"\nSaved GSM8K summary to {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="Comma-separated GPU IDs for training")
    ap.add_argument("--sweep-root", default=str(BEEGFS_ARTIFACTS / "lora_sweep"),
                    help="Output directory for sweep results")
    ap.add_argument("--phase", default="all",
                    choices=["all", "train-only", "compile-only", "eval-only"],
                    help="Which phase to run")
    ap.add_argument("--eval-top-k", type=int, default=5,
                    help="Evaluate top K configs on GSM8K")
    ap.add_argument("--eval-gpu", type=int, default=1,
                    help="GPU for GSM8K evaluation")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="Limit GSM8K eval samples (0=full)")
    return ap.parse_args()


def main():
    args = parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    sweep_root = Path(args.sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments()

    # Save experiment plan
    plan_file = sweep_root / "experiment_plan.json"
    plan_file.write_text(json.dumps(experiments, indent=2))
    print(f"Experiment plan ({len(experiments)} experiments) saved to {plan_file}")
    print(f"GPUs: {gpus}  |  Sweep root: {sweep_root}")

    # Phase 1: Training
    if args.phase in ("all", "train-only"):
        print(f"\n{'#' * 70}")
        print(f"  TRAINING PHASE: {len(experiments)} experiments, {len(gpus)} GPUs")
        print(f"{'#' * 70}")
        t0 = time.time()
        run_sweep(experiments, gpus, sweep_root)
        elapsed = time.time() - t0
        print(f"\nTraining phase completed in {elapsed/60:.1f} minutes")

    # Phase 2: Compile results
    if args.phase in ("all", "train-only", "compile-only", "eval-only"):
        results = compile_results(sweep_root)
        print_summary(results)
        summary_file = sweep_root / "sweep_summary.json"
        summary_file.write_text(json.dumps(results, indent=2))
        print(f"\nFull summary saved to {summary_file}")

    # Phase 3: GSM8K evaluation
    if args.phase in ("all", "eval-only") and args.eval_top_k > 0:
        results = compile_results(sweep_root)
        run_gsm8k_eval_phase(results, sweep_root, args.eval_top_k,
                             args.eval_gpu, args.eval_limit)

    print("\nDone!")


if __name__ == "__main__":
    main()
