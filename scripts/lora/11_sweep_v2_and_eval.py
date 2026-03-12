#!/usr/bin/env python3
"""
Phase 2 LoRA sweep + comprehensive GSM8K evaluation.

Phase 2 training: 16 experiments informed by Phase 1 insights:
  - Fine-grained LR around sweet spot (7e-5 to 1.5e-4)
  - Combined best settings (lr + rank + alpha/r ratio)
  - Warmup ratio exploration
  - "Optimal" combined configs

Then: GSM8K evaluation on ALL adapters (Phase 1 + Phase 2 + base)
using parallel evaluation on 4 GPUs.

Usage:
    python 11_sweep_v2_and_eval.py --gpus 0,1,2,3
    python 11_sweep_v2_and_eval.py --phase eval-only --gpus 0,1,2,3
    python 11_sweep_v2_and_eval.py --phase summary-only
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
TRAIN_SCRIPT = SCRIPT_DIR / "05_finetune_lora_same_dataset.py"
HARNESS_DIR = PROJECT_ROOT / "lm-evaluation-harness"

BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
DATASET_PREFILL = BEEGFS_ARTIFACTS / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PYTHON_BIN = "/mnt/beegfs/youyang7/.conda/envs/fact/bin/python"
TASK = "gsm8k_cot_zeroshot_unified"

ALL_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
ATTN_MODULES = "q_proj,k_proj,v_proj,o_proj"


# ============================================================
# Phase 2 experiment definitions
# ============================================================

def build_phase2_experiments() -> List[Dict[str, Any]]:
    """
    Phase 1 found:
      - Best LR: 1e-4  (eval_loss=0.2682, but tested only with r=16, alpha=32)
      - Best rank: 8    (eval_loss=0.2686, but tested only with lr=2e-4)
      - Best ratio: 1   (eval_loss=0.2692, but tested only with lr=2e-4)
      - Best epochs: 3-5
      - Warmup: never varied (fixed 0.03)
      - Overfitting is dominant failure mode (109 train samples)

    Phase 2 strategy: combine winners, fine-grained search, explore warmup.
    """
    exps = []

    def add(name, lr=2e-4, r=16, alpha=32, dropout=0.05, epochs=5,
            wd=0.0, warmup=0.03, ga_steps=16, modules=ALL_MODULES):
        exps.append(dict(name=name, lr=lr, r=r, alpha=alpha, dropout=dropout,
                         epochs=epochs, wd=wd, warmup=warmup,
                         ga_steps=ga_steps, modules=modules))

    # --- Group A: Fine-grained LR around sweet spot ---
    # Phase 1 gap: lr=5e-5 (0.2816) -> lr=1e-4 (0.2682) -> lr=2e-4 (0.2919)
    add("v2_lr_7e-5",   lr=7e-5)
    add("v2_lr_8e-5",   lr=8e-5)
    add("v2_lr_1.2e-4", lr=1.2e-4)
    add("v2_lr_1.5e-4", lr=1.5e-4)

    # --- Group B: Combined best settings (never tested together in Phase 1) ---
    add("v2_lr1e4_r8",      lr=1e-4, r=8, alpha=16)
    add("v2_lr1e4_r8_a1x",  lr=1e-4, r=8, alpha=8)
    add("v2_lr7e5_r8",      lr=7e-5, r=8, alpha=16)
    add("v2_lr1e4_r4_a1x",  lr=1e-4, r=4, alpha=4)

    # --- Group C: Warmup ratio + epoch tuning (with best LR) ---
    # Warmup was always fixed at 0.03 in Phase 1
    add("v2_warmup_0",    lr=1e-4, warmup=0.0)
    add("v2_warmup_0.1",  lr=1e-4, warmup=0.1)
    add("v2_lr1e4_ep3",   lr=1e-4, epochs=3)
    add("v2_lr1e4_ep7",   lr=1e-4, epochs=7)

    # --- Group D: "Optimal" combined configs ---
    add("v2_optimal_a", lr=1e-4, r=8, alpha=16, epochs=5, warmup=0.05, dropout=0.05)
    add("v2_optimal_b", lr=7e-5, r=8, alpha=16, epochs=7, warmup=0.1,
        dropout=0.1, wd=0.01)
    add("v2_optimal_c", lr=1e-4, r=8, alpha=8, epochs=4, warmup=0.05)
    add("v2_optimal_d", lr=8e-5, r=8, alpha=16, epochs=5, warmup=0.05, dropout=0.05)

    return exps


# ============================================================
# Training
# ============================================================

def launch_training(exp: Dict, gpu_id: int, sweep_root: Path) -> subprocess.Popen:
    name = exp["name"]
    output_dir = sweep_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN, str(TRAIN_SCRIPT),
        "--model-id",    MODEL_ID,
        "--dataset-path", str(DATASET_PREFILL),
        "--output-dir",  str(output_dir),
        "--lora-r",      str(exp["r"]),
        "--lora-alpha",  str(exp["alpha"]),
        "--lora-dropout", str(exp["dropout"]),
        "--target-modules", exp["modules"],
        "--num-train-epochs", str(exp["epochs"]),
        "--learning-rate",    str(exp["lr"]),
        "--weight-decay",     str(exp["wd"]),
        "--warmup-ratio",     str(exp["warmup"]),
        "--gradient-accumulation-steps", str(exp["ga_steps"]),
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
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def run_training_sweep(experiments: List[Dict], gpus: List[int], sweep_root: Path):
    batch_size = len(gpus)
    total_batches = (len(experiments) + batch_size - 1) // batch_size

    skipped = []
    to_run = []
    for exp in experiments:
        adapter_path = sweep_root / exp["name"] / "final_adapter"
        if adapter_path.exists():
            skipped.append(exp["name"])
        else:
            to_run.append(exp)

    if skipped:
        print(f"Skipping {len(skipped)} already-trained experiments: {skipped}")

    if not to_run:
        print("All Phase 2 experiments already completed!")
        return

    total_batches = (len(to_run) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch = to_run[batch_start: batch_start + batch_size]

        print(f"\n{'=' * 70}")
        print(f"BATCH {batch_idx + 1}/{total_batches}  |  "
              f"Experiments: {[e['name'] for e in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, exp in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            print(f"  Launching {exp['name']} on GPU {gpu} ...")
            proc = launch_training(exp, gpu, sweep_root)
            procs.append((exp["name"], proc))
            time.sleep(15)

        for name, proc in procs:
            proc.wait()
            proc._log_fh.close()  # type: ignore[attr-defined]
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

    print(f"\nAll Phase 2 training completed.")


# ============================================================
# GSM8K Evaluation
# ============================================================

def find_latest_results(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def extract_exact_match(results_json: Path) -> Optional[float]:
    obj = json.loads(results_json.read_text())
    task_block = (obj.get("results") or {}).get(TASK, {})
    for key in ["exact_match,none", "exact_match,flexible-extract",
                "exact_match,strict-match", "exact_match"]:
        if key in task_block:
            try:
                return float(task_block[key])
            except Exception:
                pass
    for key, val in task_block.items():
        if "exact_match" in key:
            try:
                return float(val)
            except Exception:
                continue
    return None


def run_single_eval(model_args: str, output_path: Path, gpu_id: int,
                    batch_size: int = 8) -> subprocess.Popen:
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", TASK,
        "--batch_size", str(batch_size),
        "--gen_kwargs", "max_gen_toks=2048,temperature=0,do_sample=False",
        "--output_path", str(output_path),
        "--log_samples",
        "--apply_chat_template",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"

    log_file = output_path / "eval.log"
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(HARNESS_DIR), env=env)
    proc._log_fh = fh  # type: ignore[attr-defined]
    proc._output_path = output_path  # type: ignore[attr-defined]
    return proc


def discover_all_adapters(sweep_root: Path) -> List[Dict[str, Any]]:
    adapters = []
    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        adapter_path = d / "final_adapter"
        if not adapter_path.exists():
            continue

        mf = d / "sweep_metrics.json"
        training: Dict[str, Any] = {}
        if mf.exists():
            m = json.loads(mf.read_text())
            cfg = m.get("config", {})
            training = {
                "eval_loss": m.get("eval_metrics", {}).get("eval_loss"),
                "train_loss": m.get("train_metrics", {}).get("train_loss"),
                "lr": cfg.get("learning_rate"),
                "r": cfg.get("lora_r"),
                "alpha": cfg.get("lora_alpha"),
                "dropout": cfg.get("lora_dropout"),
                "epochs": cfg.get("num_train_epochs"),
                "wd": cfg.get("weight_decay"),
                "warmup": cfg.get("warmup_ratio"),
                "modules": cfg.get("target_modules"),
                "ga_steps": cfg.get("gradient_accumulation_steps"),
            }

        adapters.append({"name": d.name, "adapter_path": str(adapter_path), **training})

    ds2 = BEEGFS_ARTIFACTS / "lora_qwen25_3b_ds2_fix_step2" / "final_adapter"
    if ds2.exists():
        adapters.append({
            "name": "ds2_fix_step2",
            "adapter_path": str(ds2),
            "r": 16, "alpha": 32, "dropout": 0.05,
        })

    return adapters


def save_incremental(path: Path, results: Dict[str, Dict]):
    path.write_text(json.dumps(list(results.values()), indent=2, default=str))


def run_parallel_gsm8k_eval(adapters: List[Dict], gpus: List[int],
                             eval_root: Path, batch_size: int = 8) -> Dict[str, Dict]:
    eval_root.mkdir(parents=True, exist_ok=True)
    incremental_file = eval_root / "all_results.json"

    done: Dict[str, Dict] = {}
    if incremental_file.exists():
        for entry in json.loads(incremental_file.read_text()):
            if entry.get("gsm8k_em") is not None:
                done[entry["name"]] = entry

    # --- Base model ---
    if "base" not in done:
        print("\n[BASE] Evaluating base model (this takes ~30-60 min) ...")
        t0 = time.time()
        model_args = f"pretrained={MODEL_ID},dtype=bfloat16,device=cuda"
        proc = run_single_eval(model_args, eval_root / "base", gpus[0], batch_size)
        proc.wait()
        proc._log_fh.close()  # type: ignore[attr-defined]
        elapsed = time.time() - t0

        rj = find_latest_results(eval_root / "base")
        em = extract_exact_match(rj) if rj else None
        done["base"] = {"name": "base", "gsm8k_em": em, "elapsed_sec": round(elapsed, 1)}
        save_incremental(incremental_file, done)
        print(f"  Base exact_match = {em} ({elapsed/60:.1f} min)")
    else:
        print(f"[BASE] Already done: exact_match = {done['base'].get('gsm8k_em')}")

    # --- Adapters ---
    to_eval = [a for a in adapters if a["name"] not in done]
    print(f"\nAdapters to evaluate: {len(to_eval)} "
          f"(skipping {len(adapters) - len(to_eval)} already done)")

    if not to_eval:
        print("All evaluations already completed!")
        return done

    n_gpus = len(gpus)
    total_batches = (len(to_eval) + n_gpus - 1) // n_gpus

    for batch_idx in range(total_batches):
        batch_start = batch_idx * n_gpus
        batch = to_eval[batch_start: batch_start + n_gpus]

        print(f"\n{'=' * 70}")
        print(f"EVAL BATCH {batch_idx + 1}/{total_batches}  |  "
              f"Adapters: {[a['name'] for a in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, adapter in enumerate(batch):
            gpu = gpus[i % n_gpus]
            name = adapter["name"]
            model_args = (f"pretrained={MODEL_ID},"
                          f"peft={adapter['adapter_path']},"
                          f"dtype=bfloat16,device=cuda")
            print(f"  Launching {name} on GPU {gpu} ...")
            proc = run_single_eval(model_args, eval_root / name, gpu, batch_size)
            procs.append((adapter, proc))
            time.sleep(5)

        for adapter, proc in procs:
            proc.wait()
            proc._log_fh.close()  # type: ignore[attr-defined]
            name = adapter["name"]

            rj = find_latest_results(eval_root / name)
            em = extract_exact_match(rj) if rj else None
            status = f"exact_match = {em}" if em is not None else "FAILED"
            print(f"  {name}: {status}")

            done[name] = {
                "name": name,
                "gsm8k_em": em,
                "eval_loss": adapter.get("eval_loss"),
                "train_loss": adapter.get("train_loss"),
                "lr": adapter.get("lr"),
                "r": adapter.get("r"),
                "alpha": adapter.get("alpha"),
                "dropout": adapter.get("dropout"),
                "epochs": adapter.get("epochs"),
                "wd": adapter.get("wd"),
                "warmup": adapter.get("warmup"),
                "modules": adapter.get("modules"),
            }

        save_incremental(incremental_file, done)
        elapsed_batch = sum(
            done.get(a["name"], {}).get("elapsed_sec", 0) for a in batch
        )
        remaining = len(to_eval) - (batch_start + len(batch))
        print(f"  Remaining: {remaining} adapters")

    return done


# ============================================================
# Compile results (training metrics for ALL adapters)
# ============================================================

def compile_training_results(sweep_root: Path) -> List[Dict]:
    results = []
    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        mf = d / "sweep_metrics.json"
        if not mf.exists():
            continue

        m = json.loads(mf.read_text())
        eval_loss = m.get("eval_metrics", {}).get("eval_loss")
        train_loss = m.get("train_metrics", {}).get("train_loss")
        cfg = m.get("config", {})

        log_history = m.get("log_history", [])
        eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
        best_eval = min(eval_losses) if eval_losses else eval_loss

        results.append({
            "name": d.name,
            "eval_loss": eval_loss,
            "best_eval_loss": best_eval,
            "train_loss": train_loss,
            "lr": cfg.get("learning_rate"),
            "r": cfg.get("lora_r"),
            "alpha": cfg.get("lora_alpha"),
            "dropout": cfg.get("lora_dropout"),
            "epochs": cfg.get("num_train_epochs"),
            "wd": cfg.get("weight_decay"),
            "warmup": cfg.get("warmup_ratio"),
            "modules": cfg.get("target_modules"),
            "phase": "v2" if d.name.startswith("v2_") else "v1",
        })

    results.sort(key=lambda x: x.get("eval_loss") or float("inf"))
    return results


# ============================================================
# Summary & reporting
# ============================================================

def print_training_summary(results: List[Dict]):
    print(f"\n{'=' * 105}")
    print(f"{'ALL TRAINING RESULTS (Phase 1 + Phase 2)':^105}")
    print(f"{'=' * 105}")
    print(f"{'Rk':<4} {'Name':<28} {'Phase':<6} {'Eval Loss':<11} {'Best EL':<11} "
          f"{'Train Loss':<11} {'LR':<10} {'r':<4} {'α':<5} {'Ep':<5} "
          f"{'Drop':<6} {'WD':<6} {'WU':<6}")
    print("-" * 105)

    for i, r in enumerate(results):
        el = f"{r['eval_loss']:.4f}" if r.get('eval_loss') is not None else "N/A"
        bel = f"{r['best_eval_loss']:.4f}" if r.get('best_eval_loss') is not None else "N/A"
        tl = f"{r['train_loss']:.4f}" if r.get('train_loss') is not None else "N/A"
        lr_s = f"{r['lr']:.0e}" if r.get('lr') is not None else "?"
        wu = f"{r.get('warmup', '?')}"
        print(f"{i+1:<4} {r['name']:<28} {r['phase']:<6} {el:<11} {bel:<11} "
              f"{tl:<11} {lr_s:<10} {str(r.get('r','?')):<4} "
              f"{str(r.get('alpha','?')):<5} {str(r.get('epochs','?')):<5} "
              f"{str(r.get('dropout','?')):<6} {str(r.get('wd','?')):<6} {wu:<6}")

    v1 = [r for r in results if r["phase"] == "v1"]
    v2 = [r for r in results if r["phase"] == "v2"]
    print(f"\nPhase 1: {len(v1)} experiments | Phase 2: {len(v2)} experiments")

    if results:
        best = results[0]
        print(f"\nOverall best: {best['name']} (eval_loss={best['eval_loss']:.4f})")
    if v2:
        best_v2 = v2[0] if v2[0]["eval_loss"] is not None else None
        if best_v2:
            print(f"Best Phase 2: {best_v2['name']} (eval_loss={best_v2['eval_loss']:.4f})")


def print_gsm8k_summary(results: Dict[str, Dict]):
    base_em = results.get("base", {}).get("gsm8k_em")
    adapter_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )

    print(f"\n{'=' * 110}")
    print(f"{'COMPREHENSIVE GSM8K EVALUATION RESULTS':^110}")
    print(f"{'=' * 110}")
    print(f"\nBase model ({MODEL_ID}) exact_match: {base_em}")
    print(f"\n{'Rk':<4} {'Name':<28} {'GSM8K EM':<12} {'Delta':<10} "
          f"{'Eval Loss':<11} {'LR':<10} {'r':<4} {'α':<5} {'Ep':<5} "
          f"{'Drop':<6} {'WD':<6}")
    print("-" * 110)

    for i, (name, r) in enumerate(adapter_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        if em is not None and base_em is not None:
            delta_str = f"{em - base_em:+.4f}"
        else:
            delta_str = "N/A"
        el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "N/A"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "N/A"
        print(f"{i+1:<4} {name:<28} {em_str:<12} {delta_str:<10} "
              f"{el:<11} {lr_str:<10} {str(r.get('r', 'N/A')):<4} "
              f"{str(r.get('alpha', 'N/A')):<5} {str(r.get('epochs', 'N/A')):<5} "
              f"{str(r.get('dropout', 'N/A')):<6} {str(r.get('wd', 'N/A')):<6}")

    ok = [(n, r) for n, r in adapter_results if r.get("gsm8k_em") is not None]
    if ok and base_em is not None:
        improved = [x for x in ok if x[1]["gsm8k_em"] > base_em]
        print(f"\n{len(improved)}/{len(ok)} adapters improved over base model")
        if ok:
            best = ok[0]
            print(f"Best: {best[0]} (EM={best[1]['gsm8k_em']:.4f}, "
                  f"delta={best[1]['gsm8k_em'] - base_em:+.4f})")


def generate_final_markdown(results: Dict[str, Dict], output_path: Path):
    base_em = results.get("base", {}).get("gsm8k_em")
    adapter_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )

    lines = [
        "# Comprehensive LoRA Sweep: Phase 1 + Phase 2 + GSM8K Evaluation",
        "",
        f"**Base Model**: `{MODEL_ID}`",
        f"**Task**: `{TASK}` (GSM8K zero-shot CoT, 1319 test samples)",
        f"**Training Data**: 109 train / 6 eval samples",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Base Model GSM8K EM**: {f'{base_em:.4f}' if base_em else 'N/A'}",
        "",
        "## Results (Ranked by GSM8K Exact Match)",
        "",
        "| Rank | Name | GSM8K EM | Delta | Eval Loss | Train Loss | LR | r | Alpha"
        " | Epochs | Dropout | WD |",
        "|------|------|----------|-------|-----------|------------|-----|---|-------"
        "|--------|---------|-----|",
    ]

    for i, (name, r) in enumerate(adapter_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        delta = (em - base_em) if (em is not None and base_em is not None) else None
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "N/A"
        tl = f"{r.get('train_loss', 0):.4f}" if r.get('train_loss') is not None else "N/A"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "N/A"
        lines.append(
            f"| {i+1} | {name} | {em_str} | {delta_str} | {el} | {tl} | "
            f"{lr_str} | {r.get('r', 'N/A')} | {r.get('alpha', 'N/A')} | "
            f"{r.get('epochs', 'N/A')} | {r.get('dropout', 'N/A')} | "
            f"{r.get('wd', 'N/A')} |"
        )

    ok = [(n, r) for n, r in adapter_results if r.get("gsm8k_em") is not None]
    if ok and base_em is not None:
        best_name, best_r = ok[0]
        worst_name, worst_r = ok[-1]
        improved = [x for x in ok if x[1]["gsm8k_em"] > base_em]

        lines.extend([
            "",
            "## Key Findings",
            "",
            f"- **Best adapter**: `{best_name}` "
            f"(EM={best_r['gsm8k_em']:.4f}, "
            f"delta={best_r['gsm8k_em'] - base_em:+.4f})",
            f"- **Worst adapter**: `{worst_name}` "
            f"(EM={worst_r['gsm8k_em']:.4f})",
            f"- **{len(improved)}/{len(ok)}** adapters improved over base",
        ])

        p1 = [(n, r) for n, r in ok if not n.startswith("v2_")]
        p2 = [(n, r) for n, r in ok if n.startswith("v2_")]

        if p1:
            best_p1 = max(p1, key=lambda x: x[1]["gsm8k_em"])
            lines.append(f"- Best Phase 1: `{best_p1[0]}` "
                         f"(EM={best_p1[1]['gsm8k_em']:.4f})")
        if p2:
            best_p2 = max(p2, key=lambda x: x[1]["gsm8k_em"])
            lines.append(f"- Best Phase 2: `{best_p2[0]}` "
                         f"(EM={best_p2[1]['gsm8k_em']:.4f})")

        lines.extend(["", "## Eval Loss vs GSM8K EM Correlation", ""])
        lines.append("| Name | Eval Loss | GSM8K EM | Consistent? |")
        lines.append("|------|-----------|----------|-------------|")
        for n, r in ok[:10]:
            el = r.get("eval_loss")
            em_val = r["gsm8k_em"]
            el_str = f"{el:.4f}" if el is not None else "N/A"
            lines.append(f"| {n} | {el_str} | {em_val:.4f} | |")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFinal report saved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--phase", default="all",
                    choices=["all", "train-only", "eval-only", "summary-only"])
    ap.add_argument("--eval-batch-size", type=int, default=8)
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    sweep_root = BEEGFS_ARTIFACTS / "lora_sweep"
    eval_root = sweep_root / "_gsm8k_comprehensive_eval"

    t_global = time.time()

    # ---- Phase 2 Training ----
    if args.phase in ("all", "train-only"):
        experiments = build_phase2_experiments()

        print(f"\n{'#' * 70}")
        print(f"  PHASE 2 TRAINING: {len(experiments)} experiments, {len(gpus)} GPUs")
        print(f"{'#' * 70}")

        plan_file = sweep_root / "v2_experiment_plan.json"
        plan_file.write_text(json.dumps(experiments, indent=2))

        t0 = time.time()
        run_training_sweep(experiments, gpus, sweep_root)
        elapsed = time.time() - t0
        print(f"\nPhase 2 training completed in {elapsed/60:.1f} minutes")

    # ---- Compile training results ----
    if args.phase in ("all", "train-only", "summary-only"):
        training_results = compile_training_results(sweep_root)
        print_training_summary(training_results)

        summary_file = sweep_root / "sweep_v2_training_summary.json"
        summary_file.write_text(json.dumps(training_results, indent=2, default=str))
        print(f"Training summary saved to: {summary_file}")

    # ---- GSM8K Evaluation ----
    if args.phase in ("all", "eval-only"):
        adapters = discover_all_adapters(sweep_root)

        print(f"\n{'#' * 70}")
        print(f"  GSM8K EVALUATION: {len(adapters)} adapters + base, "
              f"{len(gpus)} GPUs")
        print(f"{'#' * 70}")

        t0 = time.time()
        results = run_parallel_gsm8k_eval(
            adapters, gpus, eval_root, args.eval_batch_size)
        elapsed = time.time() - t0
        print(f"\nGSM8K evaluation completed in {elapsed/60:.1f} minutes")

        print_gsm8k_summary(results)
        generate_final_markdown(
            results,
            PROJECT_ROOT / "documents" / "gsm8k_comprehensive_results.md",
        )

    # ---- Summary only ----
    if args.phase == "summary-only":
        results_file = eval_root / "all_results.json"
        if results_file.exists():
            results = {r["name"]: r
                       for r in json.loads(results_file.read_text())}
            print_gsm8k_summary(results)
            generate_final_markdown(
                results,
                PROJECT_ROOT / "documents" / "gsm8k_comprehensive_results.md",
            )
        else:
            print("No GSM8K results found yet.")

    total_elapsed = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"ALL DONE! Total time: {total_elapsed/60:.1f} minutes "
          f"({total_elapsed/3600:.1f} hours)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
