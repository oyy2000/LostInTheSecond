#!/usr/bin/env python3
"""
Full-parameter fine-tuning sweep + GSM8K evaluation.

Experiment design rationale (top-AI-PhD level):
  With only 109 training samples on a 3B model, overfitting is the dominant
  failure mode. LoRA sweep showed lower capacity → better generalization
  (best: attn_only r=16, and r=4 alpha=4). Full FT goes the opposite
  direction (3B trainable params), so we MUST use:
    - Ultra-low learning rates (1e-6 to 5e-5, vs 1e-4 for LoRA)
    - Strong weight decay (0.01–0.1)
    - Few epochs (1–5)
    - Large warmup ratio for stability

  16 experiments across 4 axes: LR, epochs, weight decay, combined.

Usage:
    python 14_full_ft_sweep_and_eval.py --gpus 0,1,2,3
    python 14_full_ft_sweep_and_eval.py --phase eval-only --gpus 0,1,2,3
    python 14_full_ft_sweep_and_eval.py --phase summary-only
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
TRAIN_SCRIPT = SCRIPT_DIR / "13_full_finetune.py"
HARNESS_DIR = PROJECT_ROOT / "lm-evaluation-harness"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
DATASET_PREFILL = BEEGFS_ARTIFACTS / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json"
SWEEP_ROOT = BEEGFS_ARTIFACTS / "full_ft_sweep"
EVAL_ROOT = SWEEP_ROOT / "_gsm8k_vllm_eval"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PYTHON_BIN = "/mnt/beegfs/youyang7/.conda/envs/fact/bin/python"
TASK = "gsm8k_cot_zeroshot_unified"

VLLM_COMMON_ARGS = (
    "dtype=bfloat16,"
    "gpu_memory_utilization=0.6,"
    "max_model_len=3072,"
    "max_num_seqs=32"
)


# ============================================================
# Experiment definitions
# ============================================================

def build_experiments() -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = []

    def add(name: str, lr: float = 1e-5, epochs: float = 3, wd: float = 0.01,
            warmup: float = 0.1, ga_steps: int = 16):
        exps.append(dict(
            name=name, lr=lr, epochs=epochs, wd=wd,
            warmup=warmup, ga_steps=ga_steps,
        ))

    # --- Group A: Learning rate sweep (most critical axis) ---
    # Full FT requires 10-100x lower LR than LoRA
    add("ft_lr1e6",  lr=1e-6)
    add("ft_lr5e6",  lr=5e-6)
    add("ft_lr1e5",  lr=1e-5)
    add("ft_lr2e5",  lr=2e-5)
    add("ft_lr5e5",  lr=5e-5)

    # --- Group B: Epoch sweep (overfitting trajectory) ---
    add("ft_ep1", epochs=1)
    add("ft_ep2", epochs=2)
    # ep=3 already covered by ft_lr1e5
    add("ft_ep5", epochs=5)

    # --- Group C: Weight decay sweep (regularization) ---
    add("ft_wd0",    wd=0.0)
    add("ft_wd005",  wd=0.05)
    # wd=0.01 already covered by ft_lr1e5
    add("ft_wd01",   wd=0.1)

    # --- Group D: Combined best + variations ---
    add("ft_combo_a", lr=2e-5,  epochs=2, wd=0.01)   # aggressive short
    add("ft_combo_b", lr=5e-6,  epochs=5, wd=0.05)   # conservative long
    add("ft_combo_c", lr=1e-5,  epochs=3, warmup=0.0) # no warmup
    add("ft_combo_d", lr=1e-5,  epochs=3, warmup=0.2) # high warmup
    add("ft_combo_e", lr=1e-5,  epochs=2, wd=0.05)    # balanced

    return exps


# ============================================================
# Training
# ============================================================

def launch_training(exp: Dict, gpu_id: int) -> subprocess.Popen:
    name = exp["name"]
    output_dir = SWEEP_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = " ".join([
        f"--model-id {MODEL_ID}",
        f"--dataset-path {DATASET_PREFILL}",
        f"--output-dir {output_dir}",
        f"--learning-rate {exp['lr']}",
        f"--num-train-epochs {exp['epochs']}",
        f"--weight-decay {exp['wd']}",
        f"--warmup-ratio {exp['warmup']}",
        f"--gradient-accumulation-steps {exp['ga_steps']}",
        "--per-device-train-batch-size 1",
        "--logging-steps 1",
        "--eval-steps 7",
        "--save-steps 7",
        "--save-total-limit 2",
        "--seed 42",
    ])

    bash_cmd = (
        f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
        f"export PYTHONUNBUFFERED=1; "
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
        f"{PYTHON_BIN} {TRAIN_SCRIPT} {train_args}"
    )

    log_file = output_dir / "training.log"
    fh = open(log_file, "w")
    proc = subprocess.Popen(
        ["/bin/bash", "-c", bash_cmd],
        stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def run_training_sweep(experiments: List[Dict], gpus: List[int]):
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    skipped, to_run = [], []
    for exp in experiments:
        model_path = SWEEP_ROOT / exp["name"] / "best_model"
        if model_path.exists() and (model_path / "config.json").exists():
            skipped.append(exp["name"])
        else:
            to_run.append(exp)

    if skipped:
        print(f"Skipping {len(skipped)} already-trained: {skipped}")
    if not to_run:
        print("All experiments already completed!")
        return

    n_gpus = len(gpus)
    total_batches = (len(to_run) + n_gpus - 1) // n_gpus

    for batch_idx in range(total_batches):
        batch_start = batch_idx * n_gpus
        batch = to_run[batch_start: batch_start + n_gpus]

        print(f"\n{'=' * 70}")
        print(f"TRAIN BATCH {batch_idx + 1}/{total_batches}  |  "
              f"{[e['name'] for e in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, exp in enumerate(batch):
            gpu = gpus[i % n_gpus]
            print(f"  Launching {exp['name']} on GPU {gpu} ...")
            proc = launch_training(exp, gpu)
            procs.append((exp["name"], proc))
            time.sleep(10)

        for name, proc in procs:
            proc.wait()
            proc._log_fh.close()  # type: ignore[attr-defined]
            if proc.returncode != 0:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except (ProcessLookupError, PermissionError):
                    pass
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

    print(f"\nAll training completed.")


# ============================================================
# GSM8K Evaluation (vLLM)
# ============================================================

def find_latest_results(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def extract_exact_match(results_json: Path) -> Optional[float]:
    obj = json.loads(results_json.read_text())
    task_block = (obj.get("results") or {}).get(TASK, {})
    for key in ["exact_match,flexible-extract", "exact_match,none",
                "exact_match,strict-match", "exact_match"]:
        if key in task_block:
            try:
                return float(task_block[key])
            except Exception:
                pass
    for key, val in task_block.items():
        if "exact_match" in key and "stderr" not in key:
            try:
                return float(val)
            except Exception:
                continue
    return None


def run_vllm_eval(model_args: str, output_path: Path, gpu_id: int,
                  batch_size: str = "auto", limit: int = 0) -> subprocess.Popen:
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", TASK,
        "--batch_size", batch_size,
        "--gen_kwargs", "max_gen_toks=2048,temperature=0,do_sample=False",
        "--output_path", str(output_path),
        "--log_samples",
        "--apply_chat_template",
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"

    log_file = output_path / "eval.log"
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(HARNESS_DIR), env=env)
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def load_training_metrics(exp_dir: Path) -> Dict[str, Any]:
    mf = exp_dir / "sweep_metrics.json"
    if not mf.exists():
        return {}
    m = json.loads(mf.read_text())
    cfg = m.get("config", {})
    log_history = m.get("log_history", [])
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    best_eval = min(eval_losses) if eval_losses else m.get("eval_metrics", {}).get("eval_loss")

    return {
        "eval_loss": m.get("eval_metrics", {}).get("eval_loss"),
        "best_eval_loss": best_eval,
        "train_loss": m.get("train_metrics", {}).get("train_loss"),
        "lr": cfg.get("learning_rate"),
        "epochs": cfg.get("num_train_epochs"),
        "wd": cfg.get("weight_decay"),
        "warmup": cfg.get("warmup_ratio"),
        "optim": cfg.get("optim"),
        "trainable_params": cfg.get("trainable_params"),
        "train_samples": cfg.get("train_samples"),
    }


def discover_all_models() -> List[Dict[str, Any]]:
    models = []
    for d in sorted(SWEEP_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        model_path = d / "best_model"
        if not model_path.exists() or not (model_path / "config.json").exists():
            continue
        training = load_training_metrics(d)
        models.append({
            "name": d.name,
            "model_path": str(model_path),
            **training,
        })
    return models


def save_incremental(path: Path, results: Dict[str, Dict]):
    path.write_text(json.dumps(list(results.values()), indent=2, default=str))


def run_parallel_eval(models: List[Dict], gpus: List[int],
                      batch_size: str, limit: int) -> Dict[str, Dict]:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    incremental_file = EVAL_ROOT / "all_results.json"

    done: Dict[str, Dict] = {}
    if incremental_file.exists():
        for entry in json.loads(incremental_file.read_text()):
            if entry.get("gsm8k_em") is not None:
                done[entry["name"]] = entry

    # --- Base model ---
    if "base" not in done:
        print(f"\n[BASE] Evaluating base model on GPU {gpus[0]} ...")
        t0 = time.time()
        model_args = f"pretrained={MODEL_ID},{VLLM_COMMON_ARGS}"
        proc = run_vllm_eval(model_args, EVAL_ROOT / "base", gpus[0], batch_size, limit)
        proc.wait()
        proc._log_fh.close()  # type: ignore[attr-defined]
        elapsed = time.time() - t0

        rj = find_latest_results(EVAL_ROOT / "base")
        em = extract_exact_match(rj) if rj else None
        done["base"] = {"name": "base", "gsm8k_em": em, "elapsed_sec": round(elapsed, 1)}
        save_incremental(incremental_file, done)
        print(f"  Base exact_match = {em}  ({elapsed:.0f}s)")
    else:
        print(f"[BASE] Already done: exact_match = {done['base'].get('gsm8k_em')}")

    # --- Full FT models in parallel ---
    to_eval = [m for m in models if m["name"] not in done]
    print(f"\nModels to evaluate: {len(to_eval)} "
          f"(skipping {len(models) - len(to_eval)} already done)")

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
              f"{[m['name'] for m in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, model_info in enumerate(batch):
            gpu = gpus[i % n_gpus]
            name = model_info["name"]
            model_args = f"pretrained={model_info['model_path']},{VLLM_COMMON_ARGS}"
            print(f"  Launching {name} on GPU {gpu} ...")
            proc = run_vllm_eval(
                model_args, EVAL_ROOT / name, gpu, batch_size, limit)
            procs.append((model_info, proc, time.time()))
            time.sleep(3)

        for model_info, proc, t0 in procs:
            proc.wait()
            proc._log_fh.close()  # type: ignore[attr-defined]
            elapsed = time.time() - t0
            name = model_info["name"]

            rj = find_latest_results(EVAL_ROOT / name)
            em = extract_exact_match(rj) if rj else None
            status = f"EM = {em:.4f}" if em is not None else "FAILED"
            print(f"  {name}: {status}  ({elapsed:.0f}s)")

            done[name] = {
                "name": name,
                "gsm8k_em": em,
                "elapsed_sec": round(elapsed, 1),
                "method": "full_ft",
                "eval_loss": model_info.get("eval_loss"),
                "best_eval_loss": model_info.get("best_eval_loss"),
                "train_loss": model_info.get("train_loss"),
                "lr": model_info.get("lr"),
                "epochs": model_info.get("epochs"),
                "wd": model_info.get("wd"),
                "warmup": model_info.get("warmup"),
                "train_samples": model_info.get("train_samples"),
            }

        save_incremental(incremental_file, done)
        remaining = len(to_eval) - (batch_start + len(batch))
        print(f"  Saved. Remaining: {remaining} models")

    return done


# ============================================================
# Report generation
# ============================================================

def generate_report(results: Dict[str, Dict], output_path: Path):
    base_em = results.get("base", {}).get("gsm8k_em")

    # Also load LoRA best for comparison
    lora_eval_file = BEEGFS_ARTIFACTS / "lora_sweep" / "_gsm8k_vllm_eval" / "all_results.json"
    lora_best_em = None
    lora_best_name = None
    if lora_eval_file.exists():
        lora_results = json.loads(lora_eval_file.read_text())
        lora_adapters = [r for r in lora_results if r.get("name") != "base" and r.get("gsm8k_em")]
        if lora_adapters:
            best_lora = max(lora_adapters, key=lambda x: x["gsm8k_em"])
            lora_best_em = best_lora["gsm8k_em"]
            lora_best_name = best_lora["name"]

    ft_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )
    ok = [(n, r) for n, r in ft_results if r.get("gsm8k_em") is not None]

    lines = [
        "# Full Fine-Tuning Sweep — GSM8K Evaluation Results",
        "",
        f"**Base Model**: `{MODEL_ID}`",
        f"**Method**: Full parameter fine-tuning (no LoRA)",
        f"**Optimizer**: paged_adamw_8bit (bitsandbytes)",
        f"**Task**: `{TASK}` (GSM8K zero-shot CoT, 1319 test samples)",
        f"**Training Data**: 109 train / 6 eval samples",
        f"**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## Base Model: GSM8K EM = **{base_em:.4f}**" if base_em else "## Base Model: N/A",
    ]

    if lora_best_em and lora_best_name:
        lines.append(f"## Best LoRA (comparison): `{lora_best_name}` EM = **{lora_best_em:.4f}**")
    lines.append("")

    # Main results table
    lines.extend([
        "## Full Results (Ranked by GSM8K Exact Match)",
        "",
        "| Rank | Name | GSM8K EM | Delta | Eval Loss | Best EL | Train Loss "
        "| LR | Epochs | WD | Warmup |",
        "|------|------|----------|-------|-----------|---------|------------"
        "|-----|--------|-----|--------|",
    ])

    for i, (name, r) in enumerate(ft_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        delta = (em - base_em) if (em is not None and base_em is not None) else None
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        el = f"{r.get('eval_loss'):.4f}" if r.get('eval_loss') is not None else "—"
        bel = f"{r.get('best_eval_loss'):.4f}" if r.get('best_eval_loss') is not None else "—"
        tl = f"{r.get('train_loss'):.4f}" if r.get('train_loss') is not None else "—"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "—"
        lines.append(
            f"| {i+1} | {name} | {em_str} | {delta_str} | {el} | {bel} | {tl} "
            f"| {lr_str} | {r.get('epochs', '—')} | {r.get('wd', '—')} "
            f"| {r.get('warmup', '—')} |"
        )

    # Key findings
    if ok and base_em is not None:
        best_name, best_r = ok[0]
        worst_name, worst_r = ok[-1]
        improved = [x for x in ok if x[1]["gsm8k_em"] > base_em]

        lines.extend(["", "## Key Findings", ""])
        lines.append(f"- **Best full FT**: `{best_name}` "
                     f"(EM = {best_r['gsm8k_em']:.4f}, "
                     f"delta = {best_r['gsm8k_em'] - base_em:+.4f})")
        lines.append(f"- **Worst full FT**: `{worst_name}` "
                     f"(EM = {worst_r['gsm8k_em']:.4f}, "
                     f"delta = {worst_r['gsm8k_em'] - base_em:+.4f})")
        lines.append(f"- **{len(improved)}/{len(ok)}** experiments improved over base")

        if lora_best_em:
            if best_r['gsm8k_em'] > lora_best_em:
                lines.append(f"- Full FT **beats** best LoRA by "
                             f"{best_r['gsm8k_em'] - lora_best_em:+.4f}")
            else:
                lines.append(f"- Full FT **loses** to best LoRA by "
                             f"{best_r['gsm8k_em'] - lora_best_em:+.4f}")

    # HP analysis
    lines.extend(["", "## Hyperparameter Analysis", ""])

    # LR
    lr_groups: Dict[str, List[float]] = {}
    for n, r in ok:
        lr_val = r.get("lr")
        if lr_val is not None:
            key = f"{lr_val:.0e}"
            lr_groups.setdefault(key, []).append(r["gsm8k_em"])
    if lr_groups:
        lines.extend(["### Learning Rate", "",
                       "| LR | Count | Mean EM | Best EM | Worst EM |",
                       "|----|-------|---------|---------|----------|"])
        for lr_key in sorted(lr_groups.keys(), key=lambda x: float(x)):
            vals = lr_groups[lr_key]
            lines.append(f"| {lr_key} | {len(vals)} | {sum(vals)/len(vals):.4f} "
                         f"| {max(vals):.4f} | {min(vals):.4f} |")

    # Epochs
    ep_groups: Dict[float, List[float]] = {}
    for n, r in ok:
        ep = r.get("epochs")
        if ep is not None:
            ep_groups.setdefault(float(ep), []).append(r["gsm8k_em"])
    if ep_groups:
        lines.extend(["", "### Epochs", "",
                       "| Epochs | Count | Mean EM | Best EM |",
                       "|--------|-------|---------|---------|"])
        for ep in sorted(ep_groups.keys()):
            vals = ep_groups[ep]
            lines.append(f"| {ep:.0f} | {len(vals)} | {sum(vals)/len(vals):.4f} "
                         f"| {max(vals):.4f} |")

    # WD
    wd_groups: Dict[float, List[float]] = {}
    for n, r in ok:
        wd = r.get("wd")
        if wd is not None:
            wd_groups.setdefault(float(wd), []).append(r["gsm8k_em"])
    if wd_groups:
        lines.extend(["", "### Weight Decay", "",
                       "| WD | Count | Mean EM | Best EM |",
                       "|----|-------|---------|---------|"])
        for wd in sorted(wd_groups.keys()):
            vals = wd_groups[wd]
            lines.append(f"| {wd} | {len(vals)} | {sum(vals)/len(vals):.4f} "
                         f"| {max(vals):.4f} |")

    # Overfitting analysis
    lines.extend(["", "## Overfitting Analysis", "",
                   "| Name | Train Loss | Eval Loss | Gap | GSM8K EM |",
                   "|------|-----------|-----------|-----|----------|"])
    for n, r in ft_results:
        tl = r.get("train_loss")
        el = r.get("eval_loss")
        em = r.get("gsm8k_em")
        if tl is not None and el is not None and em is not None:
            gap = el - tl
            lines.append(f"| {n} | {tl:.4f} | {el:.4f} | {gap:+.4f} | {em:.4f} |")

    lines.extend(["", "---", "",
                   f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
                   f"*Full JSON: `{EVAL_ROOT / 'all_results.json'}`*", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


def print_summary(results: Dict[str, Dict]):
    base_em = results.get("base", {}).get("gsm8k_em")
    ft_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )

    print(f"\n{'=' * 100}")
    print(f"{'FULL FT — GSM8K EVALUATION RESULTS':^100}")
    print(f"{'=' * 100}")
    print(f"\nBase model: {MODEL_ID}  |  GSM8K EM = {base_em}")
    print(f"\n{'Rk':<4} {'Name':<20} {'GSM8K EM':<10} {'Delta':<8} "
          f"{'EvalLoss':<10} {'TrainLoss':<10} {'LR':<8} {'Ep':<5} "
          f"{'WD':<6} {'WU':<6}")
    print("-" * 100)

    for i, (name, r) in enumerate(ft_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        delta_str = f"{em - base_em:+.4f}" if (em and base_em) else "N/A"
        el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "—"
        tl = f"{r.get('train_loss', 0):.4f}" if r.get('train_loss') is not None else "—"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "—"
        print(f"{i+1:<4} {name:<20} {em_str:<10} {delta_str:<8} "
              f"{el:<10} {tl:<10} {lr_str:<8} "
              f"{str(r.get('epochs', '—')):<5} {str(r.get('wd', '—')):<6} "
              f"{str(r.get('warmup', '—')):<6}")

    ok = [(n, r) for n, r in ft_results if r.get("gsm8k_em") is not None]
    if ok and base_em:
        improved = sum(1 for _, r in ok if r["gsm8k_em"] > base_em)
        print(f"\n{improved}/{len(ok)} experiments beat base model")
        best = ok[0]
        print(f"Best: {best[0]} (EM={best[1]['gsm8k_em']:.4f}, "
              f"delta={best[1]['gsm8k_em'] - base_em:+.4f})")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--phase", default="all",
                    choices=["all", "train-only", "eval-only", "summary-only"])
    ap.add_argument("--eval-batch-size", default="auto")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="Limit GSM8K samples (0=full 1319)")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    t_global = time.time()

    # ---- Training sweep ----
    if args.phase in ("all", "train-only"):
        experiments = build_experiments()

        print(f"\n{'#' * 70}")
        print(f"  FULL FT TRAINING SWEEP: {len(experiments)} experiments, "
              f"{len(gpus)} GPUs")
        print(f"{'#' * 70}")

        plan_file = SWEEP_ROOT / "experiment_plan.json"
        SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps(experiments, indent=2))
        print(f"Plan: {plan_file}")

        t0 = time.time()
        run_training_sweep(experiments, gpus)
        elapsed = time.time() - t0
        print(f"\nTraining completed in {elapsed/60:.1f} min")

    # ---- GSM8K Evaluation ----
    if args.phase in ("all", "eval-only"):
        models = discover_all_models()

        print(f"\n{'#' * 70}")
        print(f"  GSM8K EVALUATION (vLLM)  |  {len(models)} models + base  |  "
              f"{len(gpus)} GPUs")
        print(f"  Limit: {'full (1319)' if args.eval_limit == 0 else args.eval_limit}")
        print(f"{'#' * 70}")

        t0 = time.time()
        results = run_parallel_eval(
            models, gpus, args.eval_batch_size, args.eval_limit)
        elapsed = time.time() - t0
        print(f"\nEvaluation completed in {elapsed/60:.1f} min")

        print_summary(results)
        generate_report(results, DOCUMENTS_DIR / "full_ft_gsm8k_results.md")

    # ---- Summary only ----
    if args.phase == "summary-only":
        results_file = EVAL_ROOT / "all_results.json"
        if not results_file.exists():
            print("No results found. Run evaluation first.")
            return
        results = {r["name"]: r for r in json.loads(results_file.read_text())}
        print_summary(results)
        generate_report(results, DOCUMENTS_DIR / "full_ft_gsm8k_results.md")

    total_elapsed = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"ALL DONE! Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.1f} hr)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
