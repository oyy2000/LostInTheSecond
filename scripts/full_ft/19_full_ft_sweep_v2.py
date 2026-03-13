#!/usr/bin/env python3
"""
Full FT Phase 2 sweep + GSM8K evaluation (vLLM).

Phase 1 found LR=1e-6 with epoch=3 as the most promising regime.
Phase 2 explores weight decay and warmup at this fixed LR/epoch:
  - Group A: WD sweep   (warmup=0.1 fixed)
  - Group B: Warmup sweep (wd=0.01 fixed)
  - Group C: Cross of promising WD x warmup

Usage:
    python 19_full_ft_sweep_v2.py --gpus 0,1,2,3
    python 19_full_ft_sweep_v2.py --phase eval-only --gpus 0,1,2,3
    python 19_full_ft_sweep_v2.py --phase train-only --gpus 0,1,2,3
    python 19_full_ft_sweep_v2.py --phase summary-only
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
# Phase 2 experiment definitions
# ============================================================

def build_phase2_experiments() -> List[Dict[str, Any]]:
    """
    Phase 1 found:
      - LR=1e-6, ep=3, wd=0.01, warmup=0.1 → EM=0.8446 (#2 overall)
      - LR=5e-6, ep=3, wd=0.01, warmup=0.1 → EM=0.8461 (#1 overall)
      - WD and warmup were underexplored at the ultra-low LR regime.

    Phase 2 fixes lr=1e-6, epochs=3, and sweeps WD and warmup.
    ft_lr1e6 (wd=0.01, warmup=0.1) already exists — skipped below.
    """
    exps: List[Dict[str, Any]] = []

    def add(name: str, desc: str = "", lr: float = 1e-6, epochs: float = 3,
            wd: float = 0.01, warmup: float = 0.1, ga_steps: int = 16):
        exps.append(dict(
            name=name, desc=desc, lr=lr, epochs=epochs, wd=wd,
            warmup=warmup, ga_steps=ga_steps,
        ))

    # --- Group A: WD sweep (warmup=0.1 fixed, lr=1e-6, ep=3) ---
    add("ft2_wd0",    desc="WD=0.0 at lr=1e-6",   wd=0.0)
    add("ft2_wd0005", desc="WD=0.005 at lr=1e-6",  wd=0.005)
    add("ft2_wd005",  desc="WD=0.05 at lr=1e-6",   wd=0.05)
    add("ft2_wd01",   desc="WD=0.1 at lr=1e-6",    wd=0.1)

    # --- Group B: Warmup sweep (wd=0.01 fixed, lr=1e-6, ep=3) ---
    add("ft2_wu0",   desc="Warmup=0.0 at lr=1e-6",  warmup=0.0)
    add("ft2_wu005", desc="Warmup=0.05 at lr=1e-6",  warmup=0.05)
    add("ft2_wu02",  desc="Warmup=0.2 at lr=1e-6",   warmup=0.2)
    add("ft2_wu03",  desc="Warmup=0.3 at lr=1e-6",   warmup=0.3)

    # --- Group C: Cross of promising WD x warmup ---
    add("ft2_wd0_wu02",  desc="WD=0 + warmup=0.2",
        wd=0.0, warmup=0.2)
    add("ft2_wd01_wu02", desc="WD=0.1 + warmup=0.2",
        wd=0.1, warmup=0.2)

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
        print("All Phase 2 experiments already completed!")
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

    print(f"\nAll Phase 2 full FT training completed.")


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
        phase = "v2" if d.name.startswith("ft2_") else "v1"
        models.append({
            "name": d.name,
            "model_path": str(model_path),
            "phase": phase,
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

    if "base" not in done:
        print(f"\n[BASE] Evaluating base model on GPU {gpus[0]} ...")
        t0 = time.time()
        model_args = f"pretrained={MODEL_ID},{VLLM_COMMON_ARGS}"
        proc = run_vllm_eval(model_args, EVAL_ROOT / "base",
                             gpus[0], batch_size, limit)
        proc.wait()
        proc._log_fh.close()  # type: ignore[attr-defined]
        elapsed = time.time() - t0

        rj = find_latest_results(EVAL_ROOT / "base")
        em = extract_exact_match(rj) if rj else None
        done["base"] = {"name": "base", "gsm8k_em": em,
                        "elapsed_sec": round(elapsed, 1)}
        save_incremental(incremental_file, done)
        print(f"  Base exact_match = {em}  ({elapsed:.0f}s)")
    else:
        print(f"[BASE] Already done: exact_match = {done['base'].get('gsm8k_em')}")

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
                "name": name, "gsm8k_em": em,
                "elapsed_sec": round(elapsed, 1),
                "phase": model_info.get("phase"),
                "method": "full_ft",
                "eval_loss": model_info.get("eval_loss"),
                "best_eval_loss": model_info.get("best_eval_loss"),
                "train_loss": model_info.get("train_loss"),
                "lr": model_info.get("lr"),
                "epochs": model_info.get("epochs"),
                "wd": model_info.get("wd"),
                "warmup": model_info.get("warmup"),
            }

        save_incremental(incremental_file, done)
        remaining = len(to_eval) - (batch_start + len(batch))
        print(f"  Saved. Remaining: {remaining} models")

    return done


# ============================================================
# Report
# ============================================================

def generate_report(results: Dict[str, Dict], output_path: Path,
                    experiments: Optional[List[Dict]] = None):
    base_em = results.get("base", {}).get("gsm8k_em")
    ft_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )
    ok = [(n, r) for n, r in ft_results if r.get("gsm8k_em") is not None]

    lines = [
        "# Full Fine-Tuning Sweep — Phase 1+2 GSM8K Results",
        "",
        f"**Base Model**: `{MODEL_ID}`",
        "**Method**: Full parameter fine-tuning (no LoRA)",
        "**Optimizer**: paged_adamw_8bit (bitsandbytes)",
        f"**Task**: `{TASK}` (GSM8K zero-shot CoT, 1319 test samples)",
        "**Training Data**: ~109 train / ~6 eval samples",
        "**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## Base Model: GSM8K EM = **{base_em:.4f}**" if base_em
        else "## Base Model: N/A",
        "",
        "## Full Results (Ranked by GSM8K Exact Match)", "",
        "| Rank | Name | Phase | GSM8K EM | Delta | Eval Loss | Best EL "
        "| Train Loss | LR | Epochs | WD | Warmup |",
        "|------|------|-------|----------|-------|-----------|--------"
        "|------------|-----|--------|-----|--------|",
    ]

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
        phase = r.get("phase", "—")
        lines.append(
            f"| {i+1} | {name} | {phase} | {em_str} | {delta_str} | "
            f"{el} | {bel} | {tl} | {lr_str} | "
            f"{r.get('epochs', '—')} | {r.get('wd', '—')} | "
            f"{r.get('warmup', '—')} |"
        )

    if ok and base_em is not None:
        best_name, best_r = ok[0]
        worst_name, worst_r = ok[-1]
        improved = [x for x in ok if x[1]["gsm8k_em"] > base_em]
        lines.extend(["", "## Key Findings", "",
            f"- **Best full FT**: `{best_name}` "
            f"(EM = {best_r['gsm8k_em']:.4f}, "
            f"delta = {best_r['gsm8k_em'] - base_em:+.4f})",
            f"- **Worst full FT**: `{worst_name}` "
            f"(EM = {worst_r['gsm8k_em']:.4f}, "
            f"delta = {worst_r['gsm8k_em'] - base_em:+.4f})",
            f"- **{len(improved)}/{len(ok)}** experiments improved over base",
        ])

    lines.extend(["", "---", "",
                   f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", ""])

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
    print(f"{'FULL FT Phase 1+2 — GSM8K RESULTS (vLLM)':^100}")
    print(f"{'=' * 100}")
    print(f"\nBase model: {MODEL_ID}  |  GSM8K EM = {base_em}")
    print(f"\n{'Rk':<4} {'Name':<20} {'Ph':<4} {'GSM8K EM':<10} {'Delta':<8} "
          f"{'EvalLoss':<10} {'LR':<8} {'Ep':<5} {'WD':<6} {'WU':<6}")
    print("-" * 100)

    for i, (name, r) in enumerate(ft_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        delta_str = f"{em - base_em:+.4f}" if (em and base_em) else "N/A"
        el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "—"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "—"
        ph = r.get("phase", "—")
        print(f"{i+1:<4} {name:<20} {ph:<4} {em_str:<10} {delta_str:<8} "
              f"{el:<10} {lr_str:<8} {str(r.get('epochs', '—')):<5} "
              f"{str(r.get('wd', '—')):<6} {str(r.get('warmup', '—')):<6}")

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

    # ---- Phase 2 Training ----
    if args.phase in ("all", "train-only"):
        experiments = build_phase2_experiments()

        print(f"\n{'#' * 70}")
        print(f"  FULL FT PHASE 2 TRAINING: {len(experiments)} experiments, "
              f"{len(gpus)} GPUs")
        print(f"{'#' * 70}")

        plan_file = SWEEP_ROOT / "v2_experiment_plan.json"
        SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps(experiments, indent=2))
        print(f"Plan: {plan_file}")

        t0 = time.time()
        run_training_sweep(experiments, gpus)
        elapsed = time.time() - t0
        print(f"\nPhase 2 training completed in {elapsed/60:.1f} min")

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
