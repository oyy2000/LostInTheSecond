#!/usr/bin/env python3
"""
Full FT sweep + GSM8K/MATH-500 evaluation for BASE (non-instruct) models.

Same 10-experiment grid as 19/19b but applied to base models:
  - Qwen/Qwen2.5-3B
  - meta-llama/Meta-Llama-3-8B

Key differences from 19/19b:
  - No --apply_chat_template in eval (base models lack chat templates)
  - Evaluates on both gsm8k_cot_zeroshot_unified AND hendrycks_math_500
  - Output goes to Ocean filesystem
  - Uses adafactor optimizer (no bitsandbytes)

Usage:
    python 24_full_ft_sweep_base_models.py --model-id Qwen/Qwen2.5-3B --dataset prefill --gpus 0,1,2,3
    python 24_full_ft_sweep_base_models.py --model-id meta-llama/Meta-Llama-3-8B --dataset wait_recompute --gpus 0,1,2,3
    python 24_full_ft_sweep_base_models.py --model-id Qwen/Qwen2.5-3B --dataset prefill --phase eval-only --gpus 0,1,2,3
    python 24_full_ft_sweep_base_models.py --model-id Qwen/Qwen2.5-3B --dataset prefill --phase summary-only
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

OCEAN_ROOT = Path("/ocean/projects/cis250050p/swang47/yang/LostInTheSecond")
ARTIFACTS_ROOT = OCEAN_ROOT / "artifacts"

DATASET_PREFILL = PROJECT_ROOT / "artifacts_real" / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json"
DATASET_WR = PROJECT_ROOT / "artifacts_real" / "samples_gsm8k_train_ds2_wait_recompute_gpt_prefill.json"

PYTHON_BIN = "/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/python"

TASKS = ["gsm8k_cot_zeroshot_unified", "hendrycks_math_500"]

MODEL_SHORT_NAMES = {
    "Qwen/Qwen2.5-3B": "qwen25_3b_base",
    "meta-llama/Meta-Llama-3-8B": "llama3_8b_base",
}

VLLM_ARGS_BY_MODEL = {
    "Qwen/Qwen2.5-3B": (
        "dtype=float16,"
        "gpu_memory_utilization=0.9,"
        "max_model_len=2048,"
        "max_num_seqs=16,"
        "enforce_eager=True"
    ),
    "meta-llama/Meta-Llama-3-8B": (
        "dtype=float16,"
        "gpu_memory_utilization=0.9,"
        "max_model_len=2048,"
        "max_num_seqs=16,"
        "enforce_eager=True"
    ),
}


def get_sweep_root(model_id: str, dataset: str) -> Path:
    short = MODEL_SHORT_NAMES[model_id]
    suffix = "_wr" if dataset == "wait_recompute" else ""
    return ARTIFACTS_ROOT / f"full_ft_sweep_{short}{suffix}"


def get_dataset_path(dataset: str) -> Path:
    return DATASET_WR if dataset == "wait_recompute" else DATASET_PREFILL


# ============================================================
# Phase 2 experiment definitions (same grid as 19/19b)
# ============================================================

def build_phase2_experiments() -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = []

    def add(name: str, desc: str = "", lr: float = 1e-6, epochs: float = 3,
            wd: float = 0.01, warmup: float = 0.1, ga_steps: int = 16):
        exps.append(dict(
            name=name, desc=desc, lr=lr, epochs=epochs, wd=wd,
            warmup=warmup, ga_steps=ga_steps,
        ))

    # Group A: WD sweep (warmup=0.1 fixed, lr=1e-6, ep=3)
    add("ft2_wd0",    desc="WD=0.0 at lr=1e-6",   wd=0.0)
    add("ft2_wd0005", desc="WD=0.005 at lr=1e-6",  wd=0.005)
    add("ft2_wd005",  desc="WD=0.05 at lr=1e-6",   wd=0.05)
    add("ft2_wd01",   desc="WD=0.1 at lr=1e-6",    wd=0.1)

    # Group B: Warmup sweep (wd=0.01 fixed, lr=1e-6, ep=3)
    add("ft2_wu0",   desc="Warmup=0.0 at lr=1e-6",  warmup=0.0)
    add("ft2_wu005", desc="Warmup=0.05 at lr=1e-6",  warmup=0.05)
    add("ft2_wu02",  desc="Warmup=0.2 at lr=1e-6",   warmup=0.2)
    add("ft2_wu03",  desc="Warmup=0.3 at lr=1e-6",   warmup=0.3)

    # Group C: Cross of promising WD x warmup
    add("ft2_wd0_wu02",  desc="WD=0 + warmup=0.2",  wd=0.0, warmup=0.2)
    add("ft2_wd01_wu02", desc="WD=0.1 + warmup=0.2", wd=0.1, warmup=0.2)

    return exps


# ============================================================
# Training
# ============================================================

def launch_training(exp: Dict, gpu_id: int, model_id: str,
                    dataset_path: Path, sweep_root: Path) -> subprocess.Popen:
    name = exp["name"]
    output_dir = sweep_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = " ".join([
        f"--model-id {model_id}",
        f"--dataset-path {dataset_path}",
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
        "--optim adafactor",
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


def run_training_sweep(experiments: List[Dict], gpus: List[int],
                       model_id: str, dataset_path: Path, sweep_root: Path):
    sweep_root.mkdir(parents=True, exist_ok=True)

    skipped, to_run = [], []
    for exp in experiments:
        model_path = sweep_root / exp["name"] / "best_model"
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
            proc = launch_training(exp, gpu, model_id, dataset_path, sweep_root)
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

    print(f"\nAll full FT training completed.")


# ============================================================
# Evaluation (vLLM, no chat template)
# ============================================================

def find_latest_results(run_dir: Path, task: str) -> Optional[Path]:
    candidates = sorted(run_dir.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def extract_exact_match(results_json: Path, task: str) -> Optional[float]:
    obj = json.loads(results_json.read_text())
    task_block = (obj.get("results") or {}).get(task, {})
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
                  task: str, batch_size: str = "auto",
                  limit: int = 0) -> subprocess.Popen:
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", task,
        "--batch_size", batch_size,
        "--gen_kwargs", "max_gen_toks=2048,temperature=0,do_sample=False",
        "--output_path", str(output_path),
        "--log_samples",
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


def discover_all_models(sweep_root: Path) -> List[Dict[str, Any]]:
    models = []
    if not sweep_root.exists():
        return models
    for d in sorted(sweep_root.iterdir()):
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


def run_eval_all(ft_models: List[Dict], gpus: List[int],
                 model_id: str, sweep_root: Path,
                 batch_size: str, limit: int) -> Dict[str, Dict]:
    vllm_args = VLLM_ARGS_BY_MODEL[model_id]
    results_all: Dict[str, Dict] = {}

    for task in TASKS:
        task_short = "gsm8k" if "gsm8k" in task else "math500"
        eval_root = sweep_root / f"_{task_short}_eval"
        eval_root.mkdir(parents=True, exist_ok=True)
        incremental_file = eval_root / "all_results.json"

        done: Dict[str, Dict] = {}
        if incremental_file.exists():
            for entry in json.loads(incremental_file.read_text()):
                em_key = f"{task_short}_em"
                if entry.get(em_key) is not None:
                    done[entry["name"]] = entry

        em_key = f"{task_short}_em"

        # Base model eval
        if "base" not in done:
            print(f"\n[BASE] Evaluating {model_id} on {task} (GPU {gpus[0]}) ...")
            t0 = time.time()
            model_args = f"pretrained={model_id},{vllm_args}"
            proc = run_vllm_eval(model_args, eval_root / "base",
                                 gpus[0], task, batch_size, limit)
            proc.wait()
            proc._log_fh.close()  # type: ignore[attr-defined]
            elapsed = time.time() - t0

            rj = find_latest_results(eval_root / "base", task)
            em = extract_exact_match(rj, task) if rj else None
            done["base"] = {"name": "base", em_key: em,
                            "elapsed_sec": round(elapsed, 1)}
            save_incremental(incremental_file, done)
            print(f"  Base {task_short} EM = {em}  ({elapsed:.0f}s)")
        else:
            print(f"[BASE] Already done on {task}: EM = {done['base'].get(em_key)}")

        # Evaluate finetuned models
        to_eval = [m for m in ft_models if m["name"] not in done]
        print(f"\nModels to evaluate on {task}: {len(to_eval)} "
              f"(skipping {len(ft_models) - len(to_eval)} already done)")

        if to_eval:
            gpu = gpus[0]
            for idx, model_info in enumerate(to_eval):
                name = model_info["name"]
                model_args = f"pretrained={model_info['model_path']},{vllm_args}"

                print(f"\n[EVAL {idx + 1}/{len(to_eval)}] {name} on {task} (GPU {gpu})")

                t0 = time.time()
                proc = run_vllm_eval(
                    model_args, eval_root / name, gpu, task, batch_size, limit)
                proc.wait()
                proc._log_fh.close()  # type: ignore[attr-defined]
                elapsed = time.time() - t0

                rj = find_latest_results(eval_root / name, task)
                em = extract_exact_match(rj, task) if rj else None
                status = f"EM = {em:.4f}" if em is not None else "FAILED"
                print(f"  {name}: {status}  ({elapsed:.0f}s)")

                done[name] = {
                    "name": name, em_key: em,
                    "elapsed_sec": round(elapsed, 1),
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

        for name, entry in done.items():
            if name not in results_all:
                results_all[name] = {"name": name}
            results_all[name].update(entry)

    return results_all


# ============================================================
# Report
# ============================================================

def generate_report(results: Dict[str, Dict], model_id: str,
                    dataset: str, output_path: Path):
    short = MODEL_SHORT_NAMES[model_id]
    ds_label = "Wait+Recompute" if dataset == "wait_recompute" else "GPT-Prefill"

    base = results.get("base", {})
    base_gsm8k = base.get("gsm8k_em")
    base_math = base.get("math500_em")

    ft_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )
    ok = [(n, r) for n, r in ft_results if r.get("gsm8k_em") is not None]

    lines = [
        f"# Full FT Sweep on {model_id} (Base) -- {ds_label}",
        "",
        f"**Model**: `{model_id}` (base, non-instruct)",
        f"**Dataset**: {ds_label}",
        f"**Method**: Full parameter fine-tuning (Adafactor optimizer)",
        f"**Eval Tasks**: GSM8K (1319 test) + MATH-500 (500 test)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## Baseline: GSM8K EM = **{base_gsm8k}** | MATH-500 EM = **{base_math}**",
        "",
        "## Full Results (Ranked by GSM8K EM)", "",
        "| Rank | Name | GSM8K EM | GSM8K Delta | MATH-500 EM | MATH Delta "
        "| LR | Epochs | WD | Warmup |",
        "|------|------|----------|-------------|-------------|------------"
        "|----|--------|-----|--------|",
    ]

    for i, (name, r) in enumerate(ft_results):
        gsm = r.get("gsm8k_em")
        math = r.get("math500_em")
        gsm_str = f"{gsm:.4f}" if gsm is not None else "FAIL"
        math_str = f"{math:.4f}" if math is not None else "FAIL"
        gsm_d = f"{gsm - base_gsm8k:+.4f}" if (gsm is not None and base_gsm8k is not None) else "N/A"
        math_d = f"{math - base_math:+.4f}" if (math is not None and base_math is not None) else "N/A"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "-"
        lines.append(
            f"| {i+1} | {name} | {gsm_str} | {gsm_d} | {math_str} | {math_d} "
            f"| {lr_str} | {r.get('epochs', '-')} | {r.get('wd', '-')} "
            f"| {r.get('warmup', '-')} |"
        )

    if ok and base_gsm8k is not None:
        best_name, best_r = ok[0]
        improved = [x for x in ok if x[1]["gsm8k_em"] > base_gsm8k]
        lines.extend(["", "## Key Findings", "",
            f"- **Best full FT**: `{best_name}` "
            f"(GSM8K EM = {best_r['gsm8k_em']:.4f}, "
            f"delta = {best_r['gsm8k_em'] - base_gsm8k:+.4f})",
            f"- **{len(improved)}/{len(ok)}** experiments improved over base on GSM8K",
        ])

    lines.extend(["", "---", "",
                   f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


def print_summary(results: Dict[str, Dict], model_id: str):
    base = results.get("base", {})
    base_gsm8k = base.get("gsm8k_em")
    base_math = base.get("math500_em")

    ft_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )

    print(f"\n{'=' * 110}")
    print(f"{'FULL FT BASE MODEL EVAL — ' + model_id:^110}")
    print(f"{'=' * 110}")
    print(f"\nBase: GSM8K EM = {base_gsm8k}  |  MATH-500 EM = {base_math}")
    print(f"\n{'Rk':<4} {'Name':<20} {'GSM8K':<10} {'Delta':<8} "
          f"{'MATH500':<10} {'Delta':<8} {'LR':<8} {'Ep':<5} {'WD':<6} {'WU':<6}")
    print("-" * 110)

    for i, (name, r) in enumerate(ft_results):
        gsm = r.get("gsm8k_em")
        math = r.get("math500_em")
        gsm_str = f"{gsm:.4f}" if gsm is not None else "FAIL"
        math_str = f"{math:.4f}" if math is not None else "FAIL"
        gsm_d = f"{gsm - base_gsm8k:+.4f}" if (gsm is not None and base_gsm8k) else "N/A"
        math_d = f"{math - base_math:+.4f}" if (math is not None and base_math) else "N/A"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "-"
        print(f"{i+1:<4} {name:<20} {gsm_str:<10} {gsm_d:<8} "
              f"{math_str:<10} {math_d:<8} {lr_str:<8} "
              f"{str(r.get('epochs', '-')):<5} "
              f"{str(r.get('wd', '-')):<6} {str(r.get('warmup', '-')):<6}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True,
                    choices=list(MODEL_SHORT_NAMES.keys()))
    ap.add_argument("--dataset", required=True,
                    choices=["prefill", "wait_recompute"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--phase", default="all",
                    choices=["all", "train-only", "eval-only", "summary-only"])
    ap.add_argument("--eval-batch-size", default="auto")
    ap.add_argument("--eval-limit", type=int, default=0)
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    sweep_root = get_sweep_root(args.model_id, args.dataset)
    dataset_path = get_dataset_path(args.dataset)
    short = MODEL_SHORT_NAMES[args.model_id]
    ds_label = "wait_recompute" if args.dataset == "wait_recompute" else "prefill"
    t_global = time.time()

    print(f"\n{'#' * 70}")
    print(f"  FULL FT SWEEP — {args.model_id} (base) — {ds_label}")
    print(f"  Sweep root: {sweep_root}")
    print(f"  Dataset: {dataset_path}")
    print(f"{'#' * 70}")

    # Training
    if args.phase in ("all", "train-only"):
        experiments = build_phase2_experiments()

        print(f"\n  Training: {len(experiments)} experiments, {len(gpus)} GPUs")

        plan_file = sweep_root / "experiment_plan.json"
        sweep_root.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps(experiments, indent=2))

        t0 = time.time()
        run_training_sweep(experiments, gpus, args.model_id,
                           dataset_path, sweep_root)
        elapsed = time.time() - t0
        print(f"\nTraining completed in {elapsed/60:.1f} min")

    # Evaluation
    if args.phase in ("all", "eval-only"):
        ft_models = discover_all_models(sweep_root)
        print(f"\n  Evaluating {len(ft_models)} models + base on "
              f"{len(TASKS)} tasks, {len(gpus)} GPUs")

        t0 = time.time()
        results = run_eval_all(
            ft_models, gpus, args.model_id, sweep_root,
            args.eval_batch_size, args.eval_limit)
        elapsed = time.time() - t0
        print(f"\nEvaluation completed in {elapsed/60:.1f} min")

        print_summary(results, args.model_id)
        report_name = f"full_ft_{short}{'_wr' if args.dataset == 'wait_recompute' else ''}_results.md"
        generate_report(results, args.model_id, args.dataset,
                        DOCUMENTS_DIR / report_name)

    # Summary only
    if args.phase == "summary-only":
        results: Dict[str, Dict] = {}
        for task in TASKS:
            task_short = "gsm8k" if "gsm8k" in task else "math500"
            eval_root = sweep_root / f"_{task_short}_eval"
            results_file = eval_root / "all_results.json"
            if results_file.exists():
                for entry in json.loads(results_file.read_text()):
                    name = entry["name"]
                    if name not in results:
                        results[name] = {"name": name}
                    results[name].update(entry)

        if not results:
            print("No results found. Run evaluation first.")
            return

        print_summary(results, args.model_id)
        report_name = f"full_ft_{short}{'_wr' if args.dataset == 'wait_recompute' else ''}_results.md"
        generate_report(results, args.model_id, args.dataset,
                        DOCUMENTS_DIR / report_name)

    total_elapsed = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"ALL DONE! Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.1f} hr)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
