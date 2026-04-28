#!/usr/bin/env python3
"""
Train and evaluate Llama-3-8B-Base on 3 datasets x 2 methods (LoRA + Full FT),
then generate a comparison report.

Datasets:
  - LEMMA 1K unique (~1000 alpaca-format entries)
  - Prefill 245 (all fix_step2 samples, alpaca format)
  - Prefill 50  (correct-only fix_step2 samples, alpaca format)

Methods: LoRA, Full parameter fine-tuning

Usage:
    python scripts/eval/29_train_and_eval_comparison.py --gpus 0,1
    python scripts/eval/29_train_and_eval_comparison.py --gpus 0,1 --phase train-only
    python scripts/eval/29_train_and_eval_comparison.py --gpus 0,1 --phase eval-only
    python scripts/eval/29_train_and_eval_comparison.py --phase report-only
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
TRAIN_SCRIPT = SCRIPTS_ROOT / "data_prep" / "15_finetune_lemma.py"
HARNESS_DIR = PROJECT_ROOT / "lm-evaluation-harness"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

OCEAN_ROOT = Path("/ocean/projects/cis250050p/swang47/yang/LostInTheSecond")
COMPARISON_ROOT = OCEAN_ROOT / "artifacts" / "comparison_llama8b"

PYTHON_BIN = "/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/python"

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
TASKS = ["gsm8k_cot_zeroshot_unified", "hendrycks_math_500"]
ALL_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

VLLM_ARGS = (
    "dtype=float16,"
    "gpu_memory_utilization=0.9,"
    "max_model_len=2048,"
    "max_num_seqs=16,"
    "enforce_eager=True"
)


def build_experiments() -> List[Dict[str, Any]]:
    """Define all 6 experiments."""
    data_root = PROJECT_ROOT / "artifacts_real"
    exps = []

    # ---- LoRA experiments ----
    exps.append({
        "name": "lora_lemma1k",
        "method": "lora",
        "dataset": str(data_root / "lemma_1k_original_sft_unique.json"),
        "dataset_label": "LEMMA 1K",
        "n_samples": 1000,
        "lr": 5e-6, "epochs": 3, "ga": 16,
        "lora_r": 4, "lora_alpha": 4, "lora_dropout": 0.05,
        "target_modules": ALL_MODULES,
        "warmup": 0.03, "wd": 0.0,
    })
    exps.append({
        "name": "lora_prefill245",
        "method": "lora",
        "dataset": str(data_root / "lemma_sft_fix_step2_all.json"),
        "dataset_label": "Prefill 245",
        "n_samples": 245,
        "lr": 5e-6, "epochs": 5, "ga": 8,
        "lora_r": 4, "lora_alpha": 4, "lora_dropout": 0.05,
        "target_modules": ALL_MODULES,
        "warmup": 0.03, "wd": 0.0,
    })
    exps.append({
        "name": "lora_prefill50",
        "method": "lora",
        "dataset": str(data_root / "lemma_sft_fix_step2.json"),
        "dataset_label": "Prefill 50",
        "n_samples": 50,
        "lr": 5e-6, "epochs": 10, "ga": 4,
        "lora_r": 4, "lora_alpha": 4, "lora_dropout": 0.1,
        "target_modules": ALL_MODULES,
        "warmup": 0.05, "wd": 0.0,
    })

    # ---- Full FT experiments ----
    exps.append({
        "name": "ft_lemma1k",
        "method": "full",
        "dataset": str(data_root / "lemma_1k_original_sft_unique.json"),
        "dataset_label": "LEMMA 1K",
        "n_samples": 1000,
        "lr": 1e-6, "epochs": 3, "ga": 16,
        "warmup": 0.1, "wd": 0.05,
        "optim": "adafactor",
    })
    exps.append({
        "name": "ft_prefill245",
        "method": "full",
        "dataset": str(data_root / "lemma_sft_fix_step2_all.json"),
        "dataset_label": "Prefill 245",
        "n_samples": 245,
        "lr": 1e-6, "epochs": 5, "ga": 8,
        "warmup": 0.1, "wd": 0.1,
        "optim": "adafactor",
    })
    exps.append({
        "name": "ft_prefill50",
        "method": "full",
        "dataset": str(data_root / "lemma_sft_fix_step2.json"),
        "dataset_label": "Prefill 50",
        "n_samples": 50,
        "lr": 1e-6, "epochs": 10, "ga": 4,
        "warmup": 0.2, "wd": 0.1,
        "optim": "adafactor",
    })

    return exps


# ============================================================
# Training
# ============================================================

def is_trained(exp: Dict) -> bool:
    exp_dir = COMPARISON_ROOT / exp["name"]
    if exp["method"] == "lora":
        return (exp_dir / "final_adapter" / "adapter_config.json").exists()
    else:
        return (exp_dir / "best_model" / "config.json").exists()


def build_train_cmd(exp: Dict) -> List[str]:
    exp_dir = COMPARISON_ROOT / exp["name"]
    cmd = [
        PYTHON_BIN, str(TRAIN_SCRIPT),
        "--method", exp["method"],
        "--model-id", MODEL_ID,
        "--dataset-path", exp["dataset"],
        "--output-dir", str(exp_dir),
        "--learning-rate", str(exp["lr"]),
        "--num-train-epochs", str(exp["epochs"]),
        "--gradient-accumulation-steps", str(exp["ga"]),
        "--warmup-ratio", str(exp["warmup"]),
        "--weight-decay", str(exp["wd"]),
        "--per-device-train-batch-size", "1",
        "--logging-steps", "1",
        "--save-total-limit", "2",
        "--seed", "42",
    ]

    if exp["method"] == "lora":
        cmd += [
            "--lora-r", str(exp["lora_r"]),
            "--lora-alpha", str(exp["lora_alpha"]),
            "--lora-dropout", str(exp["lora_dropout"]),
            "--target-modules", exp["target_modules"],
            "--eval-steps", "9999",
            "--save-steps", "9999",
        ]
    else:
        cmd += [
            "--optim", exp.get("optim", "adafactor"),
            "--eval-steps", "7",
            "--save-steps", "7",
        ]

    return cmd


def launch_training(exp: Dict, gpu_id: int) -> subprocess.Popen:
    exp_dir = COMPARISON_ROOT / exp["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_train_cmd(exp)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_file = exp_dir / "training.log"
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    proc._log_fh = fh
    return proc


def run_training(experiments: List[Dict], gpus: List[int]):
    to_run = [e for e in experiments if not is_trained(e)]
    skipped = [e for e in experiments if is_trained(e)]

    if skipped:
        print(f"Skipping {len(skipped)} already-trained: "
              f"{[e['name'] for e in skipped]}")
    if not to_run:
        print("All experiments already trained!")
        return

    batch_size = len(gpus)
    n_batches = (len(to_run) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch = to_run[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        print(f"\n{'=' * 70}")
        print(f"TRAIN BATCH {batch_idx + 1}/{n_batches}  |  "
              f"{[e['name'] for e in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, exp in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            print(f"  Launching {exp['name']} ({exp['method']}) on GPU {gpu}")
            proc = launch_training(exp, gpu)
            procs.append((exp["name"], proc))
            time.sleep(10)

        for name, proc in procs:
            proc.wait()
            proc._log_fh.close()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")


# ============================================================
# Evaluation
# ============================================================

def find_latest_results(run_dir: Path) -> Optional[Path]:
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


def run_vllm_eval(model_path: str, output_path: Path, gpu_id: int,
                  task: str) -> subprocess.Popen:
    output_path.mkdir(parents=True, exist_ok=True)
    model_args = f"pretrained={model_path},{VLLM_ARGS}"
    cmd = [
        PYTHON_BIN, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", task,
        "--batch_size", "auto",
        "--gen_kwargs", "max_gen_toks=2048,temperature=0,do_sample=False",
        "--output_path", str(output_path),
        "--log_samples",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"

    log_file = output_path / "eval.log"
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(HARNESS_DIR), env=env)
    proc._log_fh = fh
    return proc


def merge_lora_adapter(adapter_path: str, output_dir: Path) -> Path:
    merged_path = output_dir / "merged_model"
    if merged_path.exists() and (merged_path / "config.json").exists():
        print(f"    Merged model already exists: {merged_path}")
        return merged_path
    merged_path.mkdir(parents=True, exist_ok=True)
    merge_script = (
        f"import torch; "
        f"from peft import PeftModel; "
        f"from transformers import AutoModelForCausalLM, AutoTokenizer; "
        f"base = AutoModelForCausalLM.from_pretrained('{MODEL_ID}', torch_dtype=torch.float16); "
        f"model = PeftModel.from_pretrained(base, '{adapter_path}'); "
        f"merged = model.merge_and_unload(); "
        f"merged.save_pretrained('{merged_path}'); "
        f"AutoTokenizer.from_pretrained('{MODEL_ID}').save_pretrained('{merged_path}'); "
        f"print('Merged to', '{merged_path}')"
    )
    proc = subprocess.run(
        [PYTHON_BIN, "-c", merge_script],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    )
    if proc.returncode != 0:
        print(f"    Merge FAILED: {proc.stderr[-500:]}")
        return Path("")
    return merged_path


def get_eval_model_path(exp: Dict) -> Optional[Path]:
    exp_dir = COMPARISON_ROOT / exp["name"]
    if exp["method"] == "lora":
        adapter_path = exp_dir / "final_adapter"
        if not adapter_path.exists():
            return None
        print(f"  Merging LoRA adapter for {exp['name']}...")
        merged = merge_lora_adapter(str(adapter_path), exp_dir)
        return merged if merged.exists() else None
    else:
        best = exp_dir / "best_model"
        return best if best.exists() and (best / "config.json").exists() else None


def run_eval_all(experiments: List[Dict], gpus: List[int]):
    for task in TASKS:
        task_short = "gsm8k" if "gsm8k" in task else "math500"
        eval_root = COMPARISON_ROOT / f"_{task_short}_eval"
        eval_root.mkdir(parents=True, exist_ok=True)
        incremental_file = eval_root / "all_results.json"

        done = {}
        if incremental_file.exists():
            for entry in json.loads(incremental_file.read_text()):
                em_key = f"{task_short}_em"
                if entry.get(em_key) is not None:
                    done[entry["name"]] = entry

        em_key = f"{task_short}_em"

        # Base model eval
        if "base" not in done:
            print(f"\n[BASE] Evaluating {MODEL_ID} on {task} (GPU {gpus[0]})")
            t0 = time.time()
            proc = run_vllm_eval(MODEL_ID, eval_root / "base", gpus[0], task)
            proc.wait()
            proc._log_fh.close()
            elapsed = time.time() - t0

            rj = find_latest_results(eval_root / "base")
            em = extract_exact_match(rj, task) if rj else None
            done["base"] = {"name": "base", em_key: em,
                            "elapsed_sec": round(elapsed, 1)}
            _save_incremental(incremental_file, done)
            print(f"  Base {task_short} EM = {em}  ({elapsed:.0f}s)")
        else:
            print(f"[BASE] Already done on {task}: "
                  f"EM = {done['base'].get(em_key)}")

        # Evaluate each experiment
        to_eval = [e for e in experiments if e["name"] not in done]
        print(f"\nModels to evaluate on {task}: {len(to_eval)} "
              f"(skipping {len(experiments) - len(to_eval)} already done)")

        for idx, exp in enumerate(to_eval):
            model_path = get_eval_model_path(exp)
            if model_path is None:
                print(f"  Skipping {exp['name']} (no trained model)")
                continue

            gpu = gpus[idx % len(gpus)]
            print(f"\n[EVAL {idx + 1}/{len(to_eval)}] "
                  f"{exp['name']} on {task} (GPU {gpu})")

            t0 = time.time()
            proc = run_vllm_eval(
                str(model_path), eval_root / exp["name"], gpu, task)
            proc.wait()
            proc._log_fh.close()
            elapsed = time.time() - t0

            rj = find_latest_results(eval_root / exp["name"])
            em = extract_exact_match(rj, task) if rj else None
            status = f"EM = {em:.4f}" if em is not None else "FAILED"
            print(f"  {exp['name']}: {status}  ({elapsed:.0f}s)")

            done[exp["name"]] = {
                "name": exp["name"],
                "method": exp["method"],
                "dataset_label": exp["dataset_label"],
                "n_samples": exp["n_samples"],
                em_key: em,
                "elapsed_sec": round(elapsed, 1),
            }
            _save_incremental(incremental_file, done)


def _save_incremental(path: Path, results: Dict[str, Dict]):
    path.write_text(json.dumps(list(results.values()), indent=2, default=str))


# ============================================================
# Report
# ============================================================

def load_all_results() -> Dict[str, Dict]:
    results = {}
    for task in TASKS:
        task_short = "gsm8k" if "gsm8k" in task else "math500"
        eval_root = COMPARISON_ROOT / f"_{task_short}_eval"
        results_file = eval_root / "all_results.json"
        if results_file.exists():
            for entry in json.loads(results_file.read_text()):
                name = entry["name"]
                if name not in results:
                    results[name] = {"name": name}
                results[name].update(entry)
    return results


def generate_report(experiments: List[Dict]):
    results = load_all_results()
    if not results:
        print("No results found. Run evaluation first.")
        return

    base = results.get("base", {})
    base_gsm8k = base.get("gsm8k_em")
    base_math = base.get("math500_em")

    lines = [
        "# Prefill vs LEMMA 1K Comparison: Llama-3-8B-Base",
        "",
        f"**Model**: `{MODEL_ID}` (base, non-instruct)",
        f"**Eval Tasks**: GSM8K (1319 test, zero-shot CoT) + MATH-500 (500 test)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## Baseline: GSM8K EM = **{base_gsm8k}** | MATH-500 EM = **{base_math}**",
        "",
    ]

    # Build table for each method
    for method_label, method_key in [("LoRA", "lora"), ("Full FT", "full")]:
        method_exps = [e for e in experiments if e["method"] == method_key]
        lines.append(f"## {method_label} Results")
        lines.append("")

        # Parameter summary
        lines.append(f"### {method_label} Training Parameters")
        lines.append("")
        if method_key == "lora":
            lines.append("| Param | " + " | ".join(
                e["dataset_label"] for e in method_exps) + " |")
            lines.append("|---|" + "|".join(["---"] * len(method_exps)) + "|")
            lines.append("| N samples | " + " | ".join(
                str(e["n_samples"]) for e in method_exps) + " |")
            lines.append("| LR | " + " | ".join(
                f"{e['lr']:.0e}" for e in method_exps) + " |")
            lines.append("| Epochs | " + " | ".join(
                str(e["epochs"]) for e in method_exps) + " |")
            lines.append("| r / alpha | " + " | ".join(
                f"{e['lora_r']} / {e['lora_alpha']}" for e in method_exps) + " |")
            lines.append("| Dropout | " + " | ".join(
                str(e["lora_dropout"]) for e in method_exps) + " |")
            lines.append("| GA steps | " + " | ".join(
                str(e["ga"]) for e in method_exps) + " |")
            lines.append("| Warmup | " + " | ".join(
                str(e["warmup"]) for e in method_exps) + " |")
        else:
            lines.append("| Param | " + " | ".join(
                e["dataset_label"] for e in method_exps) + " |")
            lines.append("|---|" + "|".join(["---"] * len(method_exps)) + "|")
            lines.append("| N samples | " + " | ".join(
                str(e["n_samples"]) for e in method_exps) + " |")
            lines.append("| LR | " + " | ".join(
                f"{e['lr']:.0e}" for e in method_exps) + " |")
            lines.append("| Epochs | " + " | ".join(
                str(e["epochs"]) for e in method_exps) + " |")
            lines.append("| Weight Decay | " + " | ".join(
                str(e["wd"]) for e in method_exps) + " |")
            lines.append("| Warmup | " + " | ".join(
                str(e["warmup"]) for e in method_exps) + " |")
            lines.append("| GA steps | " + " | ".join(
                str(e["ga"]) for e in method_exps) + " |")
            lines.append("| Optim | " + " | ".join(
                e.get("optim", "adamw") for e in method_exps) + " |")

        lines.append("")

        # Eval results table
        lines.append(f"### {method_label} Evaluation Results")
        lines.append("")
        lines.append("| Dataset | N | GSM8K EM | GSM8K Delta | MATH-500 EM | MATH Delta |")
        lines.append("|---|---|---|---|---|---|")

        for exp in method_exps:
            r = results.get(exp["name"], {})
            gsm = r.get("gsm8k_em")
            math = r.get("math500_em")
            gsm_str = f"{gsm * 100:.2f}%" if gsm is not None else "N/A"
            math_str = f"{math * 100:.2f}%" if math is not None else "N/A"
            gsm_d = (f"{(gsm - base_gsm8k) * 100:+.2f}%"
                     if gsm is not None and base_gsm8k is not None else "N/A")
            math_d = (f"{(math - base_math) * 100:+.2f}%"
                      if math is not None and base_math is not None else "N/A")
            lines.append(
                f"| {exp['dataset_label']} | {exp['n_samples']} | "
                f"{gsm_str} | {gsm_d} | {math_str} | {math_d} |")

        lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append("| Method | Dataset | N | GSM8K EM | MATH-500 EM |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Baseline | - | - | "
        f"{base_gsm8k * 100:.2f}% | {base_math * 100:.2f}% |"
        if base_gsm8k is not None and base_math is not None
        else "| Baseline | - | - | N/A | N/A |")

    for exp in experiments:
        r = results.get(exp["name"], {})
        gsm = r.get("gsm8k_em")
        math = r.get("math500_em")
        method_label = "LoRA" if exp["method"] == "lora" else "Full FT"
        gsm_str = f"{gsm * 100:.2f}%" if gsm is not None else "N/A"
        math_str = f"{math * 100:.2f}%" if math is not None else "N/A"
        lines.append(
            f"| {method_label} | {exp['dataset_label']} | "
            f"{exp['n_samples']} | {gsm_str} | {math_str} |")

    lines.extend(["", "---", "",
                   f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", ""])

    report_path = DOCUMENTS_DIR / "prefill_vs_lemma1k_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def print_summary(experiments: List[Dict]):
    results = load_all_results()
    base = results.get("base", {})
    base_gsm8k = base.get("gsm8k_em")
    base_math = base.get("math500_em")

    print(f"\n{'=' * 100}")
    print(f"{'COMPARISON: Prefill vs LEMMA 1K — ' + MODEL_ID:^100}")
    print(f"{'=' * 100}")
    print(f"\nBase: GSM8K EM = {base_gsm8k}  |  MATH-500 EM = {base_math}")
    print(f"\n{'Method':<10} {'Dataset':<16} {'N':>5} {'GSM8K':>10} "
          f"{'Delta':>8} {'MATH500':>10} {'Delta':>8}")
    print("-" * 100)

    for exp in experiments:
        r = results.get(exp["name"], {})
        gsm = r.get("gsm8k_em")
        math = r.get("math500_em")
        method_label = "LoRA" if exp["method"] == "lora" else "Full FT"
        gsm_str = f"{gsm:.4f}" if gsm is not None else "N/A"
        math_str = f"{math:.4f}" if math is not None else "N/A"
        gsm_d = (f"{gsm - base_gsm8k:+.4f}"
                 if gsm is not None and base_gsm8k is not None else "N/A")
        math_d = (f"{math - base_math:+.4f}"
                  if math is not None and base_math is not None else "N/A")
        print(f"{method_label:<10} {exp['dataset_label']:<16} "
              f"{exp['n_samples']:>5} {gsm_str:>10} {gsm_d:>8} "
              f"{math_str:>10} {math_d:>8}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--phase", default="all",
                    choices=["all", "train-only", "eval-only", "report-only"])
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    experiments = build_experiments()
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    print(f"\n{'#' * 70}")
    print(f"  COMPARISON: Prefill vs LEMMA 1K — {MODEL_ID}")
    print(f"  Root: {COMPARISON_ROOT}")
    print(f"  GPUs: {gpus}")
    print(f"  Experiments: {len(experiments)}")
    print(f"{'#' * 70}")

    # Save experiment plan
    plan_file = COMPARISON_ROOT / "experiment_plan.json"
    plan_file.write_text(json.dumps(experiments, indent=2, default=str))

    if args.phase in ("all", "train-only"):
        t0 = time.time()
        run_training(experiments, gpus)
        print(f"\nTraining completed in {(time.time() - t0) / 60:.1f} min")

    if args.phase in ("all", "eval-only"):
        t0 = time.time()
        run_eval_all(experiments, gpus)
        print(f"\nEvaluation completed in {(time.time() - t0) / 60:.1f} min")

    if args.phase in ("all", "eval-only", "report-only"):
        print_summary(experiments)
        generate_report(experiments)

    total = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"ALL DONE! Total time: {total / 60:.1f} min ({total / 3600:.1f} hr)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
