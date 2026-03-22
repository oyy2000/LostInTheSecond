#!/usr/bin/env python3
"""
Train and evaluate Llama-3-8B-Base on 2 full-pipeline prefill datasets x 2 methods,
then evaluate on GSM8K using LEMMA's eval script.

Datasets:
  - fix_step2:           498 correct-only prefill samples
  - wait_recompute_all:  2224 all prefill samples (no correctness filter)

Methods: LoRA, Full parameter fine-tuning

Usage:
    python scripts/eval/31_train_and_eval_full_prefill.py --gpus 0,1
    python scripts/eval/31_train_and_eval_full_prefill.py --gpus 0,1 --phase train-only
    python scripts/eval/31_train_and_eval_full_prefill.py --gpus 0,1 --phase eval-only
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

OCEAN_ROOT = Path("/ocean/projects/cis250050p/swang47/yang/LostInTheSecond")
COMPARISON_ROOT = OCEAN_ROOT / "artifacts" / "full_prefill_llama8b"

PYTHON_BIN = "/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/python"
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
ALL_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

LEMMA_EVAL_DIR = Path("/jet/home/swang47/yang/projects/LEMMA/evaluation")


def build_experiments() -> List[Dict[str, Any]]:
    data_root = PROJECT_ROOT / "artifacts_real" / "full"
    exps = []

    # ---- LoRA experiments ----
    exps.append({
        "name": "lora_fix498",
        "method": "lora",
        "dataset": str(data_root / "lemma_sft_fix_step2.json"),
        "dataset_label": "Fix Step2 Correct (498)",
        "n_samples": 498,
        "lr": 5e-6, "epochs": 5, "ga": 8,
        "lora_r": 4, "lora_alpha": 4, "lora_dropout": 0.05,
        "target_modules": ALL_MODULES,
        "warmup": 0.03, "wd": 0.0,
    })
    exps.append({
        "name": "lora_wr2224",
        "method": "lora",
        "dataset": str(data_root / "lemma_sft_wait_recompute_all.json"),
        "dataset_label": "Wait Recompute All (2224)",
        "n_samples": 2224,
        "lr": 5e-6, "epochs": 3, "ga": 16,
        "lora_r": 4, "lora_alpha": 4, "lora_dropout": 0.05,
        "target_modules": ALL_MODULES,
        "warmup": 0.03, "wd": 0.0,
    })

    # ---- Full FT experiments ----
    exps.append({
        "name": "ft_fix498",
        "method": "full",
        "dataset": str(data_root / "lemma_sft_fix_step2.json"),
        "dataset_label": "Fix Step2 Correct (498)",
        "n_samples": 498,
        "lr": 1e-6, "epochs": 5, "ga": 8,
        "warmup": 0.1, "wd": 0.05,
        "optim": "adafactor",
    })
    exps.append({
        "name": "ft_wr2224",
        "method": "full",
        "dataset": str(data_root / "lemma_sft_wait_recompute_all.json"),
        "dataset_label": "Wait Recompute All (2224)",
        "n_samples": 2224,
        "lr": 1e-6, "epochs": 3, "ga": 16,
        "warmup": 0.1, "wd": 0.05,
        "optim": "adafactor",
    })

    return exps


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
        print(f"Skipping {len(skipped)} already-trained: {[e['name'] for e in skipped]}")
    if not to_run:
        print("All experiments already trained!")
        return

    batch_size = len(gpus)
    n_batches = (len(to_run) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        batch = to_run[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        print(f"\n{'=' * 70}")
        print(f"TRAIN BATCH {batch_idx + 1}/{n_batches}  |  {[e['name'] for e in batch]}")
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
# LoRA Merge
# ============================================================

def merge_lora(exp: Dict, gpu_id: int):
    exp_dir = COMPARISON_ROOT / exp["name"]
    adapter_dir = exp_dir / "final_adapter"
    merged_dir = exp_dir / "merged_model"
    if merged_dir.exists() and (merged_dir / "config.json").exists():
        print(f"  {exp['name']}: merged model exists, skipping")
        return
    print(f"  Merging LoRA for {exp['name']}...")
    merge_script = f"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{MODEL_ID}", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "{adapter_dir}")
merged = model.merge_and_unload()
merged.save_pretrained("{merged_dir}")
AutoTokenizer.from_pretrained("{MODEL_ID}").save_pretrained("{merged_dir}")
print("Merged -> {merged_dir}")
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.run(
        [PYTHON_BIN, "-c", merge_script],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"    MERGE FAILED: {proc.stderr[-500:]}")
    else:
        print(f"    OK: {merged_dir}")


def run_merge(experiments: List[Dict], gpus: List[int]):
    lora_exps = [e for e in experiments if e["method"] == "lora"]
    for i, exp in enumerate(lora_exps):
        merge_lora(exp, gpus[i % len(gpus)])


# ============================================================
# Evaluation (LEMMA GSM8K)
# ============================================================

def get_model_path(exp: Dict) -> str:
    exp_dir = COMPARISON_ROOT / exp["name"]
    if exp["method"] == "lora":
        return str(exp_dir / "merged_model")
    else:
        return str(exp_dir / "best_model")


def run_lemma_eval(model_name: str, model_path: str, gpu_id: int):
    output_dir = Path(model_path) / "math_eval" / "test_cot-meta-math_zero-shot"
    metrics_pattern = "gsm8k"

    existing = list(output_dir.rglob("*metrics*")) if output_dir.exists() else []
    if existing:
        print(f"  {model_name}: eval already exists, skipping")
        return

    print(f"  Evaluating {model_name} on GSM8K (GPU {gpu_id})...")
    cmd = [
        PYTHON_BIN, "-u", str(LEMMA_EVAL_DIR / "math_eval.py"),
        "--data_names", "gsm8k",
        "--model_name_or_path", model_path,
        "--output_dir", str(output_dir),
        "--split", "test",
        "--prompt_type", "cot-meta-math",
        "--num_test_sample", "-1",
        "--seed", "0",
        "--temperature", "0",
        "--n_sampling", "1",
        "--top_p", "1",
        "--start", "0",
        "--end", "-1",
        "--use_vllm",
        "--save_outputs",
        "--num_shots", "0",
        "--dtype", "float16",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    log_path = COMPARISON_ROOT / f"eval_{model_name}.log"
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                              cwd=str(LEMMA_EVAL_DIR))
    if proc.returncode != 0:
        print(f"    EVAL FAILED (rc={proc.returncode})")
    else:
        print(f"    OK")


def extract_gsm8k_acc(model_path: str) -> Optional[float]:
    eval_dir = Path(model_path) / "math_eval" / "test_cot-meta-math_zero-shot"
    if not eval_dir.exists():
        return None
    for f in sorted(eval_dir.rglob("*metrics*")):
        try:
            obj = json.loads(f.read_text())
            if "acc" in obj:
                return float(obj["acc"])
        except Exception:
            pass
    for f in sorted(eval_dir.rglob("*gsm8k*")):
        try:
            text = f.read_text()
            for line in text.splitlines():
                if "'acc'" in line:
                    import re
                    m = re.search(r"'acc':\s*([\d.]+)", line)
                    if m:
                        return float(m.group(1))
        except Exception:
            pass
    return None


def run_eval(experiments: List[Dict], gpus: List[int]):
    models = [("base", MODEL_ID)]
    for exp in experiments:
        models.append((exp["name"], get_model_path(exp)))

    batch_size = len(gpus)
    n_batches = (len(models) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        batch = models[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        procs = []
        for i, (name, path) in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            run_lemma_eval(name, path, gpu)


def generate_report(experiments: List[Dict]):
    print(f"\n{'=' * 70}")
    print("RESULTS: Full Prefill Pipeline Comparison (GSM8K)")
    print(f"{'=' * 70}")

    base_acc = extract_gsm8k_acc(MODEL_ID)
    print(f"\n  Base Model ({MODEL_ID}):")
    print(f"    GSM8K: {base_acc if base_acc else 'N/A'}")

    results = []
    for exp in experiments:
        model_path = get_model_path(exp)
        acc = extract_gsm8k_acc(model_path)
        results.append({
            "name": exp["name"],
            "method": exp["method"],
            "dataset_label": exp["dataset_label"],
            "n_samples": exp["n_samples"],
            "gsm8k_acc": acc,
        })
        print(f"\n  {exp['name']} ({exp['dataset_label']}, {exp['method']}):")
        print(f"    GSM8K: {acc if acc else 'N/A'}")

    report_path = PROJECT_ROOT / "documents" / "full_prefill_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Prefill Pipeline: GSM8K Comparison",
        "",
        "Base model: `meta-llama/Meta-Llama-3-8B`",
        f"Evaluation: LEMMA math_eval.py (cot-meta-math, zero-shot)",
        "",
        "## Results",
        "",
        "| Experiment | Method | Dataset | N | GSM8K Acc |",
        "|------------|--------|---------|---|-----------|",
        f"| base | - | - | - | {base_acc or 'N/A'} |",
    ]
    for r in results:
        acc_str = f"{r['gsm8k_acc']:.1%}" if r['gsm8k_acc'] is not None else "N/A"
        lines.append(
            f"| {r['name']} | {r['method']} | {r['dataset_label']} | "
            f"{r['n_samples']} | {acc_str} |"
        )
    lines += ["", "## Training Parameters", ""]
    for exp in experiments:
        lines.append(f"### {exp['name']}")
        lines.append(f"- Method: {exp['method']}")
        lines.append(f"- Dataset: {exp['dataset_label']} ({exp['n_samples']} samples)")
        lines.append(f"- LR: {exp['lr']}, Epochs: {exp['epochs']}, GA: {exp['ga']}")
        if exp['method'] == 'lora':
            lines.append(f"- LoRA: r={exp['lora_r']}, alpha={exp['lora_alpha']}, "
                         f"dropout={exp['lora_dropout']}")
        else:
            lines.append(f"- Weight decay: {exp['wd']}, Warmup: {exp['warmup']}")
        lines.append("")

    report_path.write_text("\n".join(lines), "utf-8")
    print(f"\nReport: {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--phase", default="all",
                     choices=["all", "train-only", "eval-only", "report-only"])
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    experiments = build_experiments()

    print(f"Experiments: {[e['name'] for e in experiments]}")
    print(f"GPUs: {gpus}")
    print(f"Output root: {COMPARISON_ROOT}")

    if args.phase in ("all", "train-only"):
        run_training(experiments, gpus)
        run_merge(experiments, gpus)

    if args.phase in ("all", "eval-only"):
        run_eval(experiments, gpus)

    if args.phase in ("all", "eval-only", "report-only"):
        generate_report(experiments)


if __name__ == "__main__":
    main()
