#!/usr/bin/env python3
"""
Phase 3 LoRA sweep + GSM8K evaluation (vLLM).

Phase 3 training: 13 experiments exploring lower learning rates with
rank {4, 16} and {attn_only, all-proj} target modules:
  - Group A: Low LR + r=16 + attn_only
  - Group B: Low LR + r=4  + all-proj  (alpha=1x)
  - Group C: Low LR + r=4  + attn_only (alpha=1x)
  - Group D: Low LR + r=16 + all-proj
  - Group E: Alpha ratio exploration at LR=1e-5

Evaluation via vLLM on all adapters (Phase 1+2+3 + base).

Usage:
    python 18_sweep_v3_and_eval.py --gpus 0,1,2,3
    python 18_sweep_v3_and_eval.py --phase eval-only --gpus 0,1,2,3
    python 18_sweep_v3_and_eval.py --phase train-only --gpus 0,1,2,3
    python 18_sweep_v3_and_eval.py --phase summary-only
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
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

ARTIFACTS_LOCAL = PROJECT_ROOT / "artifacts_local"
DATASET_PREFILL = PROJECT_ROOT / "artifacts_real" / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json"
SWEEP_ROOT = ARTIFACTS_LOCAL / "lora_sweep"
EVAL_ROOT = SWEEP_ROOT / "_gsm8k_vllm_eval"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PYTHON_BIN = "/common/home/sl2148/anaconda3/envs/sft/bin/python"
TASK = "gsm8k_cot_zeroshot_unified"

VLLM_COMMON_ARGS = (
    "dtype=float16,"
    "gpu_memory_utilization=0.9,"
    "max_model_len=2048,"
    "max_num_seqs=16,"
    "enforce_eager=True"
)

ALL_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
ATTN_MODULES = "q_proj,k_proj,v_proj,o_proj"


# ============================================================
# Phase 3 experiment definitions
# ============================================================

def build_phase3_experiments() -> List[Dict[str, Any]]:
    """
    Phase 1+2 found:
      - attn_only (lr=2e-4, r=16, alpha=32):  EM = 0.8461 (best overall)
      - v2_lr1e4_r4_a1x (lr=1e-4, r=4, a=4): EM = 0.8446 (#2)
      - lr_1e-5 (r=16, alpha=32, all-proj):   EM = 0.8423 (#3)

    Phase 3 strategy: push LR lower (1e-6, 5e-6, 1e-5) across the two
    winning rank configs (r=4, r=16) and both module targets (attn_only,
    all-proj). Fixed epochs=5.
    """
    exps: List[Dict[str, Any]] = []

    def add(name: str, lr: float = 1e-5, r: int = 16, alpha: int = 32,
            dropout: float = 0.05, epochs: float = 5, wd: float = 0.0,
            warmup: float = 0.03, ga_steps: int = 16,
            modules: str = ALL_MODULES):
        exps.append(dict(
            name=name, lr=lr, r=r, alpha=alpha, dropout=dropout,
            epochs=epochs, wd=wd, warmup=warmup, ga_steps=ga_steps,
            modules=modules,
        ))

    # --- Group A: Low LR + r=16 + attn_only ---
    add("v3_lr1e6_r16_attn", lr=1e-6, r=16, alpha=32, modules=ATTN_MODULES)
    add("v3_lr5e6_r16_attn", lr=5e-6, r=16, alpha=32, modules=ATTN_MODULES)
    add("v3_lr1e5_r16_attn", lr=1e-5, r=16, alpha=32, modules=ATTN_MODULES)

    # --- Group B: Low LR + r=4 + alpha=1x + all-proj ---
    add("v3_lr1e6_r4_a1x", lr=1e-6, r=4, alpha=4)
    add("v3_lr5e6_r4_a1x", lr=5e-6, r=4, alpha=4)
    add("v3_lr1e5_r4_a1x", lr=1e-5, r=4, alpha=4)

    # --- Group C: Low LR + r=4 + attn_only ---
    add("v3_lr1e6_r4_attn", lr=1e-6, r=4, alpha=4, modules=ATTN_MODULES)
    add("v3_lr5e6_r4_attn", lr=5e-6, r=4, alpha=4, modules=ATTN_MODULES)
    add("v3_lr1e5_r4_attn", lr=1e-5, r=4, alpha=4, modules=ATTN_MODULES)

    # --- Group D: Low LR + r=16 + all-proj ---
    add("v3_lr1e6_r16_all", lr=1e-6, r=16, alpha=32)
    add("v3_lr5e6_r16_all", lr=5e-6, r=16, alpha=32)

    # --- Group E: Alpha ratio exploration at LR=1e-5 ---
    add("v3_lr1e5_r16_a1x_attn", lr=1e-5, r=16, alpha=16,
        modules=ATTN_MODULES)
    add("v3_lr1e5_r4_a2x_attn",  lr=1e-5, r=4,  alpha=8,
        modules=ATTN_MODULES)

    return exps


# ============================================================
# Training
# ============================================================

def launch_training(exp: Dict, gpu_id: int) -> subprocess.Popen:
    name = exp["name"]
    output_dir = SWEEP_ROOT / name
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


def run_training_sweep(experiments: List[Dict], gpus: List[int]):
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    batch_size = len(gpus)

    skipped, to_run = [], []
    for exp in experiments:
        adapter_path = SWEEP_ROOT / exp["name"] / "final_adapter"
        if adapter_path.exists():
            skipped.append(exp["name"])
        else:
            to_run.append(exp)

    if skipped:
        print(f"Skipping {len(skipped)} already-trained: {skipped}")
    if not to_run:
        print("All Phase 3 experiments already completed!")
        return

    total_batches = (len(to_run) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch = to_run[batch_start: batch_start + batch_size]

        print(f"\n{'=' * 70}")
        print(f"TRAIN BATCH {batch_idx + 1}/{total_batches}  |  "
              f"{[e['name'] for e in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, exp in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            print(f"  Launching {exp['name']} on GPU {gpu} ...")
            proc = launch_training(exp, gpu)
            procs.append((exp["name"], proc))
            time.sleep(15)

        for name, proc in procs:
            proc.wait()
            proc._log_fh.close()  # type: ignore[attr-defined]
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

    print(f"\nAll Phase 3 training completed.")


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
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    proc._log_fh = fh  # type: ignore[attr-defined]
    return proc


def load_training_metrics(adapter_dir: Path) -> Dict[str, Any]:
    mf = adapter_dir / "sweep_metrics.json"
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
        "r": cfg.get("lora_r"),
        "alpha": cfg.get("lora_alpha"),
        "dropout": cfg.get("lora_dropout"),
        "epochs": cfg.get("num_train_epochs"),
        "wd": cfg.get("weight_decay"),
        "warmup": cfg.get("warmup_ratio"),
        "modules": cfg.get("target_modules"),
        "train_samples": cfg.get("train_samples"),
    }


def discover_all_adapters() -> List[Dict[str, Any]]:
    adapters = []
    for d in sorted(SWEEP_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        adapter_path = d / "final_adapter"
        if not adapter_path.exists():
            continue
        training = load_training_metrics(d)

        if d.name.startswith("v3_"):
            phase = "v3"
        elif d.name.startswith("v2_"):
            phase = "v2"
        else:
            phase = "v1"

        adapters.append({
            "name": d.name,
            "adapter_path": str(adapter_path),
            "phase": phase,
            **training,
        })
    return adapters


def save_incremental(path: Path, results: Dict[str, Dict]):
    path.write_text(json.dumps(list(results.values()), indent=2, default=str))


def merge_lora_adapter(adapter_path: str, output_dir: Path) -> Path:
    """Merge LoRA adapter into base model, return path to merged model."""
    merged_path = output_dir / "merged_model"
    if merged_path.exists() and (merged_path / "config.json").exists():
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
        print(f"  Merge FAILED: {proc.stderr[-500:]}")
        return Path("")
    return merged_path


def run_parallel_eval(adapters: List[Dict], gpus: List[int],
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

    to_eval = [a for a in adapters if a["name"] not in done]
    print(f"\nAdapters to evaluate: {len(to_eval)} "
          f"(skipping {len(adapters) - len(to_eval)} already done)")

    if not to_eval:
        print("All evaluations already completed!")
        return done

    # Merge LoRA adapters into base model (CPU only, sequential)
    print(f"\nMerging {len(to_eval)} LoRA adapters into base model ...")
    for adapter in to_eval:
        name = adapter["name"]
        merged = merge_lora_adapter(
            adapter["adapter_path"], SWEEP_ROOT / name)
        adapter["merged_path"] = str(merged)
        if merged.exists():
            print(f"  {name}: merged OK")
        else:
            print(f"  {name}: merge FAILED")

    gpu = gpus[0]
    for idx, adapter in enumerate(to_eval):
        name = adapter["name"]
        merged_path = adapter.get("merged_path", "")
        if not merged_path or not Path(merged_path).exists():
            print(f"  Skipping {name} (no merged model)")
            continue

        print(f"\n{'=' * 70}")
        print(f"EVAL {idx + 1}/{len(to_eval)}  |  {name}  |  GPU {gpu}")
        print(f"{'=' * 70}")

        model_args = f"pretrained={merged_path},{VLLM_COMMON_ARGS}"
        t0 = time.time()
        proc = run_vllm_eval(
            model_args, EVAL_ROOT / name, gpu, batch_size, limit)
        proc.wait()
        proc._log_fh.close()  # type: ignore[attr-defined]
        elapsed = time.time() - t0

        rj = find_latest_results(EVAL_ROOT / name)
        em = extract_exact_match(rj) if rj else None
        status = f"EM = {em:.4f}" if em is not None else "FAILED"
        print(f"  {name}: {status}  ({elapsed:.0f}s)")

        done[name] = {
            "name": name, "gsm8k_em": em,
            "elapsed_sec": round(elapsed, 1),
            "phase": adapter.get("phase"),
            "eval_loss": adapter.get("eval_loss"),
            "best_eval_loss": adapter.get("best_eval_loss"),
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
        print(f"  Saved. Remaining: {len(to_eval) - idx - 1} adapters")

    return done


# ============================================================
# Report
# ============================================================

def generate_report(results: Dict[str, Dict], output_path: Path):
    base_em = results.get("base", {}).get("gsm8k_em")
    adapter_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )
    ok = [(n, r) for n, r in adapter_results if r.get("gsm8k_em") is not None]

    lines = [
        "# GSM8K LoRA Sweep — Phase 1+2+3 Evaluation Results",
        "",
        f"**Base Model**: `{MODEL_ID}`",
        f"**Task**: `{TASK}` (GSM8K zero-shot CoT, 1319 test samples)",
        "**Training Data**: 109 train / 6 eval samples "
        "(GSM8K step-2 GPT-prefilled corrections)",
        "**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## Base Model: GSM8K EM = **{base_em:.4f}**" if base_em
        else "## Base Model: N/A",
        "",
        "## Full Results (Ranked by GSM8K Exact Match)", "",
        "| Rank | Name | Phase | GSM8K EM | Delta | Eval Loss | Best EL "
        "| Train Loss | LR | r | Alpha | Epochs | Drop | WD |",
        "|------|------|-------|----------|-------|-----------|--------"
        "|------------|-----|---|-------|--------|------|-----|",
    ]

    for i, (name, r) in enumerate(adapter_results):
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
            f"{r.get('r', '—')} | {r.get('alpha', '—')} | "
            f"{r.get('epochs', '—')} | {r.get('dropout', '—')} | "
            f"{r.get('wd', '—')} |"
        )

    if ok and base_em is not None:
        best_name, best_r = ok[0]
        worst_name, worst_r = ok[-1]
        improved = [x for x in ok if x[1]["gsm8k_em"] > base_em]

        lines.extend(["", "## Key Findings", "",
            f"- **Best adapter**: `{best_name}` "
            f"(EM = {best_r['gsm8k_em']:.4f}, "
            f"delta = {best_r['gsm8k_em'] - base_em:+.4f})",
            f"- **Worst adapter**: `{worst_name}` "
            f"(EM = {worst_r['gsm8k_em']:.4f}, "
            f"delta = {worst_r['gsm8k_em'] - base_em:+.4f})",
            f"- **{len(improved)}/{len(ok)}** adapters improved over base",
        ])

        for phase_label in ["v1", "v2", "v3"]:
            phase_ok = [(n, r) for n, r in ok if r.get("phase") == phase_label]
            if phase_ok:
                bp = max(phase_ok, key=lambda x: x[1]["gsm8k_em"])
                lines.append(f"- Best Phase {phase_label[-1]}: `{bp[0]}` "
                             f"(EM = {bp[1]['gsm8k_em']:.4f})")

    lines.extend(["", "---", "",
                   f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
                   f"*Full JSON: `{EVAL_ROOT / 'all_results.json'}`*", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


def print_summary(results: Dict[str, Dict]):
    base_em = results.get("base", {}).get("gsm8k_em")
    adapter_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )

    print(f"\n{'=' * 110}")
    print(f"{'GSM8K LoRA EVALUATION — Phase 1+2+3 (vLLM)':^110}")
    print(f"{'=' * 110}")
    print(f"\nBase model: {MODEL_ID}  |  GSM8K EM = {base_em}")
    print(f"\n{'Rk':<4} {'Name':<30} {'Ph':<4} {'GSM8K EM':<10} "
          f"{'Delta':<8} {'EvalLoss':<10} {'LR':<8} {'r':<4} {'α':<5} {'Ep':<5}")
    print("-" * 110)

    for i, (name, r) in enumerate(adapter_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        delta_str = f"{em - base_em:+.4f}" if (em and base_em) else "N/A"
        el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "—"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "—"
        ph = r.get("phase", "—")
        print(f"{i+1:<4} {name:<30} {ph:<4} {em_str:<10} {delta_str:<8} "
              f"{el:<10} {lr_str:<8} {str(r.get('r', '—')):<4} "
              f"{str(r.get('alpha', '—')):<5} {str(r.get('epochs', '—')):<5}")

    ok = [(n, r) for n, r in adapter_results if r.get("gsm8k_em") is not None]
    if ok and base_em:
        improved = sum(1 for _, r in ok if r["gsm8k_em"] > base_em)
        print(f"\n{improved}/{len(ok)} adapters beat base model")
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

    # ---- Phase 3 Training ----
    if args.phase in ("all", "train-only"):
        experiments = build_phase3_experiments()

        print(f"\n{'#' * 70}")
        print(f"  PHASE 3 LoRA TRAINING: {len(experiments)} experiments, "
              f"{len(gpus)} GPUs")
        print(f"{'#' * 70}")

        plan_file = SWEEP_ROOT / "v3_experiment_plan.json"
        SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps(experiments, indent=2))
        print(f"Plan: {plan_file}")

        t0 = time.time()
        run_training_sweep(experiments, gpus)
        elapsed = time.time() - t0
        print(f"\nPhase 3 training completed in {elapsed/60:.1f} min")

    # ---- GSM8K Evaluation ----
    if args.phase in ("all", "eval-only"):
        adapters = discover_all_adapters()

        print(f"\n{'#' * 70}")
        print(f"  GSM8K EVALUATION (vLLM)  |  {len(adapters)} adapters + base  |  "
              f"{len(gpus)} GPUs")
        print(f"  Limit: {'full (1319)' if args.eval_limit == 0 else args.eval_limit}")
        print(f"{'#' * 70}")

        t0 = time.time()
        results = run_parallel_eval(
            adapters, gpus, args.eval_batch_size, args.eval_limit)
        elapsed = time.time() - t0
        print(f"\nEvaluation completed in {elapsed/60:.1f} min")

        print_summary(results)
        generate_report(
            results,
            DOCUMENTS_DIR / "LoRA_gsm8k_eval_results.md",
        )

    # ---- Summary only ----
    if args.phase == "summary-only":
        results_file = EVAL_ROOT / "all_results.json"
        if not results_file.exists():
            print("No results found. Run evaluation first.")
            return
        results = {r["name"]: r for r in json.loads(results_file.read_text())}
        print_summary(results)
        generate_report(
            results,
            DOCUMENTS_DIR / "LoRA_gsm8k_eval_results.md",
        )

    total_elapsed = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"ALL DONE! Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.1f} hr)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
