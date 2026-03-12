#!/usr/bin/env python3
"""
Batch GSM8K evaluation for all trained LoRA adapters and the base model.

Evaluates sequentially on a single GPU, saves incremental results,
and produces a final Markdown summary table.
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
HARNESS_DIR = PROJECT_ROOT / "lm-evaluation-harness"
BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
SWEEP_ROOT = BEEGFS_ARTIFACTS / "lora_sweep"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PYTHON_BIN = "/mnt/beegfs/youyang7/.conda/envs/fact/bin/python"
TASK = "gsm8k_cot_zeroshot_unified"


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


def extract_all_metrics(results_json: Path) -> Dict[str, Any]:
    obj = json.loads(results_json.read_text())
    task_block = (obj.get("results") or {}).get(TASK, {})
    return {k: v for k, v in task_block.items() if isinstance(v, (int, float))}


def run_lm_eval(model_args: str, output_path: Path, gpu_id: int,
                batch_size: int = 8, limit: int = 0) -> Optional[Dict]:
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
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"

    log_file = output_path / "eval.log"
    print(f"  [CMD] {' '.join(cmd)}")

    with open(log_file, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(HARNESS_DIR), env=env)

    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode})")
        return None

    rj = find_latest_results(output_path)
    if rj is None:
        print(f"  No results JSON found")
        return None

    metrics = extract_all_metrics(rj)
    em = extract_exact_match(rj)
    print(f"  exact_match = {em}")
    return {"exact_match": em, "all_metrics": metrics, "results_path": str(rj)}


def load_training_metrics(adapter_dir: Path) -> Dict[str, Any]:
    mf = adapter_dir / "sweep_metrics.json"
    if not mf.exists():
        return {}
    m = json.loads(mf.read_text())
    cfg = m.get("config", {})
    return {
        "eval_loss": m.get("eval_metrics", {}).get("eval_loss"),
        "train_loss": m.get("train_metrics", {}).get("train_loss"),
        "lr": cfg.get("learning_rate"),
        "r": cfg.get("lora_r"),
        "alpha": cfg.get("lora_alpha"),
        "dropout": cfg.get("lora_dropout"),
        "epochs": cfg.get("num_train_epochs"),
        "wd": cfg.get("weight_decay"),
        "modules": cfg.get("target_modules"),
    }


def discover_adapters() -> List[Dict[str, Any]]:
    adapters = []

    for d in sorted(SWEEP_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        adapter_path = d / "final_adapter"
        if not adapter_path.exists():
            continue
        training = load_training_metrics(d)
        adapters.append({
            "name": d.name,
            "adapter_path": str(adapter_path),
            "source": "lora_sweep",
            **training,
        })

    ds2_adapter = BEEGFS_ARTIFACTS / "lora_qwen25_3b_ds2_fix_step2" / "final_adapter"
    if ds2_adapter.exists():
        adapters.append({
            "name": "ds2_fix_step2 (earlier)",
            "adapter_path": str(ds2_adapter),
            "source": "standalone",
            "r": 16, "alpha": 32, "dropout": 0.05,
            "modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        })

    return adapters


def generate_markdown_table(base_result: Dict, adapter_results: List[Dict],
                            output_path: Path):
    lines = []
    lines.append("# GSM8K Evaluation Results — LoRA Sweep")
    lines.append("")
    lines.append(f"**Base Model**: `{MODEL_ID}`")
    lines.append(f"**Task**: `{TASK}` (GSM8K zero-shot CoT)")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    base_em = base_result.get("exact_match")
    base_str = f"{base_em:.4f}" if base_em is not None else "N/A"
    lines.append(f"## Base Model Accuracy: **{base_str}**")
    lines.append("")

    lines.append("## Full Results Table")
    lines.append("")
    lines.append("| Rank | Name | GSM8K EM | Delta vs Base | Eval Loss | Train Loss | LR | Rank(r) | Alpha | Epochs | Dropout | WD |")
    lines.append("|------|------|----------|---------------|-----------|------------|-----|---------|-------|--------|---------|-----|")

    sorted_results = sorted(adapter_results,
                            key=lambda x: -(x.get("gsm8k_em") or -1))

    for i, r in enumerate(sorted_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        delta = (em - base_em) if (em is not None and base_em is not None) else None
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "N/A"
        tl = f"{r.get('train_loss', 0):.4f}" if r.get('train_loss') is not None else "N/A"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "N/A"
        rank = r.get("r", "N/A")
        alpha = r.get("alpha", "N/A")
        epochs = r.get("epochs", "N/A")
        dropout = r.get("dropout", "N/A")
        wd = r.get("wd", "N/A")

        lines.append(f"| {i+1} | {r['name']} | {em_str} | {delta_str} | "
                     f"{el} | {tl} | {lr_str} | {rank} | {alpha} | "
                     f"{epochs} | {dropout} | {wd} |")

    lines.append("")
    lines.append("## Key Observations")
    lines.append("")

    ok_results = [r for r in sorted_results if r.get("gsm8k_em") is not None]
    if ok_results:
        best = ok_results[0]
        worst = ok_results[-1]
        lines.append(f"- **Best adapter**: `{best['name']}` with GSM8K EM = {best['gsm8k_em']:.4f}")
        if base_em is not None:
            delta_best = best['gsm8k_em'] - base_em
            lines.append(f"  - Delta vs base: {delta_best:+.4f} ({delta_best*100:+.1f} pp)")
        lines.append(f"- **Worst adapter**: `{worst['name']}` with GSM8K EM = {worst['gsm8k_em']:.4f}")

        improved = [r for r in ok_results
                    if base_em is not None and r['gsm8k_em'] is not None
                    and r['gsm8k_em'] > base_em]
        lines.append(f"- **{len(improved)}/{len(ok_results)}** adapters improved over the base model")

    lines.append("")
    lines.append("## Overfitting Analysis")
    lines.append("")
    lines.append("| Name | Train Loss | Eval Loss | Gap (Overfit) | GSM8K EM |")
    lines.append("|------|-----------|-----------|---------------|----------|")
    for r in sorted_results:
        tl = r.get("train_loss")
        el = r.get("eval_loss")
        if tl is not None and el is not None:
            gap = el - tl
            em = r.get("gsm8k_em")
            em_str = f"{em:.4f}" if em is not None else "FAIL"
            lines.append(f"| {r['name']} | {tl:.4f} | {el:.4f} | {gap:.4f} | {em_str} |")

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown summary saved to: {output_path}")


def save_incremental(results_file: Path, entry: Dict):
    existing = []
    if results_file.exists():
        existing = json.loads(results_file.read_text())
    existing.append(entry)
    results_file.write_text(json.dumps(existing, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit GSM8K samples (0=full test set)")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--only", default="",
                    help="Comma-separated names to evaluate (empty=all)")
    args = ap.parse_args()

    eval_root = SWEEP_ROOT / "_gsm8k_batch_eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    incremental_file = eval_root / "incremental_results.json"

    already_done = set()
    if incremental_file.exists():
        for entry in json.loads(incremental_file.read_text()):
            if entry.get("gsm8k_em") is not None:
                already_done.add(entry["name"])

    adapters = discover_adapters()
    if args.only:
        names = set(args.only.split(","))
        adapters = [a for a in adapters if a["name"] in names]

    total = len(adapters) + (0 if args.skip_base else 1)
    print(f"\n{'=' * 70}")
    print(f"BATCH GSM8K EVALUATION")
    print(f"  Models to evaluate: {total} (base + {len(adapters)} adapters)")
    print(f"  GPU: {args.gpu}  |  Batch size: {args.batch_size}")
    print(f"  Limit: {'full' if args.limit == 0 else args.limit}")
    print(f"  Already done: {len(already_done)}")
    print(f"{'=' * 70}\n")

    base_result = {}

    if not args.skip_base:
        if "base" in already_done:
            print("[SKIP] Base model already evaluated")
            for entry in json.loads(incremental_file.read_text()):
                if entry["name"] == "base":
                    base_result = entry
                    break
        else:
            print(f"[1/{total}] Evaluating BASE model: {MODEL_ID}")
            t0 = time.time()
            model_args = f"pretrained={MODEL_ID},dtype=bfloat16,device=cuda"
            result = run_lm_eval(model_args, eval_root / "base",
                                 args.gpu, args.batch_size, args.limit)
            elapsed = time.time() - t0
            base_result = {
                "name": "base",
                "gsm8k_em": result["exact_match"] if result else None,
                "all_metrics": result.get("all_metrics", {}) if result else {},
                "elapsed_sec": round(elapsed, 1),
            }
            save_incremental(incremental_file, base_result)
            print(f"  Completed in {elapsed/60:.1f} min\n")

    for idx, adapter in enumerate(adapters):
        step = idx + (2 if not args.skip_base else 1)
        name = adapter["name"]

        if name in already_done:
            print(f"[{step}/{total}] [SKIP] {name} already evaluated")
            continue

        print(f"[{step}/{total}] Evaluating: {name}")
        t0 = time.time()
        model_args = (f"pretrained={MODEL_ID},"
                      f"peft={adapter['adapter_path']},"
                      f"dtype=bfloat16,device=cuda")
        eval_out = eval_root / name
        result = run_lm_eval(model_args, eval_out,
                             args.gpu, args.batch_size, args.limit)
        elapsed = time.time() - t0

        entry = {
            "name": name,
            "gsm8k_em": result["exact_match"] if result else None,
            "all_metrics": result.get("all_metrics", {}) if result else {},
            "elapsed_sec": round(elapsed, 1),
            "eval_loss": adapter.get("eval_loss"),
            "train_loss": adapter.get("train_loss"),
            "lr": adapter.get("lr"),
            "r": adapter.get("r"),
            "alpha": adapter.get("alpha"),
            "dropout": adapter.get("dropout"),
            "epochs": adapter.get("epochs"),
            "wd": adapter.get("wd"),
            "modules": adapter.get("modules"),
            "source": adapter.get("source"),
        }
        save_incremental(incremental_file, entry)
        print(f"  Completed in {elapsed/60:.1f} min\n")

    print(f"\n{'=' * 70}")
    print("GENERATING FINAL REPORT")
    print(f"{'=' * 70}")

    all_entries = json.loads(incremental_file.read_text())
    base_entry = next((e for e in all_entries if e["name"] == "base"), {})
    adapter_entries = [e for e in all_entries if e["name"] != "base"]

    generate_markdown_table(
        base_entry, adapter_entries,
        DOCUMENTS_DIR / "gsm8k_eval_results.md",
    )

    full_json = eval_root / "full_results.json"
    full_json.write_text(json.dumps(all_entries, indent=2, default=str))
    print(f"Full JSON results saved to: {full_json}")

    print("\nDone!")


if __name__ == "__main__":
    main()
