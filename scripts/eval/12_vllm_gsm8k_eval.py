#!/usr/bin/env python3
"""
Parallel GSM8K evaluation for all LoRA adapters + base using vLLM backend.

Uses lm-eval with vLLM for ~10x faster generation than HF backend.
Runs evaluations in parallel across multiple GPUs.
Produces a comprehensive Markdown report.

Usage:
    python 12_vllm_gsm8k_eval.py --gpus 0,1,2,3
    python 12_vllm_gsm8k_eval.py --gpus 0,1,2,3 --limit 50   # smoke test
    python 12_vllm_gsm8k_eval.py --summary-only                # regenerate report
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
HARNESS_DIR = PROJECT_ROOT / "lm-evaluation-harness"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
SWEEP_ROOT = BEEGFS_ARTIFACTS / "lora_sweep"
EVAL_ROOT = SWEEP_ROOT / "_gsm8k_vllm_eval"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PYTHON_BIN = "/mnt/beegfs/youyang7/.conda/envs/fact/bin/python"
TASK = "gsm8k_cot_zeroshot_unified"

VLLM_COMMON_ARGS = (
    "dtype=bfloat16,"
    "gpu_memory_utilization=0.9,"
    "max_model_len=3072,"
    "max_num_seqs=64"
)


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


def extract_all_metrics(results_json: Path) -> Dict[str, Any]:
    obj = json.loads(results_json.read_text())
    task_block = (obj.get("results") or {}).get(TASK, {})
    return {k: v for k, v in task_block.items() if isinstance(v, (int, float))}


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
    proc._log_fh = fh
    proc._output_path = output_path
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
        phase = "v2" if d.name.startswith("v2_") else "v1"
        adapters.append({
            "name": d.name,
            "adapter_path": str(adapter_path),
            "phase": phase,
            **training,
        })

    ds2 = BEEGFS_ARTIFACTS / "lora_qwen25_3b_ds2_fix_step2" / "final_adapter"
    if ds2.exists():
        adapters.append({
            "name": "ds2_fix_step2",
            "adapter_path": str(ds2),
            "phase": "standalone",
            "r": 16, "alpha": 32, "dropout": 0.05,
        })

    return adapters


def save_incremental(path: Path, results: Dict[str, Dict]):
    path.write_text(json.dumps(list(results.values()), indent=2, default=str))


def run_parallel_eval(adapters: List[Dict], gpus: List[int],
                      batch_size: str, limit: int) -> Dict[str, Dict]:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    incremental_file = EVAL_ROOT / "all_results.json"

    done: Dict[str, Dict] = {}
    if incremental_file.exists():
        for entry in json.loads(incremental_file.read_text()):
            if entry.get("gsm8k_em") is not None:
                done[entry["name"]] = entry

    # --- Base model (single GPU, must finish first) ---
    if "base" not in done:
        print(f"\n[BASE] Evaluating base model on GPU {gpus[0]} ...")
        t0 = time.time()
        model_args = f"pretrained={MODEL_ID},{VLLM_COMMON_ARGS}"
        proc = run_vllm_eval(model_args, EVAL_ROOT / "base", gpus[0], batch_size, limit)
        proc.wait()
        proc._log_fh.close()
        elapsed = time.time() - t0

        rj = find_latest_results(EVAL_ROOT / "base")
        em = extract_exact_match(rj) if rj else None
        done["base"] = {
            "name": "base", "gsm8k_em": em,
            "elapsed_sec": round(elapsed, 1),
        }
        save_incremental(incremental_file, done)
        print(f"  Base exact_match = {em}  ({elapsed:.0f}s)")
    else:
        print(f"[BASE] Already done: exact_match = {done['base'].get('gsm8k_em')}")

    # --- Adapters in parallel ---
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
              f"{[a['name'] for a in batch]}")
        print(f"{'=' * 70}")

        procs = []
        for i, adapter in enumerate(batch):
            gpu = gpus[i % n_gpus]
            name = adapter["name"]
            model_args = (
                f"pretrained={MODEL_ID},"
                f"lora_local_path={adapter['adapter_path']},"
                f"max_lora_rank=64,{VLLM_COMMON_ARGS}"
            )
            print(f"  Launching {name} on GPU {gpu} ...")
            proc = run_vllm_eval(
                model_args, EVAL_ROOT / name, gpu, batch_size, limit)
            procs.append((adapter, proc, time.time()))
            time.sleep(3)

        for adapter, proc, t0 in procs:
            proc.wait()
            proc._log_fh.close()
            elapsed = time.time() - t0
            name = adapter["name"]

            rj = find_latest_results(EVAL_ROOT / name)
            em = extract_exact_match(rj) if rj else None
            status = f"EM = {em:.4f}" if em is not None else "FAILED"
            print(f"  {name}: {status}  ({elapsed:.0f}s)")

            done[name] = {
                "name": name,
                "gsm8k_em": em,
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
                "train_samples": adapter.get("train_samples"),
            }

        save_incremental(incremental_file, done)
        remaining = len(to_eval) - (batch_start + len(batch))
        print(f"  Saved. Remaining: {remaining} adapters")

    return done


# ============================================================
# Report generation
# ============================================================

def generate_report(results: Dict[str, Dict], output_path: Path):
    base_em = results.get("base", {}).get("gsm8k_em")
    adapter_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )
    ok = [(n, r) for n, r in adapter_results if r.get("gsm8k_em") is not None]

    lines = [
        "# GSM8K LoRA Sweep — Comprehensive Evaluation Results",
        "",
        f"**Base Model**: `{MODEL_ID}`",
        f"**Task**: `{TASK}` (GSM8K zero-shot CoT, 1319 test samples)",
        f"**Training Data**: 109 train / 6 eval samples (GSM8K step-2 GPT-prefilled corrections)",
        f"**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## Base Model: GSM8K EM = **{base_em:.4f}**" if base_em else "## Base Model: N/A",
        "",
    ]

    # --- Main results table ---
    lines.extend([
        "## Full Results (Ranked by GSM8K Exact Match)",
        "",
        "| Rank | Name | Phase | GSM8K EM | Delta | Eval Loss | Best EL "
        "| Train Loss | LR | r | Alpha | Epochs | Drop | WD |",
        "|------|------|-------|----------|-------|-----------|--------"
        "|------------|-----|---|-------|--------|------|-----|",
    ])

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

    # --- Key findings ---
    if ok and base_em is not None:
        best_name, best_r = ok[0]
        worst_name, worst_r = ok[-1]
        improved = [x for x in ok if x[1]["gsm8k_em"] > base_em]

        lines.extend([
            "",
            "## Key Findings",
            "",
            f"- **Best adapter**: `{best_name}` "
            f"(EM = {best_r['gsm8k_em']:.4f}, "
            f"delta = {best_r['gsm8k_em'] - base_em:+.4f})",
            f"- **Worst adapter**: `{worst_name}` "
            f"(EM = {worst_r['gsm8k_em']:.4f}, "
            f"delta = {worst_r['gsm8k_em'] - base_em:+.4f})",
            f"- **{len(improved)}/{len(ok)}** adapters improved over base",
        ])

        p1 = [(n, r) for n, r in ok if r.get("phase") == "v1"]
        p2 = [(n, r) for n, r in ok if r.get("phase") == "v2"]
        if p1:
            best_p1 = max(p1, key=lambda x: x[1]["gsm8k_em"])
            lines.append(f"- Best Phase 1: `{best_p1[0]}` "
                         f"(EM = {best_p1[1]['gsm8k_em']:.4f})")
        if p2:
            best_p2 = max(p2, key=lambda x: x[1]["gsm8k_em"])
            lines.append(f"- Best Phase 2: `{best_p2[0]}` "
                         f"(EM = {best_p2[1]['gsm8k_em']:.4f})")

    # --- Hyperparameter analysis ---
    lines.extend(["", "## Hyperparameter Analysis", ""])

    # LR analysis
    lr_groups: Dict[str, List] = {}
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

    # Rank analysis
    rank_groups: Dict[int, List] = {}
    for n, r in ok:
        rv = r.get("r")
        if rv is not None:
            rank_groups.setdefault(int(rv), []).append(r["gsm8k_em"])

    if rank_groups:
        lines.extend(["", "### LoRA Rank", "",
                       "| Rank | Count | Mean EM | Best EM |",
                       "|------|-------|---------|---------|"])
        for rk in sorted(rank_groups.keys()):
            vals = rank_groups[rk]
            lines.append(f"| {rk} | {len(vals)} | {sum(vals)/len(vals):.4f} "
                         f"| {max(vals):.4f} |")

    # Epochs analysis
    ep_groups: Dict[float, List] = {}
    for n, r in ok:
        ep = r.get("epochs")
        if ep is not None:
            ep_groups.setdefault(float(ep), []).append(r["gsm8k_em"])

    if ep_groups:
        lines.extend(["", "### Training Epochs", "",
                       "| Epochs | Count | Mean EM | Best EM |",
                       "|--------|-------|---------|---------|"])
        for ep in sorted(ep_groups.keys()):
            vals = ep_groups[ep]
            lines.append(f"| {ep:.0f} | {len(vals)} | {sum(vals)/len(vals):.4f} "
                         f"| {max(vals):.4f} |")

    # Eval loss vs GSM8K EM correlation
    lines.extend(["", "## Eval Loss vs GSM8K EM (Top 10)", "",
                   "| Rank | Name | Eval Loss | Best EL | GSM8K EM | Delta |",
                   "|------|------|-----------|---------|----------|-------|"])
    for i, (n, r) in enumerate(ok[:10]):
        el = f"{r.get('eval_loss'):.4f}" if r.get('eval_loss') is not None else "—"
        bel = f"{r.get('best_eval_loss'):.4f}" if r.get('best_eval_loss') is not None else "—"
        delta = r["gsm8k_em"] - base_em if base_em else 0
        lines.append(f"| {i+1} | {n} | {el} | {bel} | {r['gsm8k_em']:.4f} | {delta:+.4f} |")

    # Overfitting analysis
    lines.extend(["", "## Overfitting Analysis", "",
                   "| Name | Train Loss | Eval Loss | Gap | GSM8K EM |",
                   "|------|-----------|-----------|-----|----------|"])
    for n, r in adapter_results:
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
    adapter_results = sorted(
        [(n, r) for n, r in results.items() if n != "base"],
        key=lambda x: -(x[1].get("gsm8k_em") or -1)
    )

    print(f"\n{'=' * 100}")
    print(f"{'GSM8K EVALUATION RESULTS (vLLM)':^100}")
    print(f"{'=' * 100}")
    print(f"\nBase model: {MODEL_ID}  |  GSM8K EM = {base_em}")
    print(f"\n{'Rk':<4} {'Name':<28} {'Ph':<4} {'GSM8K EM':<10} "
          f"{'Delta':<8} {'EvalLoss':<10} {'LR':<8} {'r':<4} {'α':<5} {'Ep':<5}")
    print("-" * 100)

    for i, (name, r) in enumerate(adapter_results):
        em = r.get("gsm8k_em")
        em_str = f"{em:.4f}" if em is not None else "FAIL"
        delta_str = f"{em - base_em:+.4f}" if (em and base_em) else "N/A"
        el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "—"
        lr_val = r.get("lr")
        lr_str = f"{lr_val:.0e}" if lr_val is not None else "—"
        ph = r.get("phase", "—")[:3]
        print(f"{i+1:<4} {name:<28} {ph:<4} {em_str:<10} {delta_str:<8} "
              f"{el:<10} {lr_str:<8} {str(r.get('r', '—')):<4} "
              f"{str(r.get('alpha', '—')):<5} {str(r.get('epochs', '—')):<5}")

    ok = [(n, r) for n, r in adapter_results if r.get("gsm8k_em") is not None]
    if ok and base_em:
        improved = sum(1 for _, r in ok if r["gsm8k_em"] > base_em)
        print(f"\n{improved}/{len(ok)} adapters beat base model")
        best = ok[0]
        print(f"Best: {best[0]} (EM={best[1]['gsm8k_em']:.4f}, "
              f"delta={best[1]['gsm8k_em'] - base_em:+.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit samples (0=full 1319)")
    ap.add_argument("--summary-only", action="store_true",
                    help="Only regenerate report from existing results")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]

    if args.summary_only:
        results_file = EVAL_ROOT / "all_results.json"
        if not results_file.exists():
            print("No results found. Run evaluation first.")
            return
        results = {r["name"]: r for r in json.loads(results_file.read_text())}
    else:
        adapters = discover_all_adapters()
        total = len(adapters) + 1

        print(f"\n{'#' * 70}")
        print(f"  GSM8K EVALUATION (vLLM)  |  {total} models  |  {len(gpus)} GPUs")
        print(f"  Limit: {'full (1319)' if args.limit == 0 else args.limit}")
        print(f"{'#' * 70}")

        t0 = time.time()
        results = run_parallel_eval(adapters, gpus, args.batch_size, args.limit)
        elapsed = time.time() - t0

        print(f"\n{'=' * 70}")
        print(f"EVALUATION COMPLETED in {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
        print(f"{'=' * 70}")

    print_summary(results)

    report_path = DOCUMENTS_DIR / "gsm8k_eval_results.md"
    generate_report(results, report_path)

    full_json = EVAL_ROOT / "all_results.json"
    if not args.summary_only:
        full_json.write_text(json.dumps(
            list(results.values()), indent=2, default=str))
    print(f"Full JSON: {full_json}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
