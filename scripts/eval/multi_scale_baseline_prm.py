#!/usr/bin/env python3
"""
Multi-scale baseline: generate MATH-500 solutions + PRM scoring for Qwen2.5
family {0.5B, 1.5B, 3B, 7B, 14B} Instruct, plus optional Base variants.

Phase 1: Generate baselines via lm-evaluation-harness (one model per GPU).
Phase 2: PRM-score all baselines with Qwen2.5-Math-PRM-7B.
Phase 3: Plot per-step PRM curves per model + overlay comparison.

Usage:
    # Full pipeline (generate + score + plot)
    python 22_multi_scale_baseline.py --gpus 0,1,2

    # Score + plot only (if baselines already generated)
    python 22_multi_scale_baseline.py --skip-generation --gpus 0,1

    # Plot only (if PRM scores already computed)
    python 22_multi_scale_baseline.py --only-plot

    # Specific models only
    python 22_multi_scale_baseline.py --models 0.5B,1.5B,3B --gpus 0,1,2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.prm.scoring import (
    PRMSampleLite,
    StepScorer,
    plot_per_step_avg_correctness,
    print_avg_score_per_step,
    split_steps,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
HARNESS_DIR = PROJECT_ROOT / "lm-evaluation-harness"

MODEL_REGISTRY = {
    "0.5B":  {"id": "Qwen/Qwen2.5-0.5B-Instruct",  "layers": 25, "family": "instruct"},
    "1.5B":  {"id": "Qwen/Qwen2.5-1.5B-Instruct",  "layers": 29, "family": "instruct"},
    "3B":    {"id": "Qwen/Qwen2.5-3B-Instruct",     "layers": 37, "family": "instruct"},
    "7B":    {"id": "Qwen/Qwen2.5-7B-Instruct",     "layers": 28, "family": "instruct"},
    "14B":   {"id": "Qwen/Qwen2.5-14B-Instruct",    "layers": 48, "family": "instruct"},
    "0.5B-base": {"id": "Qwen/Qwen2.5-0.5B",        "layers": 25, "family": "base"},
    "1.5B-base": {"id": "Qwen/Qwen2.5-1.5B",        "layers": 29, "family": "base"},
    "3B-base":   {"id": "Qwen/Qwen2.5-3B",          "layers": 37, "family": "base"},
    "7B-base":   {"id": "Qwen/Qwen2.5-7B",          "layers": 28, "family": "base"},
}

PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
PRM_DTYPE = "float16"
TASK = "hendrycks_math_500"
STEP_SPLIT = "auto"
PRM_THRESHOLD = 0.72


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-scale baseline + PRM profiling")
    ap.add_argument("--models", default="0.5B,1.5B,3B,7B",
                    help="Comma-separated model tags from MODEL_REGISTRY")
    ap.add_argument("--gpus", default="auto",
                    help="Comma-separated GPU ids, or 'auto' to detect free GPUs")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--runs-root", default=str(PROJECT_ROOT / "runs" / "multi_scale_baselines"))
    ap.add_argument("--prm-out-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_prm"))
    ap.add_argument("--batch-size", default="16")
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--gen-kwargs", default="max_gen_toks=2048,temperature=0,do_sample=False")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--prm-threshold", type=float, default=PRM_THRESHOLD)
    ap.add_argument("--skip-generation", action="store_true",
                    help="Skip lm-eval generation, go straight to PRM scoring")
    ap.add_argument("--only-plot", action="store_true",
                    help="Skip generation and PRM scoring, only plot from existing results")
    ap.add_argument("--min-free-mem-mb", type=int, default=15000,
                    help="Minimum free GPU memory (MB) to consider a GPU available")
    return ap.parse_args()


def query_gpu_free_mem() -> Dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return {i: int(line.strip()) for i, line in enumerate(out.strip().splitlines()) if line.strip()}
    except Exception as e:
        print(f"[WARN] nvidia-smi failed: {e}")
        return {}


def select_gpus(requested: str, min_free_mb: int) -> List[int]:
    free_mem = query_gpu_free_mem()
    if not free_mem:
        print("[WARN] Cannot query GPUs; defaulting to [0]")
        return [0]

    if requested == "auto":
        available = sorted(
            [gid for gid, mem in free_mem.items() if mem >= min_free_mb],
            key=lambda g: -free_mem[g],
        )
        if not available:
            print(f"[WARN] No GPU with >= {min_free_mb}MB free. Using GPU with most free memory.")
            available = [max(free_mem, key=free_mem.get)]
        print(f"[GPU Auto-select] Available GPUs (>={min_free_mb}MB free): {available}")
        for gid in available:
            print(f"  GPU {gid}: {free_mem[gid]}MB free")
        return available

    requested_ids = [int(x.strip()) for x in requested.split(",") if x.strip()]
    usable = [g for g in requested_ids if free_mem.get(g, 0) >= min_free_mb]
    if not usable:
        print(f"[WARN] None of requested GPUs {requested_ids} have enough memory. Using all requested.")
        return requested_ids
    return usable


def estimate_gpu_need(model_tag: str) -> int:
    """Estimate how many GPUs a model needs."""
    if "14B" in model_tag:
        return 2
    return 1


def find_samples_jsonl(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.rglob("samples_*.jsonl"))
    return cands[-1] if cands else None


# ── Phase 1: Generate baselines ──


def run_baseline_generation(
    model_id: str,
    model_tag: str,
    gpu_id: int,
    args,
) -> Path:
    """Run lm-eval for one model on one GPU. Returns output directory."""
    out_dir = Path(args.runs_root) / f"baseline_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = find_samples_jsonl(out_dir)
    if existing:
        print(f"[SKIP] {model_tag}: samples already exist at {existing}")
        return out_dir

    python_bin = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        python_bin, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_id},dtype={args.dtype}",
        "--tasks", args.task,
        "--device", "cuda:0",
        "--num_fewshot", "0",
        "--batch_size", args.batch_size,
        "--gen_kwargs", args.gen_kwargs,
        "--output_path", str(out_dir),
        "--log_samples",
    ]
    if MODEL_REGISTRY[model_tag].get("family") == "instruct":
        cmd.append("--apply_chat_template")
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    log_file = out_dir / "generation.log"
    print(f"[GEN] {model_tag} on GPU {gpu_id} -> {out_dir}")
    print(f"[CMD] {' '.join(cmd)}")

    with log_file.open("w") as f:
        proc = subprocess.run(
            cmd, cwd=str(HARNESS_DIR), env=env,
            stdout=f, stderr=subprocess.STDOUT, text=True,
        )

    if proc.returncode != 0:
        print(f"[FAIL] {model_tag} generation failed (rc={proc.returncode}). See {log_file}")
    else:
        print(f"[OK] {model_tag} generation done.")

    return out_dir


def phase1_generate(model_tags: List[str], gpus: List[int], args):
    """Generate baselines for all models, scheduling across available GPUs."""
    if args.skip_generation or args.only_plot:
        print("[SKIP] Phase 1: generation skipped by flag")
        return

    import queue
    from threading import Thread

    job_q: queue.Queue = queue.Queue()
    for tag in model_tags:
        job_q.put(tag)

    def worker(gpu_id: int):
        while True:
            try:
                tag = job_q.get_nowait()
            except queue.Empty:
                return
            info = MODEL_REGISTRY[tag]
            gpu_need = estimate_gpu_need(tag)
            if gpu_need > 1:
                print(f"[SKIP] {tag} needs {gpu_need} GPUs; run separately with tensor parallelism")
                continue
            run_baseline_generation(info["id"], tag, gpu_id, args)

    threads = []
    for gid in gpus:
        t = Thread(target=worker, args=(gid,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("[Phase 1] All generation jobs done.\n")


# ── Phase 2: PRM scoring ──


def prm_score_one_model(
    gpu_id: int,
    model_tag: str,
    runs_root: Path,
    prm_out_dir: Path,
):
    """Score one model's baseline outputs with PRM."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer

    run_dir = runs_root / f"baseline_{model_tag}"
    jsonl_path = find_samples_jsonl(run_dir)
    if jsonl_path is None:
        print(f"[PRM SKIP] {model_tag}: no samples found in {run_dir}")
        return

    out_file = prm_out_dir / f"prm_{model_tag}.json"
    if out_file.exists():
        print(f"[PRM SKIP] {model_tag}: results already exist at {out_file}")
        return

    print(f"[PRM] {model_tag} on GPU {gpu_id}: loading PRM model...")
    scorer = StepScorer(PRM_MODEL, PRM_DTYPE)

    model_id = MODEL_REGISTRY[model_tag]["id"]
    gen_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

    data = [json.loads(line) for line in open(jsonl_path) if line.strip()]
    samples_out = []

    from tqdm import tqdm
    for idx, d in enumerate(tqdm(data, desc=f"PRM {model_tag}", unit="s")):
        if d.get("filter") == "strict-match":
            continue

        cot = ""
        fr = d.get("filtered_resps", [])
        if isinstance(fr, list) and fr:
            cot = (fr[0] or "").strip()
        if not cot:
            rs = d.get("resps", [])
            if rs and isinstance(rs[0], list) and rs[0]:
                cot = (rs[0][0] or "").strip()
        if not cot:
            continue

        steps = split_steps(cot, mode=STEP_SPLIT)
        if not steps:
            continue

        try:
            query = d.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
            if not query:
                query = d.get("doc", {}).get("problem", "")
            scores = scorer.score_steps(query, steps)
        except Exception as e:
            print(f"[PRM WARN] {model_tag} sample {idx}: {e}")
            continue

        if len(scores) != len(steps):
            continue

        step_tok_lens = [len(gen_tok.encode(s, add_special_tokens=False)) for s in steps]
        samples_out.append({
            "doc_id": d.get("doc_id", -1),
            "exact_match": int(d.get("exact_match", 0)),
            "step_scores": scores,
            "steps_text": steps,
            "step_token_len": step_tok_lens,
            "n_steps": len(steps),
        })

    result = {
        "model_tag": model_tag,
        "model_id": model_id,
        "prm_model": PRM_MODEL,
        "n_samples": len(samples_out),
        "n_correct": sum(1 for s in samples_out if s["exact_match"] >= 1),
        "n_wrong": sum(1 for s in samples_out if s["exact_match"] < 1),
        "accuracy": sum(s["exact_match"] for s in samples_out) / max(len(samples_out), 1),
        "samples": samples_out,
    }

    os.makedirs(str(prm_out_dir), exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PRM OK] {model_tag}: {len(samples_out)} samples -> {out_file}")


def phase2_prm_score(model_tags: List[str], gpus: List[int], args):
    """PRM-score all models. PRM model is heavy, so one GPU at a time per model."""
    if args.only_plot:
        print("[SKIP] Phase 2: PRM scoring skipped by flag")
        return

    prm_out = Path(args.prm_out_dir)
    os.makedirs(str(prm_out), exist_ok=True)
    runs_root = Path(args.runs_root)

    import queue
    from threading import Thread

    job_q: queue.Queue = queue.Queue()
    for tag in model_tags:
        job_q.put(tag)

    def worker(gpu_id: int):
        while True:
            try:
                tag = job_q.get_nowait()
            except queue.Empty:
                return
            prm_score_one_model(gpu_id, tag, runs_root, prm_out)

    # PRM-7B needs a full GPU, so use one GPU per worker
    threads = []
    for gid in gpus[:max(1, len(gpus) // 2)]:
        t = Thread(target=worker, args=(gid,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("[Phase 2] PRM scoring done.\n")


# ── Phase 3: Plots and analysis ──


def _avg_curve(samples: list) -> Tuple[List[int], List[float]]:
    if not samples:
        return [], []
    max_steps = max((len(s["step_scores"]) for s in samples), default=0)
    xs, ys = [], []
    for i in range(max_steps):
        vals = [s["step_scores"][i] for s in samples if i < len(s["step_scores"])]
        if vals:
            xs.append(i + 1)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def compute_mode_stability_metrics(samples: list, threshold: float = 0.85) -> Dict[str, Any]:
    """Extract quantitative mode stability metrics from PRM scores."""
    wrong = [s for s in samples if s["exact_match"] < 1]
    correct = [s for s in samples if s["exact_match"] >= 1]

    def _metrics_for_group(group, label):
        xs, ys = _avg_curve(group)
        if len(ys) < 2:
            return {f"{label}_dip_depth": None, f"{label}_dip_onset": None,
                    f"{label}_recovery_step": None, f"{label}_stability_ratio": None}

        step1_score = ys[0]
        min_score = min(ys[:min(10, len(ys))])
        min_idx = ys.index(min_score)

        dip_depth = step1_score - min_score
        dip_onset = None
        for i, y in enumerate(ys):
            if y < threshold:
                dip_onset = i + 1
                break

        recovery_step = None
        if min_idx < len(ys) - 1:
            for i in range(min_idx + 1, len(ys)):
                if ys[i] >= threshold:
                    recovery_step = i + 1
                    break

        late_scores = ys[8:] if len(ys) > 8 else ys[len(ys)//2:]
        stability_ratio = (sum(late_scores) / len(late_scores)) / step1_score if step1_score > 0 and late_scores else None

        return {
            f"{label}_dip_depth": round(dip_depth, 4),
            f"{label}_dip_onset": dip_onset,
            f"{label}_recovery_step": recovery_step,
            f"{label}_stability_ratio": round(stability_ratio, 4) if stability_ratio else None,
            f"{label}_min_score": round(min_score, 4),
            f"{label}_min_step": min_idx + 1,
            f"{label}_step1_score": round(step1_score, 4),
        }

    metrics = {
        "n_samples": len(samples),
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "accuracy": round(len(correct) / max(len(samples), 1), 4),
    }
    metrics.update(_metrics_for_group(wrong, "wrong"))
    metrics.update(_metrics_for_group(correct, "correct"))
    metrics.update(_metrics_for_group(samples, "all"))
    return metrics


def plot_single_model(
    samples: list,
    model_tag: str,
    out_dir: Path,
    threshold: float,
):
    """Generate the 3-line per-step PRM plot for one model."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_s = samples
    correct = [s for s in samples if s["exact_match"] >= 1]
    wrong = [s for s in samples if s["exact_match"] < 1]

    xa, ya = _avg_curve(all_s)
    xc, yc = _avg_curve(correct)
    xw, yw = _avg_curve(wrong)
    max_x = max([0] + xa + xc + xw)
    if max_x <= 0:
        return

    model_id = MODEL_REGISTRY.get(model_tag, {}).get("id", model_tag)

    plt.figure(figsize=(10, 6))
    if xa:
        plt.plot(xa, ya, marker="o", linewidth=2, label=f"All (n@1={len(all_s)})")
    if xc:
        plt.plot(xc, yc, marker="s", linewidth=2, label=f"Correct (n@1={len(correct)})")
    if xw:
        plt.plot(xw, yw, marker="x", linewidth=2, label=f"Wrong (n@1={len(wrong)})")
    plt.axhline(y=threshold, linestyle="--", linewidth=1.5, label=f"thr={threshold:.2f}")
    plt.xlabel("Step k")
    plt.ylabel("Avg PRM Step Score (across samples)")
    plt.title(f"Per-step Avg Correctness\n{model_id} | L=L1 | λ=BASELINE")
    plt.xlim(1, max_x)
    plt.ylim(0.55, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"per_step_prm_{model_tag}.png", dpi=180)
    plt.close()


def plot_cross_scale_overlay(
    all_results: Dict[str, dict],
    out_dir: Path,
    threshold: float,
):
    """Overlay wrong-answer PRM curves for all model sizes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_results)))

    # Wrong-answer overlay
    plt.figure(figsize=(12, 7))
    for idx, (tag, result) in enumerate(sorted(all_results.items(), key=lambda x: _model_params(x[0]))):
        wrong = [s for s in result["samples"] if s["exact_match"] < 1]
        xw, yw = _avg_curve(wrong)
        if xw:
            model_id = result.get("model_id", tag)
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            n_wrong = len(wrong)
            acc = result.get("accuracy", 0)
            plt.plot(xw, yw, marker="o", linewidth=2, color=COLORS[idx],
                     label=f"{short} (acc={acc:.2f}, n_wrong={n_wrong})")

    plt.axhline(y=threshold, linestyle="--", linewidth=1.5, color="gray",
                alpha=0.7, label=f"thr={threshold:.2f}")
    plt.xlabel("Step k", fontsize=13)
    plt.ylabel("Avg PRM Step Score (wrong-answer samples)", fontsize=13)
    plt.title("Per-step PRM Score: Wrong Answers Across Model Scales\nQwen2.5-Instruct Family", fontsize=14)
    plt.ylim(0.5, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "cross_scale_wrong_overlay.png", dpi=200)
    plt.close()

    # All-answer overlay
    plt.figure(figsize=(12, 7))
    for idx, (tag, result) in enumerate(sorted(all_results.items(), key=lambda x: _model_params(x[0]))):
        xa, ya = _avg_curve(result["samples"])
        if xa:
            model_id = result.get("model_id", tag)
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            plt.plot(xa, ya, marker="o", linewidth=2, color=COLORS[idx],
                     label=f"{short} (n={len(result['samples'])})")

    plt.axhline(y=threshold, linestyle="--", linewidth=1.5, color="gray", alpha=0.7)
    plt.xlabel("Step k", fontsize=13)
    plt.ylabel("Avg PRM Step Score (all samples)", fontsize=13)
    plt.title("Per-step PRM Score: All Samples Across Model Scales\nQwen2.5-Instruct Family", fontsize=14)
    plt.ylim(0.5, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "cross_scale_all_overlay.png", dpi=200)
    plt.close()


def plot_scaling_summary(
    metrics_by_model: Dict[str, dict],
    out_dir: Path,
):
    """Plot mode stability metrics vs model size."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = sorted(metrics_by_model.keys(), key=_model_params)
    params = [_model_params(t) for t in tags]

    metric_keys = [
        ("wrong_dip_depth", "Dip Depth (wrong)", "lower = more stable"),
        ("wrong_min_score", "Min PRM Score (wrong)", "higher = more stable"),
        ("wrong_stability_ratio", "Stability Ratio (wrong)", "closer to 1 = more stable"),
    ]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, (key, title, note) in zip(axes, metric_keys):
        vals = []
        valid_params = []
        valid_tags = []
        for tag, p in zip(tags, params):
            v = metrics_by_model[tag].get(key)
            if v is not None:
                vals.append(v)
                valid_params.append(p)
                valid_tags.append(tag)

        if not vals:
            ax.set_title(f"{title}\n(no data)")
            continue

        ax.plot(valid_params, vals, marker="o", linewidth=2, markersize=8)
        for p, v, t in zip(valid_params, vals, valid_tags):
            ax.annotate(t, (p, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title}\n({note})", fontsize=11)
        ax.grid(alpha=0.3)

    plt.suptitle("Mode Stability Scaling: Qwen2.5-Instruct Family", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "scaling_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def _model_params(tag: str) -> float:
    """Extract approximate parameter count for sorting."""
    tag_lower = tag.lower().replace("-base", "")
    for size, val in [("0.5b", 5e8), ("1.5b", 1.5e9), ("3b", 3e9),
                      ("7b", 7e9), ("14b", 14e9), ("32b", 32e9)]:
        if size in tag_lower:
            return val
    return 1e9


def phase3_plot(model_tags: List[str], args):
    """Load PRM results and generate all plots + metrics."""
    prm_out = Path(args.prm_out_dir)
    plot_dir = prm_out / "plots"
    os.makedirs(str(plot_dir), exist_ok=True)

    all_results: Dict[str, dict] = {}
    metrics_by_model: Dict[str, dict] = {}

    for tag in model_tags:
        prm_file = prm_out / f"prm_{tag}.json"
        if not prm_file.exists():
            print(f"[PLOT SKIP] {tag}: no PRM results at {prm_file}")
            continue

        result = json.loads(prm_file.read_text(encoding="utf-8"))
        all_results[tag] = result

        plot_single_model(result["samples"], tag, plot_dir / tag, args.prm_threshold)
        metrics = compute_mode_stability_metrics(result["samples"], threshold=0.85)
        metrics["model_tag"] = tag
        metrics["model_id"] = result.get("model_id", "")
        metrics_by_model[tag] = metrics

        print(f"\n=== {tag} ({result.get('model_id', '')}) ===")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['n_correct']}/{metrics['n_samples']})")
        print(f"  Wrong samples: {metrics['n_wrong']}")
        if metrics.get("wrong_dip_depth") is not None:
            print(f"  Dip depth (wrong): {metrics['wrong_dip_depth']:.4f}")
            print(f"  Min score (wrong): {metrics['wrong_min_score']:.4f} at step {metrics['wrong_min_step']}")
            print(f"  Stability ratio (wrong): {metrics.get('wrong_stability_ratio', 'N/A')}")

    if len(all_results) >= 2:
        plot_cross_scale_overlay(all_results, plot_dir, args.prm_threshold)
        plot_scaling_summary(metrics_by_model, plot_dir)

    summary = {
        "models": metrics_by_model,
        "prm_model": PRM_MODEL,
        "task": args.task,
        "threshold": args.prm_threshold,
        "generated_at": datetime.now().isoformat(),
    }
    summary_path = prm_out / "multi_scale_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[Summary] Saved to {summary_path}")
    print(f"[Plots] Saved to {plot_dir}")


def main():
    args = parse_args()

    model_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    invalid = [t for t in model_tags if t not in MODEL_REGISTRY]
    if invalid:
        print(f"[ERROR] Unknown model tags: {invalid}")
        print(f"  Available: {list(MODEL_REGISTRY.keys())}")
        return

    print("=" * 60)
    print("  Multi-Scale Baseline + PRM Profiling")
    print("=" * 60)
    print(f"Models: {model_tags}")
    print(f"Task: {args.task}")
    print(f"Runs root: {args.runs_root}")
    print(f"PRM output: {args.prm_out_dir}")
    print()

    gpus = select_gpus(args.gpus, args.min_free_mem_mb)
    print(f"Selected GPUs: {gpus}\n")

    phase1_generate(model_tags, gpus, args)
    phase2_prm_score(model_tags, gpus, args)
    phase3_plot(model_tags, args)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
