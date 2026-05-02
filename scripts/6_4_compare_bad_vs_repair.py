#!/usr/bin/env python3
"""
Experiment: Compare rollback resample vs from-scratch resample (majority vote).

Two conditions for each wrong trajectory with known first-error step tau:
  - R_rollback: keep prefix up to step tau-1 (before the error), continue N times,
                majority vote.  Reuses 6_5 (rollback_one_step) output.
  - R_scratch:  generate N fresh completions from scratch, majority vote.

Multi-GPU data-parallel for the from-scratch generation.

Usage:
    python scripts/6_4_compare_bad_vs_repair.py \
        --rollback-file results/gsm8k_3b_multi_sample/rollback_one_step/continuations.jsonl \
        --early-file results/gsm8k_3b_multi_sample/first_error/bucket_early.json \
        --late-file results/gsm8k_3b_multi_sample/first_error/bucket_late.json \
        --out-dir results/gsm8k_3b_multi_sample/rollback_vs_scratch \
        --n-continuations 32 \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# SECTION: args


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare rollback resample vs from-scratch resample (majority vote)")
    ap.add_argument("--rollback-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/rollback_one_step/continuations.jsonl"),
        help="Continuations from 6_5 (rollback one step recovery)")
    ap.add_argument("--early-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/bucket_early.json"))
    ap.add_argument("--late-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/bucket_late.json"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/rollback_vs_scratch"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/rollback_vs_scratch"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--n-continuations", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="Limit samples per bucket; 0=all")
    ap.add_argument("--skip-generation", action="store_true",
                    help="Skip from-scratch generation; use existing scratch file")
    # Internal shard worker args
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_n-shards", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()

# SECTION: helpers


def build_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_boxed_answer(text: str) -> str:
    idx = (text or "").rfind("\\boxed")
    if idx < 0:
        return ""
    i, depth, start = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            if depth == 0:
                start = i
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start + 1 : i].strip()
        i += 1
    return ""


def normalize_answer(text: str) -> str:
    text = (text or "").strip().replace("$", "").replace(",", "")
    text = re.sub(r"\\boxed\{(.*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()

# SECTION: majority_vote


def majority_vote(pred_answers: List[str]) -> str:
    """Return the most common normalized answer among predictions."""
    normed = [normalize_answer(a) for a in pred_answers if normalize_answer(a)]
    if not normed:
        return ""
    return Counter(normed).most_common(1)[0][0]


def load_samples(early_file: str, late_file: str, limit: int) -> List[Dict[str, Any]]:
    samples = []
    for path_str in [early_file, late_file]:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        data = json.loads(path.read_text("utf-8"))
        samples.extend(data)
    if limit > 0:
        early = [s for s in samples if s["bucket"] == "early"][:limit]
        late = [s for s in samples if s["bucket"] == "late"][:limit]
        samples = early + late
    return samples

# SECTION: shard_worker


def run_shard(args) -> None:
    """Generate from-scratch completions on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    task_file = Path(args._task_file)
    tasks = json.loads(task_file.read_text("utf-8"))
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: {len(tasks)} prompts")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="half",
    )

    prompts = [t["prompt"] for t in tasks]
    outputs = llm.generate(prompts, sampling_params)

    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for task, output in zip(tasks, outputs):
            full_response = output.outputs[0].text.strip()
            pred_answer = extract_boxed_answer(full_response)
            gold = task["gold_answer"]
            is_correct = float(
                normalize_answer(pred_answer) == normalize_answer(gold)
            ) if gold else 0.0
            rec = {
                "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "continuation_idx": task["continuation_idx"],
                "bucket": task["bucket"],
                "tau": task["tau"],
                "n_steps": task["n_steps"],
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "exact_match": is_correct,
                "source": "scratch",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Shard {args._shard_id}] Done -> {out_path}")

# SECTION: generation


def generate_scratch_completions(args, samples, out_dir: Path) -> Path:
    """Generate from-scratch completions via multi-GPU sharding."""
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)

    all_tasks = []
    for s in samples:
        prompt = build_chat_prompt(s["question"])
        for ci in range(args.n_continuations):
            all_tasks.append({
                "doc_id": s["doc_id"],
                "sample_idx": s["sample_idx"],
                "continuation_idx": ci,
                "bucket": s["bucket"],
                "tau": s["tau"],
                "n_steps": s["n_steps"],
                "gold_answer": s["gold_answer"],
                "prompt": prompt,
            })

    print(f"From-scratch generation: {len(all_tasks)} prompts across {n_shards} GPUs")

    shard_dir = out_dir / "_shards_scratch"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_tasks = [[] for _ in range(n_shards)]
    for i, task in enumerate(all_tasks):
        shard_tasks[i % n_shards].append(task)

    script_path = str(Path(__file__).resolve())
    procs = []
    shard_out_files = []

    for si, gpu_id in enumerate(gpu_ids):
        task_file = shard_dir / f"tasks_{si}.json"
        task_file.write_text(
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8")
        shard_out = shard_dir / f"shard_{si}.jsonl"
        shard_out_files.append(shard_out)

        cmd = [
            sys.executable, script_path,
            "--model-id", args.model_id,
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--_shard-id", str(si),
            "--_n-shards", str(n_shards),
            "--_shard-out", str(shard_out),
            "--_gpu-id", gpu_id,
            "--_task-file", str(task_file),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        print(f"  Launching shard {si} on GPU {gpu_id} "
              f"({len(shard_tasks[si])} prompts)...")
        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w", encoding="utf-8")
        p = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p, log_fh))

    failed = []
    for si, gpu_id, p, log_fh in procs:
        p.wait()
        log_fh.close()
        rc = p.returncode
        log_path = shard_dir / f"log_{si}.txt"
        output_text = log_path.read_text("utf-8", errors="replace")
        lines = output_text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"\n--- Shard {si} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        if rc != 0:
            failed.append(si)
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))
        else:
            log_path.unlink(missing_ok=True)

    if failed:
        print(f"\nERROR: Shards {failed} failed!")
        sys.exit(1)

    merged_path = out_dir / "scratch_continuations.jsonl"
    n_total = 0
    with merged_path.open("w", encoding="utf-8") as fout:
        for sf in shard_out_files:
            if sf.exists():
                for line in sf.read_text("utf-8").splitlines():
                    if line.strip():
                        fout.write(line + "\n")
                        n_total += 1

    print(f"Merged: {n_total} scratch continuations -> {merged_path}")

    for sf in shard_out_files:
        sf.unlink(missing_ok=True)
    for si in range(n_shards):
        tf = shard_dir / f"tasks_{si}.json"
        tf.unlink(missing_ok=True)
    shard_dir.rmdir()

    return merged_path

# SECTION: analysis


def compute_majority_vote_accuracy(
    rows: List[Dict], n_continuations: int
) -> Dict[tuple, Dict]:
    """Group by (doc_id, sample_idx), majority-vote, return per-sample result."""
    by_sample: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r["doc_id"], r["sample_idx"])
        by_sample[key].append(r)

    results = {}
    for key, recs in by_sample.items():
        pred_answers = [r["pred_answer"] for r in recs]
        mv_answer = majority_vote(pred_answers)
        gold = normalize_answer(recs[0]["gold_answer"])
        correct = 1.0 if (mv_answer and mv_answer == gold) else 0.0
        results[key] = {
            "mv_correct": correct,
            "mv_answer": mv_answer,
            "n_continuations": len(recs),
            "bucket": recs[0]["bucket"],
            "tau": recs[0]["tau"],
            "n_steps": recs[0]["n_steps"],
            "gold_answer": recs[0]["gold_answer"],
            "per_cont_rate": sum(
                1 for r in recs if r["exact_match"] >= 1.0) / len(recs),
        }
    return results

# SECTION: figures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats


def make_figures(paired, fig_dir: Path):
    """Generate comparison figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_bucket = defaultdict(list)
    for p in paired:
        by_bucket[p["bucket"]].append(p)

    summary = {}
    for bucket in ["early", "late"]:
        items = by_bucket[bucket]
        if not items:
            continue
        r_rep = np.array([x["mv_repair"] for x in items])
        r_scr = np.array([x["mv_scratch"] for x in items])
        n = len(items)
        summary[bucket] = {
            "n": n,
            "rollback_mean": float(np.mean(r_rep)),
            "rollback_se": float(np.std(r_rep) / np.sqrt(n)),
            "scratch_mean": float(np.mean(r_scr)),
            "scratch_se": float(np.std(r_scr) / np.sqrt(n)),
            "delta_mean": float(np.mean(r_scr - r_rep)),
            "delta_se": float(np.std(r_scr - r_rep) / np.sqrt(n)),
        }

    # Fig 1: grouped bar -- repair vs scratch by bucket
    _fig_grouped_bar(summary, fig_dir)
    # Fig 2: by tau line plot
    _fig_by_tau(paired, fig_dir)
    # Fig 3: per-sample scatter
    _fig_scatter(paired, fig_dir)

    return summary


def _fig_grouped_bar(summary, fig_dir):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    buckets = ["early", "late"]
    x = np.arange(len(buckets))
    w = 0.3

    rep_m = [summary[b]["rollback_mean"] for b in buckets]
    rep_e = [1.96 * summary[b]["rollback_se"] for b in buckets]
    scr_m = [summary[b]["scratch_mean"] for b in buckets]
    scr_e = [1.96 * summary[b]["scratch_se"] for b in buckets]

    ax.bar(x - w / 2, rep_m, w, yerr=rep_e, capsize=4,
           label="Rollback (from $\\tau$-1)", color="#e74c3c", alpha=0.85)
    ax.bar(x + w / 2, scr_m, w, yerr=scr_e, capsize=4,
           label="Scratch (from start)", color="#3498db", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["early ($\\tau$=2,3)", "late ($\\tau$=4,5,6)"])
    ax.set_ylabel("Majority-Vote Accuracy")
    ax.set_title("Rollback Resample vs From-Scratch Resample")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(rep_m + scr_m) * 1.4
    ax.set_ylim(0, min(ymax, 1.05))
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_rollback_vs_scratch.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_rollback_vs_scratch.pdf'}")

# SECTION: fig_by_tau


def _fig_by_tau(paired, fig_dir):
    by_tau = defaultdict(list)
    for p in paired:
        by_tau[p["tau"]].append(p)

    taus = sorted(by_tau.keys())
    rep_means, scr_means = [], []
    for t in taus:
        items = by_tau[t]
        rep_means.append(np.mean([x["mv_repair"] for x in items]))
        scr_means.append(np.mean([x["mv_scratch"] for x in items]))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    ax.plot(taus, rep_means, "o-", color="#e74c3c", linewidth=2,
            markersize=7, label="Rollback (from $\\tau$-1)")
    ax.plot(taus, scr_means, "s-", color="#3498db", linewidth=2,
            markersize=7, label="Scratch (from start)")
    ax.fill_between(taus, rep_means, scr_means, alpha=0.12, color="#3498db")

    for t, rm, sm in zip(taus, rep_means, scr_means):
        d = sm - rm
        ax.annotate(f"$\\Delta$={d:.3f}",
                    (t, (rm + sm) / 2),
                    textcoords="offset points", xytext=(15, 0),
                    fontsize=8, color="#2c3e50")

    ax.set_xlabel("First Error Step ($\\tau$)")
    ax.set_ylabel("Majority-Vote Accuracy")
    ax.set_title("Accuracy by Error Position")
    ax.set_xticks(taus)
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_by_tau.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_by_tau.pdf'}")

# SECTION: fig_scatter


def _fig_scatter(paired, fig_dir):
    """Per-sample scatter: repair rate vs scratch rate."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    rep = np.array([p["rate_repair"] for p in paired])
    scr = np.array([p["rate_scratch"] for p in paired])
    colors = ["#e74c3c" if p["bucket"] == "early" else "#2ecc71"
              for p in paired]

    rng = np.random.default_rng(42)
    jitter_r = rng.uniform(-0.01, 0.01, len(rep))
    jitter_s = rng.uniform(-0.01, 0.01, len(scr))
    ax.scatter(rep + jitter_r, scr + jitter_s,
               c=colors, alpha=0.35, s=18, edgecolors="none")

    lims = [0, 1.05]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8)
    ax.set_xlabel("Per-continuation Rate (Rollback)")
    ax.set_ylabel("Per-continuation Rate (Scratch)")
    ax.set_title("Per-sample: Rollback vs Scratch")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="early"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=8, label="late"),
    ]
    ax.legend(handles=legend_elems, loc="lower right")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_scatter_rollback_scratch.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_scatter_rollback_scratch.pdf'}")

# SECTION: main


def main() -> None:
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Load rollback continuations (from 6_5) ---
    rollback_path = Path(args.rollback_file)
    if not rollback_path.exists():
        print(f"ERROR: rollback file not found: {rollback_path}")
        sys.exit(1)
    rollback_rows = [json.loads(l)
                     for l in rollback_path.read_text("utf-8").splitlines()
                     if l.strip()]
    print(f"Loaded {len(rollback_rows)} rollback continuations")

    # --- Load sample metadata for questions ---
    samples = load_samples(args.early_file, args.late_file, args.limit)
    keys_in_rollback = {(r["doc_id"], r["sample_idx"]) for r in rollback_rows}
    samples = [s for s in samples
               if (s["doc_id"], s["sample_idx"]) in keys_in_rollback]
    print(f"Samples with rollback data: {len(samples)}")

    # --- Generate from-scratch completions ---
    scratch_path = out_dir / "scratch_continuations.jsonl"
    if args.skip_generation and scratch_path.exists():
        print(f"Skipping generation, using existing: {scratch_path}")
    else:
        scratch_path = generate_scratch_completions(args, samples, out_dir)

    scratch_rows = [json.loads(l)
                    for l in scratch_path.read_text("utf-8").splitlines()
                    if l.strip()]
    print(f"Loaded {len(scratch_rows)} scratch continuations")

    # --- Majority vote for both conditions ---
    rollback_mv = compute_majority_vote_accuracy(
        rollback_rows, args.n_continuations)
    scratch_mv = compute_majority_vote_accuracy(
        scratch_rows, args.n_continuations)

    common_keys = set(rollback_mv.keys()) & set(scratch_mv.keys())
    print(f"Common samples for comparison: {len(common_keys)}")

    paired = []
    for key in common_keys:
        paired.append({
            "doc_id": key[0],
            "sample_idx": key[1],
            "bucket": rollback_mv[key]["bucket"],
            "tau": rollback_mv[key]["tau"],
            "n_steps": rollback_mv[key]["n_steps"],
            "mv_repair": rollback_mv[key]["mv_correct"],
            "mv_scratch": scratch_mv[key]["mv_correct"],
            "rate_repair": rollback_mv[key]["per_cont_rate"],
            "rate_scratch": scratch_mv[key]["per_cont_rate"],
        })

    # --- Print summary ---
    by_bucket = defaultdict(list)
    for p in paired:
        by_bucket[p["bucket"]].append(p)

    print("\n=== Majority-Vote Accuracy ===")
    summary_data = {}
    for bucket in ["early", "late"]:
        items = by_bucket[bucket]
        if not items:
            continue
        mv_rep = [x["mv_repair"] for x in items]
        mv_scr = [x["mv_scratch"] for x in items]
        n = len(items)
        acc_rep = sum(mv_rep) / n
        acc_scr = sum(mv_scr) / n
        delta = acc_scr - acc_rep

        # McNemar test
        both_right = sum(1 for r, s in zip(mv_rep, mv_scr)
                         if r == 1 and s == 1)
        rep_only = sum(1 for r, s in zip(mv_rep, mv_scr)
                       if r == 1 and s == 0)
        scr_only = sum(1 for r, s in zip(mv_rep, mv_scr)
                       if r == 0 and s == 1)
        both_wrong = sum(1 for r, s in zip(mv_rep, mv_scr)
                         if r == 0 and s == 0)

        summary_data[bucket] = {
            "n": n,
            "acc_rollback": acc_rep,
            "acc_scratch": acc_scr,
            "delta": delta,
            "both_right": both_right,
            "rollback_only": rep_only,
            "scratch_only": scr_only,
            "both_wrong": both_wrong,
        }

        print(f"  {bucket} (n={n}):")
        print(f"    Rollback MV acc = {acc_rep:.4f}")
        print(f"    Scratch MV acc  = {acc_scr:.4f}")
        print(f"    Delta (scratch - rollback) = {delta:+.4f}")
        print(f"    Contingency: both_right={both_right}, "
              f"rollback_only={rep_only}, scratch_only={scr_only}, "
              f"both_wrong={both_wrong}")

    # --- Per-continuation rate comparison ---
    print("\n=== Per-Continuation Rate (not majority vote) ===")
    for bucket in ["early", "late"]:
        items = by_bucket[bucket]
        if not items:
            continue
        rate_rep = np.mean([x["rate_repair"] for x in items])
        rate_scr = np.mean([x["rate_scratch"] for x in items])
        print(f"  {bucket}: rollback={rate_rep:.4f}, scratch={rate_scr:.4f}")

    # --- Figures ---
    summary = make_figures(paired, fig_dir)

    # --- Save summary ---
    out_summary = out_dir / "comparison_summary.json"
    full_out = {
        "majority_vote": summary_data,
        "figure_summary": summary,
        "n_paired": len(paired),
        "n_continuations": args.n_continuations,
    }
    out_summary.write_text(
        json.dumps(full_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary: {out_summary}")
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
