#!/usr/bin/env python3
"""
Compare PRM step-level scores: prefilled_k1 vs raw_cot_n8.

Only scores samples whose (doc_id, sample_idx) appear in BOTH files.
Uses src/prm/scoring.StepScorer (Qwen2.5-Math-PRM-7B).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/compare_prm_prefilled_vs_raw.py
    # plot-only from cache:
    python scripts/eval/compare_prm_prefilled_vs_raw.py --plot-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prefilled(path: Path) -> Dict[Tuple[int,int], dict]:
    """Load prefilled_k1.jsonl keyed by (doc_id, sample_idx)."""
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            key = (rec["doc_id"], rec.get("sample_idx", 0))
            data[key] = {
                "doc_id": rec["doc_id"],
                "sample_idx": rec.get("sample_idx", 0),
                "question": rec["question"],
                "gold_answer": rec.get("gold_answer", ""),
                "steps": rec.get("all_steps") or rec.get("steps") or [],
                "exact_match": float(rec.get("exact_match", 0)),
                "condition": rec.get("condition", ""),
                "source": "prefilled_k1",
            }
    return data


def load_raw_cot(path: Path) -> Dict[Tuple[int,int], dict]:
    """Load raw_cot_n8.jsonl keyed by (doc_id, sample_idx)."""
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            key = (rec["doc_id"], rec.get("sample_idx", 0))
            data[key] = {
                "doc_id": rec["doc_id"],
                "sample_idx": rec.get("sample_idx", 0),
                "question": rec["question"],
                "gold_answer": rec.get("gold_answer", ""),
                "steps": rec.get("steps") or [],
                "exact_match": float(rec.get("exact_match", 0)),
                "source": "raw_cot_n8",
            }
    return data


# ---------------------------------------------------------------------------
# PRM scoring
# ---------------------------------------------------------------------------

def score_records(records: List[dict], scorer, label: str) -> List[dict]:
    """Score each record's steps with PRM."""
    for i, rec in enumerate(records):
        scores = scorer.score_steps(rec["question"], rec["steps"])
        rec["step_scores"] = scores
        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"  [{label}] {i+1}/{len(records)}", flush=True)
    return records


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def avg_curve_absolute(records: List[dict]):
    by_step: Dict[int, List[float]] = {}
    for r in records:
        for k, s in enumerate(r.get("step_scores", [])):
            by_step.setdefault(k, []).append(s)
    xs, ys, ns = [], [], []
    for k in sorted(by_step):
        vals = by_step[k]
        xs.append(k + 1)
        ys.append(np.mean(vals))
        ns.append(len(vals))
    return xs, ys, ns


def avg_curve_normalized(records: List[dict], n_bins: int = 10):
    positions, values = [], []
    for r in records:
        scores = r.get("step_scores", [])
        n = len(scores)
        if n < 2:
            continue
        for k, s in enumerate(scores):
            positions.append(k / (n - 1))
            values.append(s)
    if not positions:
        return [], [], []
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for p, v in zip(positions, values):
        idx = min(int(p * n_bins), n_bins - 1)
        sums[idx] += v
        counts[idx] += 1
    mask = counts > 0
    means = np.where(counts > 0, sums / counts, np.nan)
    return centers[mask].tolist(), means[mask].tolist(), counts[mask].tolist()


def compute_summary(records: List[dict]) -> dict:
    """Compute aggregate stats for a set of scored records."""
    all_scores = [s for r in records for s in r.get("step_scores", [])]
    min_scores = [min(r["step_scores"]) for r in records if r.get("step_scores")]
    last_scores = [r["step_scores"][-1] for r in records if r.get("step_scores")]
    n_steps_list = [len(r.get("step_scores", [])) for r in records]
    em = [r["exact_match"] for r in records]
    return {
        "n_samples": len(records),
        "exact_match_rate": np.mean(em) if em else 0,
        "avg_all_step_scores": np.mean(all_scores) if all_scores else 0,
        "avg_min_step_score": np.mean(min_scores) if min_scores else 0,
        "avg_last_step_score": np.mean(last_scores) if last_scores else 0,
        "avg_n_steps": np.mean(n_steps_list) if n_steps_list else 0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlay(prefilled: List[dict], raw: List[dict],
                 out_dir: Path, n_bins: int = 10):
    """Plot PRM curves: prefilled vs raw, with correct/wrong splits."""
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "prefilled_k1": prefilled,
        "raw_cot_n8": raw,
    }
    colors = {"prefilled_k1": "tab:blue", "raw_cot_n8": "tab:orange"}

    # --- 1. Normalized overlay: all / correct / wrong ---
    for split_name, filter_fn in [
        ("all", lambda r: True),
        ("correct", lambda r: r["exact_match"] >= 1.0),
        ("wrong", lambda r: r["exact_match"] < 1.0),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, recs in datasets.items():
            subset = [r for r in recs if filter_fn(r)]
            if not subset:
                continue
            xs, ys, ns = avg_curve_normalized(subset, n_bins)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                        color=colors[label],
                        label=f"{label} ({split_name}, n={len(subset)})")
        ax.set_xlabel("Normalized step position (0=start, 1=end)")
        ax.set_ylabel("Avg PRM score")
        ax.set_title(f"PRM Step Score — prefilled_k1 vs raw_cot_n8 — {split_name}")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"prm_norm_{split_name}.png", dpi=180)
        plt.close(fig)
        print(f"  Saved: prm_norm_{split_name}.png")

    # --- 2. Absolute step overlay: all ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, recs in datasets.items():
        xs, ys, ns = avg_curve_absolute(recs)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                    color=colors[label],
                    label=f"{label} (n={len(recs)})")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Avg PRM score")
    ax.set_title("PRM Step Score (absolute) — prefilled_k1 vs raw_cot_n8")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "prm_absolute_all.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: prm_absolute_all.png")

    # --- 3. Paired delta histogram (min-step score diff) ---
    deltas = []
    for r_p, r_r in zip(prefilled, raw):
        sp = r_p.get("step_scores", [])
        sr = r_r.get("step_scores", [])
        if sp and sr:
            deltas.append(min(sp) - min(sr))
    if deltas:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(deltas, bins=40, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Δ min-step PRM score (prefilled − raw)")
        ax.set_ylabel("Count")
        ax.set_title(f"Paired min-step PRM diff (n={len(deltas)}, mean={np.mean(deltas):.4f})")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "paired_min_step_delta.png", dpi=180)
        plt.close(fig)
        print(f"  Saved: paired_min_step_delta.png")

    # --- 4. Paired delta histogram (last-step score diff) ---
    deltas_last = []
    for r_p, r_r in zip(prefilled, raw):
        sp = r_p.get("step_scores", [])
        sr = r_r.get("step_scores", [])
        if sp and sr:
            deltas_last.append(sp[-1] - sr[-1])
    if deltas_last:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(deltas_last, bins=40, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Δ last-step PRM score (prefilled − raw)")
        ax.set_ylabel("Count")
        ax.set_title(f"Paired last-step PRM diff (n={len(deltas_last)}, mean={np.mean(deltas_last):.4f})")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "paired_last_step_delta.png", dpi=180)
        plt.close(fig)
        print(f"  Saved: paired_last_step_delta.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare PRM scores: prefilled_k1 vs raw_cot_n8 (matched samples only)")
    parser.add_argument("--prefilled", type=str,
                        default="results/gsm8k_3b_prefix_correction/prefilled_k1.jsonl")
    parser.add_argument("--raw", type=str,
                        default="results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl")
    parser.add_argument("--output-dir", type=str,
                        default="runs/prm_prefilled_vs_raw")
    parser.add_argument("--prm-model", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    # Resolve paths
    prefilled_path = Path(args.prefilled)
    if not prefilled_path.is_absolute():
        prefilled_path = PROJECT_ROOT / prefilled_path
    raw_path = Path(args.raw)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "scored_cache.jsonl"

    # --- Load data ---
    print("Loading prefilled_k1 ...", flush=True)
    prefilled_map = load_prefilled(prefilled_path)
    print(f"  {len(prefilled_map)} records")

    print("Loading raw_cot_n8 ...", flush=True)
    raw_map = load_raw_cot(raw_path)
    print(f"  {len(raw_map)} records")

    # Intersect on (doc_id, sample_idx)
    common_keys = sorted(set(prefilled_map) & set(raw_map))
    print(f"Common (doc_id, sample_idx) pairs: {len(common_keys)}")

    prefilled_list = [prefilled_map[k] for k in common_keys]
    raw_list = [raw_map[k] for k in common_keys]

    # --- PRM scoring ---
    if not args.plot_only:
        # Check for cached scores
        if cache_path.exists():
            print(f"Loading cached scores from {cache_path} ...")
            cached = {}
            with cache_path.open("r") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    cached[obj["key"]] = obj["step_scores"]
            # Apply cached scores
            n_cached = 0
            for rec in prefilled_list + raw_list:
                ck = f"{rec['source']}|{rec['doc_id']}|{rec['sample_idx']}"
                if ck in cached:
                    rec["step_scores"] = cached[ck]
                    n_cached += 1
            print(f"  Applied {n_cached} cached scores")

        # Find records still needing scoring
        to_score_p = [r for r in prefilled_list if "step_scores" not in r]
        to_score_r = [r for r in raw_list if "step_scores" not in r]

        if to_score_p or to_score_r:
            print(f"\nNeed to score: {len(to_score_p)} prefilled + {len(to_score_r)} raw")
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from prm.scoring import StepScorer

            print("Loading PRM model ...", flush=True)
            t0 = time.time()
            scorer = StepScorer(args.prm_model, args.dtype)
            print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

            if to_score_p:
                print(f"\nScoring prefilled_k1 ({len(to_score_p)} records) ...")
                score_records(to_score_p, scorer, "prefilled")
            if to_score_r:
                print(f"\nScoring raw_cot_n8 ({len(to_score_r)} records) ...")
                score_records(to_score_r, scorer, "raw")

            # Save to cache
            print("Saving scores to cache ...")
            with cache_path.open("a", encoding="utf-8") as f:
                for rec in to_score_p + to_score_r:
                    obj = {
                        "key": f"{rec['source']}|{rec['doc_id']}|{rec['sample_idx']}",
                        "step_scores": rec.get("step_scores", []),
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            print("All scores already cached.")
    else:
        # plot-only: must load from cache
        if not cache_path.exists():
            print(f"ERROR: --plot-only but no cache at {cache_path}")
            sys.exit(1)
        print(f"Loading cached scores from {cache_path} ...")
        cached = {}
        with cache_path.open("r") as f:
            for line in f:
                obj = json.loads(line.strip())
                cached[obj["key"]] = obj["step_scores"]
        for rec in prefilled_list + raw_list:
            ck = f"{rec['source']}|{rec['doc_id']}|{rec['sample_idx']}"
            if ck in cached:
                rec["step_scores"] = cached[ck]
        n_scored = sum(1 for r in prefilled_list + raw_list if "step_scores" in r)
        print(f"  Loaded {n_scored} scored records")

    # --- Summary stats ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, recs in [("prefilled_k1", prefilled_list), ("raw_cot_n8", raw_list)]:
        scored = [r for r in recs if r.get("step_scores")]
        s = compute_summary(scored)
        print(f"\n  [{label}]")
        print(f"    n_samples:           {s['n_samples']}")
        print(f"    exact_match_rate:    {s['exact_match_rate']:.4f}")
        print(f"    avg_all_step_scores: {s['avg_all_step_scores']:.4f}")
        print(f"    avg_min_step_score:  {s['avg_min_step_score']:.4f}")
        print(f"    avg_last_step_score: {s['avg_last_step_score']:.4f}")
        print(f"    avg_n_steps:         {s['avg_n_steps']:.1f}")

    # Paired comparison
    scored_pairs = [(p, r) for p, r in zip(prefilled_list, raw_list)
                    if p.get("step_scores") and r.get("step_scores")]
    if scored_pairs:
        min_deltas = [min(p["step_scores"]) - min(r["step_scores"]) for p, r in scored_pairs]
        last_deltas = [p["step_scores"][-1] - r["step_scores"][-1] for p, r in scored_pairs]
        avg_deltas = [np.mean(p["step_scores"]) - np.mean(r["step_scores"]) for p, r in scored_pairs]
        print(f"\n  [Paired Δ: prefilled − raw] (n={len(scored_pairs)})")
        print(f"    Δ avg_step_score:  {np.mean(avg_deltas):+.4f} ± {np.std(avg_deltas):.4f}")
        print(f"    Δ min_step_score:  {np.mean(min_deltas):+.4f} ± {np.std(min_deltas):.4f}")
        print(f"    Δ last_step_score: {np.mean(last_deltas):+.4f} ± {np.std(last_deltas):.4f}")
        print(f"    prefilled > raw (min): {sum(1 for d in min_deltas if d > 0)}/{len(min_deltas)}")
        print(f"    prefilled > raw (last): {sum(1 for d in last_deltas if d > 0)}/{len(last_deltas)}")

    # Save summary JSON
    summary = {
        "prefilled": compute_summary([r for r in prefilled_list if r.get("step_scores")]),
        "raw": compute_summary([r for r in raw_list if r.get("step_scores")]),
        "n_common_pairs": len(common_keys),
    }
    if scored_pairs:
        min_deltas = [min(p["step_scores"]) - min(r["step_scores"]) for p, r in scored_pairs]
        last_deltas = [p["step_scores"][-1] - r["step_scores"][-1] for p, r in scored_pairs]
        avg_deltas = [np.mean(p["step_scores"]) - np.mean(r["step_scores"]) for p, r in scored_pairs]
        summary["paired_delta"] = {
            "n": len(scored_pairs),
            "avg_step_mean": float(np.mean(avg_deltas)),
            "avg_step_std": float(np.std(avg_deltas)),
            "min_step_mean": float(np.mean(min_deltas)),
            "min_step_std": float(np.std(min_deltas)),
            "last_step_mean": float(np.mean(last_deltas)),
            "last_step_std": float(np.std(last_deltas)),
        }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_dir / 'summary.json'}")

    # --- Plotting ---
    print(f"\n=== Plotting ===")
    scored_p = [r for r in prefilled_list if r.get("step_scores")]
    scored_r = [r for r in raw_list if r.get("step_scores")]
    plot_overlay(scored_p, scored_r, out_dir / "figures", args.n_bins)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
