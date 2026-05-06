#!/usr/bin/env python3
"""
Offline evaluation of PRM rollback variants on MATH-500.

Variants tested:
  1. prm_argmin       — original argmin(step_scores)
  2. prm_thresh_T     — argmin, but only if min < T; else no rollback
  3. prm_drop_D       — rollback at largest score drop > D
  4. prm_skip0        — argmin over steps[1:]
  5. prm_entropy_gate — argmin, gated by entropy spike at same step

All use existing checkpoint data (no new generation needed).
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompt_templates import check_answer, split_steps
from src.sweep_datasets import load_dataset_by_name

DATASET = "math500"


def load_checkpoint():
    ckpt_path = ROOT / "results/math500_entropy_triggered_sweep/checkpoint.jsonl"
    records = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l.strip()]
    drafts = {(r["doc_id"], r["draft_idx"]): r
              for r in records if r.get("task_type") == "draft"}
    se_map = {(r["doc_id"], r.get("draft_idx", 0)): r.get("step_scores", [])
              for r in records if r.get("task_type") == "self_eval"}
    lp_recs = [r for r in records if r.get("task_type") == "logprob"]
    lp_map = {(r["doc_id"], r.get("draft_idx", 0)): r for r in lp_recs}
    sfx_map = {}
    for s in records:
        if s.get("task_type") == "suffix":
            key = (s["doc_id"], s["draft_idx"], s["rollback_step"], s["suffix_idx"])
            sfx_map[key] = s
    sc_recs = [r for r in records if r.get("task_type") == "fullsc"]
    return drafts, se_map, lp_map, sfx_map, sc_recs


def step_char_bounds(response, steps):
    bounds, pos = [], 0
    for s in steps:
        idx = response.find(s, pos)
        if idx < 0:
            idx = pos
        bounds.append((idx, idx + len(s)))
        pos = idx + len(s)
    return bounds


def compute_step_entropies(lps, offs, bounds):
    """Return per-step mean entropy."""
    entropies = []
    ti = 0
    for s0, s1 in bounds:
        while ti < len(offs) and offs[ti] < s0:
            ti += 1
        shs = []
        scan = ti
        while scan < len(offs) and offs[scan] < s1:
            lp = lps[scan]
            if lp < 0:
                p = math.exp(lp)
                shs.append(-p * lp)
            else:
                shs.append(0.0)
            scan += 1
        entropies.append(np.mean(shs) if shs else 0.0)
    return entropies


def compute_rb_points(questions, drafts, se_map, lp_map):
    """Compute rollback points for all PRM variants."""
    q_map = {q["doc_id"]: q for q in questions}
    nd_max = 4

    THRESHOLDS = [0.3, 0.5, 0.7]
    DROP_DELTAS = [0.2, 0.3, 0.5]

    rb = {}
    for q in questions:
        did = q["doc_id"]
        for di in range(nd_max):
            d = drafts.get((did, di))
            if not d or not d.get("draft_steps"):
                continue
            steps = d["draft_steps"]
            n = len(steps)
            if n < 2:
                continue

            scores = se_map.get((did, di), [])
            if not scores or len(scores) < n:
                continue
            scores = scores[:n]

            worst = int(np.argmin(scores))
            rb[(did, di, "prm_argmin")] = worst

            for T in THRESHOLDS:
                if scores[worst] < T:
                    rb[(did, di, f"prm_thresh_{T}")] = worst

            for D in DROP_DELTAS:
                drops = [scores[i] - scores[i + 1]
                         for i in range(n - 1)]
                if drops:
                    max_drop_idx = int(np.argmax(drops))
                    if drops[max_drop_idx] > D:
                        rb[(did, di, f"prm_drop_{D}")] = max_drop_idx + 1

            worst_skip0 = min(range(1, n), key=lambda t: scores[t])
            rb[(did, di, "prm_skip0")] = worst_skip0

            lp_rec = lp_map.get((did, di))
            if lp_rec and lp_rec.get("token_logprobs"):
                bounds = step_char_bounds(d["draft_text"], steps)
                ents = compute_step_entropies(
                    lp_rec["token_logprobs"],
                    lp_rec["token_offsets"], bounds)
                mean_ent = np.mean(ents) if ents else 0
                if ents[worst] > mean_ent:
                    rb[(did, di, "prm_ent_gate")] = worst

    return rb


def _vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def evaluate_variants(questions, drafts, sfx_map, sc_recs, rb_points):
    """Evaluate each PRM variant: acc and tokens_per_q."""
    nq = len(questions)
    draft_map = drafts

    method_keys = sorted(set(mk for (_, _, mk) in rb_points.keys()))
    nd_vals = [1, 2]
    k_vals = [2, 3]

    results = []

    greedy_correct = 0
    greedy_toks = 0
    for q in questions:
        d = draft_map.get((q["doc_id"], 0))
        if d and check_answer(DATASET, d.get("draft_answer", ""), q["gold_answer"]):
            greedy_correct += 1
        greedy_toks += d["draft_tokens"] if d else 0
    results.append(("greedy", greedy_toks / nq, greedy_correct / nq))

    for mkey in method_keys:
        for nd in nd_vals:
            for K in k_vals:
                correct = 0
                total_toks = 0
                for q in questions:
                    did = q["doc_id"]
                    answers = []
                    q_toks = 0
                    for di in range(nd):
                        d = draft_map.get((did, di))
                        if not d:
                            continue
                        answers.append(d.get("draft_answer", ""))
                        q_toks += d["draft_tokens"]
                        rb = rb_points.get((did, di, mkey))
                        if rb is not None:
                            b = max(1, rb)
                            for si in range(K - 1):
                                s = sfx_map.get((did, di, b, si))
                                if s:
                                    answers.append(s.get("suffix_answer", ""))
                                    q_toks += s.get("suffix_tokens", 0)
                    if check_answer(DATASET, _vote(answers), q["gold_answer"]):
                        correct += 1
                    total_toks += q_toks
                label = f"{mkey}_nd{nd}_K{K}"
                results.append((label, total_toks / nq, correct / nq))

    return results


def plot_pareto(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ROOT / "figures/entropy_triggered_sweep"
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for m, t, a in rows:
        if m == "greedy":
            g = "Greedy"
        elif "argmin" in m:
            g = "PRM argmin (original)"
        elif "thresh" in m:
            g = "PRM + threshold"
        elif "drop" in m:
            g = "PRM score-drop"
        elif "skip0" in m:
            g = "PRM skip-step0"
        elif "ent_gate" in m:
            g = "PRM + entropy gate"
        else:
            g = "other"
        groups.setdefault(g, []).append((m, t, a))

    colors = {
        "Greedy": "#333333",
        "PRM argmin (original)": "#9C27B0",
        "PRM + threshold": "#E53935",
        "PRM score-drop": "#2196F3",
        "PRM skip-step0": "#FF9800",
        "PRM + entropy gate": "#4CAF50",
    }
    markers = {
        "Greedy": "*",
        "PRM argmin (original)": "P",
        "PRM + threshold": "s",
        "PRM score-drop": "D",
        "PRM skip-step0": "^",
        "PRM + entropy gate": "o",
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    for g, pts in groups.items():
        xs = [t / 1000 for _, t, _ in pts]
        ys = [a * 100 for _, _, a in pts]
        labels = [m for m, _, _ in pts]
        ax.scatter(xs, ys, label=g, color=colors.get(g, "#666"),
                   marker=markers.get(g, "o"), s=60, alpha=0.8, zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            short = lab.replace("prm_", "").replace("_nd", " d").replace("_K", " K")
            ax.annotate(short, (x, y), fontsize=5.5, alpha=0.7,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Tokens per question (x1000)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("PRM Rollback Variants on MATH-500 (Qwen2.5-3B)", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"prm_variants_pareto_math500.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Pareto figure -> {fig_dir}/prm_variants_pareto_math500.png")


def main():
    questions = load_dataset_by_name(DATASET, 0, seed=42)
    print(f"Loaded {len(questions)} questions")

    drafts, se_map, lp_map, sfx_map, sc_recs = load_checkpoint()
    print(f"Drafts: {len(drafts)}, PRM: {len(se_map)}, "
          f"Logprob: {len(lp_map)}, Suffixes: {len(sfx_map)}")

    rb_points = compute_rb_points(questions, drafts, se_map, lp_map)

    method_counts = Counter(mk for (_, _, mk) in rb_points.keys())
    print("\nRollback points per variant:")
    for mk, cnt in sorted(method_counts.items()):
        print(f"  {mk}: {cnt}")

    results = evaluate_variants(questions, drafts, sfx_map, sc_recs, rb_points)

    print(f"\n{'Method':<35s} {'Tok/Q':>8s} {'Acc':>7s}")
    print("-" * 52)
    for m, t, a in sorted(results, key=lambda x: x[1]):
        print(f"{m:<35s} {t:>8.1f} {a*100:>6.1f}%")

    out_dir = ROOT / "results/math500_entropy_triggered_sweep"
    res_path = out_dir / "prm_variants_results.json"
    rows = [{"method": m, "tokens_per_q": t, "acc": a} for m, t, a in results]
    res_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nResults -> {res_path}")

    plot_pareto(results, out_dir)


if __name__ == "__main__":
    main()
