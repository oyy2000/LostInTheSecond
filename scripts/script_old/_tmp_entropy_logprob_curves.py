#!/usr/bin/env python3
"""Plot entropy & logprob step curves for the same 12 samples as prm_step_curves."""

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.prompt_templates import check_answer

DATASET = "math500"

def step_char_bounds(response, steps):
    bounds, pos = [], 0
    for s in steps:
        idx = response.find(s, pos)
        if idx < 0: idx = pos
        bounds.append((idx, idx + len(s)))
        pos = idx + len(s)
    return bounds

ckpt_path = ROOT / "results/math500_entropy_triggered_sweep/checkpoint.jsonl"
records = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l.strip()]

drafts = {(r["doc_id"], r["draft_idx"]): r for r in records if r.get("task_type") == "draft"}
se_map = {(r["doc_id"], r.get("draft_idx", 0)): r.get("step_scores", [])
          for r in records if r.get("task_type") == "self_eval"}
lp_map = {(r["doc_id"], r.get("draft_idx", 0)): r
          for r in records if r.get("task_type") == "logprob"}

# ---- reuse the same _tok_entropy / compute_step_metrics from the sweep ----
def _tok_entropy(lp):
    p = np.exp(lp) if lp > -50 else 0.0
    p = np.clip(p, 1e-12, 1.0)
    return -p * np.log2(p) - (1 - p) * np.log2(max(1 - p, 1e-12))

def compute_step_metrics(lps, offs, bounds):
    metrics = []
    ti = 0
    for s0, s1 in bounds:
        while ti < len(offs) and offs[ti] < s0:
            ti += 1
        slps, shs = [], []
        scan = ti
        while scan < len(offs) and offs[scan] < s1:
            slps.append(lps[scan])
            shs.append(_tok_entropy(lps[scan]))
            scan += 1
        if slps:
            m = dict(mean_entropy=np.mean(shs), max_entropy=max(shs),
                     mean_logprob=np.mean(slps), min_logprob=min(slps),
                     n_tokens=len(slps))
        else:
            m = dict(mean_entropy=0, max_entropy=0,
                     mean_logprob=0, min_logprob=0, n_tokens=0)
        metrics.append(m)
    return metrics

# ---- select same 12 samples as prm_step_curves ----------------------------
samples = []
for (did, di), scores in se_map.items():
    d = drafts.get((did, di))
    if not d or not d.get("draft_steps"):
        continue
    n = len(d["draft_steps"])
    if n < 3 or len(scores) < n:
        continue
    lp_rec = lp_map.get((did, di))
    if not lp_rec:
        continue
    scores_trimmed = scores[:n]
    worst_prm = int(np.argmin(scores_trimmed))
    correct = check_answer(DATASET, d.get("draft_answer", ""), d.get("gold_answer", ""))

    bounds = step_char_bounds(d["draft_text"], d["draft_steps"])
    sm = compute_step_metrics(
        lp_rec["token_logprobs"], lp_rec["token_offsets"], bounds)

    samples.append(dict(
        doc_id=did, draft_idx=di, n_steps=n,
        prm_scores=scores_trimmed, rollback_prm=worst_prm,
        correct=correct,
        mean_entropy=[m["mean_entropy"] for m in sm],
        max_entropy=[m["max_entropy"] for m in sm],
        mean_logprob=[m["mean_logprob"] for m in sm],
        min_logprob=[m["min_logprob"] for m in sm],
    ))

random.seed(42)
chosen = random.sample(samples, min(12, len(samples)))
chosen.sort(key=lambda x: x["n_steps"])

# ---- Plot 1: Entropy curves -----------------------------------------------
ncols = 4
nrows = (len(chosen) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
axes = np.array(axes).flatten()

for i, s in enumerate(chosen):
    ax = axes[i]
    xs = list(range(s["n_steps"]))
    ax.plot(xs, s["mean_entropy"], "o-", color="#1976D2", label="mean entropy", markersize=5)
    ax.plot(xs, s["max_entropy"], "s--", color="#E91E63", label="max entropy", markersize=4, alpha=0.7)
    # mark PRM rollback
    ax.axvline(s["rollback_prm"], color="orange", ls="--", lw=1.5, alpha=0.8, label="PRM rollback")
    # mark entropy argmax
    worst_ent = int(np.argmax(s["mean_entropy"]))
    ax.axvline(worst_ent, color="green", ls=":", lw=1.5, alpha=0.8, label="entropy argmax")
    tag = "CORRECT" if s["correct"] else "WRONG"
    color = "#388E3C" if s["correct"] else "#D32F2F"
    ax.set_title(f"{s['doc_id']} ({s['n_steps']} steps)\ndraft: {tag}", fontsize=9, color=color)
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Entropy", fontsize=8)
    ax.set_xticks(xs)
    if i == 0:
        ax.legend(fontsize=6, loc="best")

for j in range(len(chosen), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Entropy Step Curves (MATH-500, Qwen2.5-3B)  |  orange=PRM rollback, green=entropy argmax",
             fontsize=11, y=1.01)
fig.tight_layout()

out_dir = ROOT / "figures/entropy_triggered_sweep"
out_dir.mkdir(parents=True, exist_ok=True)
for fmt in ("png", "pdf"):
    fig.savefig(out_dir / f"entropy_step_curves_math500.{fmt}", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved entropy curves -> {out_dir}/entropy_step_curves_math500.png")

# ---- Plot 2: Logprob curves -----------------------------------------------
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
axes2 = np.array(axes2).flatten()

for i, s in enumerate(chosen):
    ax = axes2[i]
    xs = list(range(s["n_steps"]))
    ax.plot(xs, s["mean_logprob"], "o-", color="#1976D2", label="mean logprob", markersize=5)
    ax.plot(xs, s["min_logprob"], "s--", color="#E91E63", label="min logprob", markersize=4, alpha=0.7)
    # mark PRM rollback
    ax.axvline(s["rollback_prm"], color="orange", ls="--", lw=1.5, alpha=0.8, label="PRM rollback")
    # mark min_logprob argmin
    worst_lp = int(np.argmin(s["min_logprob"]))
    ax.axvline(worst_lp, color="green", ls=":", lw=1.5, alpha=0.8, label="logprob argmin")
    tag = "CORRECT" if s["correct"] else "WRONG"
    color = "#388E3C" if s["correct"] else "#D32F2F"
    ax.set_title(f"{s['doc_id']} ({s['n_steps']} steps)\ndraft: {tag}", fontsize=9, color=color)
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Log-prob", fontsize=8)
    ax.set_xticks(xs)
    if i == 0:
        ax.legend(fontsize=6, loc="best")

for j in range(len(chosen), len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle("Logprob Step Curves (MATH-500, Qwen2.5-3B)  |  orange=PRM rollback, green=logprob argmin",
              fontsize=11, y=1.01)
fig2.tight_layout()

for fmt in ("png", "pdf"):
    fig2.savefig(out_dir / f"logprob_step_curves_math500.{fmt}", dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved logprob curves -> {out_dir}/logprob_step_curves_math500.png")
