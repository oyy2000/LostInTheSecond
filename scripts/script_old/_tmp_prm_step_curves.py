#!/usr/bin/env python3
"""Plot PRM step-score curves for random samples from MATH-500."""

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

ckpt_path = ROOT / "results/math500_entropy_triggered_sweep/checkpoint.jsonl"
records = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l.strip()]

drafts = {(r["doc_id"], r["draft_idx"]): r for r in records if r.get("task_type") == "draft"}
se_map = {(r["doc_id"], r.get("draft_idx", 0)): r.get("step_scores", [])
          for r in records if r.get("task_type") == "self_eval"}

samples = []
for (did, di), scores in se_map.items():
    d = drafts.get((did, di))
    if not d or not d.get("draft_steps"):
        continue
    n = len(d["draft_steps"])
    if n < 3 or len(scores) < n:
        continue
    scores_trimmed = scores[:n]
    worst = int(np.argmin(scores_trimmed))
    correct = check_answer(DATASET, d.get("draft_answer", ""), d.get("gold_answer", ""))
    samples.append(dict(doc_id=did, draft_idx=di, n_steps=n,
                        scores=scores_trimmed, rollback=worst,
                        correct=correct))

random.seed(42)
chosen = random.sample(samples, min(12, len(samples)))
chosen.sort(key=lambda x: x["n_steps"])

ncols = 4
nrows = (len(chosen) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
axes = np.array(axes).flatten()

for i, s in enumerate(chosen):
    ax = axes[i]
    xs = list(range(s["n_steps"]))
    ys = s["scores"]
    rb = s["rollback"]

    correct = s["correct"]
    line_color = "#43A047" if correct else "#E53935"
    tag = "CORRECT" if correct else "WRONG"

    ax.plot(xs, ys, "o-", color=line_color, markersize=5, linewidth=1.5, zorder=2,
            label=f"draft: {tag}")
    ax.scatter([rb], [ys[rb]], color="#FF6F00", s=110, zorder=3,
               marker="v", edgecolors="black", linewidths=0.6,
               label=f"rollback = step {rb}")
    ax.axhline(np.mean(ys), color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("PRM score", fontsize=9)
    ax.set_title(f"{s['doc_id']}  ({s['n_steps']} steps)", fontsize=9)
    ax.set_xticks(xs)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.2)
    ax.set_ylim(-0.05, 1.05)

for j in range(len(chosen), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("PRM Step Scores (MATH-500, Qwen2.5-3B)  |  green=correct, red=wrong, orange=rollback",
             fontsize=11, y=1.01)
fig.tight_layout()

out_dir = ROOT / "figures/entropy_triggered_sweep"
out_dir.mkdir(parents=True, exist_ok=True)
for fmt in ("png", "pdf"):
    fig.savefig(out_dir / f"prm_step_curves_math500.{fmt}", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_dir}/prm_step_curves_math500.png")
