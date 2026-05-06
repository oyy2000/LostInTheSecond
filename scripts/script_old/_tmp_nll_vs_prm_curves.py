#!/usr/bin/env python3
"""Side-by-side mean-NLL vs PRM step curves for GSM8K samples.

Selects samples that include wrong drafts so we can visually compare
where PRM and NLL each flag the problematic step.
"""

import json, random, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.prompt_templates import check_answer

DATASET = "gsm8k"
N_SAMPLES = 12
N_WRONG_MIN = 6
SEED = 42


def step_char_bounds(response, steps):
    bounds, pos = [], 0
    for s in steps:
        idx = response.find(s, pos)
        if idx < 0:
            idx = pos
        bounds.append((idx, idx + len(s)))
        pos = idx + len(s)
    return bounds


def compute_step_mean_nll(lps, offs, bounds):
    nlls = []
    ti = 0
    for s0, s1 in bounds:
        while ti < len(offs) and offs[ti] < s0:
            ti += 1
        slps = []
        scan = ti
        while scan < len(offs) and offs[scan] < s1:
            slps.append(lps[scan])
            scan += 1
        nlls.append(-np.mean(slps) if slps else 0.0)
    return nlls


# ---- load data ----
ckpt_path = ROOT / "results/gsm8k_entropy_triggered_sweep/checkpoint.jsonl"
records = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l.strip()]

drafts = {(r["doc_id"], r["draft_idx"]): r
          for r in records if r.get("task_type") == "draft"}
se_map = {(r["doc_id"], r.get("draft_idx", 0)): r.get("step_scores", [])
          for r in records if r.get("task_type") == "self_eval"}
lp_map = {(r["doc_id"], r.get("draft_idx", 0)): r
          for r in records if r.get("task_type") == "logprob"}

# ---- build samples ----
wrong, right = [], []
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

    scores_t = scores[:n]
    bounds = step_char_bounds(d["draft_text"], d["draft_steps"])
    nlls = compute_step_mean_nll(
        lp_rec["token_logprobs"], lp_rec["token_offsets"], bounds)
    correct = check_answer(DATASET, d.get("draft_answer", ""),
                           d.get("gold_answer", ""))

    rec = dict(doc_id=did, draft_idx=di, n_steps=n,
               prm_scores=scores_t, mean_nll=nlls, correct=correct)
    (right if correct else wrong).append(rec)

print(f"Available: {len(right)} correct, {len(wrong)} wrong")

random.seed(SEED)
n_wrong = min(len(wrong), max(N_WRONG_MIN, N_SAMPLES // 2))
n_right = min(len(right), N_SAMPLES - n_wrong)
chosen = random.sample(wrong, n_wrong) + random.sample(right, n_right)
chosen.sort(key=lambda x: (x["correct"], x["n_steps"]))

print(f"Selected: {sum(not c['correct'] for c in chosen)} wrong, "
      f"{sum(c['correct'] for c in chosen)} correct")

# ---- plot ----
out_dir = ROOT / "figures/entropy_triggered_sweep_gsm8k"
out_dir.mkdir(parents=True, exist_ok=True)

ncols = 4
nrows = (len(chosen) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
axes = np.array(axes).flatten()

for i, s in enumerate(chosen):
    ax = axes[i]
    xs = list(range(s["n_steps"]))
    prm = s["prm_scores"]
    nll = s["mean_nll"]

    nll_min, nll_max = min(nll), max(nll)
    nll_range = nll_max - nll_min if nll_max > nll_min else 1.0
    nll_norm = [(v - nll_min) / nll_range for v in nll]

    ax.plot(xs, prm, "o-", color="#FF9800", label="PRM score",
            markersize=5, linewidth=1.8)
    ax.plot(xs, nll_norm, "s--", color="#7B1FA2",
            label="mean NLL (norm)", markersize=4, linewidth=1.5, alpha=0.85)

    prm_worst = int(np.argmin(prm))
    nll_worst = int(np.argmax(nll))
    ax.axvline(prm_worst, color="#FF9800", ls="--", lw=1.3, alpha=0.7)
    ax.axvline(nll_worst, color="#7B1FA2", ls=":", lw=1.3, alpha=0.7)

    tag = "CORRECT" if s["correct"] else "WRONG"
    clr = "#388E3C" if s["correct"] else "#D32F2F"
    ax.set_title(f"{s['doc_id']} ({s['n_steps']} steps)  [{tag}]\n"
                 f"PRM worst={prm_worst}  NLL worst={nll_worst}",
                 fontsize=8.5, color=clr)
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Score (0-1 range)", fontsize=8)
    ax.set_xticks(xs)
    ax.set_ylim(-0.08, 1.12)
    if i == 0:
        ax.legend(fontsize=7, loc="lower left")

for j in range(len(chosen), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("PRM Score vs Mean NLL per Step  (GSM8K, Qwen2.5-3B)\n"
             "dashed=PRM worst step, dotted=NLL worst step",
             fontsize=12, y=1.02)
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(out_dir / f"nll_vs_prm_step_curves.{fmt}",
                dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_dir}/nll_vs_prm_step_curves.png")
