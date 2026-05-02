#!/usr/bin/env python3
"""Compare PRM-drop and NLL-drop triggers against GPT ground-truth first-error step.

Metrics reported:
  - Exact match: predicted rollback step == tau (GPT first error step)
  - Within-1: predicted == tau OR predicted == tau-1 (error step or one before)
  - Before-or-at: predicted <= tau (conservative: we rollback no later than error)
  - Mean absolute distance: |predicted - tau|

Trigger definitions:
  PRM-drop:  g^ = first i such that (score[i] - score[i+1]) > threshold
             fallback to argmin(score) if no drop exceeds threshold
  NLL-drop:  n_i = mean NLL of step i
             delta_i = n_i - n_{i-1}
             g^ = first i such that delta_i > tau_n (NLL threshold)
             fallback to argmax(n_i) if no delta exceeds threshold

Usage:
    python scripts/_tmp_trigger_vs_ground_truth.py
"""

import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.prompt_templates import check_answer

DATASET = "gsm8k"

PRM_DROP_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
NLL_DROP_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]


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


def prm_drop_predict(scores, threshold):
    """Return predicted error step (0-indexed) using PRM score drop."""
    n = len(scores)
    if n < 2:
        return 0
    drops = [scores[i] - scores[i + 1] for i in range(n - 1)]
    for i, d in enumerate(drops):
        if d > threshold:
            return i + 1
    return n - 1


def nll_drop_predict(nlls, threshold):
    """Return predicted error step (0-indexed) using NLL delta."""
    n = len(nlls)
    if n < 2:
        return 0
    deltas = [nlls[i] - nlls[i - 1] for i in range(1, n)]
    for i, d in enumerate(deltas):
        if d > threshold:
            return i + 1
    return n - 1


def evaluate_method(predictions, ground_truths):
    """Compute metrics given lists of (predicted_step, gt_step) both 0-indexed.

    within1: predicted is at gt or one step before gt (i.e. pred == gt or pred == gt-1).
    This means we flagged the error step itself or the step just before it.
    """
    n = len(predictions)
    if n == 0:
        return {}
    exact = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    within1 = sum(1 for p, g in zip(predictions, ground_truths)
                  if p == g or p == g - 1)
    before_or_at = sum(1 for p, g in zip(predictions, ground_truths) if p <= g)
    dists = [abs(p - g) for p, g in zip(predictions, ground_truths)]
    return dict(
        n=n,
        exact=exact / n,
        within1=within1 / n,
        before_or_at=before_or_at / n,
        mean_dist=np.mean(dists),
        median_dist=np.median(dists),
    )


def main():
    # ---- Load data ----
    ckpt_path = ROOT / "results/gsm8k_entropy_triggered_sweep/checkpoint.jsonl"
    records = [json.loads(l) for l in ckpt_path.read_text().splitlines() if l.strip()]

    drafts = {(r["doc_id"], r["draft_idx"]): r
              for r in records if r.get("task_type") == "draft"}
    lp_map = {(r["doc_id"], r.get("draft_idx", 0)): r
              for r in records if r.get("task_type") == "logprob"}
    se_map = {(r["doc_id"], r.get("draft_idx", 0)): r.get("step_scores", [])
              for r in records if r.get("task_type") == "self_eval"}

    gpt_path = ROOT / "results/gsm8k_3b_multi_sample/first_error/gpt_first_error_cache.jsonl"
    gpt_recs = [json.loads(l) for l in gpt_path.read_text().splitlines() if l.strip()]
    gpt_map = {}
    for r in gpt_recs:
        gpt_map[(f"gsm8k_{r['doc_id']}", r["sample_idx"])] = r

    # ---- Build evaluation set ----
    samples = []
    for (did, di), g in gpt_map.items():
        tau = g.get("tau")
        if tau is None:
            continue
        d = drafts.get((did, di))
        if not d or not d.get("draft_steps"):
            continue
        n = len(d["draft_steps"])
        if n < 2:
            continue

        prm_scores = se_map.get((did, di), [])
        lp_rec = lp_map.get((did, di))
        if not lp_rec or len(prm_scores) < n:
            continue

        bounds = step_char_bounds(d["draft_text"], d["draft_steps"])
        nlls = compute_step_mean_nll(
            lp_rec["token_logprobs"], lp_rec["token_offsets"], bounds)

        gt_step = int(tau) - 1  # tau is 1-indexed, convert to 0-indexed

        samples.append(dict(
            doc_id=did, draft_idx=di, n_steps=n,
            prm_scores=prm_scores[:n], nlls=nlls,
            gt_step=gt_step,
        ))

    print(f"Evaluation samples: {len(samples)} (wrong drafts with GPT annotation)")
    print()

    # ---- Evaluate PRM-drop ----
    print("=" * 72)
    print("PRM Score Drop Trigger")
    print("  Rule: g^ = first i where score[i]-score[i+1] > threshold")
    print("        fallback: argmin(score)")
    print("=" * 72)
    print(f"{'threshold':>10} {'exact':>8} {'within1':>8} {'before':>8} "
          f"{'mean_d':>8} {'med_d':>8}")
    print("-" * 60)

    for thr in PRM_DROP_THRESHOLDS:
        preds = [prm_drop_predict(s["prm_scores"], thr) for s in samples]
        gts = [s["gt_step"] for s in samples]
        m = evaluate_method(preds, gts)
        print(f"{thr:>10.2f} {m['exact']:>8.3f} {m['within1']:>8.3f} "
              f"{m['before_or_at']:>8.3f} {m['mean_dist']:>8.2f} "
              f"{m['median_dist']:>8.1f}")

    # ---- Evaluate NLL-drop ----
    print()
    print("=" * 72)
    print("Mean NLL Drop Trigger")
    print("  Rule: delta_i = nll[i] - nll[i-1]")
    print("        g^ = first i where delta_i > threshold")
    print("        fallback: argmax(nll)")
    print("=" * 72)
    print(f"{'threshold':>10} {'exact':>8} {'within1':>8} {'before':>8} "
          f"{'mean_d':>8} {'med_d':>8}")
    print("-" * 60)

    for thr in NLL_DROP_THRESHOLDS:
        preds = [nll_drop_predict(s["nlls"], thr) for s in samples]
        gts = [s["gt_step"] for s in samples]
        m = evaluate_method(preds, gts)
        print(f"{thr:>10.2f} {m['exact']:>8.3f} {m['within1']:>8.3f} "
              f"{m['before_or_at']:>8.3f} {m['mean_dist']:>8.2f} "
              f"{m['median_dist']:>8.1f}")

    # ---- Baselines ----
    print()
    print("=" * 72)
    print("Baselines")
    print("=" * 72)
    print(f"{'method':>20} {'exact':>8} {'within1':>8} {'before':>8} "
          f"{'mean_d':>8} {'med_d':>8}")
    print("-" * 68)

    # Random baseline
    rng = np.random.default_rng(42)
    preds_rand = [rng.integers(0, s["n_steps"]) for s in samples]
    gts = [s["gt_step"] for s in samples]
    m = evaluate_method(preds_rand, gts)
    print(f"{'random':>20} {m['exact']:>8.3f} {m['within1']:>8.3f} "
          f"{m['before_or_at']:>8.3f} {m['mean_dist']:>8.2f} "
          f"{m['median_dist']:>8.1f}")

    # Always-last baseline
    preds_last = [s["n_steps"] - 1 for s in samples]
    m = evaluate_method(preds_last, gts)
    print(f"{'always_last':>20} {m['exact']:>8.3f} {m['within1']:>8.3f} "
          f"{m['before_or_at']:>8.3f} {m['mean_dist']:>8.2f} "
          f"{m['median_dist']:>8.1f}")

    # PRM argmin (no threshold, just worst step)
    preds_prm_argmin = [int(np.argmin(s["prm_scores"])) for s in samples]
    m = evaluate_method(preds_prm_argmin, gts)
    print(f"{'prm_argmin':>20} {m['exact']:>8.3f} {m['within1']:>8.3f} "
          f"{m['before_or_at']:>8.3f} {m['mean_dist']:>8.2f} "
          f"{m['median_dist']:>8.1f}")

    # NLL argmax (no threshold, just highest NLL step)
    preds_nll_argmax = [int(np.argmax(s["nlls"])) for s in samples]
    m = evaluate_method(preds_nll_argmax, gts)
    print(f"{'nll_argmax':>20} {m['exact']:>8.3f} {m['within1']:>8.3f} "
          f"{m['before_or_at']:>8.3f} {m['mean_dist']:>8.2f} "
          f"{m['median_dist']:>8.1f}")

    # ---- Combined: PRM-drop OR NLL-drop ----
    print()
    print("=" * 72)
    print("Combined: PRM-drop(0.1) + NLL-drop(threshold)")
    print("  Rule: use PRM if drop>0.1 exists, else use NLL-drop")
    print("=" * 72)
    print(f"{'nll_thr':>10} {'exact':>8} {'within1':>8} {'before':>8} "
          f"{'mean_d':>8} {'med_d':>8}")
    print("-" * 60)

    for nll_thr in NLL_DROP_THRESHOLDS:
        preds = []
        for s in samples:
            scores = s["prm_scores"]
            n = len(scores)
            drops = [scores[i] - scores[i + 1] for i in range(n - 1)]
            prm_fired = any(d > 0.1 for d in drops)
            if prm_fired:
                preds.append(prm_drop_predict(scores, 0.1))
            else:
                preds.append(nll_drop_predict(s["nlls"], nll_thr))
        m = evaluate_method(preds, gts)
        print(f"{nll_thr:>10.2f} {m['exact']:>8.3f} {m['within1']:>8.3f} "
              f"{m['before_or_at']:>8.3f} {m['mean_dist']:>8.2f} "
              f"{m['median_dist']:>8.1f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
