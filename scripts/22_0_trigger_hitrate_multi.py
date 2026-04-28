#!/usr/bin/env python3
"""
Experiment 2: Trajectory-level trigger hit-rate analysis (multi-combo).

For each (model, dataset) combo, computes:
  - Hit@k (k=0,1,2): fraction of trajectories where trigger fires within
    +/-k steps of the first-error step tau.
  - False-trigger rate: avg non-error triggers per trajectory.

Sweeps thresholds for each individual metric + a logistic regression combo.
Plots 4-panel ROC-style figure (x=FTR, y=Hit@1).

Usage:
    python scripts/22_0_trigger_hitrate_multi.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.keystep_utils import load_jsonl

MODELS = [
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B"),
]
DATASETS = [("gsm8k", "GSM8K"), ("math500", "MATH500")]
FIG_DIR = ROOT / "figures" / "trigger_hitrate"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_METRICS = {"logprob_drop": False, "min_logprob": False}
LOOKAHEAD_METRICS = {"entropy_delta": True, "mean_entropy": True}
ALL_METRICS = {**LOOKBACK_METRICS, **LOOKAHEAD_METRICS}
N_THRESHOLDS = 50


def _ms(mid):
    return mid.split("/")[-1].lower().replace("-", "_")


def load_trajs_per_step(path):
    rows = load_jsonl(path)
    trajs = defaultdict(list)
    for r in rows:
        trajs[(r["doc_id"], r.get("sample_idx", 0))].append(r)
    for k in trajs:
        trajs[k].sort(key=lambda x: x["step_idx"])
    return trajs


def load_trajs_pareto(pareto_dir, fe_file):
    sm = load_jsonl(pareto_dir / "step_metrics.jsonl")
    fe = load_jsonl(fe_file)
    fe_map = {r["doc_id"]: r["tau"] for r in fe if r.get("tau") is not None}
    trajs = {}
    for rec in sm:
        did = rec["doc_id"]
        tau = fe_map.get(did)
        if tau is None or tau < 1:
            continue
        steps = rec.get("step_metrics", [])
        if not steps:
            continue
        traj = []
        for si, m in enumerate(steps):
            row = dict(m)
            row.update(doc_id=did, sample_idx=0, step_idx=si,
                       tau=tau, n_steps=len(steps))
            traj.append(row)
        trajs[(did, 0)] = traj
    return trajs


def sweep_metric(trajs, metric, higher_triggers):
    vals = []
    for steps in trajs.values():
        for s in steps:
            v = s.get(metric)
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    arr = np.array(vals)
    thrs = np.percentile(arr, np.linspace(1, 99, N_THRESHOLDS))
    is_lb = metric in LOOKBACK_METRICS
    h0, h1, h2, ftr = [], [], [], []
    for thr in thrs:
        hits = {0: 0, 1: 0, 2: 0}
        nt, tf = 0, 0
        for steps in trajs.values():
            tau = steps[0].get("tau")
            if tau is None or tau < 0:
                continue
            nt += 1
            n = len(steps)
            fired = [s["step_idx"] for s in steps
                     if (s.get(metric, 0) > thr) == higher_triggers]
            targets = set()
            for ft in fired:
                if is_lb and ft - 1 >= 0:
                    targets.add(ft - 1)
                targets.add(ft)
            for k in (0, 1, 2):
                if any(abs(idx - tau) <= k for idx in targets):
                    hits[k] += 1
            adj = {tau + dk for dk in (-1, 0, 1) if 0 <= tau + dk < n}
            tf += sum(1 for ft in fired if ft not in adj)
        if nt == 0:
            continue
        h0.append(hits[0] / nt)
        h1.append(hits[1] / nt)
        h2.append(hits[2] / nt)
        ftr.append(tf / nt)
    return dict(hit0=np.array(h0), hit1=np.array(h1),
                hit2=np.array(h2), ftr=np.array(ftr))


def sweep_logistic_combo(trajs):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    mnames = list(ALL_METRICS.keys()) + ["max_entropy"]
    X, y, keys = [], [], []
    for key, steps in trajs.items():
        tau = steps[0].get("tau")
        if tau is None or tau < 0:
            continue
        for s in steps:
            feats = [s.get(m, 0.0) or 0.0 for m in mnames]
            X.append(feats)
            y.append(1.0 if abs(s["step_idx"] - tau) <= 1 else 0.0)
            keys.append(key)
    if len(X) < 50:
        return None
    X, y = np.array(X), np.array(y)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X), y)
    scores = clf.predict_proba(sc.transform(X))[:, 1]
    scored_trajs = defaultdict(list)
    idx = 0
    for key, steps in trajs.items():
        tau = steps[0].get("tau")
        if tau is None or tau < 0:
            continue
        for s in steps:
            scored_trajs[key].append({**s, "combo_score": float(scores[idx])})
            idx += 1
    return sweep_metric(scored_trajs, "combo_score", higher_triggers=True)


COLORS = {"logprob_drop": "#C44E52", "min_logprob": "#DD8452",
           "entropy_delta": "#4C72B0", "mean_entropy": "#55A868",
           "combo_score": "#8172B2"}
LABELS = {"logprob_drop": "logprob_drop (lookback)",
           "min_logprob": "min_logprob (lookback)",
           "entropy_delta": "entropy_delta (lookahead)",
           "mean_entropy": "mean_entropy (lookahead)",
           "combo_score": "Logistic Combo"}


def plot_panel(ax, results, title):
    for m, r in results.items():
        if r is None:
            continue
        order = np.argsort(r["ftr"])
        ax.plot(r["ftr"][order], r["hit1"][order],
                color=COLORS.get(m, "gray"), label=LABELS.get(m, m),
                lw=2.5 if m == "combo_score" else 1.5, alpha=0.85,
                ls="--" if m == "combo_score" else "-")
    ax.set_xlabel("False Trigger Rate (per traj)", fontsize=9)
    ax.set_ylabel("Hit@1", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trigger Hit-Rate: Hit@1 vs False-Trigger Rate",
                 fontsize=14, fontweight="bold", y=1.01)
    summaries = {}
    for ri, (mid, ml) in enumerate(MODELS):
        for ci, (ds, dl) in enumerate(DATASETS):
            ax = axes[ri, ci]
            title = f"{ml} / {dl}"
            ms = _ms(mid)
            psm = ROOT / f"results/{ds}_3b_multi_sample/step_uncertainty/per_step_metrics.jsonl"
            pdir = ROOT / "results" / f"{ds}_{ms}_pareto"
            fe = ROOT / f"results/{ds}_3b_multi_sample/first_error/gpt_first_error_cache.jsonl"
            trajs = None
            if psm.exists() and "qwen" in ms:
                trajs = load_trajs_per_step(psm)
            elif (pdir / "step_metrics.jsonl").exists() and fe.exists():
                trajs = load_trajs_pareto(pdir, fe)
            if not trajs:
                ax.set_title(f"{title} (no data)")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                continue
            print(f"{title}: {len(trajs)} trajectories")
            results = {}
            for m, ht in ALL_METRICS.items():
                r = sweep_metric(trajs, m, ht)
                if r:
                    results[m] = r
                    bi = np.argmax(r["hit1"])
                    print(f"  {m}: Hit@1={r['hit1'][bi]:.3f} FTR={r['ftr'][bi]:.3f}")
            combo = sweep_logistic_combo(trajs)
            if combo:
                results["combo_score"] = combo
                bi = np.argmax(combo["hit1"])
                print(f"  combo: Hit@1={combo['hit1'][bi]:.3f} FTR={combo['ftr'][bi]:.3f}")
            plot_panel(ax, results, title)
            summaries[f"{ml}_{dl}"] = {
                m: {"best_hit1": float(r["hit1"][np.argmax(r["hit1"])]),
                    "ftr_at_best": float(r["ftr"][np.argmax(r["hit1"])])}
                for m, r in results.items()
            }
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_trigger_hitrate_multi.{ext}",
                    dpi=200, bbox_inches="tight")
    print(f"\nSaved to {FIG_DIR}")
    plt.close(fig)
    (FIG_DIR / "trigger_hitrate_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
