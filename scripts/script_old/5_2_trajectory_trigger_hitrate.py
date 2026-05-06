"""
Trajectory-level trigger hit-rate analysis.

Reads per_step_metrics.jsonl, groups by trajectory, sweeps thresholds
per metric, and computes Hit@k and false-trigger rates.
Produces fig_trigger_roc.png/pdf.
"""

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.keystep_utils import load_jsonl

# ---------------------------------------------------------------------------
ROOT = pathlib.Path("/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond")
DATA = ROOT / "results/gsm8k_3b_multi_sample/step_uncertainty/per_step_metrics.jsonl"
FIG_DIR = ROOT / "figures/step_uncertainty"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric configs: (metric_name, higher_triggers, hit_check)
# ---------------------------------------------------------------------------
LOOKBACK_METRICS = {
    "logprob_drop": {"higher_triggers": False},
    "min_logprob":  {"higher_triggers": False},
}
LOOKAHEAD_METRICS = {
    "entropy_delta": {"higher_triggers": True},
    "mean_entropy":  {"higher_triggers": True},
}
ALL_METRICS = {**LOOKBACK_METRICS, **LOOKAHEAD_METRICS}

N_THRESHOLDS = 50


def load_trajectories(path: pathlib.Path):
    """Group records into trajectories keyed by (doc_id, sample_idx)."""
    rows = load_jsonl(path)
    trajs = defaultdict(list)
    for r in rows:
        key = (r["doc_id"], r["sample_idx"])
        trajs[key].append(r)
    for key in trajs:
        trajs[key].sort(key=lambda x: x["step_idx"])
    return trajs


def sweep_metric(trajs, metric_name, higher_triggers):
    """Sweep thresholds for one metric. Return arrays for ROC curve."""
    all_vals = []
    for steps in trajs.values():
        for s in steps:
            v = s.get(metric_name)
            if v is not None:
                all_vals.append(v)
    if not all_vals:
        return None
    all_vals = np.array(all_vals)
    pcts = np.linspace(1, 99, N_THRESHOLDS)
    thresholds = np.percentile(all_vals, pcts)

    hit0_arr, hit1_arr, hit2_arr, ftr_arr = [], [], [], []

    for thr in thresholds:
        hits = {0: 0, 1: 0, 2: 0}
        n_traj = 0
        total_false = 0
        total_steps = 0

        for steps in trajs.values():
            tau = steps[0].get("tau")
            if tau is None or tau < 0:
                continue
            n_traj += 1
            n = len(steps)
            fired_steps = []

            for s in steps:
                t = s["step_idx"]
                val = s.get(metric_name, 0.0)
                fire = (val > thr) if higher_triggers else (val < thr)
                if fire:
                    fired_steps.append(t)

            is_lookback = metric_name in LOOKBACK_METRICS
            hit_indices = set()
            for ft in fired_steps:
                if is_lookback:
                    target = ft - 1
                else:
                    target = ft
                hit_indices.add(target)
                if is_lookback:
                    hit_indices.add(ft)

            for k in (0, 1, 2):
                found = any(abs(idx - tau) <= k for idx in hit_indices)
                if found:
                    hits[k] += 1

            error_adjacent = set()
            for dk in (-1, 0, 1):
                adj = tau + dk
                if 0 <= adj < n:
                    error_adjacent.add(adj)
            n_false = sum(1 for ft in fired_steps if ft not in error_adjacent)
            total_false += n_false
            total_steps += n

        if n_traj == 0:
            continue
        hit0_arr.append(hits[0] / n_traj)
        hit1_arr.append(hits[1] / n_traj)
        hit2_arr.append(hits[2] / n_traj)
        ftr_arr.append(total_false / n_traj)

    return {
        "thresholds": thresholds[:len(hit0_arr)],
        "hit0": np.array(hit0_arr),
        "hit1": np.array(hit1_arr),
        "hit2": np.array(hit2_arr),
        "ftr": np.array(ftr_arr),
    }


def main():
    print(f"Loading {DATA}")
    trajs = load_trajectories(DATA)
    print(f"  {len(trajs)} trajectories loaded")

    results = {}
    for metric_name, cfg in ALL_METRICS.items():
        r = sweep_metric(trajs, metric_name, cfg["higher_triggers"])
        if r is not None:
            results[metric_name] = r
            best_idx = np.argmax(r["hit1"])
            print(f"  {metric_name}: best Hit@1={r['hit1'][best_idx]:.3f} "
                  f"at FTR={r['ftr'][best_idx]:.3f}")

    # --- ROC-style figure ---
    COLORS = {"logprob_drop": "#C44E52", "min_logprob": "#DD8452",
              "entropy_delta": "#4C72B0", "mean_entropy": "#55A868"}
    LABELS = {"logprob_drop": "logprob_drop (lookback)",
              "min_logprob": "min_logprob (lookback)",
              "entropy_delta": "entropy_delta (lookahead)",
              "mean_entropy": "mean_entropy (lookahead)"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric_name, r in results.items():
        order = np.argsort(r["ftr"])
        ax.plot(r["ftr"][order], r["hit1"][order],
                color=COLORS.get(metric_name, "gray"),
                label=LABELS.get(metric_name, metric_name),
                linewidth=2, alpha=0.85)

    ax.set_xlabel("False Trigger Rate (avg non-error triggers per trajectory)", fontsize=11)
    ax.set_ylabel("Hit@1 (fraction of trajectories)", fontsize=11)
    ax.set_title("Trigger Performance: Hit@1 vs False-Trigger Rate", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_trigger_roc.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved fig_trigger_roc to {FIG_DIR}")
    plt.close(fig)

    # --- Summary table ---
    print("\n" + "=" * 80)
    print(f"{'Metric':<22} {'Best Hit@0':>10} {'Best Hit@1':>10} "
          f"{'Best Hit@2':>10} {'FTR@best':>10}")
    print("-" * 80)
    for metric_name, r in results.items():
        bi = np.argmax(r["hit1"])
        print(f"{metric_name:<22} {r['hit0'][bi]:>10.3f} {r['hit1'][bi]:>10.3f} "
              f"{r['hit2'][bi]:>10.3f} {r['ftr'][bi]:>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
