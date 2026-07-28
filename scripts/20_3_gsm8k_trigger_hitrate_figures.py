#!/usr/bin/env python3
"""
Publication-style GSM8K trajectory-level trigger hit-rate figures.

Uses the step-level uncertainty file produced by scripts/4_1 and makes:
  1. Hit@1 vs false-trigger rate per trajectory.
  2. Hit@0/1/2 panels vs false-trigger rate per trajectory.

Note: this FTR is measured on non-error positions inside incorrect
trajectories, matching scripts/20_2.py. A stricter correct-trajectory FTR
requires collecting step metrics for correct trajectories too.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures" / "trigger_hitrate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT / "results/gsm8k_3b_multi_sample/step_uncertainty/per_step_metrics.jsonl"


def load_20_2():
    spec = importlib.util.spec_from_file_location("trigger20_2", ROOT / "scripts/20_2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sorted_curve(result, key):
    order = np.argsort(result["ftr"])
    return result["ftr"][order], result[key][order]


def closest_op(result, target_ftr=1.0):
    idx = int(np.argmin(np.abs(result["ftr"] - target_ftr)))
    return {
        "ftr": float(result["ftr"][idx]),
        "hit0": float(result["hit0"][idx]),
        "hit1": float(result["hit1"][idx]),
        "hit2": float(result["hit2"][idx]),
    }


def best_under(result, max_ftr=1.0):
    mask = result["ftr"] <= max_ftr
    if not mask.any():
        return closest_op(result, max_ftr)
    scores = np.where(mask, result["hit1"], -1.0)
    idx = int(np.argmax(scores))
    return {
        "ftr": float(result["ftr"][idx]),
        "hit0": float(result["hit0"][idx]),
        "hit1": float(result["hit1"][idx]),
        "hit2": float(result["hit2"][idx]),
    }


def main():
    mod = load_20_2()
    trajs = mod.load_trajs_per_step(DATA_PATH)
    results = {}
    for metric, higher in mod.ALL_METRICS.items():
        r = mod.sweep_metric(trajs, metric, higher)
        if r is not None:
            results[metric] = r
    combo = mod.sweep_logistic_combo(trajs)
    if combo is not None:
        results["combo_score"] = combo

    summary = {
        "dataset": "gsm8k",
        "model": "Qwen2.5-3B-Instruct",
        "n_trajectories": len(trajs),
        "ftr_definition": "non-error positions in incorrect trajectories",
        "operating_points": {
            name: {
                "closest_ftr_1": closest_op(res, 1.0),
                "best_hit1_under_ftr_1": best_under(res, 1.0),
            }
            for name, res in results.items()
        },
    }
    (OUT_DIR / "gsm8k_trigger_hitrate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, res in results.items():
        x, y = sorted_curve(res, "hit1")
        lw = 2.8 if name == "combo_score" else 2.0
        ls = "--" if name == "combo_score" else "-"
        ax.plot(x, y, label=mod.LABELS.get(name, name), lw=lw, ls=ls,
                color=mod.COLORS.get(name), alpha=0.9)
        op = closest_op(res, 1.0)
        ax.scatter([op["ftr"]], [op["hit1"]], s=28,
                   color=mod.COLORS.get(name), zorder=3)
    ax.axvline(1.0, color="black", lw=1.2, ls=":", alpha=0.75)
    ax.text(1.02, 0.04, "1 false trigger / traj", rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom", ha="left",
            fontsize=9)
    ax.set_title("GSM8K Trajectory-Level Trigger Hit-Rate")
    ax.set_xlabel("False-trigger rate per trajectory")
    ax.set_ylabel("Hit@1")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gsm8k_trigger_hit1_roc.png", dpi=220)
    fig.savefig(OUT_DIR / "gsm8k_trigger_hit1_roc.pdf", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), sharex=True, sharey=True)
    for ax, hit_key, title in zip(
        axes, ["hit0", "hit1", "hit2"], ["Hit@0", "Hit@1", "Hit@2"]
    ):
        for name, res in results.items():
            x, y = sorted_curve(res, hit_key)
            lw = 2.5 if name == "combo_score" else 1.7
            ls = "--" if name == "combo_score" else "-"
            ax.plot(x, y, label=mod.LABELS.get(name, name), lw=lw, ls=ls,
                    color=mod.COLORS.get(name), alpha=0.9)
        ax.axvline(1.0, color="black", lw=1.0, ls=":", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("False-trigger rate / traj")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Trajectory hit rate")
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(0, 1.03)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("GSM8K Trigger Localization at Different Tolerances")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gsm8k_trigger_hitk_panels.png", dpi=220)
    fig.savefig(OUT_DIR / "gsm8k_trigger_hitk_panels.pdf", dpi=220)
    plt.close(fig)

    print(f"Saved figures and summary to {OUT_DIR}")
    for name, vals in summary["operating_points"].items():
        op = vals["closest_ftr_1"]
        print(
            f"{name:14s} FTR={op['ftr']:.3f} "
            f"Hit@0={op['hit0']:.3f} Hit@1={op['hit1']:.3f} Hit@2={op['hit2']:.3f}"
        )


if __name__ == "__main__":
    main()
