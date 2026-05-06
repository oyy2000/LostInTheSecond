#!/usr/bin/env python3
"""
Three-way analysis: Bad prefix vs Minimal repair vs Correct-prefix control.

Controls for question difficulty by comparing:
  - R_bad: continue from bad prefix (includes error)
  - R_fix: continue from repaired prefix (error step corrected)
  - R_good: continue from correct prefix (same length, same question)

Key insight: if R_good is high across all tau, then low R_bad is due to the
bad prefix (not question difficulty). The "damage" of a bad prefix is:
  Damage = R_good - R_bad

Usage:
    python scripts/6_6_three_way_analysis.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Three-way comparison analysis")
    ap.add_argument("--bad-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/bad_prefix_recovery/continuations.jsonl"))
    ap.add_argument("--fix-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/minimal_repair/continuations.jsonl"))
    ap.add_argument("--good-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/correct_prefix_control/continuations.jsonl"))
    ap.add_argument("--fig-dir", default=str(
        PROJECT_ROOT / "figures/bad_prefix_recovery"))
    ap.add_argument("--out-summary", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/three_way_summary.json"))
    return ap.parse_args()


def load_per_sample_rates(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    by_sample = defaultdict(list)
    for r in rows:
        key = (r["doc_id"], r["sample_idx"])
        by_sample[key].append(r)
    out = {}
    for key, recs in by_sample.items():
        n_correct = sum(1 for r in recs if r["exact_match"] >= 1.0)
        out[key] = {
            "rate": n_correct / len(recs),
            "bucket": recs[0]["bucket"],
            "tau": recs[0]["tau"],
            "n_steps": recs[0]["n_steps"],
        }
    return out


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    bad_path = Path(args.bad_file)
    fix_path = Path(args.fix_file)
    good_path = Path(args.good_file)

    for p, name in [(bad_path, "bad"), (fix_path, "fix"), (good_path, "good")]:
        if not p.exists():
            print(f"ERROR: {name} file not found: {p}")
            sys.exit(1)

    bad_rates = load_per_sample_rates(bad_path)
    fix_rates = load_per_sample_rates(fix_path)
    good_rates = load_per_sample_rates(good_path)
    print(f"Bad: {len(bad_rates)}, Fix: {len(fix_rates)}, Good: {len(good_rates)}")

    # Common samples across all three
    common = set(bad_rates.keys()) & set(fix_rates.keys()) & set(good_rates.keys())
    print(f"Common samples (all 3 conditions): {len(common)}")

    paired = []
    for key in common:
        tau = bad_rates[key]["tau"]
        n_steps = bad_rates[key]["n_steps"]
        paired.append({
            "doc_id": key[0],
            "sample_idx": key[1],
            "bucket": bad_rates[key]["bucket"],
            "tau": tau,
            "n_steps": n_steps,
            "rel_pos": tau / n_steps,
            "r_bad": bad_rates[key]["rate"],
            "r_fix": fix_rates[key]["rate"],
            "r_good": good_rates[key]["rate"],
            "delta_fix": fix_rates[key]["rate"] - bad_rates[key]["rate"],
            "damage": good_rates[key]["rate"] - bad_rates[key]["rate"],
        })

    # --- By bucket ---
    by_bucket = defaultdict(list)
    for p in paired:
        by_bucket[p["bucket"]].append(p)

    print("\n=== Three-way results by bucket ===")
    summary = {}
    for bucket in ["early", "late"]:
        items = by_bucket[bucket]
        if not items:
            continue
        n = len(items)
        r_bad = np.array([x["r_bad"] for x in items])
        r_fix = np.array([x["r_fix"] for x in items])
        r_good = np.array([x["r_good"] for x in items])
        damage = np.array([x["damage"] for x in items])
        delta_fix = np.array([x["delta_fix"] for x in items])

        summary[bucket] = {
            "n": n,
            "r_bad": float(np.mean(r_bad)),
            "r_fix": float(np.mean(r_fix)),
            "r_good": float(np.mean(r_good)),
            "damage_mean": float(np.mean(damage)),
            "damage_se": float(np.std(damage, ddof=1) / np.sqrt(n)),
            "delta_fix_mean": float(np.mean(delta_fix)),
            "delta_fix_se": float(np.std(delta_fix, ddof=1) / np.sqrt(n)),
        }
        print(f"  {bucket} (n={n}):")
        print(f"    R_bad  = {np.mean(r_bad):.4f}")
        print(f"    R_fix  = {np.mean(r_fix):.4f}")
        print(f"    R_good = {np.mean(r_good):.4f}")
        print(f"    Damage (R_good - R_bad) = {np.mean(damage):.4f} +/- {np.std(damage,ddof=1)/np.sqrt(n):.4f}")
        print(f"    Delta_fix = {np.mean(delta_fix):.4f}")

    # Between-bucket test on Damage
    damage_early = np.array([x["damage"] for x in by_bucket["early"]])
    damage_late = np.array([x["damage"] for x in by_bucket["late"]])
    t_dmg, p_dmg = sp_stats.ttest_ind(damage_early, damage_late, equal_var=False)
    print(f"\n  Between-bucket Damage test: t={t_dmg:.3f}, p={p_dmg:.2e}")

    # --- By relative position ---
    rel_positions = np.array([p["rel_pos"] for p in paired])
    damages_all = np.array([p["damage"] for p in paired])
    r_good_all = np.array([p["r_good"] for p in paired])
    r_bad_all = np.array([p["r_bad"] for p in paired])

    rho_dmg, p_rho_dmg = spearmanr(rel_positions, damages_all)
    rho_good, p_rho_good = spearmanr(rel_positions, r_good_all)
    rho_bad, p_rho_bad = spearmanr(rel_positions, r_bad_all)

    print(f"\n=== Relative position correlations ===")
    print(f"  Spearman(rel_pos, R_good): rho={rho_good:.4f}, p={p_rho_good:.2e}")
    print(f"  Spearman(rel_pos, R_bad):  rho={rho_bad:.4f}, p={p_rho_bad:.2e}")
    print(f"  Spearman(rel_pos, Damage): rho={rho_dmg:.4f}, p={p_rho_dmg:.2e}")

    # --- By tau ---
    by_tau = defaultdict(list)
    for p in paired:
        by_tau[p["tau"]].append(p)

    print("\n=== By tau ===")
    tau_summary = {}
    for tau in sorted(by_tau.keys()):
        items = by_tau[tau]
        n = len(items)
        tau_summary[tau] = {
            "n": n,
            "r_bad": float(np.mean([x["r_bad"] for x in items])),
            "r_fix": float(np.mean([x["r_fix"] for x in items])),
            "r_good": float(np.mean([x["r_good"] for x in items])),
            "damage": float(np.mean([x["damage"] for x in items])),
        }
        print(f"  tau={tau} (n={n}): R_bad={tau_summary[tau]['r_bad']:.4f}, "
              f"R_fix={tau_summary[tau]['r_fix']:.4f}, "
              f"R_good={tau_summary[tau]['r_good']:.4f}, "
              f"Damage={tau_summary[tau]['damage']:.4f}")

    # --- Figure: Three-way grouped bar ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.arange(2)
    width = 0.25
    buckets_list = ["early", "late"]
    r_bad_m = [summary[b]["r_bad"] for b in buckets_list]
    r_fix_m = [summary[b]["r_fix"] for b in buckets_list]
    r_good_m = [summary[b]["r_good"] for b in buckets_list]

    ax.bar(x - width, r_bad_m, width, label="$R_{bad}$ (bad prefix)",
           color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x, r_fix_m, width, label="$R_{fix}$ (repaired)",
           color="#3498db", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width, r_good_m, width, label="$R_{good}$ (correct prefix)",
           color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["early ($\\tau$=2,3)", "late ($\\tau$=4,5,6)"])
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Three-way Comparison (difficulty-controlled)")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(r_good_m) * 1.3)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_three_way.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_three_way.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'fig_three_way.pdf'}")

    # --- Figure: Three curves by tau ---
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    taus = sorted(tau_summary.keys())
    ax.plot(taus, [tau_summary[t]["r_bad"] for t in taus], "o-",
            color="#e74c3c", linewidth=2, markersize=7, label="$R_{bad}$")
    ax.plot(taus, [tau_summary[t]["r_fix"] for t in taus], "s-",
            color="#3498db", linewidth=2, markersize=7, label="$R_{fix}$")
    ax.plot(taus, [tau_summary[t]["r_good"] for t in taus], "^-",
            color="#2ecc71", linewidth=2, markersize=7, label="$R_{good}$ (control)")
    ax.fill_between(taus, [tau_summary[t]["r_bad"] for t in taus],
                    [tau_summary[t]["r_good"] for t in taus],
                    alpha=0.1, color="#e74c3c", label="Damage")
    ax.set_xlabel("First Error Step ($\\tau$)")
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Recovery by Condition and Error Position")
    ax.set_xticks(taus)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_three_way_by_tau.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_three_way_by_tau.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_three_way_by_tau.pdf'}")

    # --- Figure: Damage by relative position ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(rel_positions, damages_all, alpha=0.12, s=12, color="#2c3e50")
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bc, bm, bse = [], [], []
    for i in range(n_bins):
        mask = (rel_positions >= bin_edges[i]) & (rel_positions < bin_edges[i+1])
        if mask.sum() >= 5:
            bc.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bm.append(np.mean(damages_all[mask]))
            bse.append(1.96 * np.std(damages_all[mask]) / np.sqrt(mask.sum()))
    ax.errorbar(bc, bm, yerr=bse, color="#e74c3c", linewidth=2,
                marker="o", markersize=5, capsize=3, label="binned mean")
    ax.set_xlabel("Relative Error Position ($\\tau / N_{steps}$)")
    ax.set_ylabel("Damage ($R_{good} - R_{bad}$)")
    ax.set_title(f"Prefix Damage vs Position ($\\rho_s$={rho_dmg:.3f}, p={p_rho_dmg:.1e})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_damage_by_relpos.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "fig_damage_by_relpos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'fig_damage_by_relpos.pdf'}")

    # Save summary
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_summary = {
        "by_bucket": summary,
        "by_tau": {str(k): v for k, v in tau_summary.items()},
        "between_bucket_damage_test": {
            "welch_t": float(t_dmg), "welch_p": float(p_dmg),
        },
        "relative_position": {
            "spearman_rho_damage": float(rho_dmg),
            "spearman_p_damage": float(p_rho_dmg),
            "spearman_rho_r_good": float(rho_good),
            "spearman_p_r_good": float(p_rho_good),
        },
        "n_samples": len(paired),
    }
    out_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
