#!/usr/bin/env python3
"""
Conditional error probability: P(next/prev step = error | current step entropy).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.keystep_utils import load_jsonl

DDIR = str(ROOT / "results/gsm8k_3b_multi_sample/step_uncertainty")
FDIR = str(ROOT / "figures/step_uncertainty")
MS = ["mean_entropy","max_entropy","entropy_delta","mean_logprob","min_logprob","logprob_drop"]
ML = dict(mean_entropy="Mean Entropy",max_entropy="Max Entropy",entropy_delta="Entropy Delta",
          mean_logprob="Mean LogProb",min_logprob="Min LogProb",logprob_drop="LogProb Drop")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DDIR)
    ap.add_argument("--fig-dir", default=FDIR)
    ap.add_argument("--n-bins", type=int, default=10)
    return ap.parse_args()

def build_datasets(per_step):
    trajs = defaultdict(list)
    for r in per_step:
        trajs[(r["doc_id"], r["sample_idx"])].append(r)
    nxt, prv = [], []
    for key, steps in trajs.items():
        steps = sorted(steps, key=lambda s: s["step_idx"])
        tau = steps[0]["tau"]
        if tau is None or tau < 1:
            continue
        n, eidx = len(steps), tau - 1
        for i, s in enumerate(steps):
            f = {m: s[m] for m in MS}
            f["step_idx"], f["n_steps"] = i, n
            f["relative_position"] = i / max(n - 1, 1)
            base = {**f, "doc_id": s["doc_id"], "sample_idx": s["sample_idx"]}
            if i < n - 1:
                nxt.append({**base, "label": int(i + 1 == eidx)})
            if i > 0:
                prv.append({**base, "label": int(i - 1 == eidx)})
    return nxt, prv

def emp_prob(rows, metric, nb=10):
    v = np.array([r[metric] for r in rows])
    lb = np.array([r["label"] for r in rows])
    try:
        edges = np.unique(np.percentile(v, np.linspace(0, 100, nb + 1)))
    except Exception:
        return [], [], [], []
    if len(edges) < 2:
        return [], [], [], []
    bi = np.digitize(v, edges[1:-1])
    cs, ps, ns, ci = [], [], [], []
    for b in range(len(edges) - 1):
        mask = bi == b
        if mask.sum() == 0:
            continue
        bl = lb[mask]
        p, nn = bl.mean(), len(bl)
        se = np.sqrt(p * (1 - p) / nn) if nn > 1 else 0
        cs.append((edges[b] + edges[min(b + 1, len(edges) - 1)]) / 2)
        ps.append(p)
        ns.append(nn)
        ci.append(1.96 * se)
    return cs, ps, ns, ci

def fit_lr(rows, feats):
    X = np.array([[r[f] for f in feats] for r in rows])
    y = np.array([r["label"] for r in rows])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    mdl = LogisticRegression(max_iter=1000, class_weight="balanced")
    mdl.fit(Xs, y)
    pr = mdl.predict_proba(Xs)[:, 1]
    try:
        auc = roc_auc_score(y, pr)
    except ValueError:
        auc = 0.5
    return mdl, sc, auc, pr

def fig_cond_prob(nxt, prv, nb, fdir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, metric in enumerate(MS):
        ax = axes[idx // 3, idx % 3]
        c1, p1, _, ci1 = emp_prob(nxt, metric, nb)
        if c1:
            ax.errorbar(c1, p1, yerr=ci1, marker="o", capsize=3,
                        color="#2196F3", linewidth=2, markersize=5,
                        label="P(next=error)")
        c2, p2, _, ci2 = emp_prob(prv, metric, nb)
        if c2:
            ax.errorbar(c2, p2, yerr=ci2, marker="s", capsize=3,
                        color="#E91E63", linewidth=2, markersize=5,
                        label="P(prev=error)")
        ax.set_xlabel(ML[metric])
        ax.set_ylabel("P(error)")
        ax.set_title(ML[metric])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Conditional Error Probability by Step-Level Uncertainty\n"
                 "(GSM8K, Qwen2.5-3B-Instruct)", fontsize=13, y=1.02)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fdir / f"fig_conditional_error_prob.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_conditional_error_prob")

def fig_lr_coefs(models_info, fdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, info) in zip(axes, models_info.items()):
        mdl = info["model"]
        fnames = info["features"]
        coefs = mdl.coef_[0]
        auc = info["auc"]
        si = np.argsort(np.abs(coefs))[::-1]
        names = [ML.get(fnames[i], fnames[i]) for i in si]
        vals = [coefs[i] for i in si]
        colors = ["#EF5350" if v > 0 else "#42A5F5" for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Logistic Regression Coefficient")
        ax.set_title(f"{label} (AUC={auc:.3f})")
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle("Feature Importance for Error Adjacency Prediction", y=1.02)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fdir / f"fig_logistic_coefficients.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_logistic_coefficients")

def fig_heatmap(nxt, prv, nb, fdir):
    for direction, rows, cmap in [("next", nxt, "Reds"), ("prev", prv, "Blues")]:
        m1 = np.array([r["mean_entropy"] for r in rows])
        m2 = np.array([r["entropy_delta"] for r in rows])
        labels = np.array([r["label"] for r in rows])
        try:
            e1 = np.percentile(m1, np.linspace(0, 100, nb + 1))
            e2 = np.percentile(m2, np.linspace(0, 100, nb + 1))
        except Exception:
            continue
        b1 = np.clip(np.digitize(m1, e1[1:-1]), 0, nb - 1)
        b2 = np.clip(np.digitize(m2, e2[1:-1]), 0, nb - 1)
        grid = np.full((nb, nb), np.nan)
        for i in range(nb):
            for j in range(nb):
                mask = (b1 == i) & (b2 == j)
                if mask.sum() >= 5:
                    grid[j, i] = labels[mask].mean()
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                       vmin=0, interpolation="nearest")
        ax.set_xlabel("Mean Entropy (quantile bin)")
        ax.set_ylabel("Entropy Delta (quantile bin)")
        tdir = "Next Step" if direction == "next" else "Previous Step"
        ax.set_title(f"P({tdir} = Error) by Entropy Features")
        plt.colorbar(im, ax=ax, label="P(error)")
        fig.tight_layout()
        for fmt in ("png", "pdf"):
            fig.savefig(fdir / f"fig_prob_heatmap_{direction}.{fmt}",
                        dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> fig_prob_heatmap_{direction}")

def main():
    args = parse_args()
    ddir = Path(args.data_dir)
    fdir = Path(args.fig_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    per_step = load_jsonl(ddir / "per_step_metrics.jsonl")
    if not per_step:
        print("No per-step data. Run 4_1 first.")
        return
    print(f"Loaded {len(per_step)} step rows")

    nxt, prv = build_datasets(per_step)
    nn = sum(r["label"] for r in nxt)
    np_ = sum(r["label"] for r in prv)
    print(f"Next-error: {len(nxt)} rows, {nn} pos ({nn/len(nxt)*100:.1f}%)")
    print(f"Prev-error: {len(prv)} rows, {np_} pos ({np_/len(prv)*100:.1f}%)")

    # Empirical binned probabilities
    sep = "=" * 70
    print(f"\n{sep}")
    print("Empirical P(error) by metric decile")
    print(sep)
    for direction, rows in [("NEXT", nxt), ("PREV", prv)]:
        print(f"\n--- {direction} step is error ---")
        for metric in MS:
            cs, ps, _, _ = emp_prob(rows, metric, args.n_bins)
            if not cs:
                continue
            lo, hi = ps[0], ps[-1]
            ratio = hi / lo if lo > 0 else float("inf")
            print(f"  {ML[metric]:<18}  lo={lo:.4f}  hi={hi:.4f}  ratio={ratio:.2f}x")

    # Logistic regression
    print(f"\n{sep}")
    print("Logistic Regression (all 6 metrics)")
    print(sep)
    mi = {}
    for direction, rows in [("P(next=error)", nxt), ("P(prev=error)", prv)]:
        mdl, sc, auc, pr = fit_lr(rows, MS)
        print(f"\n{direction}:  AUC = {auc:.4f}")
        coefs = mdl.coef_[0]
        for m, c in sorted(zip(MS, coefs), key=lambda x: -abs(x[1])):
            print(f"  {ML[m]:<18}  coef={c:+.4f}")
        mi[direction] = {"model": mdl, "scaler": sc, "auc": auc, "features": MS}

    # Univariate AUC
    print(f"\n{sep}")
    print("Univariate AUC per metric")
    print(sep)
    print(f"{'Metric':<18} {'AUC(next)':>10} {'AUC(prev)':>10}")
    print("-" * 40)
    auc_res = {}
    for metric in MS:
        aucs = {}
        for d, rows in [("next", nxt), ("prev", prv)]:
            vals = np.array([r[metric] for r in rows]).reshape(-1, 1)
            labels = np.array([r["label"] for r in rows])
            try:
                a = roc_auc_score(labels, vals)
                a = max(a, 1 - a)
            except ValueError:
                a = 0.5
            aucs[d] = round(a, 4)
        auc_res[metric] = aucs
        print(f"{ML[metric]:<18} {aucs['next']:>10.4f} {aucs['prev']:>10.4f}")

    # Save summary
    summary = {
        "next_error": {"n": len(nxt), "n_pos": nn, "lr_auc": round(mi["P(next=error)"]["auc"], 4)},
        "prev_error": {"n": len(prv), "n_pos": np_, "lr_auc": round(mi["P(prev=error)"]["auc"], 4)},
        "univariate_auc": auc_res,
    }
    sp = ddir / "conditional_error_prob_summary.json"
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {sp}")

    # Figures
    print("\nGenerating figures...")
    fig_cond_prob(nxt, prv, args.n_bins, fdir)
    fig_lr_coefs(mi, fdir)
    fig_heatmap(nxt, prv, min(args.n_bins, 8), fdir)
    print(f"\nAll figures -> {fdir}")

if __name__ == "__main__":
    main()
