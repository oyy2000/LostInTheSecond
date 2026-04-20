#!/usr/bin/env python3
"""
Scaling law analysis for reasoning mode stability.

Reads the multi_scale_summary.json from 22_multi_scale_baseline.py and fits
power-law / log-linear relationships between model size and mode stability
metrics (dip_depth, recovery_step, stability_ratio, etc.).

This is the paper-level analysis: does mode stability have its own scaling
exponent distinct from accuracy scaling?

Usage:
    python 26_scaling_law_mode_stability.py
    python 26_scaling_law_mode_stability.py --summary-file path/to/summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-file",
        default=str(PROJECT_ROOT / "runs" / "multi_scale_prm" / "multi_scale_summary.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "runs" / "scaling_law_analysis"),
    )
    return ap.parse_args()


def _model_params(tag: str) -> float:
    for size, val in [("0.5b", 5e8), ("1.5b", 1.5e9), ("3b", 3e9),
                      ("7b", 7e9), ("14b", 14e9), ("32b", 32e9)]:
        if size in tag.lower():
            return val
    return 1e9


def power_law(x, a, alpha):
    """y = a * x^(-alpha)"""
    return a * np.power(x, -alpha)


def log_linear(x, a, b):
    """y = a * log(x) + b"""
    return a * np.log(x) + b


def fit_scaling(
    params: List[float],
    values: List[float],
    metric_name: str,
) -> Dict[str, Any]:
    """Fit power law and log-linear to the data."""
    x = np.array(params)
    y = np.array(values)

    fits = {}

    # Power law: y = a * N^(-alpha)
    try:
        popt, pcov = curve_fit(power_law, x, y, p0=[1.0, 0.3], maxfev=5000)
        y_pred = power_law(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fits["power_law"] = {
            "a": round(float(popt[0]), 6),
            "alpha": round(float(popt[1]), 6),
            "r2": round(float(r2), 6),
            "formula": f"{metric_name} = {popt[0]:.4f} * N^(-{popt[1]:.4f})",
        }
    except Exception as e:
        fits["power_law"] = {"error": str(e)}

    # Log-linear: y = a * log(N) + b
    try:
        popt, pcov = curve_fit(log_linear, x, y, p0=[-0.01, 0.5], maxfev=5000)
        y_pred = log_linear(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fits["log_linear"] = {
            "a": round(float(popt[0]), 6),
            "b": round(float(popt[1]), 6),
            "r2": round(float(r2), 6),
            "formula": f"{metric_name} = {popt[0]:.4f} * ln(N) + {popt[1]:.4f}",
        }
    except Exception as e:
        fits["log_linear"] = {"error": str(e)}

    return fits


def plot_scaling_laws(
    metrics_by_model: Dict[str, dict],
    out_dir: Path,
):
    """Create the paper-level scaling law figure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tags_sorted = sorted(metrics_by_model.keys(), key=_model_params)
    params = np.array([_model_params(t) for t in tags_sorted])

    metric_configs = [
        ("wrong_dip_depth", "Dip Depth (wrong)", True),
        ("wrong_min_score", "Min PRM Score (wrong)", False),
        ("wrong_stability_ratio", "Stability Ratio (wrong)", False),
        ("accuracy", "Final Answer Accuracy", False),
    ]

    valid_metrics = []
    for key, label, invert_better in metric_configs:
        vals = [metrics_by_model[t].get(key) for t in tags_sorted]
        if all(v is not None for v in vals):
            valid_metrics.append((key, label, invert_better, vals))

    if not valid_metrics:
        print("[WARN] No valid metrics to plot")
        return {}

    n_metrics = len(valid_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(7 * ((n_metrics + 1) // 2), 12))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    all_fits = {}

    for idx, (key, label, invert_better, vals) in enumerate(valid_metrics):
        ax = axes[idx]
        vals_arr = np.array(vals, dtype=float)

        ax.scatter(params, vals_arr, s=80, zorder=5, color="steelblue")
        for p, v, t in zip(params, vals_arr, tags_sorted):
            ax.annotate(t, (p, v), textcoords="offset points",
                        xytext=(5, 8), fontsize=9, ha="left")

        fits = fit_scaling(params.tolist(), vals_arr.tolist(), key)
        all_fits[key] = fits

        x_smooth = np.logspace(np.log10(params.min() * 0.5),
                               np.log10(params.max() * 3), 50)

        best_fit_name = None
        best_r2 = -1
        for fit_name in ["power_law", "log_linear"]:
            fit = fits.get(fit_name, {})
            if "error" in fit:
                continue
            r2 = fit.get("r2", -1)
            if r2 > best_r2:
                best_r2 = r2
                best_fit_name = fit_name

        if best_fit_name == "power_law" and "error" not in fits.get("power_law", {}):
            fit = fits["power_law"]
            y_smooth = power_law(x_smooth, fit["a"], fit["alpha"])
            ax.plot(x_smooth, y_smooth, "--", color="red", linewidth=1.5,
                    label=f"Power law: α={fit['alpha']:.3f}, R²={fit['r2']:.3f}")
        elif best_fit_name == "log_linear" and "error" not in fits.get("log_linear", {}):
            fit = fits["log_linear"]
            y_smooth = log_linear(x_smooth, fit["a"], fit["b"])
            ax.plot(x_smooth, y_smooth, "--", color="orange", linewidth=1.5,
                    label=f"Log-linear: R²={fit['r2']:.3f}")

        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters (N)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Hide unused axes
    for idx in range(len(valid_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Scaling Laws of Reasoning Mode Stability\nQwen2.5-Instruct Family on MATH-500",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "scaling_laws.png", dpi=200, bbox_inches="tight")
    plt.close()

    return all_fits


def plot_dip_vs_accuracy(
    metrics_by_model: Dict[str, dict],
    out_dir: Path,
):
    """Scatter plot: dip depth vs accuracy, colored by model size."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = sorted(metrics_by_model.keys(), key=_model_params)
    accs = []
    dips = []
    valid_tags = []

    for t in tags:
        m = metrics_by_model[t]
        acc = m.get("accuracy")
        dip = m.get("wrong_dip_depth")
        if acc is not None and dip is not None:
            accs.append(acc)
            dips.append(dip)
            valid_tags.append(t)

    if len(valid_tags) < 2:
        return

    params = [_model_params(t) for t in valid_tags]
    sizes = [max(40, p / 1e8) for p in params]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(accs, dips, s=sizes, c=np.log10(params),
                         cmap="viridis", zorder=5, edgecolors="black", linewidth=0.5)
    for a, d, t in zip(accs, dips, valid_tags):
        ax.annotate(t, (a, d), textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.set_xlabel("Final Answer Accuracy", fontsize=12)
    ax.set_ylabel("PRM Dip Depth (wrong samples)", fontsize=12)
    ax.set_title("Accuracy vs Mode Stability: Are They Decoupled?", fontsize=13)
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("log10(Model Params)")

    plt.tight_layout()
    plt.savefig(out_dir / "dip_vs_accuracy.png", dpi=200)
    plt.close()


def extrapolate_disappearance(
    all_fits: Dict[str, dict],
    out_dir: Path,
):
    """Extrapolate: at what model size does the dip effectively disappear?"""
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    dip_fit = all_fits.get("wrong_dip_depth", {}).get("power_law", {})

    if "error" not in dip_fit and "a" in dip_fit and "alpha" in dip_fit:
        a, alpha = dip_fit["a"], dip_fit["alpha"]
        threshold = 0.02  # dip smaller than this is "effectively zero"
        if alpha > 0:
            # a * N^(-alpha) = threshold => N = (a / threshold)^(1/alpha)
            N_disappear = (a / threshold) ** (1.0 / alpha)
            results["dip_disappearance_threshold"] = threshold
            results["estimated_params_for_disappearance"] = float(N_disappear)
            results["estimated_params_billions"] = round(N_disappear / 1e9, 1)
            results["fit"] = dip_fit

            print(f"\n{'='*60}")
            print(f"EXTRAPOLATION: PRM dip effectively disappears at ~{N_disappear/1e9:.1f}B parameters")
            print(f"  (using threshold={threshold}, power law: dip = {a:.4f} * N^(-{alpha:.4f}))")
            print(f"{'='*60}\n")

    (out_dir / "extrapolation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return results


def main():
    args = parse_args()
    summary_path = Path(args.summary_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        print(f"[ERROR] Summary file not found: {summary_path}")
        print("  Run 22_multi_scale_baseline.py first to generate PRM baselines.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics_by_model = summary.get("models", {})

    if not metrics_by_model:
        print("[ERROR] No model metrics found in summary.")
        return

    print(f"Loaded metrics for {len(metrics_by_model)} models: {list(metrics_by_model.keys())}")
    print()

    for tag, m in sorted(metrics_by_model.items(), key=lambda x: _model_params(x[0])):
        acc = m.get("accuracy", "N/A")
        dip = m.get("wrong_dip_depth", "N/A")
        stab = m.get("wrong_stability_ratio", "N/A")
        print(f"  {tag:>8}: acc={acc}, dip_depth={dip}, stability_ratio={stab}")

    print()

    all_fits = plot_scaling_laws(metrics_by_model, out_dir)
    plot_dip_vs_accuracy(metrics_by_model, out_dir)
    extrapolation = extrapolate_disappearance(all_fits, out_dir)

    full_results = {
        "models": metrics_by_model,
        "fits": all_fits,
        "extrapolation": extrapolation,
    }
    (out_dir / "scaling_law_results.json").write_text(
        json.dumps(full_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"All results saved to {out_dir}")


if __name__ == "__main__":
    main()
