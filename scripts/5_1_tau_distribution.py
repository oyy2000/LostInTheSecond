"""
Plot the distribution of first-error step relative position (tau/N)
and absolute first-error step (tau) for GSM8K and MATH500.
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path("/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond")
DATASETS = {
    "GSM8K": ROOT / "results/gsm8k_3b_multi_sample/first_error/gpt_first_error_cache.jsonl",
    "MATH500": ROOT / "results/math500_3b_multi_sample/first_error/gpt_first_error_cache.jsonl",
}
OUT_DIR = ROOT / "figures/step_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tau(path: pathlib.Path):
    """Return arrays of tau and n_steps for valid records."""
    taus, n_steps_list = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            tau = rec.get("tau")
            n_steps = rec.get("n_steps")
            if tau is None or tau < 1 or n_steps is None or n_steps < 1:
                continue
            taus.append(tau)
            n_steps_list.append(n_steps)
    return np.array(taus, dtype=float), np.array(n_steps_list, dtype=float)


data = {}
for name, path in DATASETS.items():
    tau, n_steps = load_tau(path)
    rel = tau / n_steps
    data[name] = {"tau": tau, "n_steps": n_steps, "rel": rel}
    print(f"{name}: N={len(tau)}, mean(tau/N)={rel.mean():.3f}, "
          f"median(tau/N)={np.median(rel):.3f}, std(tau/N)={rel.std():.3f}")

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
HIST_COLOR = "#4C72B0"
MEAN_COLOR = "#C44E52"
MEDIAN_COLOR = "#55A868"
EDGE_COLOR = "white"


def _annotate(ax, values, label_prefix=""):
    """Add mean/median lines and a stats text box."""
    mu = values.mean()
    med = np.median(values)
    sd = values.std()
    n = len(values)
    ax.axvline(mu, color=MEAN_COLOR, ls="--", lw=1.5, label=f"Mean = {mu:.3f}")
    ax.axvline(med, color=MEDIAN_COLOR, ls="--", lw=1.5, label=f"Median = {med:.3f}")
    txt = f"Mean = {mu:.3f}\nMedian = {med:.3f}\nStd = {sd:.3f}\nN = {n}"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.legend(fontsize=7, loc="upper left")


# ---------------------------------------------------------------------------
# Figure 1 -- relative position tau/N
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
fig.suptitle("Distribution of First-Error Relative Position ($\\tau / N$)",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (name, d) in zip(axes, data.items()):
    ax.hist(d["rel"], bins=20, range=(0, 1), color=HIST_COLOR,
            edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.85)
    _annotate(ax, d["rel"])
    ax.set_title(f"{name} (Qwen2.5-3B-Instruct)", fontsize=10)
    ax.set_xlabel("$\\tau / N$", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.tick_params(labelsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"fig_tau_distribution.{ext}", dpi=200,
                bbox_inches="tight")
print(f"Saved fig_tau_distribution to {OUT_DIR}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- absolute tau
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
fig2.suptitle("Distribution of First-Error Step ($\\tau$, absolute)",
              fontsize=13, fontweight="bold", y=1.02)

for ax, (name, d) in zip(axes2, data.items()):
    max_tau = int(d["tau"].max())
    bins = min(20, max_tau)
    ax.hist(d["tau"], bins=bins, color=HIST_COLOR,
            edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.85)
    _annotate(ax, d["tau"])
    ax.set_title(f"{name} (Qwen2.5-3B-Instruct)", fontsize=10)
    ax.set_xlabel("$\\tau$ (step index)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.tick_params(labelsize=8)

fig2.tight_layout()
for ext in ("png", "pdf"):
    fig2.savefig(OUT_DIR / f"fig_tau_distribution_absolute.{ext}", dpi=200,
                 bbox_inches="tight")
print(f"Saved fig_tau_distribution_absolute to {OUT_DIR}")
plt.close(fig2)

print("Done.")
