#!/usr/bin/env python3
"""
Validate the relationship between step-level entropy and the first error step.

Data sources:
  - draft_logprobs.jsonl   : draft steps + token logprobs per draft
  - entropy_details.jsonl  : per-step entropy (same across draft_idx for a doc)

Analysis:
  1. Load GSM8K gold answers; classify each draft as correct/wrong.
  2. For wrong drafts (draft_idx=0 only, the original generation):
     - Identify the argmax-entropy step as the "suspected error step".
     - Compare entropy at the first error step vs other steps.
  3. Produce figures showing the entropy-error relationship.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULT_DIR = ROOT / "results" / "gsm8k_llama_3.2_3b_instruct_entropy_later"
FIG_DIR = ROOT / "figures" / "entropy_vs_first_error"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def extract_number(text: str):
    """Extract the final numeric answer from a GSM8K-style response.
    Handles #### markers and various formatting."""
    text = text.replace(",", "").replace("$", "").strip()
    marker = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if marker:
        try:
            return float(marker.group(1))
        except ValueError:
            pass
    m = re.findall(r"-?\d+\.?\d*", text)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            return None
    return None


def load_gold_answers():
    """Load GSM8K gold answers via HuggingFace datasets."""
    from datasets import load_dataset as hf_load
    ds = hf_load("openai/gsm8k", "main", split="test")
    gold = {}
    for i, row in enumerate(ds):
        ans_text = row["answer"].split("####")[-1].strip()
        gold[f"gsm8k_{i}"] = extract_number(ans_text)
    return gold


def load_drafts():
    """Load draft_logprobs.jsonl, keep only draft_idx=0."""
    drafts = {}
    with open(RESULT_DIR / "draft_logprobs.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["draft_idx"] != 0:
                continue
            drafts[d["doc_id"]] = d
    return drafts


def load_entropies():
    """Load entropy_details.jsonl, keep only draft_idx=0."""
    ents = {}
    with open(RESULT_DIR / "entropy_details.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["draft_idx"] != 0:
                continue
            ents[d["doc_id"]] = d["step_entropies"]
    return ents


def main():
    print("Loading gold answers...")
    gold = load_gold_answers()
    print(f"  {len(gold)} gold answers loaded")

    print("Loading drafts...")
    drafts = load_drafts()
    print(f"  {len(drafts)} drafts loaded (draft_idx=0)")

    print("Loading entropies...")
    entropies = load_entropies()
    print(f"  {len(entropies)} entropy records loaded")

    doc_ids = sorted(set(drafts.keys()) & set(entropies.keys()) & set(gold.keys()))
    print(f"  {len(doc_ids)} docs with all three sources")

    correct_docs = []
    wrong_docs = []

    for doc_id in doc_ids:
        d = drafts[doc_id]
        d = drafts[doc_id]
        pred = extract_number(d["draft_text"])
        g = gold[doc_id]

        if pred is not None and g is not None and abs(pred - g) < 1e-6:
            correct_docs.append(doc_id)
        else:
            wrong_docs.append(doc_id)

    print(f"\nCorrect: {len(correct_docs)}, Wrong: {len(wrong_docs)}")
    print(f"Accuracy: {len(correct_docs) / len(doc_ids):.3f}")

    # ── Analysis 1: Mean entropy profile for correct vs wrong ──
    max_steps = max(len(entropies[d]) for d in doc_ids)

    def padded_entropy_matrix(doc_list, max_len):
        """Build (N, max_len) matrix, NaN-padded."""
        mat = np.full((len(doc_list), max_len), np.nan)
        for i, doc_id in enumerate(doc_list):
            ent = entropies[doc_id]
            mat[i, :len(ent)] = ent
        return mat

    mat_correct = padded_entropy_matrix(correct_docs, max_steps)
    mat_wrong = padded_entropy_matrix(wrong_docs, max_steps)

    mean_correct = np.nanmean(mat_correct, axis=0)
    mean_wrong = np.nanmean(mat_wrong, axis=0)
    se_correct = np.nanstd(mat_correct, axis=0) / np.sqrt(np.sum(~np.isnan(mat_correct), axis=0))
    se_wrong = np.nanstd(mat_wrong, axis=0) / np.sqrt(np.sum(~np.isnan(mat_wrong), axis=0))

    steps_x = np.arange(1, max_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_x, mean_correct, "o-", color="#2196F3", label=f"Correct (n={len(correct_docs)})", markersize=5)
    ax.fill_between(steps_x, mean_correct - se_correct, mean_correct + se_correct, alpha=0.15, color="#2196F3")
    ax.plot(steps_x, mean_wrong, "s-", color="#F44336", label=f"Wrong (n={len(wrong_docs)})", markersize=5)
    ax.fill_between(steps_x, mean_wrong - se_wrong, mean_wrong + se_wrong, alpha=0.15, color="#F44336")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Mean step entropy")
    ax.set_title("Step entropy: correct vs wrong drafts (GSM8K, Llama-3.2-3B)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_entropy_correct_vs_wrong.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_entropy_correct_vs_wrong.png'}")

    # ── Analysis 2: For wrong drafts, where is the max-entropy step? ──
    argmax_positions = []
    n_steps_list = []
    for doc_id in wrong_docs:
        ent = entropies[doc_id]
        argmax_positions.append(int(np.argmax(ent)))
        n_steps_list.append(len(ent))

    argmax_relpos = [p / (n - 1) if n > 1 else 0.0 for p, n in zip(argmax_positions, n_steps_list)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(argmax_positions, bins=range(max_steps + 2), color="#FF9800", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Argmax-entropy step index (0-based)")
    ax.set_ylabel("Count")
    ax.set_title("Where is the highest-entropy step? (wrong drafts)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(argmax_relpos, bins=20, color="#9C27B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Relative position of max-entropy step")
    ax.set_ylabel("Count")
    ax.set_title("Relative position of max-entropy step (wrong drafts)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_argmax_entropy_position.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_argmax_entropy_position.png'}")

    # ── Analysis 3: Entropy at each relative step position ──
    n_bins = 10
    rel_ent_correct = defaultdict(list)
    rel_ent_wrong = defaultdict(list)

    for doc_id in correct_docs:
        ent = entropies[doc_id]
        n = len(ent)
        for i, e in enumerate(ent):
            b = min(int(i / n * n_bins), n_bins - 1)
            rel_ent_correct[b].append(e)

    for doc_id in wrong_docs:
        ent = entropies[doc_id]
        n = len(ent)
        for i, e in enumerate(ent):
            b = min(int(i / n * n_bins), n_bins - 1)
            rel_ent_wrong[b].append(e)

    bins_x = np.arange(n_bins)
    mean_c = [np.mean(rel_ent_correct[b]) for b in bins_x]
    mean_w = [np.mean(rel_ent_wrong[b]) for b in bins_x]
    se_c = [np.std(rel_ent_correct[b]) / np.sqrt(len(rel_ent_correct[b])) for b in bins_x]
    se_w = [np.std(rel_ent_wrong[b]) / np.sqrt(len(rel_ent_wrong[b])) for b in bins_x]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(bins_x - width / 2, mean_c, width, yerr=se_c, color="#2196F3", alpha=0.8, label="Correct", capsize=3)
    ax.bar(bins_x + width / 2, mean_w, width, yerr=se_w, color="#F44336", alpha=0.8, label="Wrong", capsize=3)
    ax.set_xlabel("Relative step position (decile)")
    ax.set_ylabel("Mean entropy")
    ax.set_title("Entropy by relative step position")
    ax.set_xticks(bins_x)
    ax.set_xticklabels([f"{b * 10}-{(b + 1) * 10}%" for b in bins_x], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_entropy_by_relpos.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_entropy_by_relpos.png'}")

    # ── Analysis 4: Per-draft max entropy vs correctness ──
    max_ent_correct = [max(entropies[d]) for d in correct_docs]
    max_ent_wrong = [max(entropies[d]) for d in wrong_docs]
    mean_ent_correct = [np.mean(entropies[d]) for d in correct_docs]
    mean_ent_wrong = [np.mean(entropies[d]) for d in wrong_docs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(max_ent_correct, bins=30, alpha=0.6, color="#2196F3", label="Correct", density=True)
    ax.hist(max_ent_wrong, bins=30, alpha=0.6, color="#F44336", label="Wrong", density=True)
    ax.set_xlabel("Max step entropy")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of max step entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(mean_ent_correct, bins=30, alpha=0.6, color="#2196F3", label="Correct", density=True)
    ax.hist(mean_ent_wrong, bins=30, alpha=0.6, color="#F44336", label="Wrong", density=True)
    ax.set_xlabel("Mean step entropy")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of mean step entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_entropy_distributions.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_entropy_distributions.png'}")

    # ── Analysis 5: Entropy spike detection ──
    # For wrong drafts, compute entropy "spike" = max(entropy) - mean(entropy)
    # and check if spike magnitude correlates with error severity
    spike_wrong = []
    spike_correct = []
    for doc_id in wrong_docs:
        ent = entropies[doc_id]
        spike_wrong.append(max(ent) - np.mean(ent))
    for doc_id in correct_docs:
        ent = entropies[doc_id]
        spike_correct.append(max(ent) - np.mean(ent))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(spike_correct, bins=30, alpha=0.6, color="#2196F3", label="Correct", density=True)
    ax.hist(spike_wrong, bins=30, alpha=0.6, color="#F44336", label="Wrong", density=True)
    ax.set_xlabel("Entropy spike (max - mean)")
    ax.set_ylabel("Density")
    ax.set_title("Entropy spike magnitude: correct vs wrong")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_entropy_spike.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_entropy_spike.png'}")

    # ── Analysis 6: Step-level entropy rank vs step position ──
    # For wrong drafts: is the highest-entropy step earlier or later?
    # Compare with correct drafts
    rank_of_max_correct = []
    rank_of_max_wrong = []
    for doc_id in correct_docs:
        ent = entropies[doc_id]
        n = len(ent)
        if n > 1:
            rank_of_max_correct.append(int(np.argmax(ent)) / (n - 1))
    for doc_id in wrong_docs:
        ent = entropies[doc_id]
        n = len(ent)
        if n > 1:
            rank_of_max_wrong.append(int(np.argmax(ent)) / (n - 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(rank_of_max_correct, bins=20, alpha=0.6, color="#2196F3", label="Correct", density=True)
    ax.hist(rank_of_max_wrong, bins=20, alpha=0.6, color="#F44336", label="Wrong", density=True)
    ax.set_xlabel("Relative position of max-entropy step")
    ax.set_ylabel("Density")
    ax.set_title("Max-entropy step position: correct vs wrong")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_max_entropy_relpos_compare.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_max_entropy_relpos_compare.png'}")

    # ── Summary statistics ──
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total docs: {len(doc_ids)}")
    print(f"Correct: {len(correct_docs)} ({len(correct_docs)/len(doc_ids)*100:.1f}%)")
    print(f"Wrong:   {len(wrong_docs)} ({len(wrong_docs)/len(doc_ids)*100:.1f}%)")

    print(f"\nMean entropy (correct): {np.mean(mean_ent_correct):.4f} +/- {np.std(mean_ent_correct):.4f}")
    print(f"Mean entropy (wrong):   {np.mean(mean_ent_wrong):.4f} +/- {np.std(mean_ent_wrong):.4f}")

    print(f"\nMax entropy (correct):  {np.mean(max_ent_correct):.4f} +/- {np.std(max_ent_correct):.4f}")
    print(f"Max entropy (wrong):    {np.mean(max_ent_wrong):.4f} +/- {np.std(max_ent_wrong):.4f}")

    print(f"\nEntropy spike (correct): {np.mean(spike_correct):.4f} +/- {np.std(spike_correct):.4f}")
    print(f"Entropy spike (wrong):   {np.mean(spike_wrong):.4f} +/- {np.std(spike_wrong):.4f}")

    print(f"\nMax-entropy step relpos (correct): {np.mean(rank_of_max_correct):.3f}")
    print(f"Max-entropy step relpos (wrong):   {np.mean(rank_of_max_wrong):.3f}")

    # Statistical test
    from scipy import stats
    t_mean, p_mean = stats.ttest_ind(mean_ent_correct, mean_ent_wrong)
    t_max, p_max = stats.ttest_ind(max_ent_correct, max_ent_wrong)
    t_spike, p_spike = stats.ttest_ind(spike_correct, spike_wrong)
    t_pos, p_pos = stats.ttest_ind(rank_of_max_correct, rank_of_max_wrong)

    print(f"\nt-test (mean entropy):  t={t_mean:.3f}, p={p_mean:.4f}")
    print(f"t-test (max entropy):   t={t_max:.3f}, p={p_max:.4f}")
    print(f"t-test (entropy spike): t={t_spike:.3f}, p={p_spike:.4f}")
    print(f"t-test (max-ent relpos): t={t_pos:.3f}, p={p_pos:.4f}")

    # Save summary
    summary = {
        "n_docs": len(doc_ids),
        "n_correct": len(correct_docs),
        "n_wrong": len(wrong_docs),
        "accuracy": len(correct_docs) / len(doc_ids),
        "mean_entropy_correct": float(np.mean(mean_ent_correct)),
        "mean_entropy_wrong": float(np.mean(mean_ent_wrong)),
        "max_entropy_correct": float(np.mean(max_ent_correct)),
        "max_entropy_wrong": float(np.mean(max_ent_wrong)),
        "spike_correct": float(np.mean(spike_correct)),
        "spike_wrong": float(np.mean(spike_wrong)),
        "max_ent_relpos_correct": float(np.mean(rank_of_max_correct)),
        "max_ent_relpos_wrong": float(np.mean(rank_of_max_wrong)),
        "ttest_mean_entropy_p": float(p_mean),
        "ttest_max_entropy_p": float(p_max),
        "ttest_spike_p": float(p_spike),
        "ttest_relpos_p": float(p_pos),
    }
    with open(FIG_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {FIG_DIR / 'summary.json'}")

    # ── Analysis 7: ROC curve -- mean entropy as predictor of wrong answer ──
    from sklearn.metrics import roc_curve, auc as sk_auc

    labels = []  # 1 = wrong, 0 = correct
    scores = []  # mean entropy (higher -> more likely wrong)
    for doc_id in doc_ids:
        ent = entropies[doc_id]
        scores.append(np.mean(ent))
        labels.append(1 if doc_id in set(wrong_docs) else 0)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#E91E63", lw=2, label=f"Mean entropy (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: mean entropy as predictor of wrong answer")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_roc_entropy.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_roc_entropy.png'}")

    # Also max entropy as predictor
    scores_max = [max(entropies[d]) for d in doc_ids]
    fpr2, tpr2, _ = roc_curve(labels, scores_max)
    roc_auc2 = sk_auc(fpr2, tpr2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#E91E63", lw=2, label=f"Mean entropy (AUC={roc_auc:.3f})")
    ax.plot(fpr2, tpr2, color="#FF9800", lw=2, label=f"Max entropy (AUC={roc_auc2:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: entropy features as predictors of wrong answer")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_roc_entropy_combined.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_roc_entropy_combined.png'}")
    print(f"  Mean entropy AUC: {roc_auc:.3f}")
    print(f"  Max entropy AUC:  {roc_auc2:.3f}")

    # ── Analysis 8: Normalized entropy heatmap ──
    # For wrong drafts, normalize entropy within each draft (z-score)
    # and show the average z-scored entropy at each step position
    n_pos = 10
    z_wrong = defaultdict(list)
    z_correct = defaultdict(list)

    for doc_id in wrong_docs:
        ent = np.array(entropies[doc_id])
        if ent.std() > 0:
            z = (ent - ent.mean()) / ent.std()
        else:
            z = np.zeros_like(ent)
        n = len(z)
        for i, val in enumerate(z):
            b = min(int(i / n * n_pos), n_pos - 1)
            z_wrong[b].append(val)

    for doc_id in correct_docs:
        ent = np.array(entropies[doc_id])
        if ent.std() > 0:
            z = (ent - ent.mean()) / ent.std()
        else:
            z = np.zeros_like(ent)
        n = len(z)
        for i, val in enumerate(z):
            b = min(int(i / n * n_pos), n_pos - 1)
            z_correct[b].append(val)

    pos_x = np.arange(n_pos)
    z_mean_w = [np.mean(z_wrong[b]) for b in pos_x]
    z_mean_c = [np.mean(z_correct[b]) for b in pos_x]
    z_se_w = [np.std(z_wrong[b]) / np.sqrt(len(z_wrong[b])) for b in pos_x]
    z_se_c = [np.std(z_correct[b]) / np.sqrt(len(z_correct[b])) for b in pos_x]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(pos_x, z_mean_c, yerr=z_se_c, fmt="o-", color="#2196F3",
                label="Correct", capsize=4, markersize=6)
    ax.errorbar(pos_x, z_mean_w, yerr=z_se_w, fmt="s-", color="#F44336",
                label="Wrong", capsize=4, markersize=6)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Relative step position (decile)")
    ax.set_ylabel("Z-scored entropy (within draft)")
    ax.set_title("Normalized entropy profile: correct vs wrong")
    ax.set_xticks(pos_x)
    ax.set_xticklabels([f"{b*10}-{(b+1)*10}%" for b in pos_x], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_zscore_entropy_profile.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_zscore_entropy_profile.png'}")

    summary["roc_auc_mean_entropy"] = float(roc_auc)
    summary["roc_auc_max_entropy"] = float(roc_auc2)
    with open(FIG_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated: {FIG_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
