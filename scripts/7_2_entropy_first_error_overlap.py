#!/usr/bin/env python3
"""
Core question: does the highest-entropy step overlap with the first error step?

Pipeline:
  1. Identify wrong drafts (draft_idx=0) from entropy_later data.
  2. Call GPT to annotate the first error step for each wrong draft (cached).
  3. Compare argmax-entropy step with GPT-annotated first error step.
  4. Produce overlap statistics and figures.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from step_judge import (
    build_first_error_prompt,
    call_gpt_first_error,
    load_env_file,
    make_client,
)

RESULT_DIR = ROOT / "results" / "gsm8k_llama_3.2_3b_instruct_entropy_later"
CACHE_PATH = RESULT_DIR / "gpt_first_error_cache.jsonl"
FIG_DIR = ROOT / "figures" / "entropy_vs_first_error"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def extract_number(text: str):
    text = str(text).replace(",", "").replace("$", "").strip()
    m = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.findall(r"-?\d+\.?\d*", text)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            return None
    return None


def load_gold():
    from datasets import load_dataset as hf_load
    ds = hf_load("openai/gsm8k", "main", split="test")
    gold = {}
    questions = {}
    for i, row in enumerate(ds):
        ans_text = row["answer"].split("####")[-1].strip()
        gold[f"gsm8k_{i}"] = extract_number(ans_text)
        questions[f"gsm8k_{i}"] = row["question"]
    return gold, questions


def load_drafts():
    drafts = {}
    with open(RESULT_DIR / "draft_logprobs.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["draft_idx"] != 0:
                continue
            drafts[d["doc_id"]] = d
    return drafts


def load_entropies():
    ents = {}
    with open(RESULT_DIR / "entropy_details.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["draft_idx"] != 0:
                continue
            ents[d["doc_id"]] = d["step_entropies"]
    return ents


def load_cache():
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            for line in f:
                d = json.loads(line)
                cache[d["doc_id"]] = d
    return cache


def save_cache_entry(entry):
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def annotate_first_errors(wrong_docs, drafts, gold, questions):
    """Call GPT to annotate first error step for each wrong draft."""
    load_env_file(ROOT / ".env")
    client = make_client()
    model = "gpt-4o-mini"

    cache = load_cache()
    print(f"Existing cache: {len(cache)} entries")

    to_annotate = [d for d in wrong_docs if d not in cache]
    print(f"Need to annotate: {len(to_annotate)} drafts")

    for i, doc_id in enumerate(to_annotate):
        d = drafts[doc_id]
        steps = d["draft_steps"]
        question = questions[doc_id]
        gold_ans = str(gold[doc_id])

        prompt = build_first_error_prompt(question, gold_ans, steps)
        parsed, raw = call_gpt_first_error(client, model, prompt)

        entry = {
            "doc_id": doc_id,
            "question": question,
            "steps": steps,
            "n_steps": len(steps),
            "gold_answer": gold_ans,
            "pred_answer": str(extract_number(d["draft_text"])),
            "gpt_parsed": parsed,
            "gpt_raw": raw,
        }
        save_cache_entry(entry)
        cache[doc_id] = entry

        if (i + 1) % 20 == 0 or i == len(to_annotate) - 1:
            print(f"  [{i+1}/{len(to_annotate)}] annotated")

    return cache


def main():
    print("Loading data...")
    gold, questions = load_gold()
    drafts = load_drafts()
    entropies = load_entropies()

    doc_ids = sorted(set(drafts) & set(entropies) & set(gold))
    wrong_docs = [
        d for d in doc_ids
        if (lambda p, g: g is not None and p is not None and abs(p - g) >= 1e-6)(
            extract_number(drafts[d]["draft_text"]), gold[d]
        )
    ]
    print(f"Wrong drafts: {len(wrong_docs)}")

    # Step 1: Annotate first errors
    print("\n--- GPT First-Error Annotation ---")
    cache = annotate_first_errors(wrong_docs, drafts, gold, questions)

    # Step 2: Collect paired data
    pairs = []  # (doc_id, first_error_step, argmax_entropy_step, n_steps, entropies)
    skipped = 0
    for doc_id in wrong_docs:
        if doc_id not in cache:
            skipped += 1
            continue
        entry = cache[doc_id]
        parsed = entry.get("gpt_parsed")
        if not parsed or "first_error_step" not in parsed:
            skipped += 1
            continue

        fe_step = parsed["first_error_step"]  # 1-indexed
        if fe_step == -1:
            skipped += 1
            continue

        ent = entropies[doc_id]
        n = len(ent)
        argmax_step = int(np.argmax(ent)) + 1  # convert to 1-indexed

        pairs.append({
            "doc_id": doc_id,
            "first_error_step": fe_step,
            "argmax_entropy_step": argmax_step,
            "n_steps": n,
            "entropies": ent,
        })

    print(f"\nValid pairs: {len(pairs)}, skipped: {skipped}")

    fe_steps = np.array([p["first_error_step"] for p in pairs])
    ae_steps = np.array([p["argmax_entropy_step"] for p in pairs])
    n_steps_arr = np.array([p["n_steps"] for p in pairs])

    # ── Metric 1: Exact match ──
    exact_match = np.sum(fe_steps == ae_steps)
    print(f"\nExact match (first_error == argmax_entropy): {exact_match}/{len(pairs)} = {exact_match/len(pairs):.3f}")

    # ── Metric 2: Within +/- 1 step ──
    within_1 = np.sum(np.abs(fe_steps - ae_steps) <= 1)
    print(f"Within +/-1 step: {within_1}/{len(pairs)} = {within_1/len(pairs):.3f}")

    # ── Metric 3: Entropy rank of first-error step ──
    fe_entropy_ranks = []
    for p in pairs:
        ent = p["entropies"]
        fe_idx = p["first_error_step"] - 1  # 0-indexed
        if fe_idx < len(ent):
            sorted_indices = sorted(range(len(ent)), key=lambda i: ent[i], reverse=True)
            rank = sorted_indices.index(fe_idx)
            fe_entropy_ranks.append(rank)

    print(f"\nEntropy rank of first-error step (0=highest):")
    print(f"  Mean: {np.mean(fe_entropy_ranks):.2f}")
    print(f"  Median: {np.median(fe_entropy_ranks):.1f}")
    print(f"  Top-1: {sum(1 for r in fe_entropy_ranks if r == 0)}/{len(fe_entropy_ranks)} = {sum(1 for r in fe_entropy_ranks if r == 0)/len(fe_entropy_ranks):.3f}")
    print(f"  Top-2: {sum(1 for r in fe_entropy_ranks if r <= 1)}/{len(fe_entropy_ranks)} = {sum(1 for r in fe_entropy_ranks if r <= 1)/len(fe_entropy_ranks):.3f}")
    print(f"  Top-3: {sum(1 for r in fe_entropy_ranks if r <= 2)}/{len(fe_entropy_ranks)} = {sum(1 for r in fe_entropy_ranks if r <= 2)/len(fe_entropy_ranks):.3f}")

    # ── Metric 4: Random baseline ──
    # If we picked a random step, what's the expected exact match rate?
    random_match_rate = np.mean(1.0 / n_steps_arr)
    print(f"\nRandom baseline (exact match): {random_match_rate:.3f}")
    random_top2 = np.mean(np.minimum(2.0, n_steps_arr) / n_steps_arr)
    random_top3 = np.mean(np.minimum(3.0, n_steps_arr) / n_steps_arr)
    print(f"Random baseline (top-2): {random_top2:.3f}")
    print(f"Random baseline (top-3): {random_top3:.3f}")

    # ── Metric 5: Entropy at first-error step vs other steps ──
    ent_at_fe = []
    ent_at_other = []
    for p in pairs:
        ent = p["entropies"]
        fe_idx = p["first_error_step"] - 1
        if fe_idx < len(ent):
            ent_at_fe.append(ent[fe_idx])
            for j, e in enumerate(ent):
                if j != fe_idx:
                    ent_at_other.append(e)

    print(f"\nEntropy at first-error step: {np.mean(ent_at_fe):.4f} +/- {np.std(ent_at_fe):.4f}")
    print(f"Entropy at other steps:      {np.mean(ent_at_other):.4f} +/- {np.std(ent_at_other):.4f}")
    from scipy import stats
    t, p_val = stats.ttest_ind(ent_at_fe, ent_at_other)
    print(f"t-test: t={t:.3f}, p={p_val:.6f}")

    # ═══════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════

    # ── Fig 1: Scatter of first_error_step vs argmax_entropy_step ──
    fig, ax = plt.subplots(figsize=(6, 6))
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(fe_steps))
    ax.scatter(fe_steps + jitter, ae_steps + jitter, alpha=0.5, s=30, color="#E91E63", edgecolors="none")
    lim = max(fe_steps.max(), ae_steps.max()) + 1
    ax.plot([0.5, lim], [0.5, lim], "k--", alpha=0.4, label="exact match")
    ax.set_xlabel("First error step (GPT-annotated, 1-indexed)")
    ax.set_ylabel("Argmax entropy step (1-indexed)")
    ax.set_title(f"First error vs max-entropy step (n={len(pairs)})\n"
                 f"Exact match: {exact_match/len(pairs):.1%}, within 1: {within_1/len(pairs):.1%}")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fe_vs_argmax_scatter.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved: fig_fe_vs_argmax_scatter.png")

    # ── Fig 2: Histogram of distance (argmax_entropy - first_error) ──
    diffs = ae_steps - fe_steps
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(diffs.min() - 0.5, diffs.max() + 1.5, 1)
    ax.hist(diffs, bins=bins, color="#3F51B5", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", ls="--", lw=2, label="exact match (diff=0)")
    ax.set_xlabel("Argmax entropy step - First error step")
    ax.set_ylabel("Count")
    ax.set_title("Distance between max-entropy step and first error step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fe_ae_distance_hist.png", dpi=200)
    plt.close(fig)
    print(f"Saved: fig_fe_ae_distance_hist.png")

    # ── Fig 3: Entropy rank of first-error step ──
    fig, ax = plt.subplots(figsize=(7, 5))
    rank_counts = Counter(fe_entropy_ranks)
    max_rank = max(fe_entropy_ranks)
    ranks = list(range(max_rank + 1))
    counts = [rank_counts.get(r, 0) for r in ranks]
    cumulative = np.cumsum(counts) / len(fe_entropy_ranks)

    ax.bar(ranks, counts, color="#FF9800", edgecolor="white", alpha=0.85, label="Count")
    ax2 = ax.twinx()
    ax2.plot(ranks, cumulative, "o-", color="#E91E63", markersize=5, label="Cumulative %")
    ax2.axhline(random_match_rate, color="gray", ls=":", alpha=0.6)
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_ylim(0, 1.05)

    ax.set_xlabel("Entropy rank of first-error step (0 = highest entropy)")
    ax.set_ylabel("Count")
    ax.set_title("How high is the entropy at the first error step?")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fe_entropy_rank.png", dpi=200)
    plt.close(fig)
    print(f"Saved: fig_fe_entropy_rank.png")

    # ── Fig 4: Entropy profile aligned to first-error step ──
    # Center each draft's entropy at the first-error step (position 0)
    max_offset = 6
    aligned = defaultdict(list)
    for p in pairs:
        ent = p["entropies"]
        fe_idx = p["first_error_step"] - 1
        for j, e in enumerate(ent):
            offset = j - fe_idx
            if -max_offset <= offset <= max_offset:
                aligned[offset].append(e)

    offsets = sorted(aligned.keys())
    means = [np.mean(aligned[o]) for o in offsets]
    ses = [np.std(aligned[o]) / np.sqrt(len(aligned[o])) for o in offsets]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(offsets, means, yerr=ses, fmt="o-", color="#E91E63", capsize=4, markersize=7)
    ax.axvline(0, color="red", ls="--", alpha=0.6, label="First error step")
    ax.set_xlabel("Step offset relative to first error (0 = first error)")
    ax.set_ylabel("Mean entropy")
    ax.set_title("Entropy profile aligned to first error step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_entropy_aligned_to_fe.png", dpi=200)
    plt.close(fig)
    print(f"Saved: fig_entropy_aligned_to_fe.png")

    # ── Fig 5: Box plot of entropy at first-error vs before vs after ──
    ent_before = []
    ent_at = []
    ent_after = []
    for p in pairs:
        ent = p["entropies"]
        fe_idx = p["first_error_step"] - 1
        if fe_idx < len(ent):
            ent_at.append(ent[fe_idx])
            for j in range(fe_idx):
                ent_before.append(ent[j])
            for j in range(fe_idx + 1, len(ent)):
                ent_after.append(ent[j])

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [ent_before, ent_at, ent_after],
        labels=["Before error", "At first error", "After error"],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markersize=6),
    )
    colors = ["#2196F3", "#F44336", "#FF9800"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Step entropy")
    ax.set_title("Entropy at/before/after first error step")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_entropy_before_at_after.png", dpi=200)
    plt.close(fig)
    print(f"Saved: fig_entropy_before_at_after.png")

    # ── Save summary ──
    summary = {
        "n_wrong_drafts": len(wrong_docs),
        "n_valid_pairs": len(pairs),
        "exact_match": int(exact_match),
        "exact_match_rate": float(exact_match / len(pairs)),
        "within_1_rate": float(within_1 / len(pairs)),
        "random_baseline_exact": float(random_match_rate),
        "fe_entropy_rank_mean": float(np.mean(fe_entropy_ranks)),
        "fe_entropy_rank_median": float(np.median(fe_entropy_ranks)),
        "fe_top1_rate": float(sum(1 for r in fe_entropy_ranks if r == 0) / len(fe_entropy_ranks)),
        "fe_top2_rate": float(sum(1 for r in fe_entropy_ranks if r <= 1) / len(fe_entropy_ranks)),
        "fe_top3_rate": float(sum(1 for r in fe_entropy_ranks if r <= 2) / len(fe_entropy_ranks)),
        "entropy_at_fe": float(np.mean(ent_at_fe)),
        "entropy_at_other": float(np.mean(ent_at_other)),
        "ttest_p": float(p_val),
    }
    with open(FIG_DIR / "overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: overlap_summary.json")


if __name__ == "__main__":
    main()
