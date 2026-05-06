#!/usr/bin/env python3
"""
Score MATH-500 drafts with Skywork-o1-Open-PRM-Qwen-2.5-1.5B,
then plot step-level PRM curves with drop>0.1 rollback.

Usage:
    python scripts/_tmp_skywork_prm_curves.py --gpu 0
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.prompt_templates import check_answer, split_steps

DATASET = "math500"
SKYWORK_PRM = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
DROP_THRESHOLD = 0.1


def load_drafts():
    """Load draft-0 records from the MATH-500 sweep checkpoint."""
    ckpt = ROOT / "results/math500_entropy_triggered_sweep/checkpoint.jsonl"
    records = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    drafts = {}
    for r in records:
        if r.get("task_type") == "draft" and r.get("draft_idx") == 0:
            drafts[r["doc_id"]] = r
    return drafts


def score_with_skywork(drafts, device="cuda:0"):
    """Score each draft's steps using Skywork PRM at \\n\\n granularity."""
    from transformers import AutoTokenizer

    skywork_cache = ROOT / "third_party/skywork-o1-prm-inference"
    if not skywork_cache.exists():
        skywork_cache.mkdir(parents=True, exist_ok=True)
        os.system(
            f"git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git "
            f"{skywork_cache}"
        )
    sys.path.insert(0, str(skywork_cache))
    from model_utils.prm_model import PRM_MODEL
    from model_utils.io_utils import (
        prepare_batch_input_for_model,
        derive_step_rewards,
    )

    tokenizer = AutoTokenizer.from_pretrained(SKYWORK_PRM, trust_remote_code=True, use_safetensors=True)
    model = PRM_MODEL.from_pretrained(
        SKYWORK_PRM, device_map={"": device},
        torch_dtype=torch.bfloat16,
    ).eval()

    def prepare_input_paragraph(problem, response, tokenizer):
        """Build input with reward flags at the last token of each \\n\\n paragraph."""
        prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
        reward_flags = [0] * len(prompt_ids)

        steps = split_steps(response)
        response_ids = []
        for i, step in enumerate(steps):
            sep = "\n\n" if i < len(steps) - 1 else "\n"
            chunk = step + sep
            chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
            flags = [0] * len(chunk_ids)
            flags[-1] = 1
            response_ids.extend(chunk_ids)
            reward_flags.extend(flags)

        input_ids = prompt_ids + response_ids
        return input_ids, steps, reward_flags

    results = {}
    doc_ids = sorted(drafts.keys())
    BATCH = 8
    for bi in range(0, len(doc_ids), BATCH):
        batch_ids = doc_ids[bi:bi + BATCH]
        batch_input_ids = []
        batch_steps = []
        batch_flags = []
        for did in batch_ids:
            d = drafts[did]
            question = d.get("question", "")
            response = d["draft_text"]
            input_ids, steps, reward_flags = prepare_input_paragraph(
                question, response, tokenizer,
            )
            batch_input_ids.append(input_ids)
            batch_steps.append(steps)
            batch_flags.append(reward_flags)

        padded_ids, attn_mask, padded_flags = prepare_batch_input_for_model(
            batch_input_ids, batch_flags, tokenizer.pad_token_id,
        )
        padded_ids = padded_ids.to(device)
        attn_mask = attn_mask.to(device)
        padded_flags = padded_flags.to(device)

        with torch.no_grad():
            _, _, rewards = model(
                input_ids=padded_ids,
                attention_mask=attn_mask,
                return_probs=True,
            )
        step_rewards = derive_step_rewards(rewards, padded_flags)

        for j, did in enumerate(batch_ids):
            results[did] = {
                "step_scores": step_rewards[j],
                "steps": batch_steps[j],
                "n_steps": len(batch_steps[j]),
            }
        print(f"  Scored {min(bi + BATCH, len(doc_ids))}/{len(doc_ids)}")

    return results


def find_drop_rollback(scores, delta=DROP_THRESHOLD):
    """Return rollback step using max-drop > delta, or None."""
    if len(scores) < 2:
        return None
    drops = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    md_idx = int(np.argmax(drops))
    if drops[md_idx] > delta:
        return md_idx + 1
    return None


def plot_curves(samples, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random.seed(42)
    chosen = random.sample(samples, min(12, len(samples)))
    chosen.sort(key=lambda x: x["n_steps"])

    ncols = 4
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).flatten()

    for i, s in enumerate(chosen):
        ax = axes[i]
        xs = list(range(s["n_steps"]))
        ys = s["scores"]
        rb = s["rollback"]
        correct = s["correct"]

        line_color = "#43A047" if correct else "#E53935"
        tag = "CORRECT" if correct else "WRONG"

        ax.plot(xs, ys, "o-", color=line_color,
                markersize=5, linewidth=1.5, zorder=2,
                label=f"draft: {tag}")

        if rb is not None:
            ax.scatter([rb], [ys[rb]], color="#FF6F00", s=110,
                       zorder=3, marker="v", edgecolors="black",
                       linewidths=0.6,
                       label=f"rollback = step {rb}")
        else:
            ax.annotate("no rollback", xy=(0.5, 0.05),
                        xycoords="axes fraction", fontsize=7,
                        ha="center", color="gray")

        ax.axhline(np.mean(ys), color="gray", linewidth=0.7,
                   linestyle="--", alpha=0.5)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("PRM score", fontsize=9)
        ax.set_title(f"{s['doc_id']}  ({s['n_steps']} steps)", fontsize=9)
        ax.set_xticks(xs if len(xs) <= 20 else
                      xs[::max(1, len(xs) // 10)])
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.2)
        ax.set_ylim(-0.05, 1.05)

    for j in range(len(chosen), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Skywork PRM 1.5B Step Scores (MATH-500)  |  "
        f"rollback = first drop > {DROP_THRESHOLD}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"skywork_prm_step_curves_math500.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out_dir}/skywork_prm_step_curves_math500.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0"

    print("Loading drafts...")
    drafts = load_drafts()
    print(f"  {len(drafts)} draft-0 records")

    from src.sweep_datasets import load_dataset_by_name
    questions = load_dataset_by_name(DATASET, 0, seed=42)
    q_map = {q["doc_id"]: q for q in questions}

    cache_path = ROOT / "results/math500_entropy_triggered_sweep/skywork_prm_scores.jsonl"
    if cache_path.exists():
        print(f"Loading cached Skywork PRM scores from {cache_path}")
        scored = {}
        for line in cache_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                scored[r["doc_id"]] = r
    else:
        print("Scoring with Skywork PRM...")
        for did, d in drafts.items():
            if "question" not in d:
                d["question"] = q_map.get(did, {}).get("question", "")
        scored = score_with_skywork(drafts, device=device)
        with cache_path.open("w") as f:
            for did, r in sorted(scored.items()):
                r["doc_id"] = did
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Cached scores -> {cache_path}")

    samples = []
    for did, r in scored.items():
        d = drafts.get(did)
        if not d:
            continue
        scores = r["step_scores"]
        n = r["n_steps"]
        if n < 3 or len(scores) < 2:
            continue
        scores = scores[:n]
        correct = check_answer(
            DATASET,
            d.get("draft_answer", ""),
            d.get("gold_answer", q_map.get(did, {}).get("gold_answer", "")),
        )
        rb = find_drop_rollback(scores, DROP_THRESHOLD)
        samples.append(dict(
            doc_id=did, n_steps=n, scores=scores,
            rollback=rb, correct=correct,
        ))

    n_with_rb = sum(1 for s in samples if s["rollback"] is not None)
    print(f"\nSamples: {len(samples)}, with rollback: {n_with_rb} "
          f"({n_with_rb / len(samples) * 100:.1f}%)")

    rb_steps = [s["rollback"] for s in samples if s["rollback"] is not None]
    if rb_steps:
        print(f"Rollback step: mean={np.mean(rb_steps):.1f}, "
              f"median={np.median(rb_steps):.0f}")

    out_dir = ROOT / "figures/entropy_triggered_sweep"
    plot_curves(samples, out_dir)


if __name__ == "__main__":
    main()
