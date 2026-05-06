#!/usr/bin/env python3
"""
Recount tokens from checkpoint.jsonl using the actual tokenizer.

Reads draft records, tokenizes each draft's step prefix up to the rollback
point, and writes per-draft prefix token counts into a lookup JSON that
can be used by the eval logic.

Usage:
    python scripts/6_8_1_recount_tokens.py \
        --checkpoint results/qwen2.5_3b_instruct_budget_multisignal/math500/checkpoint.jsonl \
        --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                    help="HuggingFace model ID for tokenizer")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)

    print(f"Reading checkpoint: {ckpt_path}")
    drafts = {}
    with open(ckpt_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("task_type") == "draft":
                key = (r["doc_id"], r["draft_idx"])
                drafts[key] = r

    print(f"Total draft records: {len(drafts)}")

    results = {}
    for (doc_id, draft_idx), d in drafts.items():
        steps = d.get("draft_steps", [])
        if not steps:
            continue

        full_text = d.get("draft_text", "")
        full_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

        step_token_counts = []
        for step in steps:
            n_tok = len(tokenizer.encode(step, add_special_tokens=False))
            step_token_counts.append(n_tok)

        cumulative = []
        running = 0
        for n_tok in step_token_counts:
            running += n_tok
            cumulative.append(running)

        results[f"{doc_id}|{draft_idx}"] = {
            "full_tokens": full_tokens,
            "stored_tokens": d.get("draft_tokens", 0),
            "step_tokens": step_token_counts,
            "cumulative_tokens": cumulative,
            "n_steps": len(steps),
        }

    out_path = ckpt_path.parent / "draft_step_tokens.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"Saved {len(results)} entries to: {out_path}")

    total_stored = sum(v["stored_tokens"] for v in results.values())
    total_recount = sum(v["full_tokens"] for v in results.values())
    print(f"Token comparison: stored={total_stored}, "
          f"recount={total_recount}, "
          f"ratio={total_recount/total_stored:.4f}")


if __name__ == "__main__":
    main()
