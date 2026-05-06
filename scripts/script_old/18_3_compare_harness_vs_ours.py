#!/usr/bin/env python3
"""Compare per-sample results between lm-eval-harness and our 18_1 script."""

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

HARNESS_SAMPLES = ROOT / "results/harness_math500_vllm/meta-llama__Llama-3.2-3B-Instruct/samples_hendrycks_math_500_2026-04-26T14-42-04.294052.jsonl"
OUR_SAMPLES = ROOT / "results/math500_vllm_4096_llama_3.2_3b_instruct.jsonl"


def load_harness():
    records = {}
    for line in HARNESS_SAMPLES.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        did = r["doc_id"]
        resp = r["filtered_resps"][0] if r.get("filtered_resps") else ""
        records[did] = {
            "exact_match": r.get("exact_match", 0),
            "resp_len": len(resp),
            "resp_preview": resp[:200],
            "has_boxed": "\\boxed" in resp,
        }
    return records


def load_ours():
    records = {}
    for line in OUR_SAMPLES.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        idx = int(r["doc_id"].replace("math500_", ""))
        raw = r.get("raw_output", "")
        records[idx] = {
            "correct": r["correct"],
            "pred": r["pred"],
            "n_tokens": r["n_tokens"],
            "truncated": r["truncated"],
            "raw_len": len(raw),
            "raw_preview": raw[:200],
            "has_boxed": "\\boxed" in raw,
        }
    return records


def main():
    h = load_harness()
    o = load_ours()

    common_ids = sorted(set(h.keys()) & set(o.keys()))
    print(f"Common samples: {len(common_ids)}")

    h_total = sum(h[i]["exact_match"] for i in common_ids)
    o_total = sum(o[i]["correct"] for i in common_ids)
    print(f"Harness correct: {h_total}/{len(common_ids)} = {h_total/len(common_ids):.4f}")
    print(f"Ours correct:    {o_total}/{len(common_ids)} = {o_total/len(common_ids):.4f}")
    print()

    # Categorize disagreements
    h_only, o_only, both_right, both_wrong = [], [], [], []
    for i in common_ids:
        hc = bool(h[i]["exact_match"])
        oc = bool(o[i]["correct"])
        if hc and oc:
            both_right.append(i)
        elif hc and not oc:
            h_only.append(i)
        elif not hc and oc:
            o_only.append(i)
        else:
            both_wrong.append(i)

    print(f"Both correct:     {len(both_right)}")
    print(f"Both wrong:       {len(both_wrong)}")
    print(f"Harness only:     {len(h_only)}")
    print(f"Ours only:        {len(o_only)}")
    print()

    # Analyze harness-only wins
    if h_only:
        print("=" * 80)
        print(f"HARNESS CORRECT, OURS WRONG ({len(h_only)} samples)")
        print("=" * 80)
        trunc_count = sum(o[i]["truncated"] for i in h_only)
        no_boxed_count = sum(not o[i]["has_boxed"] for i in h_only)
        print(f"  Ours truncated:    {trunc_count}")
        print(f"  Ours no \\boxed:    {no_boxed_count}")
        print()
        for i in h_only[:20]:
            print(f"  doc_id={i}")
            print(f"    harness resp:  {h[i]['resp_preview'][:120]}...")
            print(f"    ours raw:      {o[i]['raw_preview'][:120]}...")
            print(f"    ours pred:     {o[i]['pred']!r}")
            print(f"    ours tokens:   {o[i]['n_tokens']}  trunc={o[i]['truncated']}")
            print()

    # Analyze ours-only wins
    if o_only:
        print("=" * 80)
        print(f"OURS CORRECT, HARNESS WRONG ({len(o_only)} samples)")
        print("=" * 80)
        for i in o_only[:10]:
            print(f"  doc_id={i}")
            print(f"    harness resp:  {h[i]['resp_preview'][:120]}...")
            print(f"    ours raw:      {o[i]['raw_preview'][:120]}...")
            print(f"    ours pred:     {o[i]['pred']!r}")
            print()

    # Token length comparison
    print("=" * 80)
    print("TOKEN LENGTH COMPARISON (ours)")
    print("=" * 80)
    our_tokens = [o[i]["n_tokens"] for i in common_ids]
    print(f"  Mean tokens:  {sum(our_tokens)/len(our_tokens):.1f}")
    print(f"  Truncated:    {sum(o[i]['truncated'] for i in common_ids)}")
    print(f"  No \\boxed:    {sum(not o[i]['has_boxed'] for i in common_ids)}")
    h_no_boxed = sum(not h[i]["has_boxed"] for i in common_ids)
    print(f"  Harness no \\boxed: {h_no_boxed}")


if __name__ == "__main__":
    main()
