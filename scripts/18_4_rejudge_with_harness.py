#!/usr/bin/env python3
"""Re-judge our vLLM outputs using the exact harness process_results logic.

Imports is_equiv / strip_string / last_boxed_only_string / remove_boxed
directly from the harness utils.py, so there is zero divergence in grading.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "lm-evaluation-harness/lm_eval/tasks/hendrycks_math_500"))
from utils import is_equiv, last_boxed_only_string, remove_boxed

MATH500_PATH = ROOT / "lm-evaluation-harness/math_eval_data/MATH-500/test.jsonl"
OUR_SAMPLES = ROOT / "results/math500_vllm_4096_llama_3.2_3b_instruct.jsonl"
HARNESS_SAMPLES = ROOT / "results/harness_math500_vllm/meta-llama__Llama-3.2-3B-Instruct/samples_hendrycks_math_500_2026-04-26T14-42-04.294052.jsonl"


def main():
    # Load gold solutions
    gold = {}
    for i, line in enumerate(MATH500_PATH.read_text("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        gold[i] = row["solution"]

    # Load our outputs
    ours = {}
    for line in OUR_SAMPLES.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        idx = int(r["doc_id"].replace("math500_", ""))
        ours[idx] = r

    # Load harness outputs
    harness = {}
    for line in HARNESS_SAMPLES.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        harness[r["doc_id"]] = r

    common = sorted(set(ours.keys()) & set(harness.keys()))
    print(f"Samples: {len(common)}")

    # Re-judge using exact harness logic
    our_correct_harness_judge = 0
    our_correct_our_judge = 0
    harness_correct = 0
    disagree = []

    for i in common:
        raw = ours[i].get("raw_output", "")
        sol = gold[i]

        # Harness grading on our output
        boxed_pred = last_boxed_only_string(raw)
        boxed_gold = last_boxed_only_string(sol)
        h_judge = 0
        if boxed_pred is not None and boxed_gold is not None:
            if is_equiv(remove_boxed(boxed_pred), remove_boxed(boxed_gold)):
                h_judge = 1
        our_correct_harness_judge += h_judge

        # Our grading
        o_judge = 1 if ours[i]["correct"] else 0
        our_correct_our_judge += o_judge

        # Harness self-grading
        h_self = harness[i].get("exact_match", 0)
        harness_correct += h_self

        if h_judge != o_judge:
            disagree.append({
                "doc_id": i,
                "harness_judge": h_judge,
                "our_judge": o_judge,
                "harness_self": h_self,
                "our_pred": ours[i].get("pred", ""),
                "boxed_pred": remove_boxed(boxed_pred) if boxed_pred else None,
                "boxed_gold": remove_boxed(boxed_gold) if boxed_gold else None,
                "raw_preview": raw[:150],
            })

    n = len(common)
    print(f"\nHarness self-grade:          {harness_correct}/{n} = {harness_correct/n:.4f}")
    print(f"Our output, harness judge:   {our_correct_harness_judge}/{n} = {our_correct_harness_judge/n:.4f}")
    print(f"Our output, our judge:       {our_correct_our_judge}/{n} = {our_correct_our_judge/n:.4f}")
    print(f"\nJudge disagreements (harness vs ours on same output): {len(disagree)}")

    if disagree:
        print("\n" + "=" * 80)
        print("GRADING DISAGREEMENTS")
        print("=" * 80)
        for d in disagree[:20]:
            print(f"\n  doc_id={d['doc_id']}")
            print(f"    harness_judge={d['harness_judge']}  our_judge={d['our_judge']}")
            print(f"    boxed_pred: {d['boxed_pred']!r}")
            print(f"    boxed_gold: {d['boxed_gold']!r}")
            print(f"    our_pred:   {d['our_pred']!r}")
            print(f"    raw:        {d['raw_preview']!r}")


if __name__ == "__main__":
    main()
