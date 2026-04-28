#!/usr/bin/env python3
"""
Rejection sampling: generate N responses per question with temperature > 0,
keep correct ones as additional SFT training data.

Uses vLLM for efficient batched multi-sample generation.
Focuses on questions the model got WRONG in greedy decoding.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

ANSWER_PATTERNS = [
    (r"The answer is:?\s*(.+?)(?:\n|$)", True),
    (r"####\s*(.+?)(?:\n|$)", False),
]


def extract_answer(text: str) -> str:
    import re
    for pat, ignore_case in ANSWER_PATTERNS:
        flags = re.IGNORECASE if ignore_case else 0
        m = re.search(pat, text or "", flags)
        if m:
            return m.group(1).strip()
    idx = (text or "").rfind("\\boxed")
    if idx >= 0:
        i, depth = idx, 0
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[idx + len("\\boxed{"):i].strip()
            i += 1
    return ""


def normalize_answer(text: str) -> str:
    import re
    text = (text or "").strip().lower()
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text


def is_correct(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    return normalize_answer(pred) == normalize_answer(gt)


def make_sft_entry(question: str, response: str):
    instruction = f"### Instruction:\n{question}\n\n### Response: Let's think step by step."
    return {"instruction": instruction, "input": "", "output": response}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", default="artifacts_real/full/lemma_llama3_generations.json",
                    help="Original greedy generation results")
    ap.add_argument("--questions-file", default="artifacts_real/full/lemma_all_questions.jsonl")
    ap.add_argument("--out-file", default="artifacts_real/full/lemma_sft_rejection_sampled.json")
    ap.add_argument("--out-raw", default="artifacts_real/full/lemma_rejection_raw.json",
                    help="Raw generation output for further processing")
    ap.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="Number of responses to sample per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--only-previously-wrong", action="store_true", default=True)
    ap.add_argument("--no-only-previously-wrong", dest="only_previously_wrong",
                    action="store_false")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def build_llama3_messages(question: str):
    return [
        {
            "role": "system",
            "content": "You are a helpful math assistant. Solve the problem step by step. "
                       "Separate each step with a blank line. At the end, write "
                       "\"The answer is: <your_answer>\".",
        },
        {"role": "user", "content": question},
    ]


def main():
    args = parse_args()
    random.seed(args.seed)

    gen_path = Path(args.in_file)
    obj = json.loads(gen_path.read_text("utf-8"))
    all_samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj

    if args.only_previously_wrong:
        targets = [s for s in all_samples if s.get("source_exact_match") == 0.0]
        print(f"Targeting {len(targets)} previously-wrong questions (out of {len(all_samples)})")
    else:
        targets = all_samples
        print(f"Targeting all {len(targets)} questions")

    if args.limit > 0:
        targets = targets[:args.limit]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    prompts = []
    questions_meta = []
    for s in targets:
        q = s["doc"]["question"]
        msgs = build_llama3_messages(q)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        questions_meta.append({
            "question": q,
            "doc_id": s["doc"].get("id", -1),
            "gt_answer": s.get("gt_answer", ""),
            "source": s.get("source", ""),
        })

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    print(f"Generating {args.n_samples} responses x {len(prompts)} questions = "
          f"{args.n_samples * len(prompts)} total generations")

    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        max_model_len=4096,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel,
    )
    outputs = llm.generate(prompts, sampling_params)

    sft_entries = []
    raw_entries = []
    questions_with_correct = 0
    total_correct_responses = 0

    for meta, output in tqdm(zip(questions_meta, outputs), total=len(outputs), desc="Filtering"):
        gt = meta["gt_answer"]
        seen_normalized = set()
        q_has_correct = False

        for completion in output.outputs:
            text = completion.text.strip()
            pred = extract_answer(text)

            raw_entries.append({
                "doc_id": meta["doc_id"],
                "question": meta["question"],
                "response": text,
                "pred_answer": pred,
                "gt_answer": gt,
                "is_correct": is_correct(pred, gt),
                "source": meta["source"],
            })

            if is_correct(pred, gt):
                norm_key = normalize_answer(text[:200])
                if norm_key not in seen_normalized:
                    seen_normalized.add(norm_key)
                    sft_entries.append(make_sft_entry(meta["question"], text))
                    total_correct_responses += 1
                    q_has_correct = True

        if q_has_correct:
            questions_with_correct += 1

    random.shuffle(sft_entries)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sft_entries, indent=2, ensure_ascii=False), "utf-8")

    raw_path = Path(args.out_raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_entries, indent=2, ensure_ascii=False), "utf-8")

    print(f"\n=== Rejection Sampling Results ===")
    print(f"Questions targeted: {len(targets)}")
    print(f"Questions with ≥1 correct response: {questions_with_correct} "
          f"({100*questions_with_correct/len(targets):.1f}%)")
    print(f"Total correct (deduplicated) responses: {total_correct_responses}")
    print(f"SFT entries: {len(sft_entries)}")
    print(f"Output: {out_path}")
    print(f"Raw: {raw_path}")


if __name__ == "__main__":
    main()
