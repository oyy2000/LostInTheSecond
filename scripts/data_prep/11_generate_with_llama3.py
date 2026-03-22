#!/usr/bin/env python3
"""
Generate step-by-step math solutions with LLaMA-3-8B-Instruct.

Reads questions from a JSONL file (from 10_sample_lemma_questions.py),
generates solutions, splits into steps, and outputs a format compatible
with 12_gpt_fix_step2_lemma.py.

Output format (JSON with "samples" key):
{
  "samples": [
    {
      "doc": {"question": "...", "id": 123},
      "neg_response": "...",
      "neg_steps": [...],
      "source_exact_match": 0|1,
      "gt_answer": "..."
    }
  ]
}
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)


def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if "\n\n" in text:
        parts = [x.strip() for x in text.split("\n\n") if x.strip()]
        if len(parts) >= 2:
            return parts
    hits = list(STEP_RE.finditer(text))
    if hits:
        out = []
        for i, m in enumerate(hits):
            st = m.start()
            ed = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            out.append(text[st:ed].strip())
        return [x for x in out if x]
    return [x.strip() for x in text.split("\n") if x.strip()]


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def extract_model_answer(text: str) -> str:
    """Extract final answer from model output."""
    m = re.search(r"The answer is:?\s*(.+?)(?:\n|$)", text or "", re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(.+?)(?:\n|$)", text or "")
    if m:
        return m.group(1).strip()
    idx = (text or "").rfind("\\boxed")
    if idx >= 0:
        i = idx
        num_left = 0
        right_idx = None
        while i < len(text):
            if text[i] == "{":
                num_left += 1
            if text[i] == "}":
                num_left -= 1
                if num_left == 0:
                    right_idx = i
                    break
            i += 1
        if right_idx is not None:
            inner = text[idx:right_idx + 1]
            if inner.startswith("\\boxed{") and inner.endswith("}"):
                return inner[len("\\boxed{"):-1].strip()
    return ""


def check_correct(pred_answer: str, gt_answer: str) -> bool:
    if not pred_answer or not gt_answer:
        return False
    return normalize_answer(pred_answer) == normalize_answer(gt_answer)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-file",
        default="./artifacts_real/lemma_sampled_questions.jsonl",
    )
    ap.add_argument(
        "--out-file",
        default="./artifacts_real/lemma_llama3_generations.json",
    )
    ap.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--use-vllm", action="store_true", default=True)
    ap.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    return ap.parse_args()


def build_llama3_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful math assistant. Solve the problem step by step. "
                       "Separate each step with a blank line. At the end, write "
                       "\"The answer is: <your_answer>\".",
        },
        {"role": "user", "content": question},
    ]


def generate_vllm(args, questions: List[Dict[str, Any]]) -> List[str]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    prompts = []
    for q in questions:
        msgs = build_llama3_messages(q["question"])
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=1.0 if args.temperature == 0 else 0.95,
    )

    llm = LLM(model=args.model_id, trust_remote_code=True, max_model_len=4096, dtype="half")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        results.append(text)
    return results


def generate_hf(args, questions: List[Dict[str, Any]]) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
    ).eval()

    results = []
    for q in tqdm(questions, desc="Generate"):
        msgs = build_llama3_messages(q["question"])
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = 0.95

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append(text)

    return results


def main():
    args = parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"Generating for {len(questions)} questions with {args.model_id}")

    if args.use_vllm:
        responses = generate_vllm(args, questions)
    else:
        responses = generate_hf(args, questions)

    samples = []
    n_correct, n_total = 0, 0
    for q, resp in zip(questions, responses):
        steps = split_steps(resp)
        pred_answer = extract_model_answer(resp)
        gt = q.get("gt_answer", "")
        is_correct = check_correct(pred_answer, gt) if gt else False
        if is_correct:
            n_correct += 1
        n_total += 1

        samples.append({
            "doc": {"question": q["question"], "id": q["id"]},
            "neg_response": resp,
            "neg_steps": steps,
            "source_exact_match": 1.0 if is_correct else 0.0,
            "gt_answer": gt,
            "pred_answer": pred_answer,
            "source": q.get("source", ""),
        })

    out_obj = {
        "samples": samples,
        "meta": {
            "model_id": args.model_id,
            "n_questions": len(questions),
            "n_correct": n_correct,
            "n_incorrect": n_total - n_correct,
            "accuracy": n_correct / n_total if n_total > 0 else 0,
        },
    }

    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. {n_total} samples generated.")
    print(f"Correct: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
