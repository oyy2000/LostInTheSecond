#!/usr/bin/env python3
"""
Phase 18: Prompt and chat-template comparison.

Prints side-by-side the exact prompts sent to the model by:
  1. Our pipeline (sweep_engine / prompt_templates.py)
  2. lm-evaluation-harness (with and without --apply_chat_template)

Usage:
    python scripts/18_0_prompt_comparison.py
    python scripts/18_0_prompt_comparison.py --dataset gsm8k
    python scripts/18_0_prompt_comparison.py --dataset math500 --model-id Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt_templates import build_prompt, _get_user_prompt

SEPARATOR = "=" * 72


def _llama_chat_template(user_msg: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _qwen_chat_template(user_msg: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# -- Harness doc_to_text per task ------------------------------------------

HARNESS_DOC_TO_TEXT = {
    "math500": (
        "Solve the following math problem. Present the final answer in the format: "
        "Final Answer: \\boxed{{your_answer}}.\nProblem: {question}\nAnswer:"
    ),
    "gsm8k": "{question}",  # gsm8k_cot_zeroshot: raw question only
    "gsm8k_cot": (
        "Q: {question}\nA:"  # 8-shot CoT; exemplars prepended by harness
    ),
}

HARNESS_STOP_TOKENS = {
    "math500": ["Problem:"],
    "gsm8k": ["Q:", "</s>", "<|im_end|>", "<|eot_id|>", "<|start_header_id|>user<|end_header_id|>"],
    "gsm8k_cot": ["Q:", "</s>", "<|im_end|>"],
}

GSM8K_COT_EXEMPLARS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. "
     "So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
]  # abbreviated to 2; harness uses 8


def _harness_raw(dataset: str, question: str) -> str:
    """Harness prompt WITHOUT --apply_chat_template (raw text)."""
    ds = dataset.lower()
    if ds == "math500":
        return HARNESS_DOC_TO_TEXT["math500"].format(question=question)
    if ds == "gsm8k":
        return HARNESS_DOC_TO_TEXT["gsm8k"].format(question=question)
    return HARNESS_DOC_TO_TEXT.get(ds, "{question}").format(question=question)


def _harness_with_chat_template(dataset: str, question: str, model_id: str) -> str:
    """Harness prompt WITH --apply_chat_template."""
    user_msg = _harness_raw(dataset, question)
    if "llama" in model_id.lower():
        return _llama_chat_template(user_msg)
    return _qwen_chat_template(user_msg)


def _harness_gsm8k_cot_with_chat(question: str, model_id: str) -> str:
    """8-shot CoT prompt with chat template (abbreviated to 2 shots here)."""
    shots = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in GSM8K_COT_EXEMPLARS
    )
    user_msg = shots + f"\nQ: {question}\nA:"
    if "llama" in model_id.lower():
        return _llama_chat_template(user_msg)
    return _qwen_chat_template(user_msg)


def print_block(title: str, text: str, show_repr: bool = True):
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")
    if show_repr:
        print(repr(text))
    print()
    print(text)


def compare(dataset: str, model_id: str, question: str):
    ds = dataset.lower()
    is_llama = "llama" in model_id.lower()
    model_short = model_id.split("/")[-1]

    print(f"\n{SEPARATOR}")
    print(f"  Dataset : {dataset}")
    print(f"  Model   : {model_short}")
    print(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")
    print(SEPARATOR)

    # 1. Our pipeline
    our_prompt = build_prompt(model_id, dataset, question)
    print_block("1. OUR PIPELINE  (prompt_templates.build_prompt)", our_prompt)

    # 2. Harness raw (no chat template)
    harness_raw = _harness_raw(dataset, question)
    print_block("2. HARNESS  --no-apply_chat_template  (raw text)", harness_raw)

    # 3. Harness with chat template
    harness_chat = _harness_with_chat_template(dataset, question, model_id)
    print_block("3. HARNESS  --apply_chat_template", harness_chat)

    # 4. For GSM8K: also show 8-shot CoT variant
    if ds == "gsm8k":
        harness_cot = _harness_gsm8k_cot_with_chat(question, model_id)
        print_block(
            "4. HARNESS  gsm8k_cot  (8-shot CoT + chat template, 2 shots shown)",
            harness_cot,
        )

    # Diff summary
    print(f"\n{'─' * 72}")
    print("  KEY DIFFERENCES")
    print(f"{'─' * 72}")

    ours_user = _get_user_prompt(dataset, question)
    harness_user = _harness_raw(dataset, question)

    if ours_user.strip() == harness_user.strip():
        print("  User message:    IDENTICAL")
    else:
        print("  User message:    DIFFERENT")
        print(f"    Ours:    {repr(ours_user[:120])}")
        print(f"    Harness: {repr(harness_user[:120])}")

    print(f"  Chat template:   {'Llama 3.x' if is_llama else 'Qwen ChatML'} (ours)")
    print(f"  Stop tokens:")
    print(f"    Ours:    {['<|eot_id|>', '<|end_of_text|>'] if is_llama else ['<|im_end|>', '<|endoftext|>']}")
    print(f"    Harness: {HARNESS_STOP_TOKENS.get(ds, ['?'])}")

    if ds == "gsm8k":
        print("  Few-shot:        0-shot (ours)  vs  0-shot or 8-shot CoT (harness)")
    else:
        print("  Few-shot:        0-shot (both)")

    print()


def load_example_question(dataset: str) -> str:
    ds = dataset.lower()
    if ds == "math500":
        p = Path("lm-evaluation-harness/math_eval_data/MATH-500/test.jsonl")
        if p.exists():
            row = json.loads(p.read_text("utf-8").splitlines()[0])
            return row["problem"]
    if ds == "gsm8k":
        try:
            from datasets import load_dataset as hf_load
            row = next(iter(hf_load("openai/gsm8k", "main", split="test")))
            return row["question"]
        except Exception:
            pass
    return "What is 2 + 2?"


def main():
    ap = argparse.ArgumentParser(description="Prompt comparison: ours vs lm-eval-harness")
    ap.add_argument("--dataset", default="", help="Dataset name (math500, gsm8k). Empty = both.")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--question", default="", help="Custom question text")
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else ["math500", "gsm8k"]

    for ds in datasets:
        q = args.question or load_example_question(ds)
        compare(ds, args.model_id, q)


if __name__ == "__main__":
    main()
