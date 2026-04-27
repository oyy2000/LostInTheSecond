"""
Model-aware prompt formatting and answer equivalence dispatch.

Supports Qwen and Llama chat templates (official formats).
Uses a unified math prompt for GSM8K/MATH500/AIME/AMC/OlympiadBench.

Public API
----------
- build_prompt(model_id, dataset, question, tokenizer=None) -> str
- build_prompt_ids(tokenizer, dataset, question) -> List[int]
- extract_answer(dataset, text) -> str
- check_answer(dataset, pred, gold) -> bool
- split_steps(text) -> List[str]
- get_stop_tokens(model_id) -> List[str]
"""

import re
from typing import List, Optional

from src.math_answer_equiv import extract_boxed_answer, is_math_equiv
from src.hotpotqa_answer_equiv import extract_short_answer, hotpotqa_em
from src.csqa_answer_equiv import extract_choice_letter, is_choice_correct
from src.code_answer_equiv import extract_python_code, check_humaneval


# -- user-facing prompts per dataset (no chat wrapper) ---------------------

_MATH_PROMPT = (
    "Solve the following math problem. Present the final answer "
    "in the format: Final Answer: \\boxed{{your_answer}}.\n"
    "Problem: {question}\n"
    "Answer:"
)

_HOTPOTQA_PROMPT = (
    "Answer the following multi-hop question step by step. "
    "After your reasoning, write your final answer as: "
    "The answer is <your answer>.\n\n"
    "Question: {question}\n"
    "Answer:"
)

_HUMANEVAL_PROMPT = (
    "Complete the following Python function. "
    "Return ONLY the function body inside a ```python code block.\n\n"
    "{question}"
)

_CSQA_PROMPT = (
    "Answer the following multiple-choice question. "
    "Reply with ONLY the letter (A/B/C/D/E) of the correct answer.\n\n"
    "{question}"
)

MATH_DATASETS = {"gsm8k", "math500", "aime2024", "amc2023", "olympiadbench"}


def _get_user_prompt(dataset: str, question: str) -> str:
    ds = dataset.lower()
    if ds in MATH_DATASETS:
        return _MATH_PROMPT.format(question=question)
    if ds == "hotpotqa":
        return _HOTPOTQA_PROMPT.format(question=question)
    if ds == "humaneval":
        return _HUMANEVAL_PROMPT.format(question=question)
    if ds == "csqa":
        return _CSQA_PROMPT.format(question=question)
    return _MATH_PROMPT.format(question=question)


# -- official chat templates -----------------------------------------------

def _is_qwen(model_id: str) -> bool:
    return "qwen" in model_id.lower()


def _is_llama(model_id: str) -> bool:
    return "llama" in model_id.lower()


def _qwen_prompt(user_msg: str) -> str:
    """Official Qwen 2.5 ChatML format."""
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _llama_prompt(user_msg: str) -> str:
    """Official Llama 3.x Instruct format."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_prompt(model_id: str, dataset: str, question: str,
                 tokenizer=None) -> str:
    """Build a chat-formatted prompt for the given model and dataset.

    If *tokenizer* is provided, uses ``tokenizer.apply_chat_template``
    (correct special-token handling).  Otherwise falls back to the
    hand-crafted templates (legacy, may double-BOS on vLLM).
    """
    user_msg = _get_user_prompt(dataset, question)
    if tokenizer is not None:
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if _is_qwen(model_id):
        return _qwen_prompt(user_msg)
    if _is_llama(model_id):
        return _llama_prompt(user_msg)
    return _qwen_prompt(user_msg)


def build_prompt_ids(tokenizer, dataset: str, question: str) -> List[int]:
    """Build prompt and return token IDs (no duplicate BOS)."""
    prompt_str = build_prompt("", dataset, question, tokenizer=tokenizer)
    return tokenizer.encode(prompt_str, add_special_tokens=False)


def get_stop_tokens(model_id: str) -> List[str]:
    """Return stop tokens appropriate for the model family."""
    if _is_llama(model_id):
        return ["<|eot_id|>", "<|end_of_text|>"]
    return ["<|im_end|>", "<|endoftext|>"]


# -- step splitting --------------------------------------------------------

def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [x.strip() for x in text.split("\n\n") if x.strip()]
    return parts if parts else [text]


# -- answer extraction & equivalence per dataset ---------------------------

def _extract_math_answer(text: str) -> str:
    """Extract answer from unified math prompt: Final Answer: \\boxed{...} or \\boxed{...}."""
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    # fallback: "Final Answer: X" without boxed
    m = re.search(r"Final\s+Answer\s*:\s*(.+?)(?:\n|$)", text, re.I)
    if m:
        ans = m.group(1).strip()
        ans = ans.strip("$").replace(",", "").rstrip(".")
        return ans
    return ""


def _extract_gsm8k_answer(text: str) -> str:
    """GSM8K: try boxed first, then #### fallback, then last number."""
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed.strip("$").replace(",", "").rstrip(".")
    if "####" in text:
        raw = text.split("####")[-1].strip()
        raw = raw.strip("$").replace(",", "").rstrip(".").strip()
        if raw:
            return raw
    # "Final Answer: ..."
    m = re.search(r"Final\s+Answer\s*:\s*\$?\s*([0-9,]+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    # "the answer is ..."
    m = re.search(
        r"(?:final\s+answer\s*(?:is|:)|(?:the\s+)?answer\s*(?:is|:))\s*\$?\s*([0-9,]+(?:\.\d+)?)",
        text, re.I,
    )
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    # last number
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def _gsm8k_equiv(pred: str, gold: str) -> bool:
    p = pred.strip().strip("$").replace(",", "").rstrip(".")
    g = gold.strip().strip("$").replace(",", "").rstrip(".")
    try:
        return abs(float(p) - float(g)) < 1e-3
    except (ValueError, TypeError):
        return p == g


def extract_answer(dataset: str, text: str) -> str:
    ds = dataset.lower()
    if ds == "gsm8k":
        return _extract_gsm8k_answer(text)
    if ds in ("math500", "aime2024", "amc2023", "olympiadbench"):
        return _extract_math_answer(text)
    if ds == "hotpotqa":
        return extract_short_answer(text)
    if ds == "humaneval":
        return extract_python_code(text)
    if ds == "csqa":
        return extract_choice_letter(text)
    return text.strip()


def check_answer(dataset: str, pred: str, gold: str, **kw) -> bool:
    ds = dataset.lower()
    if ds == "gsm8k":
        return _gsm8k_equiv(pred, gold)
    if ds in ("math500", "aime2024", "amc2023", "olympiadbench"):
        return is_math_equiv(pred, gold)
    if ds == "hotpotqa":
        return hotpotqa_em(pred, gold)
    if ds == "csqa":
        return is_choice_correct(pred, gold)
    if ds == "humaneval":
        return check_humaneval(pred, kw.get("test", ""), kw.get("entry_point", ""))
    return pred.strip() == gold.strip()
