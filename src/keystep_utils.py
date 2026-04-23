"""
Shared utilities for KeyStep-SD pipeline.

- Step splitting and answer extraction (reused from existing codebase)
- Step-level logprob / entropy extraction from vLLM outputs
- Stall-pattern detection
- Number grounding check for prefix compatibility
- Config loading
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = str(PROJECT_ROOT / "configs" / "keystep_sd.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Step splitting
# ---------------------------------------------------------------------------

def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    steps = [x.strip() for x in text.split("\n\n") if x.strip()]
    return steps if steps else [text]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str:
    idx = (text or "").rfind("\\boxed")
    if idx < 0:
        return ""
    i, depth, start = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            if depth == 0:
                start = i
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start + 1 : i].strip()
        i += 1
    return ""


def normalize_answer(text: str) -> str:
    text = (text or "").strip().replace("$", "").replace(",", "")
    text = re.sub(r"\\boxed\{(.*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


# ---------------------------------------------------------------------------
# Chat prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_chat_prompt(question: str, system: str = SYSTEM_PROMPT) -> str:
    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_prefix_prompt(
    question: str,
    prefix_steps: List[str],
    system: str = SYSTEM_PROMPT,
) -> str:
    base = build_chat_prompt(question, system)
    if prefix_steps:
        base += "\n\n".join(prefix_steps) + "\n\n"
    return base


# ---------------------------------------------------------------------------
# Large-model single-step prompt
# ---------------------------------------------------------------------------

LARGE_MODEL_STEP_PROMPT = (
    "You are given a math/problem-solving trace.\n\n"
    "Problem:\n{question}\n\n"
    "Previous verified steps:\n{prefix}\n\n"
    "Write ONLY the next single reasoning step.\n"
    "Do not write the final answer unless the solution is complete.\n"
    "Keep the step concise and correct."
)


def build_large_step_prompt(question: str, prefix_steps: List[str]) -> str:
    prefix_text = "\n\n".join(
        f"Step {i+1}: {s}" for i, s in enumerate(prefix_steps)
    ) if prefix_steps else "(none)"
    return LARGE_MODEL_STEP_PROMPT.format(question=question, prefix=prefix_text)


# ---------------------------------------------------------------------------
# Step-level logprob / entropy from vLLM output
# ---------------------------------------------------------------------------

def step_token_boundaries(
    full_text: str, steps: List[str],
) -> List[Tuple[int, int]]:
    """Return (char_start, char_end) for each step in full_text."""
    boundaries = []
    pos = 0
    for step in steps:
        idx = full_text.find(step, pos)
        if idx < 0:
            idx = pos
        boundaries.append((idx, idx + len(step)))
        pos = idx + len(step)
    return boundaries


def compute_step_logprob_stats(
    token_logprobs: List[float],
    token_offsets: List[int],
    step_char_boundaries: List[Tuple[int, int]],
) -> List[Dict[str, float]]:
    """Compute per-step mean neg-logprob and entropy from token-level data.

    token_logprobs: list of log-probabilities for each generated token.
    token_offsets: character offset of each token in the generated text.
    step_char_boundaries: (start, end) char positions for each step.

    Returns list of dicts with keys: mean_neglogprob, entropy, n_tokens.
    """
    stats = []
    ti = 0
    for s_start, s_end in step_char_boundaries:
        step_lps = []
        while ti < len(token_offsets) and token_offsets[ti] < s_end:
            if token_offsets[ti] >= s_start:
                step_lps.append(token_logprobs[ti])
            ti += 1
        if step_lps:
            mean_neglp = -sum(step_lps) / len(step_lps)
            probs = [math.exp(lp) for lp in step_lps]
            entropy = -sum(
                p * lp for p, lp in zip(probs, step_lps) if p > 0
            ) / len(step_lps)
        else:
            mean_neglp = 0.0
            entropy = 0.0
        stats.append({
            "mean_neglogprob": mean_neglp,
            "entropy": entropy,
            "n_tokens": len(step_lps),
        })
    return stats


# ---------------------------------------------------------------------------
# Stall-pattern detection
# ---------------------------------------------------------------------------

STALL_PATTERNS = [
    re.compile(r"\bWait\b", re.IGNORECASE),
    re.compile(r"\bLet me reconsider\b", re.IGNORECASE),
    re.compile(r"\bHmm\b", re.IGNORECASE),
    re.compile(r"\bAlternatively\b", re.IGNORECASE),
    re.compile(r"\bActually\b", re.IGNORECASE),
    re.compile(r"\bOn second thought\b", re.IGNORECASE),
    re.compile(r"\bI made (a|an) (error|mistake)\b", re.IGNORECASE),
    re.compile(r"\bLet me re(do|check|calculate|think)\b", re.IGNORECASE),
]


def detect_stall(step_text: str) -> float:
    """Return 1.0 if the step contains stall/reflective patterns, else 0.0."""
    for pat in STALL_PATTERNS:
        if pat.search(step_text):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Number grounding check (prefix compatibility)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)


def number_grounding_score(
    step_text: str,
    question: str,
    prefix_steps: List[str],
) -> float:
    """Fraction of numbers in step_text that appear in question or prefix."""
    step_nums = set(extract_numbers(step_text))
    if not step_nums:
        return 1.0
    context = question + " " + " ".join(prefix_steps)
    context_nums = set(extract_numbers(context))
    grounded = step_nums & context_nums
    return len(grounded) / len(step_nums)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

import json


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
