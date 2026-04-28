"""
HotpotQA answer equivalence checker.

HotpotQA answers are short free-form strings (e.g. "yes", "Coahuila, Mexico",
"a scholar").  We use the standard SQuAD-style F1 / EM evaluation:
normalise -> tokenise -> compare.

Public API
----------
- normalise_answer(text) -> str
- extract_short_answer(text) -> str
- hotpotqa_em(pred, gold) -> bool
- hotpotqa_f1(pred, gold) -> float
"""

import re
import string
from collections import Counter


def normalise_answer(s: str) -> str:
    """Lower, strip articles / punctuation / extra whitespace."""
    s = s.lower().strip()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # collapse whitespace
    return " ".join(s.split())


def _get_tokens(s: str) -> list[str]:
    return normalise_answer(s).split()


def extract_short_answer(text: str) -> str:
    """Extract the predicted short answer from a CoT response.

    Priority:
    1. "the answer is ..." / "answer: ..."
    2. "\\boxed{...}"
    3. Last sentence of the response
    """
    if not text:
        return ""
    text = text.strip()

    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")

    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return m.group(1).strip()

    # last sentence heuristic
    sents = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
    if sents:
        return sents[-1].strip().rstrip(".")
    return text.strip()


def hotpotqa_em(pred: str, gold: str) -> bool:
    return normalise_answer(pred) == normalise_answer(gold)


def hotpotqa_f1(pred: str, gold: str) -> float:
    pred_toks = _get_tokens(pred)
    gold_toks = _get_tokens(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
