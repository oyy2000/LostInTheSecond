"""
CommonsenseQA answer equivalence checker.

Public API
----------
- extract_choice_letter(text) -> str   (returns "A"-"E" or "")
- is_choice_correct(pred, gold) -> bool
"""

import re


def extract_choice_letter(text: str) -> str:
    """Extract the predicted answer letter (A-E) from a CoT response.

    Tries multiple patterns in priority order:
    1. "the answer is (X)" / "answer: X"
    2. Boxed answer \\boxed{X}
    3. Last standalone letter A-E in the text
    """
    if not text:
        return ""
    text = text.strip()

    # Pattern 1: "the answer is (X)" / "answer is X" / "Answer: X"
    m = re.search(
        r"(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Ea-e])\)?",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Pattern 2: \boxed{X}
    m = re.search(r"\\boxed\{([A-Ea-e])\}", text)
    if m:
        return m.group(1).upper()

    # Pattern 3: "So, (X)" / "Therefore, (X)" / "Thus (X)"
    m = re.search(
        r"(?:so|therefore|thus|hence)[,\s]+\(?([A-Ea-e])\)?",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Pattern 4: last standalone letter A-E (preceded by space/newline/start)
    matches = re.findall(r"(?:^|[\s(])([A-Ea-e])(?:[.\s)\n,]|$)", text)
    if matches:
        return matches[-1].upper()

    return ""


def is_choice_correct(pred: str, gold: str) -> bool:
    """Check if predicted letter matches gold letter."""
    if not pred or not gold:
        return False
    return pred.strip().upper() == gold.strip().upper()
