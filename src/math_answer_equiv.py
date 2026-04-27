"""
Robust MATH answer equivalence checker.

Wraps the battle-tested strip_string / is_equiv logic from
lm-evaluation-harness (hendrycks_math_500) and adds a boxed-answer
extractor that handles nested braces.

Public API
----------
- extract_boxed_answer(text) -> str
- is_math_equiv(pred, gold) -> bool
"""

import re


# ── LaTeX normalisation (from lm-evaluation-harness) ────────────────────

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr or substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
        if string == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
    except (ValueError, AssertionError):
        pass
    return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    """Normalise a LaTeX answer string for comparison."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


# ── Boxed-answer extraction ─────────────────────────────────────────────

def extract_boxed_answer(text: str) -> str:
    """Extract the content of the last \\boxed{...} in *text*."""
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


# ── Equivalence check ───────────────────────────────────────────────────

def is_math_equiv(pred: str, gold: str) -> bool:
    """Return True when *pred* and *gold* represent the same MATH answer."""
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    pred = str(pred).strip()
    gold = str(gold).strip()
    if not pred or not gold:
        return False
    # Fast path: literal match after stripping $ and \boxed{}
    pred_clean = re.sub(r"\\boxed\{(.*)\}", r"\1", pred).replace("$", "").strip()
    gold_clean = re.sub(r"\\boxed\{(.*)\}", r"\1", gold).replace("$", "").strip()
    if pred_clean == gold_clean:
        return True
    try:
        return strip_string(pred_clean) == strip_string(gold_clean)
    except Exception:
        return pred_clean == gold_clean
