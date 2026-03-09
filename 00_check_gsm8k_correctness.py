#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute GSM8K correctness from lm-eval sample jsonl and export incorrect samples.

Input format:
- lm-eval output jsonl rows (doc/problem/solution/target, filtered_resps or resps)

Outputs:
1) incorrect jsonl (original rows + recomputed fields), for downstream Step2 fixing.
2) optional report json with counts and simple stats.
3) optional ids json for quick indexing.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_boxed(s: str) -> str:
    s = s.strip()
    if s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{") : -1]
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    return s


def last_boxed_only_string(string: str) -> Optional[str]:
    if not string:
        return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) <= 1:
        return string

    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr:
            continue
        if substr[0] == "{":
            new_str += substr
        else:
            if len(substr) < 2:
                return string
            a = substr[0]
            b = substr[1]
            if b != "{":
                post_substr = substr[2:] if len(substr) > 2 else ""
                new_str += "{" + a + "}{" + b + "}" + post_substr
            else:
                post_substr = substr[2:] if len(substr) > 2 else ""
                new_str += "{" + a + "}" + b + post_substr
    return new_str


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        ai = int(a)
        bi = int(b)
        if string != f"{ai}/{bi}":
            return string
        return f"\\frac{{{ai}}}{{{bi}}}"
    except Exception:
        return string


def strip_string(string: str) -> str:
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
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1: Optional[str], str2: Optional[str]) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        a = str1.lstrip("0")
        b = str2.lstrip("0")
        if "." in a and a.split(".")[1].rstrip("0") == "":
            a = a.split(".")[0]
        if "." in b and b.split(".")[1].rstrip("0") == "":
            b = b.split(".")[0]
        return strip_string(a) == strip_string(b)
    except Exception:
        return str1 == str2


def extract_generation(rec: Dict[str, Any]) -> str:
    fr = rec.get("filtered_resps", [])
    if isinstance(fr, list) and fr:
        return (fr[0] or "").strip()
    rs = rec.get("resps", [])
    if rs and isinstance(rs[0], list) and rs[0]:
        return (rs[0][0] or "").strip()
    return ""


def extract_gold_answer_text(rec: Dict[str, Any]) -> str:
    doc = rec.get("doc", {}) or {}
    candidates = [
        doc.get("answer"),
        rec.get("target"),
        doc.get("solution"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return ""


def extract_hash_answer(text: str) -> Optional[str]:
    if not text:
        return None
    if "####" not in text:
        return None
    tail = text.split("####")[-1].strip()
    return tail or None


def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:,\d{3})*(?:\.\d+)?)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_answer_from_prediction(pred: str) -> Optional[str]:
    pred = (pred or "").strip()
    if not pred:
        return None

    boxed = last_boxed_only_string(pred)
    if boxed is not None:
        return remove_boxed(boxed).strip()

    m = re.search(r"Final\s*Answer\s*:\s*(.+)$", pred, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        cand = m.group(1).strip()
        if cand:
            return cand

    h = extract_hash_answer(pred)
    if h:
        return h

    return extract_last_number(pred)


def extract_answer_from_gold(gold_text: str) -> Optional[str]:
    gold_text = (gold_text or "").strip()
    if not gold_text:
        return None

    h = extract_hash_answer(gold_text)
    if h:
        return h

    boxed = last_boxed_only_string(gold_text)
    if boxed is not None:
        return remove_boxed(boxed).strip()

    return extract_last_number(gold_text)


def evaluate_row(rec: Dict[str, Any]) -> Tuple[int, Optional[str], Optional[str]]:
    pred = extract_generation(rec)
    gold_text = extract_gold_answer_text(rec)
    pred_ans = extract_answer_from_prediction(pred)
    gold_ans = extract_answer_from_gold(gold_text)

    em = 1 if is_equiv(pred_ans, gold_ans) else 0
    return em, pred_ans, gold_ans


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Recompute GSM8K correctness and filter incorrect samples")
    ap.add_argument("--in-file", required=True, help="Input lm-eval jsonl")
    ap.add_argument("--out-incorrect-jsonl", required=True, help="Output jsonl containing only incorrect rows")
    ap.add_argument("--out-report-json", default="", help="Optional report json path")
    ap.add_argument("--out-ids-json", default="", help="Optional json path: {'ids':[...]} for incorrect doc_id")
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_file)
    out_wrong_path = Path(args.out_incorrect_jsonl)

    rows = list(read_jsonl(in_path))
    if args.limit > 0:
        rows = rows[: args.limit]

    wrong_rows: List[Dict[str, Any]] = []
    wrong_ids: List[int] = []
    n_correct, n_wrong = 0, 0

    for rec in rows:
        em, pred_ans, gold_ans = evaluate_row(rec)
        new_rec = dict(rec)
        new_rec["recomputed_exact_match"] = int(em)
        new_rec["recomputed_pred_answer"] = pred_ans
        new_rec["recomputed_gold_answer"] = gold_ans

        if em == 1:
            n_correct += 1
        else:
            n_wrong += 1
            wrong_rows.append(new_rec)
            try:
                wrong_ids.append(int(new_rec.get("doc_id", -1)))
            except Exception:
                pass

    write_jsonl(out_wrong_path, wrong_rows)

    total = len(rows)
    acc = (n_correct / total) if total > 0 else 0.0
    report = {
        "input": str(in_path),
        "incorrect_jsonl": str(out_wrong_path),
        "total": total,
        "correct": n_correct,
        "incorrect": n_wrong,
        "accuracy": acc,
    }

    if args.out_report_json:
        report_path = Path(args.out_report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_ids_json:
        ids_path = Path(args.out_ids_json)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(json.dumps({"ids": wrong_ids}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] total={total}, correct={n_correct}, incorrect={n_wrong}, acc={acc:.4f}")
    print(f"[Done] incorrect -> {out_wrong_path}")
    if args.out_report_json:
        print(f"[Done] report    -> {args.out_report_json}")
    if args.out_ids_json:
        print(f"[Done] ids       -> {args.out_ids_json}")


if __name__ == "__main__":
    main()
