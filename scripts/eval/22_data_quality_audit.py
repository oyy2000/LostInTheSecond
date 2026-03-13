#!/usr/bin/env python3
"""
Automated quality audit of the 109 training samples.

Reads the prefill training dataset and checks each sample for:
  - Does pos_response reach the correct final answer?
  - Does neg_response produce a wrong answer?
  - Is corrected_step2 present and non-empty?
  - Step count sanity (pos vs neg)
  - Prefill correctness metadata consistency

Flags problematic samples for manual review.

Output: documents/data_quality_audit.md

Usage:
    python 22_data_quality_audit.py
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
TRAINING_DATA_PATHS = [
    BEEGFS_ARTIFACTS / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json",
    ARTIFACTS_DIR / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json",
]


def find_file(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer from a response."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip().replace(",", "")
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract answer from 'Final Answer: ...' or '#### ...' patterns."""
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    m = re.search(r'####\s*(.+?)$', text, re.MULTILINE)
    if m:
        return m.group(1).strip().replace(",", "").replace("$", "")

    m = re.search(r'Final Answer:\s*(.+?)$', text, re.MULTILINE)
    if m:
        answer = m.group(1).strip()
        boxed2 = extract_boxed_answer(answer)
        return boxed2 if boxed2 else answer

    return None


def extract_gold_answer(doc: Dict) -> Optional[str]:
    """Extract gold answer from doc (GSM8K format: #### N)."""
    answer_text = doc.get("answer", "")
    m = re.search(r'####\s*(.+?)$', answer_text, re.MULTILINE)
    if m:
        return m.group(1).strip().replace(",", "").replace("$", "")
    return None


def normalize_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    return ans.strip().lower().replace(",", "").replace("$", "").replace("%", "")


def answers_match(a: Optional[str], b: Optional[str]) -> bool:
    na = normalize_answer(a)
    nb = normalize_answer(b)
    if not na or not nb:
        return False
    try:
        return abs(float(na) - float(nb)) < 1e-6
    except ValueError:
        return na == nb


def audit_sample(sample: Dict, idx: int) -> Dict[str, Any]:
    """Audit a single training sample. Return issues found."""
    doc = sample.get("doc", {})
    question = doc.get("question", "")
    qid = doc.get("id", idx)

    pos_response = sample.get("pos_response", "")
    neg_response = sample.get("neg_response", "")
    pos_steps = sample.get("pos_steps", [])
    neg_steps = sample.get("neg_steps", [])
    corrected_step2 = sample.get("corrected_step2", "")
    judge_correct = sample.get("judge_step2_correct")
    prefill_change = sample.get("prefill_correctness_change", "")
    source_em = sample.get("source_exact_match")
    result_em = sample.get("results", {}).get("exact_match")

    gold = extract_gold_answer(doc)
    pos_answer = extract_final_answer(pos_response)
    neg_answer = extract_final_answer(neg_response)

    # If doc has no answer field, derive gold from pos_response when result_em==1
    if gold is None and result_em is not None and float(result_em) >= 0.5:
        gold = pos_answer

    issues = []

    # Check 1: Does pos_response reach correct answer?
    if result_em is not None:
        pos_correct = float(result_em) >= 0.5
        if not pos_correct:
            issues.append(f"pos_response fails eval (result_em={result_em})")
    else:
        pos_correct = answers_match(pos_answer, gold)
        if not pos_correct:
            issues.append(f"pos_response answer ({pos_answer}) != gold ({gold})")

    # Check 2: Does neg_response give wrong answer?
    if gold is not None:
        neg_correct = answers_match(neg_answer, gold)
        if neg_correct:
            issues.append(f"neg_response is actually CORRECT ({neg_answer} == {gold})")
    else:
        neg_correct = False

    # Check 3: corrected_step2 present and non-empty
    if not corrected_step2 or len(corrected_step2.strip()) < 10:
        issues.append("corrected_step2 is missing or too short")

    # Check 4: Step count sanity
    if len(pos_steps) == 0:
        issues.append("pos_steps is empty")
    if len(neg_steps) == 0:
        issues.append("neg_steps is empty")
    if pos_steps and neg_steps and abs(len(pos_steps) - len(neg_steps)) > 5:
        issues.append(f"Step count mismatch: pos={len(pos_steps)}, neg={len(neg_steps)}")

    # Check 5: judge says step2 is correct but sample is in training set
    if judge_correct is True:
        issues.append("judge says step2 is CORRECT — why is it in corrections?")

    # Check 6: Prefill correctness consistency
    if prefill_change and prefill_change not in ("wrong_to_correct",
                                                   "correct_to_correct"):
        if "wrong" in prefill_change and "correct" not in prefill_change:
            issues.append(f"Prefill change is unexpected: {prefill_change}")

    # Check 7: source_em should be 0.0 (original was wrong)
    if source_em is not None and float(source_em) > 0.5:
        issues.append(f"source_exact_match = {source_em} "
                      "(original was already correct?)")

    # Check 9: prefill response exists
    prefill = sample.get("pos_response_prefill", "")
    if not prefill:
        issues.append("pos_response_prefill is missing")

    return {
        "idx": idx,
        "qid": qid,
        "question": question[:120],
        "gold_answer": gold,
        "pos_answer": pos_answer,
        "neg_answer": neg_answer,
        "pos_correct": pos_correct,
        "neg_correct": neg_correct,
        "n_pos_steps": len(pos_steps),
        "n_neg_steps": len(neg_steps),
        "corrected_step2_len": len(corrected_step2),
        "judge_step2_correct": judge_correct,
        "prefill_change": prefill_change,
        "issues": issues,
        "n_issues": len(issues),
    }


def main():
    data_path = find_file(TRAINING_DATA_PATHS)
    if data_path is None:
        print("ERROR: Training data file not found.")
        return

    print(f"Loading training data: {data_path}")
    data = json.loads(data_path.read_text())
    samples = data.get("samples", data) if isinstance(data, dict) else data
    print(f"  {len(samples)} samples loaded")

    audit_results = []
    for i, sample in enumerate(samples):
        result = audit_sample(sample, i)
        audit_results.append(result)

    flagged = [r for r in audit_results if r["n_issues"] > 0]
    clean = [r for r in audit_results if r["n_issues"] == 0]

    # Statistics
    total = len(audit_results)
    n_pos_wrong = sum(1 for r in audit_results if not r["pos_correct"])
    n_neg_right = sum(1 for r in audit_results if r["neg_correct"])
    n_no_step2 = sum(1 for r in audit_results
                     if r["corrected_step2_len"] < 10)

    print(f"\n{'=' * 60}")
    print(f"AUDIT RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples:          {total}")
    print(f"Clean (no issues):      {len(clean)} ({100*len(clean)/total:.1f}%)")
    print(f"Flagged (has issues):   {len(flagged)} ({100*len(flagged)/total:.1f}%)")
    print(f"  pos_response wrong:   {n_pos_wrong}")
    print(f"  neg_response correct: {n_neg_right}")
    print(f"  Missing step2 corr:   {n_no_step2}")

    # Generate report
    lines = [
        "# Training Data Quality Audit",
        "",
        f"**Dataset**: `{data_path.name}`",
        f"**Total samples**: {total}",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        f"| Metric | Count | Percentage |",
        f"|--------|-------|------------|",
        f"| Clean (no issues) | {len(clean)} | {100*len(clean)/total:.1f}% |",
        f"| Flagged (has issues) | {len(flagged)} | {100*len(flagged)/total:.1f}% |",
        f"| pos_response gives wrong answer | {n_pos_wrong} | {100*n_pos_wrong/total:.1f}% |",
        f"| neg_response gives correct answer | {n_neg_right} | {100*n_neg_right/total:.1f}% |",
        f"| Missing/short corrected_step2 | {n_no_step2} | {100*n_no_step2/total:.1f}% |",
        "",
    ]

    # Issue breakdown
    issue_counts: Dict[str, int] = {}
    for r in audit_results:
        for issue in r["issues"]:
            key = issue.split("(")[0].strip()
            issue_counts[key] = issue_counts.get(key, 0) + 1

    if issue_counts:
        lines.extend([
            "## Issue Type Breakdown",
            "",
            "| Issue Type | Count |",
            "|------------|-------|",
        ])
        for issue_type, count in sorted(issue_counts.items(),
                                         key=lambda x: -x[1]):
            lines.append(f"| {issue_type} | {count} |")
        lines.append("")

    # Flagged samples detail
    if flagged:
        lines.extend([
            "## Flagged Samples (Require Manual Review)",
            "",
            "| Idx | QID | Question (truncated) | Issues |",
            "|-----|-----|---------------------|--------|",
        ])
        for r in sorted(flagged, key=lambda x: -x["n_issues"]):
            q_trunc = r["question"][:80].replace("|", "/")
            issues_str = "; ".join(r["issues"]).replace("|", "/")
            lines.append(f"| {r['idx']} | {r['qid']} | {q_trunc} | {issues_str} |")
        lines.append("")

    # Sample quality statistics
    lines.extend([
        "## Per-Sample Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean pos_steps | {sum(r['n_pos_steps'] for r in audit_results)/total:.1f} |",
        f"| Mean neg_steps | {sum(r['n_neg_steps'] for r in audit_results)/total:.1f} |",
        f"| Mean corrected_step2 length | "
        f"{sum(r['corrected_step2_len'] for r in audit_results)/total:.0f} chars |",
    ])

    prefill_changes: Dict[str, int] = {}
    for r in audit_results:
        pc = r.get("prefill_change", "unknown") or "unknown"
        prefill_changes[pc] = prefill_changes.get(pc, 0) + 1

    lines.extend(["", "## Prefill Correctness Changes", "",
                   "| Change Type | Count |",
                   "|-------------|-------|"])
    for change, count in sorted(prefill_changes.items(), key=lambda x: -x[1]):
        lines.append(f"| {change} | {count} |")

    lines.extend(["", "---",
                   f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", ""])

    report_path = DOCUMENTS_DIR / "data_quality_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
