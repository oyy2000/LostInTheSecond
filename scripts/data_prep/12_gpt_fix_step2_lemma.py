#!/usr/bin/env python3
"""
Use GPT to correct Step 2 for LEMMA-sampled questions.

Input:  artifacts_real/lemma_llama3_generations.json
        (output from 11_generate_with_llama3.py)

Output: Two files:
  - artifacts_real/lemma_ds2_fix_step2_gpt.json         (corrected step2)
  - artifacts_real/lemma_ds2_wait_recompute_gpt.json     (wait + recompute)

Both are compatible with 13_prefill_llama3.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm.auto import tqdm


STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)
REPAIR_PREFIX = "Wait, the previous step is wrong. Let's recompute."


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


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


def build_judge_and_fix_prompt(question: str, step1: str, step2: str) -> str:
    return f"""You are reviewing a math solution step.

Task: Judge whether Step 2 is mathematically correct. If it is wrong, provide the corrected Step 2.

Output format (strict, follow EXACTLY):
- If Step 2 is correct, output ONLY this single line:
CORRECT
- If Step 2 is incorrect, output exactly two lines:
INCORRECT
<corrected Step 2 text here>

Hard constraints:
1) First line MUST be exactly CORRECT or INCORRECT.
2) If INCORRECT, the second line is the corrected Step 2 ONLY (no Step 1, no explanations, no markdown).
3) Keep style close to original Step 2.
4) No extra lines, notes, or fences.

Question:
{question}

Step 1:
{step1}

Step 2:
{step2}
"""


def parse_judge_and_fix(text: str):
    """Returns (judged_correct: bool, corrected_step2: str or None)."""
    text = (text or "").strip()
    if not text:
        return None, None
    lines = text.split("\n", 1)
    first = lines[0].strip().upper()
    if first == "CORRECT" or ("CORRECT" in first and "INCORRECT" not in first):
        return True, None
    if "INCORRECT" in first:
        corrected = lines[1].strip() if len(lines) > 1 else None
        return False, corrected
    return None, None


def call_gpt(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    retries: int = 4,
    sleep_base: float = 1.5,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:
            last_err = e
        time.sleep(sleep_base * (2 ** attempt))
    raise RuntimeError(f"OpenAI API failed after retries: {last_err}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=".env")
    ap.add_argument(
        "--in-file",
        default="./artifacts_real/lemma_llama3_generations.json",
    )
    ap.add_argument(
        "--out-fix",
        default="./artifacts_real/lemma_ds2_fix_step2_gpt.json",
    )
    ap.add_argument(
        "--out-wait",
        default="./artifacts_real/lemma_ds2_wait_recompute_gpt.json",
    )
    ap.add_argument(
        "--audit-json",
        default="./artifacts_real/lemma_gpt_fix_step2_audit.json",
    )
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--only-incorrect", action="store_true", default=True,
                     help="Only process samples where the model got the answer wrong")
    ap.add_argument("--no-only-incorrect", dest="only_incorrect", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep-base", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    return ap.parse_args()


def main():
    args = parse_args()
    load_env_file(Path(args.env_file))

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env or environment", file=sys.stderr)
        sys.exit(2)

    in_path = Path(args.in_file)
    obj = json.loads(in_path.read_text("utf-8"))
    all_samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj

    items = []
    for s in all_samples:
        if args.only_incorrect and s.get("source_exact_match", 0) >= 1.0:
            continue
        if not s.get("pred_answer"):
            continue
        if len(s.get("neg_steps", [])) < 2:
            continue
        items.append(s)

    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise ValueError("No valid samples to process.")

    print(f"Processing {len(items)} samples (out of {len(all_samples)} total)")

    client = OpenAI()
    fix_samples: List[Dict[str, Any]] = []
    wait_samples: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    n_rewritten, n_skipped, n_failed = 0, 0, 0
    n_judged_correct, n_judged_incorrect = 0, 0

    pbar = tqdm(items, desc="GPT Judge+Fix Step2", unit="sample")
    for item in pbar:
        doc = item.get("doc", {})
        question = doc.get("question", "")
        doc_id = doc.get("id", -1)
        neg_response = item.get("neg_response", "")
        neg_steps = item.get("neg_steps", [])
        gt_answer = item.get("gt_answer", "")

        step1 = neg_steps[0] if len(neg_steps) >= 1 else ""
        step2 = neg_steps[1] if len(neg_steps) >= 2 else ""
        tail = "\n\n".join(neg_steps[2:]) if len(neg_steps) > 2 else ""

        if not step1 or not step2:
            n_skipped += 1
            continue

        try:
            raw_response = call_gpt(
                client, args.model,
                build_judge_and_fix_prompt(question, step1, step2),
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                retries=args.retries, sleep_base=args.sleep_base,
            )
        except Exception as e:
            n_failed += 1
            pbar.write(f"[WARN] doc_id={doc_id} judge+fix failed: {e}")
            continue

        judged_correct, corrected_step2 = parse_judge_and_fix(raw_response)

        if judged_correct is True:
            n_judged_correct += 1
            n_skipped += 1
            continue

        n_judged_incorrect += 1

        if not corrected_step2 or not corrected_step2.strip():
            n_failed += 1
            pbar.write(f"[WARN] doc_id={doc_id} INCORRECT but no corrected text returned")
            continue

        corrected_step2 = corrected_step2.strip()
        n_rewritten += 1

        fix_pos_parts = [p for p in (step1, corrected_step2, tail) if p.strip()]
        fix_pos_response = "\n\n".join(fix_pos_parts)

        fix_samples.append({
            "doc": {"question": question, "id": doc_id},
            "pos_response": fix_pos_response,
            "neg_response": neg_response,
            "pos_steps": split_steps(fix_pos_response),
            "neg_steps": neg_steps,
            "corrected_step2": corrected_step2,
            "judge_step2_correct": judged_correct,
            "gt_answer": gt_answer,
            "results": {"exact_match": 1.0},
        })

        wait_pos_parts = [step1, step2, REPAIR_PREFIX, corrected_step2, tail]
        wait_pos_response = "\n\n".join(p for p in wait_pos_parts if p.strip())

        wait_samples.append({
            "doc": {"question": question, "id": doc_id},
            "pos_response": wait_pos_response,
            "neg_response": neg_response,
            "pos_steps": split_steps(wait_pos_response),
            "neg_steps": neg_steps,
            "corrected_step2": corrected_step2,
            "old_step2": step2,
            "judge_step2_correct": judged_correct,
            "gt_answer": gt_answer,
            "results": {"exact_match": 1.0},
        })

        audit_rows.append({
            "doc_id": doc_id,
            "question": question[:200],
            "step1": step1[:200],
            "step2_orig": step2[:200],
            "step2_corrected": corrected_step2[:200],
            "judged_correct": judged_correct,
            "raw_response": raw_response[:300],
        })

        pbar.set_postfix(
            rewritten=n_rewritten, skipped=n_skipped, failed=n_failed,
            j_ok=n_judged_correct, j_bad=n_judged_incorrect,
        )

    for out_path, samples, label in [
        (Path(args.out_fix), fix_samples, "fix_step2"),
        (Path(args.out_wait), wait_samples, "wait_recompute"),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_obj = {
            "samples": samples,
            "meta": {
                "variant": label,
                "model": args.model,
                "total_input": len(all_samples),
                "n_processed": len(items),
                "n_rewritten": n_rewritten,
                "n_skipped": n_skipped,
                "n_failed": n_failed,
            },
        }
        out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), "utf-8")
        print(f"[{label}] {len(samples)} samples -> {out_path}")

    audit_path = Path(args.audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_rows, indent=2, ensure_ascii=False), "utf-8")

    print(f"\nStats: rewritten={n_rewritten}, skipped={n_skipped}, failed={n_failed}")
    print(f"Judge: correct={n_judged_correct}, incorrect={n_judged_incorrect}")


if __name__ == "__main__":
    main()
