#!/usr/bin/env python3
"""
GPT fix for Step 2 AND Step 3 errors.

For each incorrect sample:
  1. Judge Step 2 → if wrong, fix it (same as before)
  2. If Step 2 is correct and ≥3 steps, judge Step 3 → if wrong, fix it

Produces two output variants per fixed step:
  - fix:  Step1 + corrected_step + remaining_steps
  - wait: Step1 + wrong_step + "Wait..." + corrected_step + remaining_steps
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm.auto import tqdm


REPAIR_PREFIX = "Wait, the previous step is wrong. Let's recompute."


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
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
    step_re = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)
    hits = list(step_re.finditer(text))
    if hits:
        out = []
        for i, m in enumerate(hits):
            st = m.start()
            ed = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            out.append(text[st:ed].strip())
        return [x for x in out if x]
    return [x.strip() for x in text.split("\n") if x.strip()]


def build_judge_and_fix_prompt(question: str, context_steps: List[str],
                                target_step: str, step_num: int) -> str:
    context_text = "\n\n".join(
        f"Step {i+1}:\n{s}" for i, s in enumerate(context_steps)
    )
    return f"""You are reviewing a math solution step.

Task: Judge whether Step {step_num} is mathematically correct given the previous steps and question. If it is wrong, provide the corrected Step {step_num}.

Output format (strict, follow EXACTLY):
- If Step {step_num} is correct, output ONLY this single line:
CORRECT
- If Step {step_num} is incorrect, output exactly two lines:
INCORRECT
<corrected Step {step_num} text here>

Hard constraints:
1) First line MUST be exactly CORRECT or INCORRECT.
2) If INCORRECT, the second line is the corrected Step {step_num} ONLY (no other steps, no explanations, no markdown).
3) Keep style close to original Step {step_num}.
4) No extra lines, notes, or fences.

Question:
{question}

{context_text}

Step {step_num}:
{target_step}
"""


def parse_judge_and_fix(text: str) -> Tuple[Optional[bool], Optional[str]]:
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


def call_gpt(client: OpenAI, model: str, prompt: str,
             temperature: float = 0.0, max_tokens: int = 1024,
             retries: int = 4, sleep_base: float = 1.5) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:
            last_err = e
        time.sleep(sleep_base * (2 ** attempt))
    raise RuntimeError(f"API failed after retries: {last_err}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--in-file", default="artifacts_real/full/lemma_llama3_generations.json")
    ap.add_argument("--out-fix", default="artifacts_real/full/lemma_ds2_fix_step23_gpt.json")
    ap.add_argument("--out-wait", default="artifacts_real/full/lemma_ds2_wait_step23_gpt.json")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    load_env_file(Path(args.env_file))

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    obj = json.loads(Path(args.in_file).read_text("utf-8"))
    all_samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj

    items = []
    for s in all_samples:
        if s.get("source_exact_match", 0) >= 1.0:
            continue
        if not s.get("pred_answer"):
            continue
        if len(s.get("neg_steps", [])) < 2:
            continue
        items.append(s)

    if args.limit > 0:
        items = items[:args.limit]

    print(f"Processing {len(items)} incorrect samples")

    client = OpenAI()
    fix_samples: List[Dict[str, Any]] = []
    wait_samples: List[Dict[str, Any]] = []

    stats = {
        "step2_incorrect": 0, "step2_correct": 0,
        "step3_incorrect": 0, "step3_correct": 0,
        "step3_skipped_short": 0, "failed": 0,
    }

    pbar = tqdm(items, desc="GPT Fix Step2+3")
    for item in pbar:
        doc = item.get("doc", {})
        question = doc.get("question", "")
        doc_id = doc.get("id", -1)
        neg_steps = item.get("neg_steps", [])
        neg_response = item.get("neg_response", "")
        gt_answer = item.get("gt_answer", "")

        step1 = neg_steps[0]
        step2 = neg_steps[1]

        # --- Judge Step 2 ---
        try:
            raw = call_gpt(client, args.model,
                           build_judge_and_fix_prompt(question, [step1], step2, 2),
                           temperature=args.temperature, max_tokens=args.max_output_tokens)
        except Exception as e:
            stats["failed"] += 1
            continue

        s2_correct, s2_fixed = parse_judge_and_fix(raw)

        if s2_correct is False and s2_fixed and s2_fixed.strip():
            stats["step2_incorrect"] += 1
            tail = "\n\n".join(neg_steps[2:]) if len(neg_steps) > 2 else ""

            fix_parts = [p for p in (step1, s2_fixed.strip(), tail) if p.strip()]
            fix_samples.append({
                "doc": {"question": question, "id": doc_id},
                "pos_response": "\n\n".join(fix_parts),
                "neg_response": neg_response,
                "fixed_step": 2,
                "corrected_text": s2_fixed.strip(),
                "gt_answer": gt_answer,
            })

            wait_parts = [step1, step2, REPAIR_PREFIX, s2_fixed.strip(), tail]
            wait_samples.append({
                "doc": {"question": question, "id": doc_id},
                "pos_response": "\n\n".join(p for p in wait_parts if p.strip()),
                "neg_response": neg_response,
                "fixed_step": 2,
                "corrected_text": s2_fixed.strip(),
                "gt_answer": gt_answer,
            })
            pbar.set_postfix(s2=stats["step2_incorrect"], s3=stats["step3_incorrect"],
                             ok2=stats["step2_correct"], ok3=stats["step3_correct"])
            continue

        if s2_correct is True:
            stats["step2_correct"] += 1
        else:
            stats["failed"] += 1
            continue

        # --- Judge Step 3 (only if Step 2 was correct and ≥3 steps) ---
        if len(neg_steps) < 3:
            stats["step3_skipped_short"] += 1
            continue

        step3 = neg_steps[2]
        try:
            raw = call_gpt(client, args.model,
                           build_judge_and_fix_prompt(question, [step1, step2], step3, 3),
                           temperature=args.temperature, max_tokens=args.max_output_tokens)
        except Exception as e:
            stats["failed"] += 1
            continue

        s3_correct, s3_fixed = parse_judge_and_fix(raw)

        if s3_correct is False and s3_fixed and s3_fixed.strip():
            stats["step3_incorrect"] += 1
            tail = "\n\n".join(neg_steps[3:]) if len(neg_steps) > 3 else ""

            fix_parts = [p for p in (step1, step2, s3_fixed.strip(), tail) if p.strip()]
            fix_samples.append({
                "doc": {"question": question, "id": doc_id},
                "pos_response": "\n\n".join(fix_parts),
                "neg_response": neg_response,
                "fixed_step": 3,
                "corrected_text": s3_fixed.strip(),
                "gt_answer": gt_answer,
            })

            wait_parts = [step1, step2, step3, REPAIR_PREFIX, s3_fixed.strip(), tail]
            wait_samples.append({
                "doc": {"question": question, "id": doc_id},
                "pos_response": "\n\n".join(p for p in wait_parts if p.strip()),
                "neg_response": neg_response,
                "fixed_step": 3,
                "corrected_text": s3_fixed.strip(),
                "gt_answer": gt_answer,
            })
        elif s3_correct is True:
            stats["step3_correct"] += 1
        else:
            stats["failed"] += 1

        pbar.set_postfix(s2=stats["step2_incorrect"], s3=stats["step3_incorrect"],
                         ok2=stats["step2_correct"], ok3=stats["step3_correct"])

    for out_path, samples, label in [
        (Path(args.out_fix), fix_samples, "fix_step23"),
        (Path(args.out_wait), wait_samples, "wait_step23"),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_obj = {"samples": samples, "meta": {
            "variant": label, "model": args.model,
            "total_input": len(all_samples), "n_processed": len(items),
            **stats,
        }}
        out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), "utf-8")
        print(f"[{label}] {len(samples)} samples -> {out_path}")

    print(f"\nStats: {json.dumps(stats, indent=2)}")
    s2_total = stats['step2_incorrect'] + stats['step2_correct']
    s3_total = stats['step3_incorrect'] + stats['step3_correct']
    print(f"Step 2: {stats['step2_incorrect']}/{s2_total} incorrect")
    print(f"Step 3: {stats['step3_incorrect']}/{s3_total} incorrect")
    print(f"Total fix samples: {len(fix_samples)} (s2: {stats['step2_incorrect']}, s3: {stats['step3_incorrect']})")


if __name__ == "__main__":
    main()
