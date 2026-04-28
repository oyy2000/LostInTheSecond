#!/usr/bin/env python3
"""
GPT fix for Step 3 errors ONLY.

Uses the already-known step 2 results: only processes samples where step 2
was previously judged correct (the ~3878 skipped samples from the step 2 run).
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


def build_judge_and_fix_step3_prompt(question: str, step1: str, step2: str, step3: str) -> str:
    return f"""You are reviewing a math solution step.

Task: Judge whether Step 3 is mathematically correct given the previous steps and question. If it is wrong, provide the corrected Step 3.

Output format (strict, follow EXACTLY):
- If Step 3 is correct, output ONLY this single line:
CORRECT
- If Step 3 is incorrect, output exactly two lines:
INCORRECT
<corrected Step 3 text here>

Hard constraints:
1) First line MUST be exactly CORRECT or INCORRECT.
2) If INCORRECT, the second line is the corrected Step 3 ONLY (no other steps, no explanations, no markdown).
3) Keep style close to original Step 3.
4) No extra lines, notes, or fences.

Question:
{question}

Step 1:
{step1}

Step 2:
{step2}

Step 3:
{step3}
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
    ap.add_argument("--step2-fix-file", default="artifacts_real/full/lemma_ds2_fix_step2_gpt.json",
                    help="Previous step2 fix output (to identify already-fixed doc_ids)")
    ap.add_argument("--out-fix", default="artifacts_real/full/lemma_ds2_fix_step3_gpt.json")
    ap.add_argument("--out-wait", default="artifacts_real/full/lemma_ds2_wait_step3_gpt.json")
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

    s2_obj = json.loads(Path(args.step2_fix_file).read_text("utf-8"))
    s2_samples = s2_obj["samples"] if isinstance(s2_obj, dict) and "samples" in s2_obj else s2_obj
    s2_fixed_ids = {s["doc"]["id"] for s in s2_samples}
    print(f"Step 2 already fixed: {len(s2_fixed_ids)} doc_ids")

    items = []
    for s in all_samples:
        if s.get("source_exact_match", 0) >= 1.0:
            continue
        if not s.get("pred_answer"):
            continue
        steps = s.get("neg_steps", [])
        if len(steps) < 3:
            continue
        doc_id = s.get("doc", {}).get("id", -1)
        if doc_id in s2_fixed_ids:
            continue
        items.append(s)

    if args.limit > 0:
        items = items[:args.limit]

    print(f"Step 3 candidates: {len(items)} (step2-correct, ≥3 steps, incorrect answer)")

    client = OpenAI()
    fix_samples: List[Dict[str, Any]] = []
    wait_samples: List[Dict[str, Any]] = []
    n_incorrect, n_correct, n_failed = 0, 0, 0

    pbar = tqdm(items, desc="GPT Fix Step3")
    for item in pbar:
        doc = item["doc"]
        question = doc.get("question", "")
        doc_id = doc.get("id", -1)
        neg_steps = item["neg_steps"]
        neg_response = item.get("neg_response", "")
        gt_answer = item.get("gt_answer", "")

        step1, step2, step3 = neg_steps[0], neg_steps[1], neg_steps[2]
        tail = "\n\n".join(neg_steps[3:]) if len(neg_steps) > 3 else ""

        try:
            raw = call_gpt(client, args.model,
                           build_judge_and_fix_step3_prompt(question, step1, step2, step3),
                           temperature=args.temperature, max_tokens=args.max_output_tokens)
        except Exception as e:
            n_failed += 1
            continue

        judged_correct, corrected = parse_judge_and_fix(raw)

        if judged_correct is True:
            n_correct += 1
        elif judged_correct is False and corrected and corrected.strip():
            n_incorrect += 1
            corrected = corrected.strip()

            fix_parts = [p for p in (step1, step2, corrected, tail) if p.strip()]
            fix_samples.append({
                "doc": {"question": question, "id": doc_id},
                "pos_response": "\n\n".join(fix_parts),
                "neg_response": neg_response,
                "pos_steps": split_steps("\n\n".join(fix_parts)),
                "neg_steps": neg_steps,
                "corrected_step3": corrected,
                "gt_answer": gt_answer,
                "results": {"exact_match": 1.0},
            })

            wait_parts = [step1, step2, step3, REPAIR_PREFIX, corrected, tail]
            wait_samples.append({
                "doc": {"question": question, "id": doc_id},
                "pos_response": "\n\n".join(p for p in wait_parts if p.strip()),
                "neg_response": neg_response,
                "pos_steps": split_steps("\n\n".join(p for p in wait_parts if p.strip())),
                "neg_steps": neg_steps,
                "corrected_step3": corrected,
                "old_step3": step3,
                "gt_answer": gt_answer,
                "results": {"exact_match": 1.0},
            })
        else:
            n_failed += 1

        pbar.set_postfix(fix=n_incorrect, ok=n_correct, fail=n_failed)

    for out_path, samples, label in [
        (Path(args.out_fix), fix_samples, "fix_step3"),
        (Path(args.out_wait), wait_samples, "wait_step3"),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_obj = {"samples": samples, "meta": {
            "variant": label, "model": args.model,
            "n_candidates": len(items),
            "n_step3_incorrect": n_incorrect,
            "n_step3_correct": n_correct,
            "n_failed": n_failed,
        }}
        out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), "utf-8")
        print(f"[{label}] {len(samples)} samples -> {out_path}")

    print(f"\nStep 3 results: incorrect={n_incorrect}, correct={n_correct}, failed={n_failed}")


if __name__ == "__main__":
    main()
