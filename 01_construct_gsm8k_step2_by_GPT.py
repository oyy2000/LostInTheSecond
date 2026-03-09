#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rewrite incorrect Step 2 for GSM8K lm-eval samples with GPT.

Input:
- lm-eval jsonl rows (fields like: doc_id, doc.problem, filtered_resps/resps, exact_match)

Output (compatible with 05_prefill_pos_tail.py):
{
  "samples": [
    {
      "doc": {"question": "...", "id": 123},
      "pos_response": "...",       # step2-fixed response
      "neg_response": "...",       # original response
      "pos_steps": [...],
      "neg_steps": [...],
      "results": {"exact_match": 1.0}
    }
  ]
}
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from tqdm.auto import tqdm


STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)


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


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_steps(text: str, mode: str = "auto") -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if mode in ("double_newline", "auto") and "\n\n" in text:
        parts = [x.strip() for x in text.split("\n\n") if x.strip()]
        if parts:
            return parts

    if mode in ("single_newline", "auto") and "\n" in text:
        parts = [x.strip() for x in text.split("\n") if x.strip()]
        if parts:
            return parts

    hits = list(STEP_RE.finditer(text))
    if hits:
        out: List[str] = []
        for i, m in enumerate(hits):
            st = m.start()
            ed = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            out.append(text[st:ed].strip())
        out = [x for x in out if x]
        if out:
            return out

    parts = [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+", text) if x.strip()]
    return parts if parts else [text]


def build_fix_step2_prompt(question: str, step1: str, step2: str) -> str:
    return f"""You are editing a math solution.

Task: Correct only Step 2.

Hard constraints:
1) Keep Step 1 unchanged (context only, do not rewrite it).
2) Output ONLY corrected Step 2 text.
3) Do NOT output Step 1.
4) Do NOT output explanations, notes, or markdown fences.
5) Keep style close to original Step 2.

Question:
{question}

Step 1:
{step1}

Step 2 (wrong):
{step2}

Corrected Step 2:
"""


def build_judge_step2_prompt(question: str, step1: str, step2: str) -> str:
    return f"""You are checking a math solution step.

Task: Determine whether Step 2 is mathematically correct given Question and Step 1 context.

Output format (strict):
First line must be exactly one token: CORRECT or INCORRECT
Second line: one short reason.

Question:
{question}

Step 1:
{step1}

Step 2:
{step2}
"""


def parse_judge_label(text: str) -> Optional[bool]:
    s = (text or "").strip().upper()
    if not s:
        return None
    first = s.splitlines()[0].strip()
    if first == "CORRECT":
        return True
    if first == "INCORRECT":
        return False
    if "INCORRECT" in first:
        return False
    if "CORRECT" in first:
        return True
    return None


def call_gpt(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    sleep_base: float,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            out = (resp.output_text or "").strip()
            if out:
                return out
        except AttributeError:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
        time.sleep(sleep_base * (2 ** attempt))
    raise RuntimeError(f"OpenAI API failed after retries: {last_err}")


def pick_exact_match(rec: Dict[str, Any]) -> float:
    if "recomputed_exact_match" in rec:
        try:
            return float(rec["recomputed_exact_match"])
        except Exception:
            return 0.0

    if "exact_match" in rec:
        try:
            return float(rec["exact_match"])
        except Exception:
            return 0.0
    for key in ("results", "metrics", "scores"):
        block = rec.get(key)
        if isinstance(block, dict) and "exact_match" in block:
            try:
                return float(block["exact_match"])
            except Exception:
                return 0.0
    return 0.0


def to_item(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    doc = rec.get("doc", {}) or {}
    question = (doc.get("problem") or doc.get("question") or "").strip()
    if not question:
        return None

    try:
        doc_id = int(rec.get("doc_id", doc.get("id", -1)))
    except Exception:
        doc_id = -1

    gen = ""
    fr = rec.get("filtered_resps", [])
    if isinstance(fr, list) and fr:
        gen = (fr[0] or "").strip()
    if not gen:
        rs = rec.get("resps", [])
        if rs and isinstance(rs[0], list) and rs[0]:
            gen = (rs[0][0] or "").strip()
    if not gen:
        return None

    steps = split_steps(gen, mode="auto")
    step1 = steps[0] if len(steps) >= 1 else ""
    step2 = steps[1] if len(steps) >= 2 else ""
    tail = "\n\n".join(steps[2:]).strip() if len(steps) > 2 else ""

    return {
        "doc_id": doc_id,
        "question": question,
        "neg_response": gen,
        "step1": step1,
        "step2": step2,
        "tail": tail,
        "source_exact_match": pick_exact_match(rec),
    }


def build_output_sample(
    question: str,
    doc_id: int,
    neg_response: str,
    step1: str,
    corrected_step2: str,
    tail: str,
    judge_step2_correct: Optional[bool],
    judge_raw: str,
    source_exact_match: float,
) -> Dict[str, Any]:
    pos_parts = [p for p in (step1, corrected_step2, tail) if p and p.strip()]
    pos_response = "\n\n".join(pos_parts).strip() if pos_parts else neg_response

    return {
        "doc": {"question": question, "id": int(doc_id)},
        "pos_response": pos_response,
        "neg_response": neg_response,
        "pos_steps": split_steps(pos_response, mode="double_newline"),
        "neg_steps": split_steps(neg_response, mode="double_newline"),
        "corrected_step2": corrected_step2,
        "judge_step2_correct": judge_step2_correct,
        "judge_raw": judge_raw,
        "source_exact_match": source_exact_match,
        "results": {"exact_match": 1.0},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fix GSM8K Step-2 errors with GPT")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--in-file", required=True, help="Input lm-eval jsonl path")
    ap.add_argument("--out-file", required=True, help="Output JSON file for 05_prefill_pos_tail.py")
    ap.add_argument("--audit-json", default="./artifacts/gsm8k_fix_step2_audit.json")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--judge-first", action="store_true")
    ap.add_argument("--force-rewrite", action="store_true")
    ap.add_argument("--only-incorrect", action="store_true", help="Only process rows where exact_match < 1.0")
    ap.add_argument("--min-doc-id", type=int, default=0)
    ap.add_argument("--max-doc-id", type=int, default=-1, help="-1 means no upper bound")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep-base", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=2, help="0 means no limit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    audit_path = Path(args.audit_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    raw_records = list(read_jsonl(in_path))
    items: List[Dict[str, Any]] = []
    for rec in raw_records:
        item = to_item(rec)
        if item is None:
            continue
        if item["doc_id"] < args.min_doc_id:
            continue
        if args.max_doc_id >= 0 and item["doc_id"] > args.max_doc_id:
            continue
        if args.only_incorrect and item["source_exact_match"] >= 1.0:
            continue
        items.append(item)

    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise ValueError("No valid records to process.")

    client = OpenAI()
    out_samples: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    n_ok, n_failed, n_skipped = 0, 0, 0
    n_judged_correct, n_judged_incorrect, n_judge_failed = 0, 0, 0

    pbar = tqdm(items, desc="Fix Step2", unit="sample", dynamic_ncols=True)
    for item in pbar:
        q = item["question"]
        doc_id = int(item["doc_id"])
        neg = item["neg_response"]
        step1 = item.get("step1", "")
        step2 = item.get("step2", "")
        tail = item.get("tail", "")
        source_exact = float(item.get("source_exact_match", 0.0))

        judged_correct: Optional[bool] = None
        judge_raw = ""

        if not step1 or not step2:
            corrected_step2 = step2
            ok = False
            n_skipped += 1
            judge_raw = "SKIPPED: <2 steps"
            pbar.write(f"[WARN] doc_id={doc_id} has <2 steps, skip")
        else:
            if args.judge_first:
                try:
                    judge_raw = call_gpt(
                        client=client,
                        model=args.model,
                        prompt=build_judge_step2_prompt(q, step1, step2),
                        temperature=0.0,
                        max_output_tokens=64,
                        retries=args.retries,
                        sleep_base=args.sleep_base,
                    )
                    judged_correct = parse_judge_label(judge_raw)
                    if judged_correct is True:
                        n_judged_correct += 1
                    elif judged_correct is False:
                        n_judged_incorrect += 1
                    else:
                        n_judge_failed += 1
                except Exception as e:
                    judged_correct = None
                    judge_raw = f"JUDGE_ERROR: {e}"
                    n_judge_failed += 1

            should_rewrite = args.force_rewrite or (judged_correct is not True)

            if should_rewrite:
                try:
                    corrected_step2 = call_gpt(
                        client=client,
                        model=args.model,
                        prompt=build_fix_step2_prompt(q, step1, step2),
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        retries=args.retries,
                        sleep_base=args.sleep_base,
                    ).strip()
                    corrected_step2 = corrected_step2.split("\n\n")[0].strip() or step2
                    ok = True
                    n_ok += 1
                except Exception as e:
                    corrected_step2 = step2
                    ok = False
                    n_failed += 1
                    pbar.write(f"[WARN] doc_id={doc_id} rewrite failed: {e}")
            else:
                corrected_step2 = step2
                ok = True
                n_skipped += 1

        pbar.set_postfix(
            ok=n_ok,
            failed=n_failed,
            skipped=n_skipped,
            j_ok=n_judged_correct,
            j_bad=n_judged_incorrect,
            j_fail=n_judge_failed,
        )

        out_sample = build_output_sample(
            question=q,
            doc_id=doc_id,
            neg_response=neg,
            step1=step1,
            corrected_step2=corrected_step2,
            tail=tail,
            judge_step2_correct=judged_correct,
            judge_raw=judge_raw,
            source_exact_match=source_exact,
        )
        out_samples.append(out_sample)

        audit_rows.append(
            {
                "doc_id": doc_id,
                "ok": ok,
                "source_exact_match": source_exact,
                "question": q,
                "step1": step1,
                "step2": step2,
                "corrected_step2": corrected_step2,
                "judge_step2_correct": judged_correct,
                "judge_raw": judge_raw,
                "model": args.model,
            }
        )

    out_obj = {
        "samples": out_samples,
        "meta": {
            "input": str(in_path),
            "model": args.model,
            "count": len(out_samples),
            "only_incorrect": args.only_incorrect,
            "min_doc_id": args.min_doc_id,
            "max_doc_id": args.max_doc_id,
            "limit": args.limit,
        },
    }
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] samples={len(out_samples)}")
    print(f"[Done] rewrite stats: ok={n_ok}, failed={n_failed}, skipped={n_skipped}")
    print(
        f"[Done] judge stats: correct={n_judged_correct}, incorrect={n_judged_incorrect}, "
        f"failed={n_judge_failed}"
    )
    print(f"[Done] output -> {out_path}")
    print(f"[Done] audit  -> {audit_path}")


if __name__ == "__main__":
    main()
