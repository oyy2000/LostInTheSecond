#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use GPT to correct Step 2 errors in model-generated solutions.

Output format matches steering-vector extraction input:
{
  "samples": [
    {
      "doc": {"question": "...", "id": 123},
      "pos_response": "...",   # GPT-corrected response
      "neg_response": "...",   # original model response
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


def load_env_file(env_path: Path):
    """Lightweight .env loader (no extra dependency)."""
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def split_steps(text: str, mode: str = "double_newline") -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if mode == "double_newline":
        parts = [x.strip() for x in text.split("\n\n") if x.strip()]
        if len(parts) >= 2:
            return parts

    if mode == "single_newline":
        parts = [x.strip() for x in text.split("\n") if x.strip()]
        if len(parts) >= 2:
            return parts

    if mode == "auto":
        if "\n\n" in text:
            parts = [x.strip() for x in text.split("\n\n") if x.strip()]
            if parts:
                return parts
        if "\n" in text:
            parts = [x.strip() for x in text.split("\n") if x.strip()]
            if parts:
                return parts

    hits = list(STEP_RE.finditer(text))
    if hits:
        out = []
        for i, m in enumerate(hits):
            st = m.start()
            ed = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            out.append(text[st:ed].strip())
        return [x for x in out if x]

    parts = [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+", text) if x.strip()]
    return parts if parts else [text]


def build_fix_step2_prompt(question: str, step1: str, step2: str) -> str:
    return f"""You are editing a math solution.

Task: Correct only Step 2.

Hard constraints:
1) Input is split by double-newline (\n\n) into steps.
2) Keep Step 1 unchanged (for context only, do not rewrite it).
3) Output ONLY the corrected Step 2 text.
4) Do NOT output Step 1.
5) Do NOT output explanations, notes, or markdown fences.
6) Keep style close to original Step 2.

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
    if "CORRECT" in first and "INCORRECT" not in first:
        return True
    if "INCORRECT" in first:
        return False
    return None


def call_gpt_rewrite(
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


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_wrong_ids(path: Optional[Path]) -> Optional[set]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("ids", [])
    return {int(x) for x in ids}


def parse_input_records(in_path: Path) -> List[Dict[str, Any]]:
    if in_path.suffix.lower() == ".jsonl":
        return list(read_jsonl(in_path))

    obj = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        return obj["samples"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported input format. Expect .jsonl or JSON with 'samples'.")


def to_item(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Case A: baseline lm-eval jsonl record
    if "resps" in rec or "filtered_resps" in rec:
        doc = rec.get("doc", {}) or {}
        question = (doc.get("problem") or doc.get("question") or "").strip()
        doc_id = int(rec.get("doc_id", -1))

        gen = ""
        fr = rec.get("filtered_resps", [])
        if isinstance(fr, list) and fr:
            gen = (fr[0] or "").strip()
        if not gen:
            rs = rec.get("resps", [])
            if rs and isinstance(rs[0], list) and rs[0]:
                gen = (rs[0][0] or "").strip()

        if not question or not gen:
            return None

        steps = split_steps(gen, mode="auto")
        step1 = steps[0] if len(steps) >= 1 else ""
        step2 = steps[1] if len(steps) >= 2 else ""
        generation_tail = "\n\n".join(steps[2:]).strip() if len(steps) > 2 else ""

        return {
            "doc_id": doc_id,
            "question": question,
            "neg_response": gen,
            "step1": step1,
            "step2": step2,
            "generation_tail": generation_tail,
        }

    # Case B: already pair-like JSON
    doc = rec.get("doc", {}) or {}
    question = (doc.get("question") or doc.get("problem") or "").strip()
    doc_id = int(doc.get("id", rec.get("doc_id", -1)))
    neg_response = (rec.get("neg_response") or "").strip()
    if question and neg_response:
        steps = split_steps(neg_response, mode="auto")
        step1 = steps[0] if len(steps) >= 1 else ""
        step2 = steps[1] if len(steps) >= 2 else ""
        generation_tail = "\n\n".join(steps[2:]).strip() if len(steps) > 2 else ""
        return {
            "doc_id": doc_id,
            "question": question,
            "neg_response": neg_response,
            "step1": step1,
            "step2": step2,
            "generation_tail": generation_tail,
        }
    return None


def build_ds2_correct_step2(question: str, doc_id: int, neg_response: str, step1: str, step2: str, generation_tail: str, corrected_step2: str, judge_step2_correct: Optional[bool], judge_raw: str) -> Dict[str, Any]:
    if step1 and step2:
        pos_parts = [step1, corrected_step2]
        if generation_tail:
            pos_parts.append(generation_tail)
        pos_parts = [p for p in pos_parts if p and p.strip()]
        pos = "\n\n".join(pos_parts).strip()
    else:
        pos = f"{neg_response}\n".strip()

    return {
        "doc": {"question": question, "id": int(doc_id)},
        "corrected_step2": corrected_step2,
        "pos_response": pos,
        "neg_response": neg_response,
        "neg_steps": split_steps(neg_response, mode="double_newline"),
        "pos_steps": split_steps(pos, mode="double_newline"),
        "judge_step2_correct": judge_step2_correct,
        "judge_raw": judge_raw,
        "results": {"exact_match": 1.0},
    }


def build_ds2_wait_recompute(question: str, doc_id: int, neg_response: str, step1: str, step2: str, generation_tail: str, corrected_step2: str, judge_step2_correct: Optional[bool], judge_raw: str) -> Dict[str, Any]:
    repair_prefix = "Wait, the previous step is wrong. Let's recompute."
    if step1 and step2:
        pos_parts = [step1, step2, repair_prefix, corrected_step2]
        if generation_tail:
            pos_parts.append(generation_tail)
        pos_parts = [p for p in pos_parts if p and p.strip()]
        pos = "\n\n".join(pos_parts).strip()
    else:
        pos = f"{neg_response}\n{repair_prefix}".strip()

    return {
        "doc": {"question": question, "id": int(doc_id)},
        "pos_response": pos,
        "neg_response": neg_response,
        "neg_steps": split_steps(neg_response, mode="double_newline"),
        "pos_steps": split_steps(pos, mode="double_newline"),
        "judge_step2_correct": judge_step2_correct,
        "judge_raw": judge_raw,
        "results": {"exact_match": 1.0},
    }


def main():
    ap = argparse.ArgumentParser(description="Fix Step-2 errors with GPT and build steering pairs")
    ap.add_argument("--env-file", default=".env", help="Path to .env file containing OPENAI_API_KEY", )
    ap.add_argument(
        "--in-file",
        default="/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond/artifacts/samples_math500_ds2_wait_recompute.json",
        help="Input .jsonl or .json",
    )
    ap.add_argument("--out-ds2-correct", default="./artifacts/samples_math500_ds2_fix_step2_gpt.json")
    ap.add_argument("--out-ds2-wait", default="./artifacts/samples_math500_ds2_wait_recompute_gpt.json")
    ap.add_argument("--audit-json", default="./artifacts/gpt_fix_step2_audit.json")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--judge-first", action="store_true", help="Use GPT to judge Step2 correctness before rewriting")
    ap.add_argument("--force-rewrite", action="store_true", help="Always rewrite Step2 even if judged correct")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep-base", type=float, default=1.5)
    ap.add_argument("--limit", dest="limit", type=int, default=2, help="Max number of samples to rewrite (0 = no limit)")
    args = ap.parse_args()

    load_env_file(Path(args.env_file))

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    in_path = Path(args.in_file)
    out_ds2_correct_path = Path(args.out_ds2_correct)
    out_ds2_wait_path = Path(args.out_ds2_wait)
    audit_path = Path(args.audit_json)
    out_ds2_correct_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds2_wait_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)


    raw_records = parse_input_records(in_path)
    items: List[Dict[str, Any]] = []
    for rec in raw_records:
        item = to_item(rec)
        if item is None:
            continue
        items.append(item)

    if args.limit > 0:
        items = items[: args.limit]

    if not items:
        raise ValueError("No valid records to process.")

    client = OpenAI()
    pairs_ds2_correct = []
    pairs_ds2_wait = []
    n_ok, n_failed, n_skipped = 0, 0, 0
    n_judged_correct, n_judged_incorrect, n_judge_failed = 0, 0, 0

    audit_rows = []
    pbar = tqdm(items, desc="GPT fix step2", unit="sample", dynamic_ncols=True)
    for item in pbar:
            q = item["question"]
            neg = item["neg_response"]
            doc_id = item["doc_id"]
            step1 = item.get("step1", "")
            step2 = item.get("step2", "")
            generation_tail = item.get("generation_tail", "")

            if not step1 or not step2:
                corrected_step2 = step2
                ok = False
                n_skipped += 1
                judged_correct = None
                judge_raw = "SKIPPED: <2 steps"
                pbar.write(f"[WARN] doc_id={doc_id} has <2 steps, skip GPT correction")
            else:
                judged_correct = None
                judge_raw = ""
                if args.judge_first:
                    try:
                        judge_prompt = build_judge_step2_prompt(q, step1, step2)
                        judge_raw = call_gpt_rewrite(
                            client=client,
                            model=args.model,
                            prompt=judge_prompt,
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
                    prompt = build_fix_step2_prompt(q, step1, step2)
                    try:
                        corrected_step2 = call_gpt_rewrite(
                            client=client,
                            model=args.model,
                            prompt=prompt,
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

            pairs_ds2_correct.append(
                build_ds2_correct_step2(
                    question=q,
                    doc_id=int(doc_id),
                    neg_response=neg,
                    step1=step1,
                    step2=step2,
                    generation_tail=generation_tail,
                    corrected_step2=corrected_step2,
                    judge_step2_correct=judged_correct,
                    judge_raw=judge_raw,
                )
            )
            pairs_ds2_wait.append(
                build_ds2_wait_recompute(
                    question=q,
                    doc_id=int(doc_id),
                    neg_response=neg,
                    step1=step1,
                    step2=step2,
                    generation_tail=generation_tail,
                    corrected_step2=corrected_step2,
                    judge_step2_correct=judged_correct,
                    judge_raw=judge_raw,
                )
            )

            audit = {
                "doc_id": int(doc_id),
                "ok": ok,
                "question": q,
                "neg_response": neg,
                "step1": step1,
                "step2": step2,
                "corrected_step2": corrected_step2,
                "judge_step2_correct": judged_correct,
                "judge_raw": judge_raw,
                "neg_steps": split_steps(neg),
                "model": args.model,
            }
            audit_rows.append(audit)

    out_ds2_correct_path.write_text(
        json.dumps({"samples": pairs_ds2_correct}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_ds2_wait_path.write_text(
        json.dumps({"samples": pairs_ds2_wait}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Done] pairs={len(pairs_ds2_correct)}")
    print(f"[Done] rewrite stats: ok={n_ok}, failed={n_failed}, skipped={n_skipped}")
    print(
        f"[Done] judge stats: correct={n_judged_correct}, incorrect={n_judged_incorrect}, "
        f"failed={n_judge_failed}"
    )
    print(f"[Done] dataset(ds2-correct) -> {out_ds2_correct_path}")
    print(f"[Done] dataset(ds2-wait)    -> {out_ds2_wait_path}")
    print(f"[Done] audit   -> {audit_path}")


if __name__ == "__main__":
    main()
