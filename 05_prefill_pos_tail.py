#!/usr/bin/env python3
"""
Regenerate pos_response tails by pre-filling with corrected early steps.

Use case:
- Existing `pos_response` has corrected Step 2, but downstream tail is still copied.
- This script keeps first N steps from pos_response, then asks model to continue.

Input format (JSON/JSONL):
{
  "samples": [
    {
      "doc": {"question": "...", "id": 123},
      "pos_response": "...",
      "neg_response": "...",
      "pos_steps": [...],
      "results": {"exact_match": 1.0}
    }
  ]
}
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prefill early corrected steps and regenerate tail")
    ap.add_argument(
        "--in-file",
        default="./artifacts/vectors_16_ds2_fix_step2_incorrect_only_id300+_step2_marker/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_openai_train_step2fix_fix_step2/Qwen2.5-3B-Instruct_L1_BASELINE/samples_gsm8k_train_ds2_fix_step2_CHANGED_ONLY.json",
        help="Input dataset json/jsonl",
    )
    ap.add_argument(
        "--out-file",
        default="./artifacts/vectors_16_ds2_fix_step2_incorrect_only_id300+_step2_marker/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_openai_train_step2fix_fix_step2/Qwen2.5-3B-Instruct_L1_BASELINE/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json",
        help="Output dataset json/jsonl",
    )
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument(
        "--step2-source-file",
        default="./artifacts/vectors_16_ds2_fix_step2_incorrect_only_id300+_step2_marker/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_openai_train_step2fix_fix_step2/Qwen2.5-3B-Instruct_L1_BASELINE/samples_gsm8k_train_ds2_fix_step2_CHANGED_ONLY.json",
        help="Dataset file used to fetch corrected Step 2 by doc.id",
    )
    ap.add_argument("--dtype", default="float16", choices=["float16"])
    ap.add_argument(
        "--ref-file",
        default="./math_eval_data/MATH-500/test.jsonl",
        help="Reference jsonl file with fields: problem, answer (used for correctness check)",
    )

    ap.add_argument(
        "--keep-steps",
        type=int,
        default=2,
        help="Keep first N steps from pos_response as prefill prefix",
    )
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--do-sample", action="store_true", default=False)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=5,
        help="Maximum number of sampled generations per sample until a correct answer is found",
    )

    ap.add_argument("--limit", type=int, default=0, help="Process first K samples; 0 means all")
    ap.add_argument("--require-exact-match", action="store_true", default=True)
    ap.add_argument("--no-require-exact-match", dest="require_exact_match", action="store_false")
    ap.add_argument(
        "--overwrite-pos-response",
        action="store_true",
        default=True,
        help="Overwrite pos_response with regenerated version",
    )
    ap.add_argument("--no-overwrite-pos-response", dest="overwrite_pos_response", action="store_false")

    return ap.parse_args()


def pick_exact_match(item: Dict[str, Any]) -> float:
    if "exact_match" in item:
        try:
            return float(item["exact_match"])
        except Exception:
            return 0.0

    for key in ("results", "metrics", "scores"):
        block = item.get(key)
        if isinstance(block, dict) and "exact_match" in block:
            try:
                return float(block["exact_match"])
            except Exception:
                return 0.0
    return 0.0


def load_input(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, {}, True

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        return obj["samples"], obj, False
    if isinstance(obj, list):
        return obj, {}, False
    raise ValueError("Unsupported input format. Expect JSON list, JSONL, or {samples:[...]}")


def save_output(path: Path, samples: List[Dict[str, Any]], root_obj: Dict[str, Any], is_jsonl: bool, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_jsonl:
        with path.open("w", encoding="utf-8") as f:
            for row in samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if root_obj:
        out_obj = dict(root_obj)
        out_obj["samples"] = samples
    else:
        out_obj = {"samples": samples}

    out_obj["prefill_meta"] = meta
    path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")


def split_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    steps = []
    if "\n\n" in text:
        steps = [x.strip() for x in text.split("\n\n") if x.strip()]
        if len(steps) >= 2:
            return steps

    hits = list(STEP_RE.finditer(text))
    if hits:
        out = []
        for i, m in enumerate(hits):
            st = m.start()
            ed = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            out.append(text[st:ed].strip())
        out = [x for x in out if x]
        if out:
            return out

    return [x.strip() for x in text.split("\n") if x.strip()]


def normalize_question(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_answer(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("$", "")
    text = re.sub(r"\\boxed\{(.*)\}", r"\1", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def load_reference_answers(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = normalize_question(row.get("problem", ""))
                a = str(row.get("answer", "")).strip()
                if q and a:
                    out[q] = a
        return out

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            q = normalize_question(row.get("problem", ""))
            a = str(row.get("answer", "")).strip()
            if q and a:
                out[q] = a
    return out


def last_boxed_only_string(text: str) -> str:
    text = text or ""
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return ""
    return text[idx : right_brace_idx + 1]


def remove_boxed(text: str) -> str:
    text = text or ""
    if "\\boxed " in text:
        return text[len("\\boxed ") :]
    if text.startswith("\\boxed{") and text.endswith("}"):
        return text[len("\\boxed{") : -1]
    return text


def extract_final_answer(text: str) -> str:
    boxed = last_boxed_only_string(text)
    if boxed:
        return remove_boxed(boxed).strip()

    m = re.search(r"Final Answer\s*:\s*(.*)", text or "", flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def qwen_chat_prompt(
    question: str,
    system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> str:
    return (
        "<|im_start|>system\n"
        f"{system}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Solve the following math problem. Present the final answer in the format: Final Answer: \\boxed{your_answer}.\n"
        f"Prolbem: {question}\n"
        "Answer:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_corrected_step2_map(source_samples: List[Dict[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for item in source_samples:
        doc = item.get("doc") or {}
        try:
            did = int(doc.get("id", -1))
        except Exception:
            continue
        if did < 0:
            continue

        if isinstance(item.get("pos_steps"), list) and len(item["pos_steps"]) >= 2:
            step2 = str(item["pos_steps"][1]).strip()
        else:
            pos_steps = split_steps(item.get("pos_response", ""))
            step2 = pos_steps[1].strip() if len(pos_steps) >= 2 else ""

        if step2:
            out[did] = step2
    return out


def get_step1_from_item(item: Dict[str, Any]) -> str:
    if isinstance(item.get("neg_steps"), list) and item["neg_steps"]:
        s = str(item["neg_steps"][0]).strip()
        if s:
            return s

    neg_steps = split_steps(item.get("neg_response", ""))
    if neg_steps:
        return neg_steps[0].strip()

    if isinstance(item.get("pos_steps"), list) and item["pos_steps"]:
        s = str(item["pos_steps"][0]).strip()
        if s:
            return s

    pos_steps = split_steps(item.get("pos_response", ""))
    return pos_steps[0].strip() if pos_steps else ""


def get_prefix_from_item(item: Dict[str, Any], keep_steps: int, corrected_step2_by_doc: Dict[int, str]) -> str:
    try:
        did = int((item.get("doc") or {}).get("id", -1))
    except Exception:
        did = -1

    step1 = get_step1_from_item(item)
    corrected_step2 = corrected_step2_by_doc.get(did, "").strip()

    if step1 and corrected_step2:
        return f"{step1}\n\n{corrected_step2}".strip()

    # fallback to old keep_steps behavior
    if isinstance(item.get("pos_steps"), list) and item["pos_steps"]:
        steps = [str(x).strip() for x in item["pos_steps"] if str(x).strip()]
    else:
        steps = split_steps(item.get("pos_response", ""))

    if not steps:
        return ""

    k = max(1, min(keep_steps, len(steps)))
    return "\n\n".join(steps[:k]).strip()


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()
    step2_source_path = Path(args.step2_source_file).resolve()
    ref_path = Path(args.ref_file).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if not step2_source_path.exists():
        raise FileNotFoundError(f"step2 source file not found: {step2_source_path}")

    samples, root_obj, is_jsonl = load_input(in_path)
    source_samples, _, _ = load_input(step2_source_path)
    corrected_step2_by_doc = build_corrected_step2_map(source_samples)
    ref_answers = load_reference_answers(ref_path)

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
    ).eval()

    processed = []
    changed = 0
    visited = 0
    solved_by_sampling = 0
    total_attempts = 0
    total_to_visit = min(len(samples), args.limit) if args.limit > 0 else len(samples)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_to_visit, desc="prefill", unit="sample")

    for i, item in enumerate(samples):
        if args.limit > 0 and visited >= args.limit:
            processed.extend(samples[i:])
            break

        visited += 1
        if pbar is not None:
            pbar.update(1)

        if args.require_exact_match and pick_exact_match(item) < 1.0:
            processed.append(item)
            continue

        doc = item.get("doc") or {}
        question = (doc.get("question") or doc.get("problem") or "").strip()
        if not question:
            processed.append(item)
            continue

        gt_answer = ref_answers.get(normalize_question(question), "")

        prefix = get_prefix_from_item(item, keep_steps=args.keep_steps, corrected_step2_by_doc=corrected_step2_by_doc)
        if not prefix:
            processed.append(item)
            continue

        prompt = qwen_chat_prompt(question)
        if not prompt:
            processed.append(item)
            continue

        prefill_text = prompt + prefix
        model_inputs = tokenizer(prefill_text, return_tensors="pt", add_special_tokens=False)
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        max_attempts = max(1, int(args.max_attempts_per_sample))
        best_regenerated = prefix
        best_tail = ""
        found_correct = False
        used_attempts = 0
        doc_id = doc.get("id", "NA")

        for attempt in range(1, max_attempts + 1):
            if pbar is not None:
                avg_attempts = (total_attempts / changed) if changed > 0 else 0.0
                pbar.set_postfix(
                    sample=f"{visited}/{total_to_visit}",
                    attempt=f"{attempt}/{max_attempts}",
                    doc_id=doc_id,
                    changed=changed,
                    solved=solved_by_sampling,
                    avg_attempts=f"{avg_attempts:.2f}",
                )

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }

            if max_attempts == 1 and not args.do_sample:
                gen_kwargs["do_sample"] = False
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)

            with torch.inference_mode():
                outputs = model.generate(**model_inputs, **gen_kwargs)

            new_ids = outputs[0][model_inputs["input_ids"].shape[1] :]
            tail = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            regenerated = prefix if not tail else (prefix + "\n\n" + tail)

            used_attempts = attempt
            best_tail = tail
            best_regenerated = regenerated

            if gt_answer:
                pred_answer = extract_final_answer(regenerated)
                if pred_answer and normalize_answer(pred_answer) == normalize_answer(gt_answer):
                    found_correct = True
                    break

        regenerated = best_regenerated
        total_attempts += used_attempts
        if found_correct:
            solved_by_sampling += 1

        new_item = dict(item)
        new_item["pos_response_original"] = item.get("pos_response", "")
        new_item["pos_response_prefill"] = regenerated
        new_item["prefill_keep_steps"] = args.keep_steps
        new_item["prefill_model"] = args.model_id
        new_item["prefill_attempts"] = used_attempts
        new_item["prefill_found_correct"] = found_correct
        if gt_answer:
            new_item["prefill_gt_answer"] = gt_answer
            new_item["prefill_pred_answer"] = extract_final_answer(regenerated)
        if best_tail:
            new_item["prefill_tail"] = best_tail

        if args.overwrite_pos_response:
            new_item["pos_response"] = regenerated

        # update pos_steps for convenience
        new_item["pos_steps"] = split_steps(regenerated)

        processed.append(new_item)
        changed += 1

        if visited % 10 == 0:
            print(f"[prefill] processed={visited}, changed={changed}")

        if pbar is not None:
            avg_attempts = (total_attempts / changed) if changed > 0 else 0.0
            pbar.set_postfix(changed=changed, solved=solved_by_sampling, avg_attempts=f"{avg_attempts:.2f}")

    if pbar is not None:
        pbar.close()

    meta = {
        "input_file": str(in_path),
        "step2_source_file": str(step2_source_path),
        "ref_file": str(ref_path),
        "step2_source_size": len(source_samples),
        "step2_map_size": len(corrected_step2_by_doc),
        "ref_answer_size": len(ref_answers),
        "model_id": args.model_id,
        "keep_steps": args.keep_steps,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_attempts_per_sample": args.max_attempts_per_sample,
        "limit": args.limit,
        "require_exact_match": args.require_exact_match,
        "overwrite_pos_response": args.overwrite_pos_response,
        "num_total": len(samples),
        "num_visited": visited,
        "num_changed": changed,
        "num_solved_by_sampling": solved_by_sampling,
        "avg_attempts": (total_attempts / changed) if changed > 0 else 0.0,
    }

    save_output(out_path, processed, root_obj, is_jsonl, meta)
    print("\nDone prefill regeneration.")
    print(f"Input : {in_path}")
    print(f"Output: {out_path}")
    print(f"Changed samples: {changed}/{visited}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
