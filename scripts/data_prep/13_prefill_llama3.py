#!/usr/bin/env python3
"""
Prefill with corrected first 2 steps and let LLaMA-3-8B-Instruct continue.

Input:  artifacts_real/lemma_ds2_fix_step2_gpt.json  OR
        artifacts_real/lemma_ds2_wait_recompute_gpt.json
Output: artifacts_real/lemma_ds2_*_prefill.json

For each sample, keep the first N steps from pos_response, then generate
the rest with the model. Uses vLLM for fast batched inference by default.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)


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


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text


def extract_answer(text: str) -> str:
    m = re.search(r"The answer is:?\s*(.+?)(?:\n|$)", text or "", re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(.+?)(?:\n|$)", text or "")
    if m:
        return m.group(1).strip()
    idx = (text or "").rfind("\\boxed")
    if idx >= 0:
        i = idx
        num_left = 0
        right_idx = None
        while i < len(text):
            if text[i] == "{":
                num_left += 1
            if text[i] == "}":
                num_left -= 1
                if num_left == 0:
                    right_idx = i
                    break
            i += 1
        if right_idx is not None:
            inner = text[idx:right_idx + 1]
            if inner.startswith("\\boxed{") and inner.endswith("}"):
                return inner[len("\\boxed{"):-1].strip()
    return ""


def build_llama3_prefill_prompt(question: str, prefix_steps: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful math assistant. Solve the problem step by step. "
        "Separate each step with a blank line. At the end, write "
        "\"The answer is: <your_answer>\"."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{prefix_steps}"
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file",
                     default="./artifacts_real/lemma_ds2_wait_recompute_gpt.json")
    ap.add_argument("--out-file",
                     default="./artifacts_real/lemma_ds2_wait_recompute_gpt_prefill.json")
    ap.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--keep-steps", type=int, default=0,
                     help="0 = use all pos_steps as prefix")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-vllm", action="store_true", default=True)
    ap.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    ap.add_argument("--tensor-parallel", type=int, default=1,
                     help="Number of GPUs for tensor parallelism (vLLM only)")
    ap.add_argument("--dtype", default="auto",
                     help="Model dtype: auto, float16, bfloat16")
    return ap.parse_args()


def prepare_samples(all_samples, keep_steps):
    """Extract (index, prefix, gt_answer, question) for samples that have valid steps."""
    prepared = []
    for idx, item in enumerate(all_samples):
        doc = item.get("doc", {})
        question = (doc.get("question") or doc.get("problem") or "").strip()
        if not question:
            continue
        pos_steps = item.get("pos_steps", [])
        if not pos_steps:
            pos_steps = split_steps(item.get("pos_response", ""))
        if not pos_steps:
            continue
        if keep_steps > 0:
            k = min(keep_steps, len(pos_steps))
            prefix = "\n\n".join(pos_steps[:k])
        else:
            prefix = "\n\n".join(pos_steps)
        gt_answer = item.get("gt_answer", "")
        prepared.append((idx, prefix, gt_answer, question))
    return prepared


def run_vllm(args):
    from vllm import LLM, SamplingParams

    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()

    obj = json.loads(in_path.read_text("utf-8"))
    all_samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
    if args.limit > 0:
        all_samples = all_samples[:args.limit]
    print(f"Processing {len(all_samples)} samples (vLLM, tp={args.tensor_parallel})")

    prepared = prepare_samples(all_samples, args.keep_steps)
    print(f"  {len(prepared)} samples have valid steps for prefill")

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=4096,
        seed=0,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    greedy_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0,
    )

    # Track best result per sample: {sample_idx: {best_regen, best_tail, found_correct, attempts}}
    best = {}
    for idx, prefix, gt_answer, question in prepared:
        best[idx] = {
            "prefix": prefix,
            "gt_answer": gt_answer,
            "question": question,
            "best_regen": prefix,
            "best_tail": "",
            "found_correct": False,
            "attempts": 0,
        }

    for attempt in range(1, args.max_attempts + 1):
        # Collect prompts for samples not yet solved
        batch_indices = []
        batch_prompts = []
        for idx, prefix, gt_answer, question in prepared:
            if best[idx]["found_correct"]:
                continue
            prompt = build_llama3_prefill_prompt(question, prefix)
            batch_indices.append(idx)
            batch_prompts.append(prompt)

        if not batch_prompts:
            print(f"  Attempt {attempt}: all solved, stopping early")
            break

        use_params = greedy_params if args.max_attempts == 1 else sampling_params
        print(f"  Attempt {attempt}/{args.max_attempts}: generating {len(batch_prompts)} prompts...")
        outputs = llm.generate(batch_prompts, use_params)

        newly_solved = 0
        for out, sample_idx in zip(outputs, batch_indices):
            tail = out.outputs[0].text.strip()
            prefix = best[sample_idx]["prefix"]
            regenerated = prefix if not tail else (prefix + "\n\n" + tail)
            best[sample_idx]["best_regen"] = regenerated
            best[sample_idx]["best_tail"] = tail
            best[sample_idx]["attempts"] = attempt

            gt = best[sample_idx]["gt_answer"]
            if gt:
                pred = extract_answer(regenerated)
                if pred and normalize_answer(pred) == normalize_answer(gt):
                    best[sample_idx]["found_correct"] = True
                    newly_solved += 1

        total_solved = sum(1 for v in best.values() if v["found_correct"])
        print(f"    -> newly_solved={newly_solved}, total_solved={total_solved}/{len(prepared)}")

    # Build results
    results = []
    changed, solved, total_attempts = 0, 0, 0
    for idx, item in enumerate(all_samples):
        if idx in best:
            b = best[idx]
            new_item = dict(item)
            new_item["pos_response"] = b["best_regen"]
            new_item["pos_steps"] = split_steps(b["best_regen"])
            new_item["prefill_tail"] = b["best_tail"]
            new_item["prefill_attempts"] = b["attempts"]
            new_item["prefill_found_correct"] = b["found_correct"]
            new_item["prefill_model"] = args.model_id
            if b["gt_answer"]:
                new_item["prefill_pred_answer"] = extract_answer(b["best_regen"])
            results.append(new_item)
            changed += 1
            total_attempts += b["attempts"]
            if b["found_correct"]:
                solved += 1
        else:
            results.append(item)

    out_obj = {
        "samples": results,
        "prefill_meta": {
            "model_id": args.model_id,
            "n_changed": changed,
            "n_solved": solved,
            "avg_attempts": (total_attempts / changed) if changed else 0,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nDone. Changed={changed}, Solved={solved}/{changed}")
    print(f"Output: {out_path}")


def run_hf(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()

    obj = json.loads(in_path.read_text("utf-8"))
    all_samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
    if args.limit > 0:
        all_samples = all_samples[:args.limit]
    print(f"Processing {len(all_samples)} samples (HuggingFace)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using dtype: {torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, trust_remote_code=True,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
    ).eval()

    results = []
    changed, solved, total_attempts = 0, 0, 0
    it = tqdm(all_samples, desc="Prefill", unit="s") if tqdm else all_samples

    for item in it:
        doc = item.get("doc", {})
        question = (doc.get("question") or doc.get("problem") or "").strip()
        if not question:
            results.append(item)
            continue
        pos_steps = item.get("pos_steps", [])
        if not pos_steps:
            pos_steps = split_steps(item.get("pos_response", ""))
        if not pos_steps:
            results.append(item)
            continue
        if args.keep_steps > 0:
            k = min(args.keep_steps, len(pos_steps))
            prefix = "\n\n".join(pos_steps[:k])
        else:
            prefix = "\n\n".join(pos_steps)

        gt_answer = item.get("gt_answer", "")
        prompt = build_llama3_prefill_prompt(question, prefix)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        best_regen, best_tail, found_correct, used_attempts = prefix, "", False, 0
        for attempt in range(1, args.max_attempts + 1):
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens, "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": args.temperature, "top_p": args.top_p,
            }
            if args.max_attempts == 1:
                gen_kwargs["do_sample"] = False
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **gen_kwargs)
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            tail = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            regenerated = prefix if not tail else (prefix + "\n\n" + tail)
            used_attempts, best_tail, best_regen = attempt, tail, regenerated
            if gt_answer:
                pred = extract_answer(regenerated)
                if pred and normalize_answer(pred) == normalize_answer(gt_answer):
                    found_correct = True
                    break

        total_attempts += used_attempts
        if found_correct:
            solved += 1
        changed += 1
        new_item = dict(item)
        new_item["pos_response"] = best_regen
        new_item["pos_steps"] = split_steps(best_regen)
        new_item["prefill_tail"] = best_tail
        new_item["prefill_attempts"] = used_attempts
        new_item["prefill_found_correct"] = found_correct
        new_item["prefill_model"] = args.model_id
        if gt_answer:
            new_item["prefill_pred_answer"] = extract_answer(best_regen)
        results.append(new_item)
        if tqdm and hasattr(it, "set_postfix"):
            avg = (total_attempts / changed) if changed else 0
            it.set_postfix(changed=changed, solved=solved, avg_att=f"{avg:.1f}")

    out_obj = {
        "samples": results,
        "prefill_meta": {
            "model_id": args.model_id, "n_changed": changed,
            "n_solved": solved,
            "avg_attempts": (total_attempts / changed) if changed else 0,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nDone. Changed={changed}, Solved={solved}/{changed}")
    print(f"Output: {out_path}")


def main():
    args = parse_args()
    if args.use_vllm:
        run_vllm(args)
    else:
        run_hf(args)


if __name__ == "__main__":
    main()
