#!/usr/bin/env python3
"""
Phase 18: vLLM vs HF full MATH500 accuracy comparison at max_tokens=4096.

Runs two full greedy passes over all 500 MATH500 questions:
  1. vLLM  with max_tokens=4096  (multi-GPU)
  2. HF    with max_new_tokens=4096  (multi-GPU via device_map)

Then reports accuracy for each and compares against the existing
max_tokens=2048 checkpoint.

Usage:
    # vLLM only (faster)
    python scripts/18_1_hf_inference_check.py --backend vllm --gpus 0,1,2,3

    # HF only
    python scripts/18_1_hf_inference_check.py --backend hf --gpus 0,1,2,3

    # Both
    python scripts/18_1_hf_inference_check.py --backend both --gpus 0,1,2,3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt_templates import extract_answer
from src.math_answer_equiv import strip_string

ROOT = Path(__file__).resolve().parent.parent
MATH500_PATH = ROOT / "lm-evaluation-harness/math_eval_data/MATH-500/test.jsonl"

TASK_PROMPT = (
    "Solve the following math problem. Present the final answer "
    "in the format: Final Answer: \\boxed{{your_answer}}.\n"
    "Problem: {question}\n"
    "Answer:"
)


def _last_boxed_only_string(string):
    """Extract last \\boxed{...} from text (matching harness logic)."""
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


def _remove_boxed(s):
    if s is None:
        return None
    if "\\fbox{" in s:
        left = "\\fbox{"
        return s[len(left) : -1] if s.startswith(left) and s.endswith("}") else s
    if "\\boxed " in s:
        return s[len("\\boxed "):]
    left = "\\boxed{"
    return s[len(left) : -1]


def _harness_equiv(pred_text: str, gold_solution: str) -> bool:
    """Match harness process_results: extract boxed from both, compare via strip_string."""
    boxed_pred = _last_boxed_only_string(pred_text)
    boxed_gold = _last_boxed_only_string(gold_solution)
    if boxed_pred is None or boxed_gold is None:
        return False
    p = _remove_boxed(boxed_pred)
    g = _remove_boxed(boxed_gold)
    if p is None or g is None:
        return False
    try:
        return strip_string(p) == strip_string(g)
    except Exception:
        return p == g


def load_math500():
    items = []
    for i, line in enumerate(MATH500_PATH.read_text("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        items.append({
            "doc_id": f"math500_{i}",
            "question": row["problem"],
            "gold_solution": row["solution"],
        })
    return items


def load_existing_checkpoint(sweep_dir: Path):
    ckpt = sweep_dir / "checkpoint.jsonl"
    if not ckpt.exists():
        return {}
    records = [json.loads(l) for l in ckpt.read_text("utf-8").splitlines() if l.strip()]
    return {
        r["doc_id"]: r
        for r in records
        if r.get("task_type") == "draft" and r.get("draft_idx") == 0
    }


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def _build_chat_prompt(tokenizer, question: str, date_string: str = None) -> str:
    """Use tokenizer.apply_chat_template to match lm-eval harness behavior.
    
    Harness with --apply_chat_template and num_fewshot=0 sends only a single
    user message (no system message) through the tokenizer's chat template.
    Pass date_string to pin the "Today Date" in the Llama 3 system header.
    """
    user_msg = TASK_PROMPT.format(question=question)
    messages = [
        {"role": "user", "content": user_msg},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if date_string:
        kwargs["date_string"] = date_string
    return tokenizer.apply_chat_template(messages, **kwargs)


def run_vllm(questions: list, model_id: str, max_tokens: int, gpus: str,
             date_string: str = None) -> list:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    n_gpus = len(gpus.split(","))
    print(f"[vLLM] Loading {model_id} on {n_gpus} GPU(s): {gpus}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=n_gpus,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=0.90,
        max_model_len=max_tokens + 512,
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["Problem:"],
    )
    prompts = [_build_chat_prompt(tokenizer, q["question"], date_string) for q in questions]
    prompt_token_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompts
    ]
    t0 = time.time()
    outputs = llm.generate(prompt_token_ids=prompt_token_ids,
                           sampling_params=sp)
    elapsed = time.time() - t0
    print(f"[vLLM] Done in {elapsed:.1f}s")

    results = []
    for q, prompt, out in zip(questions, prompts, outputs):
        text = out.outputs[0].text.strip()
        n_tok = len(out.outputs[0].token_ids)
        ok = _harness_equiv(text, q["gold_solution"])
        pred = extract_answer("math500", text)
        results.append({
            "doc_id": q["doc_id"],
            "prompt": prompt,
            "raw_output": text,
            "pred": pred,
            "correct": ok,
            "n_tokens": n_tok,
            "truncated": n_tok >= max_tokens,
        })
    return results


# ---------------------------------------------------------------------------
# HF backend (batched, multi-GPU via device_map="auto")
# ---------------------------------------------------------------------------

def run_hf(questions: list, model_id: str, max_tokens: int, gpus: str,
           batch_size: int = 8, date_string: str = None) -> list:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[HF] Loading {model_id} on GPU(s): {gpus}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompts = [_build_chat_prompt(tok, q["question"], date_string) for q in questions]
    all_texts, all_ntok = [], []

    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for j, seq in enumerate(out):
            input_len = enc["input_ids"].shape[1]
            new_toks = seq[input_len:]
            text = tok.decode(new_toks, skip_special_tokens=True)
            stop_idx = text.find("Problem:")
            if stop_idx >= 0:
                text = text[:stop_idx]
            all_texts.append(text)
            all_ntok.append(len(new_toks))
        print(f"  [{min(i+batch_size, len(prompts))}/{len(prompts)}]", flush=True)

    elapsed = time.time() - t0
    print(f"[HF] Done in {elapsed:.1f}s")

    results = []
    for q, prompt, text, n_tok in zip(questions, prompts, all_texts, all_ntok):
        ok = _harness_equiv(text, q["gold_solution"])
        pred = extract_answer("math500", text)
        results.append({
            "doc_id": q["doc_id"],
            "prompt": prompt,
            "raw_output": text,
            "pred": pred,
            "correct": ok,
            "n_tokens": n_tok,
            "truncated": n_tok >= max_tokens,
        })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(label: str, results: list, n_total: int = 500):
    correct = sum(r["correct"] for r in results)
    truncated = sum(r["truncated"] for r in results)
    acc = correct / n_total
    print(f"  {label}: {correct}/{n_total} = {acc:.4f}  (truncated: {truncated})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct") # Qwen/Qwen2.5-3B-Instruct
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--backend", choices=["vllm", "hf", "both"], default="both")
    ap.add_argument("--hf-batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate first N questions (for quick testing)")
    ap.add_argument("--date-string", default=None,
                    help="Pin 'Today Date' in Llama 3 system header (e.g. '26 Apr 2026')")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
    sweep_dir = Path(args.out_dir) if args.out_dir else \
        ROOT / "results" / f"math500_{model_short}_sweep"

    questions = load_math500()
    if args.limit:
        questions = questions[:args.limit]
    n = len(questions)
    print(f"Loaded {n} MATH500 questions")
    print(f"Model:      {args.model_id}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"GPUs:       {args.gpus}")

    # Load existing 2048-token checkpoint for baseline
    existing = load_existing_checkpoint(sweep_dir)
    baseline_correct = sum(
        _harness_equiv(existing[q["doc_id"]].get("draft_text", ""), q["gold_solution"])
        for q in questions if q["doc_id"] in existing
    )
    baseline_trunc = sum(
        1 for q in questions
        if q["doc_id"] in existing and existing[q["doc_id"]].get("draft_tokens", 0) >= 2048
    )

    print(f"\n{'='*60}")
    print("ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"  Existing vLLM (max=2048): {baseline_correct}/{n} = {baseline_correct/n:.4f}"
          f"  (truncated: {baseline_trunc})")

    out_path = ROOT / "results" / f"math500_backend_comparison_{model_short}.json"
    summary = {
        "model": args.model_id,
        "max_tokens": args.max_tokens,
        "baseline_vllm_2048": {"correct": baseline_correct, "truncated": baseline_trunc,
                                "acc": round(baseline_correct / n, 4)},
    }

    if args.backend in ("vllm", "both"):
        vllm_results = run_vllm(questions, args.model_id, args.max_tokens, args.gpus,
                               date_string=args.date_string)
        acc = report(f"vLLM (max={args.max_tokens})", vllm_results, n)
        summary["vllm_4096"] = {
            "correct": sum(r["correct"] for r in vllm_results),
            "truncated": sum(r["truncated"] for r in vllm_results),
            "acc": round(acc, 4),
        }
        # Save per-question outputs
        vllm_out = ROOT / "results" / f"math500_vllm_{args.max_tokens}_{model_short}.jsonl"
        vllm_out.write_text(
            "\n".join(json.dumps(r) for r in vllm_results), encoding="utf-8"
        )
        print(f"  vLLM outputs -> {vllm_out}")

    if args.backend in ("hf", "both"):
        hf_results = run_hf(questions, args.model_id, args.max_tokens, args.gpus,
                            batch_size=args.hf_batch_size,
                            date_string=args.date_string)
        acc = report(f"HF   (max={args.max_tokens})", hf_results, n)
        summary["hf_4096"] = {
            "correct": sum(r["correct"] for r in hf_results),
            "truncated": sum(r["truncated"] for r in hf_results),
            "acc": round(acc, 4),
        }
        hf_out = ROOT / "results" / f"math500_hf_{args.max_tokens}_{model_short}.jsonl"
        hf_out.write_text(
            "\n".join(json.dumps(r) for r in hf_results), encoding="utf-8"
        )
        print(f"  HF outputs -> {hf_out}")

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    main()
