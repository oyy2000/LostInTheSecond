#!/usr/bin/env python3
"""
Good-prefix injection experiment across model scales.

For each model size, take wrong-answer samples, replace first K steps with
correct reasoning steps, let the model continue generation, then PRM-score
the full chain.

Conditions:
  same_problem  — inject correct steps from a model-correct solution to the
                  SAME problem (matched by doc_id). If no correct solution
                  exists for that problem, inject from a same-subject problem.
  cross_problem — inject correct steps from a DIFFERENT problem (random).
  random_text   — inject generic filler steps (control).
  no_prefix     — baseline: generate from scratch.

Usage:
    # Single model on one GPU (use for parallel launcher)
    python 23_good_prefix_injection.py --models 1.5B --gpus 2

    # Plot only from existing results
    python 23_good_prefix_injection.py --only-plot
"""

import os
import sys

# Set CUDA_VISIBLE_DEVICES before torch import to ensure proper GPU isolation
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpus" and _i + 1 < len(sys.argv) and sys.argv[_i + 1] != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prm.scoring import StepScorer, split_steps

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent

MODEL_REGISTRY = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B":   "Qwen/Qwen2.5-3B-Instruct",
    "7B":   "Qwen/Qwen2.5-7B-Instruct",
    "14B":  "Qwen/Qwen2.5-14B-Instruct",
}

PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
PRM_DTYPE = "float16"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.5B,1.5B,3B,7B")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_baselines"),
                    help="Directory with lm-eval baseline outputs (for loading questions)")
    ap.add_argument("--prm-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_prm"),
                    help="Directory with prm_{tag}.json from 22_multi_scale_baseline.py")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs" / "good_prefix_exp"))
    ap.add_argument("--prefix-steps", default="1,2,3", help="Number of correct steps to inject")
    ap.add_argument("--conditions", default="same_problem,cross_problem,no_prefix",
                    help="Conditions: same_problem, cross_problem, random_text, no_prefix")
    ap.add_argument("--max-samples", type=int, default=200, help="Max wrong samples per model")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--min-free-mem-mb", type=int, default=15000)
    ap.add_argument("--only-score", action="store_true", help="Skip generation, only PRM-score")
    ap.add_argument("--only-plot", action="store_true", help="Only plot from existing results")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def query_gpu_free_mem() -> Dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return {i: int(l.strip()) for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return {}


def select_best_gpu(requested: str, min_free: int) -> int:
    free = query_gpu_free_mem()
    if not free:
        return 0
    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        if usable:
            return max(usable, key=usable.get)
        return ids[0] if ids else 0
    candidates = {g: m for g, m in free.items() if m >= min_free}
    if candidates:
        return max(candidates, key=candidates.get)
    return max(free, key=free.get)


def load_prm_results(prm_dir: Path, model_tag: str) -> Optional[dict]:
    f = prm_dir / f"prm_{model_tag}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _find_samples_jsonl(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.rglob("samples_*.jsonl"))
    return cands[-1] if cands else None


def load_questions_and_correct_responses(
    baseline_dir: Path, model_tag: str
) -> Tuple[Dict[int, str], Dict[int, Dict]]:
    """Load original questions and correct model responses from lm-eval samples.

    Returns:
        questions: {doc_id -> problem_text}
        correct_responses: {doc_id -> {"response": str, "steps": List[str],
                                       "subject": str, "level": int}}
    """
    run_dir = baseline_dir / f"baseline_{model_tag}"
    jsonl_path = _find_samples_jsonl(run_dir)
    if jsonl_path is None:
        print(f"[WARN] No lm-eval samples found in {run_dir}")
        return {}, {}

    questions = {}
    correct_responses = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("filter") == "strict-match":
                continue
            doc_id = d.get("doc_id", -1)
            doc = d.get("doc", {})
            questions[doc_id] = doc.get("problem", "")

            if d.get("exact_match", 0) >= 1:
                cot = ""
                fr = d.get("filtered_resps", [])
                if isinstance(fr, list) and fr:
                    cot = (fr[0] if isinstance(fr[0], str) else (fr[0][0] if fr[0] else "")).strip()
                if not cot:
                    rs = d.get("resps", [])
                    if rs and isinstance(rs[0], list) and rs[0]:
                        cot = (rs[0][0] or "").strip()

                if cot:
                    steps = split_steps(cot, mode="auto")
                    if len(steps) >= 2:
                        correct_responses[doc_id] = {
                            "response": cot,
                            "steps": steps,
                            "subject": doc.get("subject", ""),
                            "level": doc.get("level", 0),
                        }

    print(f"[DATA] {model_tag}: loaded {len(questions)} questions, "
          f"{len(correct_responses)} correct responses with ≥2 steps")
    return questions, correct_responses


def build_chat_prompt(question: str) -> str:
    """Build Qwen chat prompt for prefilling."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_prefill_generation(
    model,
    tokenizer,
    question: str,
    prefix_steps: List[str],
    max_new_tokens: int,
) -> Tuple[str, List[str]]:
    """Generate continuation from a prefix of correct steps."""
    prompt = build_chat_prompt(question)
    prefix_text = "\n\n".join(prefix_steps) if prefix_steps else ""

    if prefix_text:
        full_input = prompt + prefix_text
    else:
        full_input = prompt

    inputs = tokenizer(full_input, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    continuation = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if prefix_text and continuation:
        full_response = prefix_text + "\n\n" + continuation
    elif prefix_text:
        full_response = prefix_text
    else:
        full_response = continuation

    all_steps = split_steps(full_response, mode="auto")
    return full_response, all_steps


def extract_answer(text: str) -> str:
    """Extract boxed answer from model output."""
    idx = (text or "").rfind("\\boxed")
    if idx >= 0:
        i, depth, start = idx, 0, None
        while i < len(text):
            if text[i] == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start + 1:i].strip()
            i += 1
    m = re.search(r"Final Answer\s*:\s*(.*)", text or "", re.IGNORECASE)
    return m.group(1).strip() if m else ""


def normalize_answer(text: str) -> str:
    text = (text or "").strip().replace("$", "").replace(",", "")
    text = re.sub(r"\\boxed\{(.*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def run_experiment_for_model(
    model_tag: str,
    model_id: str,
    prm_result: dict,
    questions: Dict[int, str],
    correct_responses: Dict[int, Dict],
    gpu_id: int,
    args,
) -> Dict[str, Any]:
    """Run all prefix injection conditions for one model."""

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    wrong_samples = [s for s in prm_result["samples"] if s["exact_match"] < 1]

    if not wrong_samples:
        print(f"[SKIP] {model_tag}: no wrong samples")
        return {}

    random.seed(args.seed)
    if len(wrong_samples) > args.max_samples:
        wrong_samples = random.sample(wrong_samples, args.max_samples)

    prefix_step_counts = [int(x) for x in args.prefix_steps.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    print(f"[EXP] {model_tag} on GPU {gpu_id}: {len(wrong_samples)} wrong samples, "
          f"conditions={conditions}, prefix_steps={prefix_step_counts}")

    print(f"[LOAD] {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_map.get(args.dtype, torch.float16),
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    correct_by_subject: Dict[str, List[Dict]] = {}
    all_correct_list: List[Dict] = []
    for doc_id, cr in correct_responses.items():
        all_correct_list.append(cr)
        subj = cr.get("subject", "unknown")
        correct_by_subject.setdefault(subj, []).append(cr)

    all_condition_results = {}

    for condition in conditions:
        print(f"\n--- Condition: {condition} ---")
        condition_samples = []

        for K in prefix_step_counts:
            label = f"{condition}_K{K}" if condition != "no_prefix" else "no_prefix"
            if condition == "no_prefix" and K > 1:
                continue

            for sidx, wrong_s in enumerate(tqdm(wrong_samples, desc=f"{model_tag}/{label}")):
                doc_id = wrong_s.get("doc_id", sidx)
                question_text = questions.get(doc_id, "")
                if not question_text:
                    continue

                if condition == "no_prefix":
                    prefix_steps_list = []
                elif condition == "same_problem":
                    if doc_id in correct_responses:
                        donor_steps = correct_responses[doc_id]["steps"]
                    elif all_correct_list:
                        donor_steps = random.choice(all_correct_list)["steps"]
                    else:
                        continue
                    prefix_steps_list = donor_steps[:K]
                elif condition == "cross_problem":
                    candidates = [cr for did, cr in correct_responses.items()
                                  if did != doc_id]
                    if candidates:
                        prefix_steps_list = random.choice(candidates)["steps"][:K]
                    elif all_correct_list:
                        prefix_steps_list = random.choice(all_correct_list)["steps"][:K]
                    else:
                        continue
                elif condition == "random_text":
                    prefix_steps_list = [
                        f"Step {i+1}: Let me consider this problem carefully."
                        for i in range(K)
                    ]
                else:
                    continue

                try:
                    full_resp, all_steps = run_prefill_generation(
                        model, tokenizer, question_text,
                        prefix_steps_list, args.max_new_tokens,
                    )
                except Exception as e:
                    print(f"[WARN] generation failed for {doc_id}: {e}")
                    continue

                condition_samples.append({
                    "doc_id": doc_id,
                    "question": question_text,
                    "condition": condition,
                    "K": K,
                    "prefix_steps": prefix_steps_list,
                    "full_response": full_resp,
                    "all_steps": all_steps,
                    "n_steps": len(all_steps),
                    "original_step_scores": wrong_s.get("step_scores", []),
                })

            if condition == "no_prefix":
                break

        all_condition_results[condition] = condition_samples
        print(f"  Generated {len(condition_samples)} samples for {condition}")

    del model
    torch.cuda.empty_cache()

    return all_condition_results


def prm_score_condition_results(
    all_condition_results: Dict[str, list],
    model_tag: str,
    gpu_id: int,
) -> Dict[str, list]:
    """PRM-score all generated results using actual question text."""

    print(f"[PRM] Loading PRM model for {model_tag}...")
    scorer = StepScorer(PRM_MODEL, PRM_DTYPE)

    scored_results = {}
    for condition, samples in all_condition_results.items():
        scored = []
        for s in tqdm(samples, desc=f"PRM {model_tag}/{condition}"):
            steps = s["all_steps"]
            if not steps:
                continue
            try:
                query = s.get("question", "")
                if not query:
                    continue
                scores = scorer.score_steps(query, steps)
            except Exception as e:
                continue

            if len(scores) == len(steps):
                s["injected_step_scores"] = scores
                scored.append(s)

        scored_results[condition] = scored
        print(f"  PRM scored {len(scored)}/{len(samples)} for {condition}")

    return scored_results


def _avg_curve(samples: list, score_key: str = "injected_step_scores") -> Tuple[List[int], List[float]]:
    if not samples:
        return [], []
    max_steps = max((len(s.get(score_key, [])) for s in samples), default=0)
    xs, ys = [], []
    for i in range(max_steps):
        vals = [s[score_key][i] for s in samples if i < len(s.get(score_key, []))]
        if vals:
            xs.append(i + 1)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def plot_injection_results(
    all_model_results: Dict[str, Dict[str, list]],
    out_dir: Path,
):
    """Generate comparison plots across models and conditions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-model: original vs injected PRM curves
    for model_tag, condition_results in all_model_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Original wrong-answer curve
        no_prefix = condition_results.get("no_prefix", [])
        if no_prefix:
            xo, yo = _avg_curve(no_prefix, "original_step_scores")
            if xo:
                ax.plot(xo, yo, "k--", marker="x", linewidth=2, label="Original (wrong)", alpha=0.7)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        cidx = 0
        for condition, samples in condition_results.items():
            if condition == "no_prefix":
                continue
            # Group by K
            k_groups = {}
            for s in samples:
                k_groups.setdefault(s["K"], []).append(s)

            for K, k_samples in sorted(k_groups.items()):
                xi, yi = _avg_curve(k_samples, "injected_step_scores")
                if xi:
                    ax.plot(xi, yi, marker="o", linewidth=2, color=colors[cidx % 10],
                            label=f"{condition} K={K} (n={len(k_samples)})")
                    cidx += 1

        ax.set_xlabel("Step k", fontsize=12)
        ax.set_ylabel("Avg PRM Step Score", fontsize=12)
        ax.set_title(f"Good-Prefix Injection: {MODEL_REGISTRY.get(model_tag, model_tag)}", fontsize=13)
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"injection_{model_tag}.png", dpi=180)
        plt.close()

    # Cross-scale comparison: injection benefit
    if len(all_model_results) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        model_tags_sorted = sorted(all_model_results.keys(), key=_model_params)

        for condition in ["same_problem", "cross_problem"]:
            improvements = []
            tags_used = []

            for tag in model_tags_sorted:
                cr = all_model_results[tag]
                no_prefix = cr.get("no_prefix", [])
                cond_samples = [s for s in cr.get(condition, []) if s.get("K") == 2]

                if not no_prefix or not cond_samples:
                    continue

                _, yo = _avg_curve(no_prefix, "original_step_scores")
                _, yi = _avg_curve(cond_samples, "injected_step_scores")

                if yo and yi:
                    orig_mean = np.mean(yo[:min(8, len(yo))])
                    inj_mean = np.mean(yi[:min(8, len(yi))])
                    improvements.append(inj_mean - orig_mean)
                    tags_used.append(tag)

            if improvements:
                params = [_model_params(t) for t in tags_used]
                ax.plot(params, improvements, marker="o", linewidth=2, label=f"{condition} (K=2)")
                for p, imp, t in zip(params, improvements, tags_used):
                    ax.annotate(t, (p, imp), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=9)

        ax.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters", fontsize=12)
        ax.set_ylabel("PRM Score Improvement (injected - original)", fontsize=12)
        ax.set_title("Good-Prefix Injection Benefit vs Model Scale", fontsize=13)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "injection_benefit_vs_scale.png", dpi=200)
        plt.close()


def _model_params(tag: str) -> float:
    tag_lower = tag.lower().replace("-base", "")
    for size, val in [("0.5b", 5e8), ("1.5b", 1.5e9), ("3b", 3e9),
                      ("7b", 7e9), ("14b", 14e9)]:
        if size in tag_lower:
            return val
    return 1e9


def main():
    args = parse_args()
    model_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prm_dir = Path(args.prm_dir)
    baseline_dir = Path(args.baseline_dir)

    gpu_id = select_best_gpu(args.gpus, args.min_free_mem_mb)
    print(f"Using GPU {gpu_id}")

    all_model_results: Dict[str, Dict[str, list]] = {}

    if not args.only_plot:
        for tag in model_tags:
            model_id = MODEL_REGISTRY.get(tag)
            if not model_id:
                print(f"[SKIP] Unknown model tag: {tag}")
                continue

            prm_result = load_prm_results(prm_dir, tag)
            if prm_result is None:
                print(f"[SKIP] {tag}: no PRM baseline results. Run 22_multi_scale_baseline.py first.")
                continue

            result_file = out_dir / f"injection_{tag}.json"
            if result_file.exists() and args.only_score:
                print(f"[LOAD] {tag}: loading existing results")
                all_model_results[tag] = json.loads(result_file.read_text(encoding="utf-8"))
                continue

            questions, correct_responses = load_questions_and_correct_responses(
                baseline_dir, tag
            )
            if not questions:
                print(f"[SKIP] {tag}: no questions loaded from baseline samples")
                continue

            if not args.only_score:
                condition_results = run_experiment_for_model(
                    tag, model_id, prm_result, questions, correct_responses,
                    gpu_id, args,
                )
            else:
                condition_results = json.loads(result_file.read_text(encoding="utf-8"))

            if not args.only_score:
                scored = prm_score_condition_results(condition_results, tag, gpu_id)
            else:
                scored = condition_results

            all_model_results[tag] = scored
            result_file.write_text(
                json.dumps(scored, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"[SAVE] {tag} -> {result_file}")
    else:
        for tag in model_tags:
            result_file = out_dir / f"injection_{tag}.json"
            if result_file.exists():
                all_model_results[tag] = json.loads(result_file.read_text(encoding="utf-8"))

    plot_injection_results(all_model_results, out_dir / "plots")
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
