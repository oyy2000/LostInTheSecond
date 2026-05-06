#!/usr/bin/env python3
"""
Sample N CoT trajectories per CommonsenseQA question using vLLM.

Multi-GPU data-parallel: each GPU loads a full model (tp=1) and processes
1/N_GPU of the prompts. Shard outputs are merged at the end.

Usage:
    python scripts/10_0_csqa_sample_multi_cot.py \
        --n-samples 8 \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.csqa_answer_equiv import extract_choice_letter, is_choice_correct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "Think step by step, then give your final answer as a single letter "
    "(A, B, C, D, or E)."
)


def split_steps(text: str):
    text = (text or "").strip()
    if not text:
        return []
    s = [x.strip() for x in text.split("\n\n") if x.strip()]
    return s if s else [text]


def format_question(item: dict) -> str:
    """Format a CSQA item into a question string with choices."""
    q = item["question"].strip()
    labels = item["choices"]["label"]
    texts = item["choices"]["text"]
    choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"{q}\n{choices}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CommonsenseQA multi-sample CoT (vLLM)")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out-file", default=str(
        PROJECT_ROOT / "results/csqa_3b_multi_sample/raw_cot_n8.jsonl"))
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--limit", type=int, default=0, help="Process first K questions; 0=all")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--min-steps", type=int, default=3,
                    help="Minimum reasoning steps to keep a trajectory")
    # Internal shard args
    ap.add_argument("--_shard-id", type=int, default=-1)
    ap.add_argument("--_n-shards", type=int, default=-1)
    ap.add_argument("--_shard-out", type=str, default="")
    ap.add_argument("--_gpu-id", type=str, default="")
    return ap.parse_args()


def load_csqa_validation():
    """Load CommonsenseQA validation split via HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="validation")
    return list(ds)


def build_chat_prompt(question_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Shard worker ──────────────────────────────────────────────────────────

def run_shard(args):
    from vllm import LLM, SamplingParams

    questions = load_csqa_validation()
    if args.limit > 0:
        questions = questions[:args.limit]

    shard_id = args._shard_id
    n_shards = args._n_shards
    shard_qs = [q for i, q in enumerate(questions) if i % n_shards == shard_id]

    prompts, meta = [], []
    for q in shard_qs:
        q_text = format_question(q)
        prompt = build_chat_prompt(q_text)
        for si in range(args.n_samples):
            prompts.append(prompt)
            meta.append((q["id"], si, q["answerKey"], q_text))

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="half",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    outputs = llm.generate(prompts, sp)

    n_correct = 0
    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, output in enumerate(outputs):
            doc_id, sample_idx, gold, q_text = meta[idx]
            resp = output.outputs[0].text.strip()
            pred = extract_choice_letter(resp)
            correct = is_choice_correct(pred, gold)
            steps = split_steps(resp)
            if len(steps) < args.min_steps:
                continue
            rec = {
                "doc_id": doc_id,
                "sample_idx": sample_idx,
                "question": q_text,
                "gold_answer": gold,
                "response": resp,
                "pred_answer": pred,
                "exact_match": 1.0 if correct else 0.0,
                "n_steps": len(steps),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if correct:
                n_correct += 1

    total = sum(1 for _ in out_path.read_text("utf-8").splitlines() if _.strip())
    rate = n_correct / total if total else 0
    print(f"[Shard {shard_id}] Done. total={total}, correct={n_correct} "
          f"({rate:.3f}), kept={total} -> {out_path}")


# ── Coordinator ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path.parent / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_files = [shard_dir / f"shard_{i}.jsonl" for i in range(n_shards)]

    script_path = str(Path(__file__).resolve())
    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        cmd = [
            sys.executable, script_path,
            "--model-id", args.model_id,
            "--n-samples", str(args.n_samples),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--min-steps", str(args.min_steps),
            "--_shard-id", str(i),
            "--_n-shards", str(n_shards),
            "--_shard-out", str(shard_files[i]),
            "--_gpu-id", gpu_id,
        ]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"

        log_file = shard_dir / f"log_{i}.txt"
        log_fh = log_file.open("w")
        print(f"  Launching shard {i} on GPU {gpu_id}...")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((i, gpu_id, p, log_fh, log_file))

    # Wait for ALL shards in parallel
    remaining = list(range(len(procs)))
    while remaining:
        still_running = []
        for idx in remaining:
            i, gpu_id, p, log_fh, log_file = procs[idx]
            rc = p.poll()
            if rc is None:
                still_running.append(idx)
            else:
                log_fh.close()
                lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
                tail = "\n".join(lines[-5:]) if lines else "(no output)"
                print(f"\n--- Shard {i} (GPU {gpu_id}) exit={rc} ---\n{tail}")
        remaining = still_running
        if remaining:
            time.sleep(10)

    failed = []
    for i, gpu_id, p, log_fh, log_file in procs:
        if p.returncode != 0:
            failed.append(i)
            lines = log_file.read_text("utf-8", errors="replace").strip().splitlines()
            print(f"\n--- Shard {i} FAILED ---")
            print("\n".join(lines[-20:]))

    if failed:
        print(f"\nERROR: Shards {failed} failed!")
        sys.exit(1)

    # Merge
    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for sf in shard_files:
            if sf.exists():
                for line in sf.read_text("utf-8").splitlines():
                    if line.strip():
                        fout.write(line + "\n")
                        n_total += 1

    n_correct = sum(
        1 for line in out_path.read_text("utf-8").splitlines()
        if line.strip() and json.loads(line).get("exact_match", 0.0) >= 1.0
    )
    print(f"\nMerged: {n_total} trajectories (correct={n_correct}, "
          f"wrong={n_total - n_correct})")
    print(f"Output: {out_path}")

    for sf in shard_files:
        sf.unlink(missing_ok=True)
    for f in shard_dir.iterdir():
        f.unlink(missing_ok=True)
    shard_dir.rmdir()
    print("Shard files cleaned up.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
