#!/usr/bin/env python3
"""
Experiment 4 (part 1): Generate self-eval signals at step boundaries.

For each wrong trajectory, at each step boundary, prompts the model with
"Is the reasoning above correct so far?" and extracts P(Yes)/P(No).

Multi-GPU via subprocess sharding.

Usage:
    python scripts/24_1_self_eval_signal.py --gpus 0,1,2,3
    python scripts/24_1_self_eval_signal.py --model-id Qwen/Qwen2.5-3B-Instruct --dataset gsm8k
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH_PER_GPU = 4000


def _load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def _write_jsonl(p, recs):
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _ms(mid):
    return mid.split("/")[-1].lower().replace("-", "_")


def _batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--cache-file", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    return ap.parse_args()


def run_shard(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    import math
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} self-eval tasks")

    llm = LLM(
        model=args.model_id, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=10)

    BATCH = args.batch_size
    out_path = Path(args._out)
    yes_tokens = {"Yes", "yes", "YES", " Yes", " yes"}
    no_tokens = {"No", "no", "NO", " No", " no"}

    with out_path.open("w", encoding="utf-8") as fout:
        for bi in range(0, len(tasks), BATCH):
            batch = tasks[bi:bi + BATCH]
            prompts = [t["prompt"] for t in batch]
            outputs = llm.generate(prompts, sp)

            for task, output in zip(batch, outputs):
                p_yes, p_no = 0.0, 0.0
                if output.outputs and output.outputs[0].logprobs:
                    lp_dict = output.outputs[0].logprobs[0]
                    if lp_dict:
                        for tid, lp_obj in lp_dict.items():
                            decoded = (getattr(lp_obj, "decoded_token", "")
                                       or "").strip()
                            lp = getattr(lp_obj, "logprob", -100.0)
                            prob = math.exp(lp) if lp > -50 else 0.0
                            if decoded in yes_tokens:
                                p_yes += prob
                            elif decoded in no_tokens:
                                p_no += prob

                total = p_yes + p_no
                score = p_yes / total if total > 1e-10 else 0.5
                rec = {
                    "doc_id": task["doc_id"],
                    "sample_idx": task["sample_idx"],
                    "step_idx": task["step_idx"],
                    "p_yes": round(p_yes, 6),
                    "p_no": round(p_no, 6),
                    "self_eval_score": round(score, 6),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[Shard {args._sid}] batch {bi // BATCH + 1}/"
                  f"{(len(tasks) + BATCH - 1) // BATCH}")
    print(f"[Shard {args._sid}] done -> {out_path}")


def _launch(script, args, tasks, sd, gpus):
    if not tasks:
        return []
    ns = len(gpus)
    shards = [[] for _ in range(ns)]
    for i, t in enumerate(tasks):
        shards[i % ns].append(t)
    procs, outs = [], []
    for si, gid in enumerate(gpus):
        if not shards[si]:
            continue
        tf = sd / f"t_se_{si}.json"
        tf.write_text(json.dumps(shards[si], ensure_ascii=False), encoding="utf-8")
        of = sd / f"o_se_{si}.jsonl"
        outs.append(of)
        lf = sd / f"log_se_{si}.txt"
        lh = open(lf, "w")
        cmd = [sys.executable, script,
               "--model-id", args.model_id, "--dataset", args.dataset,
               "--gpu-mem", str(args.gpu_mem),
               "--max-model-len", str(args.max_model_len),
               "--batch-size", str(args.batch_size),
               "--_sid", str(si), "--_gpu", gid,
               "--_tf", str(tf), "--_out", str(of)]
        p = subprocess.Popen(cmd, stdout=lh, stderr=subprocess.STDOUT)
        procs.append((si, gid, p, lh, lf))
    for si, gid, p, lh, lf in procs:
        p.wait()
        lh.close()
        if p.returncode != 0:
            print(f"  WARNING: shard {si} failed")
        lf.unlink(missing_ok=True)
    recs = []
    for sf in outs:
        if sf.exists():
            recs.extend(_load_jsonl(sf))
            sf.unlink(missing_ok=True)
    for si in range(ns):
        (sd / f"t_se_{si}.json").unlink(missing_ok=True)
    return recs


def main():
    args = parse_args()
    if args._sid >= 0:
        run_shard(args)
        return

    from src.self_eval_signal import build_self_eval_prompts
    from src.keystep_utils import split_steps

    ms = _ms(args.model_id)
    if not args.cache_file:
        args.cache_file = str(
            ROOT / f"results/{args.dataset}_3b_multi_sample/first_error"
            / "gpt_first_error_cache.jsonl")
    if not args.out_dir:
        args.out_dir = str(
            ROOT / "results" / f"{args.dataset}_{ms}_self_eval")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = out_dir / "_shards"
    sd.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    script = str(Path(__file__).resolve())

    out_file = out_dir / "self_eval_scores.jsonl"
    if out_file.exists():
        print(f"Self-eval scores already exist: {out_file}")
        return

    cache = _load_jsonl(Path(args.cache_file))
    items = [r for r in cache if r.get("tau") is not None]
    if args.max_samples > 0:
        items = items[:args.max_samples]
    print(f"Loaded {len(items)} wrong trajectories")

    tasks = []
    for item in items:
        steps = item.get("steps", split_steps(item.get("response", "")))
        if not steps:
            continue
        prompts = build_self_eval_prompts(
            args.model_id, args.dataset, item["question"], steps)
        for si, prompt in enumerate(prompts):
            tasks.append({
                "doc_id": item["doc_id"],
                "sample_idx": item.get("sample_idx", 0),
                "step_idx": si,
                "tau": item["tau"],
                "prompt": prompt,
            })

    print(f"Total self-eval tasks: {len(tasks)}")
    records = []
    for batch in _batched(tasks, BATCH_PER_GPU * len(gpus)):
        new = _launch(script, args, batch, sd, gpus)
        records.extend(new)

    _write_jsonl(out_file, records)
    print(f"Self-eval scores: {len(records)} -> {out_file}")

    try:
        sd.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
