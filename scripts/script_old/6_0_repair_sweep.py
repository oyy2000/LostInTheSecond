"""
Entropy-triggered repair sweep.

Phases
------
1. Generate greedy drafts for all questions via vLLM.
2. Compute per-step logprob metrics via vLLM prompt_logprobs.
3. Generate repair suffixes for all possible repair points.
4. Generate SC baseline samples.

Then evaluate: greedy, SC@N, random-repair, lookback, lookahead, symmetric.

Usage
-----
    python scripts/6_0_repair_sweep.py --model Qwen/Qwen2.5-3B-Instruct \\
        --dataset gsm8k --gpus 0,1
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.prompt_templates import (
    build_prompt, check_answer, extract_answer, get_stop_tokens, split_steps,
)
from src.sweep_datasets import load_dataset_by_name
from src.repair_engine import (
    compute_step_metrics_from_logprobs, decide_repairs, METRIC_DIRECTION,
)

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/common/users/sl2148/anaconda3/envs/vllmdebug/bin/python"
BATCH_PER_GPU = 6000


# -- I/O helpers -----------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def _append_jsonl(path: Path, records: List[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fmt(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# -- shard launcher --------------------------------------------------------

def _launch(script_path, ns, tasks, shard_dir, gpu_ids):
    if not tasks:
        return []
    n_gpus = len(gpu_ids)
    shards = [[] for _ in range(n_gpus)]
    for i, t in enumerate(tasks):
        shards[i % n_gpus].append(t)

    procs, outs = [], []
    for si, gid in enumerate(gpu_ids):
        if not shards[si]:
            continue
        tf = shard_dir / f"t_{si}.json"
        tf.write_text(json.dumps(shards[si], ensure_ascii=False), "utf-8")
        of = shard_dir / f"o_{si}.jsonl"
        outs.append(of)
        log_f = shard_dir / f"log_{si}.txt"
        log_fh = open(log_f, "w")
        cmd = [
            PYTHON, script_path,
            "--model", ns.model, "--dataset", ns.dataset,
            "--temperature", str(ns.temperature),
            "--top-p", str(ns.top_p),
            "--max-tokens", str(ns.max_tokens),
            "--gpu-memory-utilization", str(ns.gpu_mem),
            "--max-model-len", str(ns.max_model_len),
            "--_sid", str(si), "--_gpu", gid,
            "--_tf", str(tf), "--_out", str(of),
        ]
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gid, p, log_fh, log_f))

    failed = []
    for si, gid, p, log_fh, log_f in procs:
        p.wait()
        log_fh.close()
        rc = p.returncode
        txt = log_f.read_text("utf-8") if log_f.exists() else ""
        tail = "\n".join(txt.strip().splitlines()[-5:]) if txt.strip() else "(no output)"
        print(f"  Shard {si} (GPU {gid}) exit={rc}\n{tail}")
        if rc != 0:
            failed.append(si)
        log_f.unlink(missing_ok=True)

    if failed:
        print(f"WARNING: shards {failed} failed")

    recs = []
    for sf in outs:
        if sf.exists():
            recs.extend(_load_jsonl(sf))
            sf.unlink(missing_ok=True)
    for si in range(n_gpus):
        (shard_dir / f"t_{si}.json").unlink(missing_ok=True)
    return recs


# -- shard worker (one per GPU, invoked via --_sid) ------------------------

def run_shard(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} tasks")

    stop = get_stop_tokens(args.model)
    llm = LLM(
        model=args.model, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
    )
    sp_greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=stop)
    sp_sample = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop=stop,
    )
    phase_map = {}
    for i, t in enumerate(tasks):
        phase_map.setdefault(t["phase"], []).append((i, t))

    results = [None] * len(tasks)

    # --- draft phase: greedy generation ---
    if "draft" in phase_map:
        idxs = [i for i, _ in phase_map["draft"]]
        prompts = [tasks[i]["prompt"] for i in idxs]
        outs = llm.generate(prompts, sp_greedy)
        for j, idx in enumerate(idxs):
            text = outs[j].outputs[0].text.strip()
            steps = split_steps(text)
            pred = extract_answer(args.dataset, text)
            toks = len(outs[j].outputs[0].token_ids)
            results[idx] = {
                "phase": "draft", "doc_id": tasks[idx]["doc_id"],
                "draft_text": text, "draft_steps": steps,
                "draft_answer": pred, "draft_tokens": toks,
            }
    # [SHARD_LOGPROB]

# [TASK_BUILDERS]

# [EVALUATION]

# [MAIN]
