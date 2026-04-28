#!/usr/bin/env python3
"""
Entropy-triggered rollback sweep for Qwen2.5-3B on GSM8K.

Pipeline:
  Phase 1: Greedy drafts (nd_max per question)
  Phase 2: Logprob scoring on draft-0 via prompt_logprobs
  Phase 2.5: Self-eval scoring on draft-0 steps (P(Yes)/P(No))
  Phase 3: Entropy-triggered suffixes (sweep thresholds x metrics)
  Phase 4: Full SC baseline
  Evaluate: acc vs total tokens (logprob cost included)

Trigger modes:
  - min_logprob (lookback): fire when min_logprob < threshold -> repair prev step
  - entropy_delta (lookback): fire when entropy_delta > threshold -> repair prev step
  - self_eval (PRM): rollback to the first step where PRM score < 0.5
  - alpha-fixed (baseline): rollback at fixed fraction alpha
  - random: fire with probability p at each step

Usage:
    python scripts/6_1_entropy_triggered_sweep.py --gpus 0,1
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
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompt_templates import (
    build_prompt, check_answer, extract_answer,
    get_stop_tokens, split_steps,
)
from src.sweep_datasets import load_dataset_by_name

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
DATASET = "gsm8k"
PYTHON = "/common/users/sl2148/anaconda3/envs/vllmdebug/bin/python"

# -- defaults ---------------------------------------------------------------
ND_MAX = 2
K_MAX = 3
FULLSC_N = 16
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_MODEL_LEN = 4096
GPU_MEM = 0.85
BATCH_PER_GPU = 256

# -- trigger thresholds to sweep -------------------------------------------
MIN_LOGPROB_THRESHOLDS = [-1.0, -1.5, -2.0, -3.0, -4.0, -5.0]
ENTROPY_DELTA_THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
PRM_DROP_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
ALPHA_VALUES = [0.3, 0.5, 0.7]
RANDOM_PROBS = [0.10, 0.20, 0.30]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--nd-max", type=int, default=ND_MAX)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--fullsc-n", type=int, default=FULLSC_N)
    ap.add_argument("--n-sample", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--_shard-id", type=int, default=-1)
    ap.add_argument("--_task-file", default="")
    ap.add_argument("--_gpu", default="0")
    ap.add_argument("--_prm", action="store_true")
    return ap.parse_args()


# ---- shard worker (one per GPU) ------------------------------------------

def run_shard(args):
    """Generate completions / logprobs for a batch of tasks on one GPU."""
    global DATASET
    DATASET = args.dataset
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    sid = args._shard_id
    print(f"[Shard {sid}] GPU {args._gpu}: {len(tasks)} tasks")

    stop = get_stop_tokens(MODEL_ID)
    llm = LLM(
        model=MODEL_ID, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    # SHARD_BODY

    sp_g = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, stop=stop)
    sp_s = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P,
        max_tokens=MAX_TOKENS, stop=stop,
    )
    sp_lp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    by_type = {}
    for i, t in enumerate(tasks):
        by_type.setdefault(t["task_type"], []).append(i)

    results = [None] * len(tasks)

    for ttype, sp in [("draft", sp_g), ("suffix", sp_s), ("fullsc", sp_s)]:
        idxs = by_type.get(ttype, [])
        if not idxs:
            continue
        IBATCH = 512
        for bi in range(0, len(idxs), IBATCH):
            chunk = idxs[bi:bi + IBATCH]
            pids = [tokenizer.encode(tasks[i]["prompt"],
                                     add_special_tokens=False) for i in chunk]
            outs = llm.generate([{"prompt_token_ids": p} for p in pids],
                                sampling_params=sp)
            for idx, o in zip(chunk, outs):
                text = o.outputs[0].text
                toks = len(o.outputs[0].token_ids)
                steps = split_steps(text) if ttype == "draft" else []
                pred = extract_answer(DATASET, text)
                rec = {k: tasks[idx][k] for k in tasks[idx] if k != "prompt"}
                if ttype == "draft":
                    rec.update(draft_text=text, draft_steps=steps,
                               draft_answer=pred, draft_tokens=toks,
                               n_steps=len(steps))
                elif ttype == "suffix":
                    rec.update(suffix_text=text, suffix_answer=pred,
                               suffix_tokens=toks)
                else:
                    rec.update(sc_text=text, sc_answer=pred, sc_tokens=toks)
                results[idx] = rec
            print(f"[Shard {sid}] {ttype} batch {bi//IBATCH+1}/{(len(idxs)+IBATCH-1)//IBATCH}")
    # SHARD_LOGPROB

    lp_idxs = by_type.get("logprob", [])
    if lp_idxs:
        IBATCH = 256
        for bi in range(0, len(lp_idxs), IBATCH):
            chunk = lp_idxs[bi:bi + IBATCH]
            pids = [tokenizer.encode(tasks[i]["prompt"],
                                     add_special_tokens=False) for i in chunk]
            outs = llm.generate([{"prompt_token_ids": p} for p in pids],
                                sampling_params=sp_lp)
            for idx, o in zip(chunk, outs):
                t = tasks[idx]
                resp_off = t["resp_char_offset"]
                lps, offs = [], []
                cum = 0
                if o.prompt_logprobs is not None:
                    for ti, lp_dict in enumerate(o.prompt_logprobs):
                        tok_id = o.prompt_token_ids[ti]
                        if lp_dict is None:
                            decoded = tokenizer.decode([tok_id])
                            cum += len(decoded)
                            continue
                        lobj = lp_dict.get(tok_id, next(iter(lp_dict.values())))
                        decoded = lobj.decoded_token or ""
                        cpos = cum; cum += len(decoded)
                        if cpos >= resp_off:
                            lps.append(lobj.logprob)
                            offs.append(cpos - resp_off)
                rec = {k: t[k] for k in t if k != "prompt"}
                rec.update(token_logprobs=lps, token_offsets=offs,
                           logprob_prompt_tokens=len(o.prompt_token_ids))
                results[idx] = rec
            print(f"[Shard {sid}] logprob batch {bi//IBATCH+1}/{(len(lp_idxs)+IBATCH-1)//IBATCH}")

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Shard {sid}] Done: {sum(1 for r in results if r)}")


def run_prm_shard(args):
    """Score steps with Qwen2.5-Math-PRM-7B on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer as ATK
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    sid = args._shard_id
    print(f"[PRM Shard {sid}] GPU {args._gpu}: {len(tasks)} tasks")

    prm_tok = ATK.from_pretrained(PRM_MODEL_ID, trust_remote_code=True)
    prm_model = AutoModel.from_pretrained(
        PRM_MODEL_ID, device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    step_sep_id = prm_tok.encode("<extra_0>")[0]

    results = [None] * len(tasks)
    for idx, t in enumerate(tasks):
        conv = prm_tok.apply_chat_template(
            t["prompt"], tokenize=False,
            add_generation_prompt=False)
        input_ids = torch.tensor(
            [prm_tok.encode(conv)], dtype=torch.long,
        ).to(prm_model.device)
        with torch.no_grad():
            logits = prm_model(input_ids=input_ids)[0]
        mask = (input_ids == step_sep_id)
        probs = F.softmax(logits, dim=-1)
        probs = probs * mask.unsqueeze(-1)
        sample = probs[0]
        pos = sample[sample != 0].view(-1, 2)[:, 1]
        scores = pos.cpu().tolist()
        rec = {k: t[k] for k in t if k != "prompt"}
        rec["step_scores"] = [round(s, 6) for s in scores]
        results[idx] = rec

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[PRM Shard {sid}] Done: {sum(1 for r in results if r)}")


# ---- launch helpers -------------------------------------------------------

def launch_shards(tasks, gpu_ids, shard_dir, prm=False):
    if not tasks:
        return []
    shard_dir.mkdir(parents=True, exist_ok=True)
    ns = len(gpu_ids)
    shards = [[] for _ in range(ns)]
    for i, t in enumerate(tasks):
        shards[i % ns].append(t)
    script = str(Path(__file__).resolve())
    procs, out_files = [], []
    for si, gid in enumerate(gpu_ids):
        if not shards[si]:
            continue
        for t in shards[si]:
            t["_out"] = str(shard_dir / f"shard_{si}.jsonl")
        tf = shard_dir / f"tasks_{si}.json"
        tf.write_text(json.dumps(shards[si]), encoding="utf-8")
        out_files.append(shard_dir / f"shard_{si}.jsonl")
        cmd = [PYTHON, script,
               "--_shard-id", str(si), "--_task-file", str(tf),
               "--_gpu", gid, "--dataset", DATASET]
        if prm:
            cmd.append("--_prm")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        env["TOKENIZERS_PARALLELISM"] = "false"
        lf = (shard_dir / f"log_{si}.txt").open("w")
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((si, gid, p, lf))
        print(f"  Shard {si} on GPU {gid}: {len(shards[si])} tasks")
    for si, gid, p, lf in procs:
        p.wait(); lf.close()
        log = (shard_dir / f"log_{si}.txt").read_text(errors="replace")
        tail = "\n".join(log.strip().splitlines()[-3:])
        print(f"  Shard {si} exit={p.returncode}: {tail}")
        if p.returncode != 0:
            print(log[-2000:])
            sys.exit(1)
    recs = []
    for of in out_files:
        if of.exists():
            for line in of.read_text().splitlines():
                if line.strip():
                    recs.append(json.loads(line))
    return recs


# ---- step-level metrics ---------------------------------------------------

def _tok_entropy(lp):
    if lp >= 0: return 0.0
    p = math.exp(lp)
    return -p * lp

def step_char_bounds(response, steps):
    bounds, pos = [], 0
    for s in steps:
        idx = response.find(s, pos)
        if idx < 0: idx = pos
        bounds.append((idx, idx + len(s)))
        pos = idx + len(s)
    return bounds

def compute_step_metrics(lps, offs, bounds):
    metrics = []
    ti = 0
    for s0, s1 in bounds:
        while ti < len(offs) and offs[ti] < s0:
            ti += 1
        slps, shs = [], []
        scan = ti
        while scan < len(offs) and offs[scan] < s1:
            slps.append(lps[scan])
            shs.append(_tok_entropy(lps[scan]))
            scan += 1
        if slps:
            m = dict(mean_entropy=np.mean(shs), max_entropy=max(shs),
                     mean_logprob=np.mean(slps), min_logprob=min(slps),
                     n_tokens=len(slps))
        else:
            m = dict(mean_entropy=0, max_entropy=0,
                     mean_logprob=0, min_logprob=0, n_tokens=0)
        metrics.append(m)
    for i, m in enumerate(metrics):
        if i == 0:
            m["entropy_delta"] = 0.0
            m["logprob_drop"] = 0.0
        else:
            m["entropy_delta"] = m["mean_entropy"] - metrics[i-1]["mean_entropy"]
            m["logprob_drop"] = m["mean_logprob"] - metrics[i-1]["mean_logprob"]
    return metrics


def pick_rollback_step(step_metrics, method, threshold):
    """Return 0-indexed step to rollback TO (keep steps[:rb])."""
    n = len(step_metrics)
    if n < 2:
        return max(0, n - 1)
    if method == "min_logprob":
        for t in range(1, n):
            if step_metrics[t]["min_logprob"] < threshold:
                return max(0, t - 1)
        return n - 1
    elif method == "entropy_delta":
        for t in range(1, n):
            if step_metrics[t]["entropy_delta"] > threshold:
                return max(0, t - 1)
        return n - 1
    elif method == "self_eval":
        scores = [step_metrics[t].get("self_eval_score", 1.0) for t in range(n)]
        worst = min(range(n), key=lambda t: scores[t])
        mean_s = sum(scores) / n
        drop = mean_s - scores[worst]
        if drop > threshold:
            return max(0, worst)
        return n - 1
    elif method == "alpha":
        return max(1, math.ceil(threshold * n)) - 1
    elif method == "random":
        import random
        for t in range(1, n):
            if random.random() < threshold:
                return max(0, t - 1)
        return n - 1
    return n - 1


# ---- evaluation -----------------------------------------------------------

def _vote(answers):
    if not answers: return ""
    return Counter(answers).most_common(1)[0][0]


def evaluate_all(questions, drafts, logprob_recs, suffix_recs, sc_recs, rb_points, args, se_recs=None):
    """Evaluate all methods and return list of (method, tokens_per_q, acc)."""
    q_map = {q["doc_id"]: q for q in questions}
    nq = len(questions)

    # organize drafts by (doc_id, draft_idx)
    draft_map = {}
    for d in drafts:
        draft_map[(d["doc_id"], d["draft_idx"])] = d

    # organize logprob by doc_id (only draft_idx=0)
    lp_map = {}
    for r in logprob_recs:
        lp_map[r["doc_id"]] = r

    # organize suffixes by (doc_id, draft_idx, rollback_step, suffix_idx)
    sfx_map = {}
    for s in suffix_recs:
        key = (s["doc_id"], s["draft_idx"], s["rollback_step"], s["suffix_idx"])
        sfx_map[key] = s

    results = []

    # --- Greedy (single draft) ---
    correct = 0
    total_toks = 0
    for q in questions:
        d = draft_map.get((q["doc_id"], 0))
        if d and check_answer(DATASET, d["draft_answer"], q["gold_answer"]):
            correct += 1
        total_toks += d["draft_tokens"] if d else 0
    results.append(("greedy", total_toks / nq, correct / nq))

    # --- Full SC baselines ---
    sc_by_doc = {}
    for s in sc_recs:
        sc_by_doc.setdefault(s["doc_id"], []).append(s)
    for N in [2, 4, 8, 16, 32]:
        correct = 0
        total_toks = 0
        for q in questions:
            recs = sc_by_doc.get(q["doc_id"], [])[:N]
            answers = [r["sc_answer"] for r in recs]
            toks = sum(r["sc_tokens"] for r in recs)
            if check_answer(DATASET, _vote(answers), q["gold_answer"]):
                correct += 1
            total_toks += toks
        results.append((f"SC@{N}", total_toks / nq, correct / nq))

    # --- Triggered rollback methods ---
    nd_vals = [1, 2, 4]
    k_vals = [2, 3]

    method_configs = []
    for th in MIN_LOGPROB_THRESHOLDS:
        method_configs.append(("min_logprob", th, f"minlp_{th}"))
    for th in ENTROPY_DELTA_THRESHOLDS:
        method_configs.append(("entropy_delta", th, f"entd_{th}"))
    for th in PRM_DROP_THRESHOLDS:
        method_configs.append(("self_eval", th, f"prm_drop_{th}"))
    for a in ALPHA_VALUES:
        method_configs.append(("alpha", a, f"alpha_{a}"))
    for p in RANDOM_PROBS:
        method_configs.append(("random", p, f"rand_{p}"))

    for method, threshold, mkey in method_configs:
        for nd in nd_vals:
            if nd > args.nd_max:
                continue
            for K in k_vals:
                if K > args.k_max:
                    continue
                correct = 0
                total_toks = 0
                for q in questions:
                    did = q["doc_id"]
                    answers = []
                    q_toks = 0
                    lp_rec = lp_map.get(did)
                    lp_cost = lp_rec["logprob_prompt_tokens"] if lp_rec else 0
                    if method in ("min_logprob", "entropy_delta"):
                        q_toks += lp_cost
                    elif method == "self_eval":
                        q_toks += lp_cost

                    rb = rb_points.get((did, mkey))
                    for di in range(nd):
                        d = draft_map.get((did, di))
                        if not d:
                            continue
                        answers.append(d["draft_answer"])
                        q_toks += d["draft_tokens"]
                        if rb is not None:
                            for si in range(K - 1):
                                s = sfx_map.get((did, di, rb, si))
                                if s:
                                    answers.append(s["suffix_answer"])
                                    q_toks += s["suffix_tokens"]
                    if check_answer(DATASET, _vote(answers), q["gold_answer"]):
                        correct += 1
                    total_toks += q_toks
                label = f"{mkey}_nd{nd}_K{K}"
                results.append((label, total_toks / nq, correct / nq))

    return results


# ---- main -----------------------------------------------------------------

def main():
    global DATASET
    args = parse_args()
    DATASET = args.dataset
    if args._shard_id >= 0:
        if args._prm:
            run_prm_shard(args)
        else:
            run_shard(args)
        return

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "results" / f"{DATASET}_entropy_triggered_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.jsonl"
    sd = out_dir / "_shards"

    questions = load_dataset_by_name(DATASET, args.n_sample, seed=42)
    nq = len(questions)
    print(f"Dataset: {DATASET}, {nq} questions, GPUs: {gpu_ids}")

    existing = []
    if ckpt.exists():
        existing = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    print(f"Existing checkpoint: {len(existing)} records")

    # ---- Phase 1: Greedy drafts -------------------------------------------
    done_drafts = {(r["doc_id"], r["draft_idx"])
                   for r in existing if r.get("task_type") == "draft"}
    draft_tasks = []
    for q in questions:
        p = build_prompt(MODEL_ID, DATASET, q["question"])
        for di in range(args.nd_max):
            if (q["doc_id"], di) in done_drafts:
                continue
            draft_tasks.append(dict(
                task_type="draft", doc_id=q["doc_id"], draft_idx=di,
                gold_answer=q["gold_answer"], prompt=p,
            ))
    if draft_tasks:
        print(f"\n--- Phase 1: {len(draft_tasks)} greedy drafts ---")
        new = launch_shards(draft_tasks, gpu_ids, sd / "p1")
        with ckpt.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing.extend(new)
    drafts = [r for r in existing if r.get("task_type") == "draft"]
    print(f"Total drafts: {len(drafts)}")

    # ---- Phase 2: Logprob scoring on draft-0 ------------------------------
    done_lp = {r["doc_id"] for r in existing if r.get("task_type") == "logprob"}
    lp_tasks = []
    d0_map = {r["doc_id"]: r for r in drafts if r["draft_idx"] == 0}
    for q in questions:
        if q["doc_id"] in done_lp:
            continue
        d0 = d0_map.get(q["doc_id"])
        if not d0:
            continue
        p = build_prompt(MODEL_ID, DATASET, q["question"])
        full = p + d0["draft_text"]
        lp_tasks.append(dict(
            task_type="logprob", doc_id=q["doc_id"],
            prompt=full, resp_char_offset=len(p),
            draft_steps=d0["draft_steps"], n_steps=d0["n_steps"],
        ))
    if lp_tasks:
        print(f"\n--- Phase 2: {len(lp_tasks)} logprob scorings ---")
        new = launch_shards(lp_tasks, gpu_ids, sd / "p2")
        with ckpt.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing.extend(new)
    lp_recs = [r for r in existing if r.get("task_type") == "logprob"]
    print(f"Total logprob records: {len(lp_recs)}")

    # ---- Phase 2.5: PRM scoring on draft-0 steps ---------------------------
    done_se = {r["doc_id"] for r in existing if r.get("task_type") == "self_eval"}
    se_tasks = []
    for q in questions:
        if q["doc_id"] in done_se:
            continue
        d0 = d0_map.get(q["doc_id"])
        if not d0 or not d0["draft_steps"]:
            continue
        steps = d0["draft_steps"]
        step_text = "<extra_0>".join(steps) + "<extra_0>"
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q["question"]},
            {"role": "assistant", "content": step_text},
        ]
        se_tasks.append(dict(
            task_type="self_eval", doc_id=q["doc_id"],
            n_steps=len(steps), prompt=messages,
        ))
    if se_tasks:
        print(f"\n--- Phase 2.5: {len(se_tasks)} PRM scorings ---")
        new = launch_shards(se_tasks, gpu_ids, sd / "p2se", prm=True)
        with ckpt.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing.extend(new)
    se_recs = [r for r in existing if r.get("task_type") == "self_eval"]
    print(f"Total PRM records: {len(se_recs)}")

    # build per-doc step scores
    se_by_doc = {}
    for r in se_recs:
        se_by_doc[r["doc_id"]] = r.get("step_scores", [])

    # ---- Compute step metrics & rollback points ---------------------------
    lp_by_doc = {r["doc_id"]: r for r in lp_recs}
    method_configs = []
    for th in MIN_LOGPROB_THRESHOLDS:
        method_configs.append(("min_logprob", th, f"minlp_{th}"))
    for th in ENTROPY_DELTA_THRESHOLDS:
        method_configs.append(("entropy_delta", th, f"entd_{th}"))
    for th in PRM_DROP_THRESHOLDS:
        method_configs.append(("self_eval", th, f"prm_drop_{th}"))
    for a in ALPHA_VALUES:
        method_configs.append(("alpha", a, f"alpha_{a}"))
    for p in RANDOM_PROBS:
        method_configs.append(("random", p, f"rand_{p}"))

    import random as _rng
    _rng.seed(42)

    q_map = {q["doc_id"]: q for q in questions}
    draft_map_all = {(r["doc_id"], r["draft_idx"]): r for r in drafts}

    rb_points = {}  # (doc_id, method_key) -> rollback_step (0-indexed)
    for q in questions:
        did = q["doc_id"]
        lp_rec = lp_by_doc.get(did)
        d0 = d0_map.get(did)
        if not d0:
            continue
        steps = d0["draft_steps"]
        if not steps:
            continue
        response = d0["draft_text"]
        bounds = step_char_bounds(response, steps)
        if lp_rec:
            sm = compute_step_metrics(
                lp_rec["token_logprobs"], lp_rec["token_offsets"], bounds)
        else:
            sm = [dict(mean_entropy=0, max_entropy=0, mean_logprob=0,
                       min_logprob=0, n_tokens=0, entropy_delta=0.0,
                       logprob_drop=0.0) for _ in steps]
        se_scores = se_by_doc.get(did, [])
        for si, m in enumerate(sm):
            m["self_eval_score"] = se_scores[si] if si < len(se_scores) else 1.0
        for method, threshold, mkey in method_configs:
            if method in ("min_logprob", "entropy_delta") and not lp_rec:
                continue
            if method == "self_eval" and did not in se_by_doc:
                continue
            rb = pick_rollback_step(sm, method, threshold)
            rb_points[(did, mkey)] = rb

    # ---- Phase 3: Suffix generation (deduplicated by rollback step) --------
    done_sfx = {(r["doc_id"], r["draft_idx"], r["rollback_step"], r["suffix_idx"])
                for r in existing if r.get("task_type") == "suffix"}
    # collect unique (doc_id, draft_idx, rollback_step) across all methods
    unique_rb = set()
    for (did, mkey), rb in rb_points.items():
        for di in range(args.nd_max):
            unique_rb.add((did, di, rb))

    sfx_tasks = []
    n_sfx = args.k_max - 1
    for did, di, rb in sorted(unique_rb):
        d = draft_map_all.get((did, di))
        if not d:
            continue
        steps = d["draft_steps"]
        T = len(steps)
        b = min(rb, T - 1)
        if b < 1:
            b = 1
        prefix = "\n\n".join(steps[:b])
        p = build_prompt(MODEL_ID, DATASET, q_map[did]["question"]) + prefix + "\n\n"
        for si in range(n_sfx):
            if (did, di, b, si) in done_sfx:
                continue
            sfx_tasks.append(dict(
                task_type="suffix", doc_id=did,
                draft_idx=di, rollback_step=b,
                suffix_idx=si, gold_answer=q_map[did]["gold_answer"],
                prompt=p,
            ))
    if sfx_tasks:
        print(f"\n--- Phase 3: {len(sfx_tasks)} suffix generations ---")
        new = launch_shards(sfx_tasks, gpu_ids, sd / "p3")
        with ckpt.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing.extend(new)
    sfx_recs = [r for r in existing if r.get("task_type") == "suffix"]
    print(f"Total suffix records: {len(sfx_recs)}")

    # ---- Phase 4: Full SC baseline ----------------------------------------
    done_sc = {(r["doc_id"], r["sc_idx"])
               for r in existing if r.get("task_type") == "fullsc"}
    sc_tasks = []
    for q in questions:
        p = build_prompt(MODEL_ID, DATASET, q["question"])
        for si in range(args.fullsc_n):
            if (q["doc_id"], si) in done_sc:
                continue
            sc_tasks.append(dict(
                task_type="fullsc", doc_id=q["doc_id"], sc_idx=si,
                gold_answer=q["gold_answer"], prompt=p,
            ))
    if sc_tasks:
        print(f"\n--- Phase 4: {len(sc_tasks)} SC samples ---")
        new = launch_shards(sc_tasks, gpu_ids, sd / "p4")
        with ckpt.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing.extend(new)
    sc_recs = [r for r in existing if r.get("task_type") == "fullsc"]
    print(f"Total SC records: {len(sc_recs)}")

    # ---- Evaluate ---------------------------------------------------------
    print("\n--- Evaluation ---")
    t0 = time.time()
    results = evaluate_all(questions, drafts, lp_recs, sfx_recs, sc_recs, rb_points, args, se_recs)
    print(f"Evaluation done in {time.time()-t0:.1f}s, {len(results)} configs")

    # ---- Save & Plot ------------------------------------------------------
    rows = [{"method": m, "tokens_per_q": t, "acc": a} for m, t, a in results]
    res_path = out_dir / "sweep_results.json"
    res_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Results -> {res_path}")

    plot_pareto(rows, out_dir, dataset=DATASET)
    print("Done.")


# ---- plotting -------------------------------------------------------------

def plot_pareto(rows, out_dir, dataset="gsm8k"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ROOT / "figures" / "entropy_triggered_sweep"
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for r in rows:
        m = r["method"]
        if m == "greedy":
            g = "Greedy"
        elif m.startswith("SC@"):
            g = "Self-Consistency"
        elif m.startswith("minlp_"):
            g = "min_logprob (lookback)"
        elif m.startswith("entd_"):
            g = "entropy_delta (lookback)"
        elif m.startswith("prm_drop_"):
            g = "PRM (relative drop)"
        elif m.startswith("alpha_"):
            g = "alpha-fixed"
        elif m.startswith("rand_"):
            g = "random-repair"
        else:
            g = "other"
        groups.setdefault(g, []).append(r)

    colors = {
        "Greedy": "#333333",
        "Self-Consistency": "#9E9E9E",
        "min_logprob (lookback)": "#2196F3",
        "entropy_delta (lookback)": "#E91E63",
        "PRM (relative drop)": "#9C27B0",
        "alpha-fixed": "#FF9800",
        "random-repair": "#4CAF50",
    }
    markers = {
        "Greedy": "*",
        "Self-Consistency": "D",
        "min_logprob (lookback)": "o",
        "entropy_delta (lookback)": "s",
        "PRM (relative drop)": "P",
        "alpha-fixed": "^",
        "random-repair": "v",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for g, pts in groups.items():
        xs = [p["tokens_per_q"] / 1000 for p in pts]
        ys = [p["acc"] * 100 for p in pts]
        ax.scatter(xs, ys, label=g, color=colors.get(g, "#666"),
                   marker=markers.get(g, "o"), s=40, alpha=0.7)
        paired = sorted(zip(xs, ys))
        front_x, front_y = [paired[0][0]], [paired[0][1]]
        for x, y in paired[1:]:
            if y >= front_y[-1]:
                front_x.append(x)
                front_y.append(y)
        if len(front_x) > 1:
            ax.plot(front_x, front_y, color=colors.get(g, "#666"),
                    linewidth=1.5, alpha=0.5)

    ax.set_xlabel("Tokens per question (x1000)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Qwen2.5-3B-Instruct on GSM8K: Acc vs Compute")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_pareto_{dataset}.{fmt}", dpi=200)
    plt.close(fig)
    print(f"Pareto figure -> {fig_dir}")

    # acc gain / 1k tokens efficiency plot
    greedy_acc = next((p["acc"] for p in rows if p["method"] == "greedy"), 0)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for g, pts in groups.items():
        if g == "Greedy":
            continue
        xs = [p["tokens_per_q"] / 1000 for p in pts]
        ys = [(p["acc"] - greedy_acc) * 100 / max(p["tokens_per_q"] / 1000, 0.01)
              for p in pts]
        ax2.scatter(xs, ys, label=g, color=colors.get(g, "#666"),
                    marker=markers.get(g, "o"), s=40, alpha=0.7)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Tokens per question (x1000)")
    ax2.set_ylabel("Acc gain per 1k tokens (pp)")
    ax2.set_title("Qwen2.5-3B-Instruct on GSM8K: Efficiency (Acc Gain / 1k tokens)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    for fmt in ("png", "pdf"):
        fig2.savefig(fig_dir / f"fig_efficiency_{dataset}.{fmt}", dpi=200)
    plt.close(fig2)
    print(f"Efficiency figure -> {fig_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

