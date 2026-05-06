#!/usr/bin/env python3
"""
Entropy-triggered rollback sweep for Qwen2.5-3B on GSM8K.

Pipeline:
  Phase 1: Greedy drafts (nd_max per question)
  Phase 2: Logprob scoring on draft-0 via prompt_logprobs
  Phase 3: Self-eval scoring on draft-0 steps (P(Yes)/P(No))
  Phase 4: Full SC baseline
  Evaluate: acc vs total tokens (logprob cost included)

Trigger modes (after slimming):
  - min_logprob (lookback): fire when min_logprob < threshold -> repair prev step
  - mean_nll: rollback to step with highest mean NLL (= lowest mean logprob)
  - PRM threshold: rollback to first step with PRM score < threshold,
                    fallback to alpha=0.8 position if none qualifies
  - alpha-fixed (baseline): rollback at fixed fraction alpha
  - random: fire with probability p at each step

Two evaluation views are produced:
  (A) including PRM scoring tokens in PRM cost
  (B) excluding PRM scoring tokens (assume PRM is "free" / amortized)

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
ND_MAX = 8
K_MAX = 3
FULLSC_N = 40
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_MODEL_LEN = 4096
GPU_MEM = 0.85
BATCH_PER_GPU = 256

# -- baseline sweep values ---------------------------------------------------
ALPHA_VALUES = [0.3, 0.5, 0.7]
RANDOM_PROBS = [0.10, 0.20, 0.30]
PRM_THRESHOLD = 0.1       # rollback to first step with PRM score < threshold
PRM_FALLBACK_ALPHA = 0.8  # if no step below threshold, rollback at this fraction


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--nd-max", type=int, default=ND_MAX)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--fullsc-n", type=int, default=FULLSC_N)
    ap.add_argument("--n-sample", type=int, default=0)
    ap.add_argument("--tag", default="", help="version tag appended to output dirs")
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

    for ttype, sp in [("draft_greedy", sp_g), ("draft_sample", sp_s),
                       ("suffix", sp_s), ("fullsc", sp_s)]:
        real_type = "draft" if ttype.startswith("draft") else ttype
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
                steps = split_steps(text) if real_type == "draft" else []
                pred = extract_answer(DATASET, text)
                rec = {k: tasks[idx][k] for k in tasks[idx] if k != "prompt"}
                if real_type == "draft":
                    rec.update(draft_text=text, draft_steps=steps,
                               draft_answer=pred, draft_tokens=toks,
                               n_steps=len(steps))
                elif real_type == "suffix":
                    rec.update(suffix_text=text, suffix_answer=pred,
                               suffix_tokens=toks)
                else:
                    rec.update(sc_text=text, sc_answer=pred, sc_tokens=toks)
                rec["task_type"] = real_type
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
            _lp_t0 = time.perf_counter()
            outs = llm.generate([{"prompt_token_ids": p} for p in pids],
                                sampling_params=sp_lp)
            _lp_wall_ms = (time.perf_counter() - _lp_t0) * 1000.0
            _lp_per_sample = _lp_wall_ms / max(len(chunk), 1)
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
                           logprob_prompt_tokens=len(o.prompt_token_ids),
                           logprob_overhead_ms=round(_lp_per_sample, 3))
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
        _prm_t0 = time.perf_counter()
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
        rec["prm_tokens"] = int(input_ids.shape[1])
        rec["prm_overhead_ms"] = round(
            (time.perf_counter() - _prm_t0) * 1000.0, 3)
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




# ---- evaluation -----------------------------------------------------------

def _vote(answers):
    if not answers: return ""
    return Counter(answers).most_common(1)[0][0]


def evaluate_all(questions, drafts, logprob_recs, suffix_recs, sc_recs,
                 rb_points, args, se_recs=None, count_prm_tokens=True):
    """Evaluate all methods and return list of (method, tokens_per_q, acc, overhead_ms_per_q).

    count_prm_tokens:
        If True,  PRM scoring cost is added to PRM-based methods.
        If False, PRM is treated as "free" (e.g. amortized / external).
    """
    q_map = {q["doc_id"]: q for q in questions}
    nq = len(questions)

    # organize drafts by (doc_id, draft_idx)
    draft_map = {}
    for d in drafts:
        draft_map[(d["doc_id"], d["draft_idx"])] = d

    # organize logprob by (doc_id, draft_idx)
    lp_map = {}
    for r in logprob_recs:
        lp_map[(r["doc_id"], r.get("draft_idx", 0))] = r

    # organize PRM tokens by (doc_id, draft_idx)
    prm_tok_map = {}
    prm_ms_map = {}
    if se_recs:
        for r in se_recs:
            key = (r["doc_id"], r.get("draft_idx", 0))
            prm_tok_map[key] = r.get("prm_tokens", 0)
            prm_ms_map[key] = r.get("prm_overhead_ms", 0.0)

    # organize logprob overhead_ms by (doc_id, draft_idx)
    lp_ms_map = {}
    for r in logprob_recs:
        lp_ms_map[(r["doc_id"], r.get("draft_idx", 0))] = r.get("logprob_overhead_ms", 0.0)

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
    results.append(("greedy", total_toks / nq, correct / nq, 0.0))

    # --- Full SC baselines ---
    sc_by_doc = {}
    for s in sc_recs:
        sc_by_doc.setdefault(s["doc_id"], []).append(s)
    for N in [8, 16, 24, 32, 40]:
        correct = 0
        total_toks = 0
        for q in questions:
            recs = sc_by_doc.get(q["doc_id"], [])[:N]
            answers = [r["sc_answer"] for r in recs]
            toks = sum(r["sc_tokens"] for r in recs)
            if check_answer(DATASET, _vote(answers), q["gold_answer"]):
                correct += 1
            total_toks += toks
        results.append((f"SC@{N}", total_toks / nq, correct / nq, 0.0))

    # --- Triggered rollback methods ---
    nd_vals = sorted(set([1, 2, 4, 8, 16]) & set(range(1, args.nd_max + 1)))
    k_vals = [2, 3]

    all_mkeys = ["minlp", "mean_nll", "prm_thr"]
    for a in ALPHA_VALUES:
        all_mkeys.append(f"alpha_{a}")
    for p in RANDOM_PROBS:
        all_mkeys.append(f"rand_{p}")

    for mkey in all_mkeys:
        for nd in nd_vals:
            if nd > args.nd_max:
                continue
            for K in k_vals:
                if K > args.k_max:
                    continue
                correct = 0
                total_toks = 0
                total_overhead_ms = 0.0
                for q in questions:
                    did = q["doc_id"]
                    answers = []
                    q_toks = 0
                    q_overhead_ms = 0.0

                    for di in range(nd):
                        d = draft_map.get((did, di))
                        if not d:
                            continue
                        answers.append(d["draft_answer"])
                        q_toks += d["draft_tokens"]
                        # Scoring cost depends on the trigger
                        if mkey in ("minlp", "mean_nll"):
                            lp_rec = lp_map.get((did, di))
                            if lp_rec:
                                q_toks += lp_rec["logprob_prompt_tokens"]
                                q_overhead_ms += lp_ms_map.get((did, di), 0.0)
                        elif mkey == "prm_thr":
                            if count_prm_tokens:
                                q_toks += prm_tok_map.get((did, di), 0)
                            q_overhead_ms += prm_ms_map.get((did, di), 0.0)
                        # alpha_* and rand_* incur no scoring cost
                        rb = rb_points.get((did, di, mkey))
                        if rb is not None:
                            for si in range(K - 1):
                                s = sfx_map.get((did, di, rb, si))
                                if s:
                                    answers.append(s["suffix_answer"])
                                    q_toks += s["suffix_tokens"]
                    if check_answer(DATASET, _vote(answers), q["gold_answer"]):
                        correct += 1
                    total_toks += q_toks
                    total_overhead_ms += q_overhead_ms
                label = f"{mkey}_nd{nd}_K{K}"
                results.append((label, total_toks / nq, correct / nq,
                                round(total_overhead_ms / nq, 3)))

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
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "results" / f"{DATASET}_entropy_triggered_sweep{suffix}")
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
            tt = "draft_greedy" if di == 0 else "draft_sample"
            draft_tasks.append(dict(
                task_type=tt, doc_id=q["doc_id"], draft_idx=di,
                gold_answer=q["gold_answer"], prompt=p,
            ))
    if draft_tasks:
        print(f"\n--- Phase 1: {len(draft_tasks)} drafts (greedy + sample) ---")
        new = launch_shards(draft_tasks, gpu_ids, sd / "p1")
        with ckpt.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing.extend(new)
    drafts = [r for r in existing if r.get("task_type") == "draft"]
    print(f"Total drafts: {len(drafts)}")

    # ---- Phase 2: Logprob scoring on ALL drafts ----------------------------
    done_lp = {(r["doc_id"], r.get("draft_idx", 0))
               for r in existing if r.get("task_type") == "logprob"}
    lp_tasks = []
    draft_map_all = {(r["doc_id"], r["draft_idx"]): r for r in drafts}
    for q in questions:
        for di in range(args.nd_max):
            if (q["doc_id"], di) in done_lp:
                continue
            d = draft_map_all.get((q["doc_id"], di))
            if not d:
                continue
            p = build_prompt(MODEL_ID, DATASET, q["question"])
            full = p + d["draft_text"]
            lp_tasks.append(dict(
                task_type="logprob", doc_id=q["doc_id"], draft_idx=di,
                prompt=full, resp_char_offset=len(p),
                draft_steps=d["draft_steps"], n_steps=d["n_steps"],
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

    # ---- Phase 2.5: PRM scoring on ALL drafts ------------------------------
    done_se = {(r["doc_id"], r.get("draft_idx", 0))
               for r in existing if r.get("task_type") == "self_eval"}
    se_tasks = []
    for q in questions:
        for di in range(args.nd_max):
            if (q["doc_id"], di) in done_se:
                continue
            d = draft_map_all.get((q["doc_id"], di))
            if not d or not d["draft_steps"]:
                continue
            steps = d["draft_steps"]
            step_text = "<extra_0>".join(steps) + "<extra_0>"
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q["question"]},
                {"role": "assistant", "content": step_text},
            ]
            se_tasks.append(dict(
                task_type="self_eval", doc_id=q["doc_id"], draft_idx=di,
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

    # build per-(doc, draft) step scores
    se_by_draft = {}
    for r in se_recs:
        se_by_draft[(r["doc_id"], r.get("draft_idx", 0))] = r.get("step_scores", [])

    # ---- Compute per-draft rollback points -----------------------------------
    lp_by_draft = {}
    for r in lp_recs:
        lp_by_draft[(r["doc_id"], r.get("draft_idx", 0))] = r

    import random as _rng
    _rng.seed(42)

    q_map = {q["doc_id"]: q for q in questions}

    rb_points = {}  # (doc_id, draft_idx, method_key) -> rollback_step
    for q in questions:
        did = q["doc_id"]
        for di in range(args.nd_max):
            d = draft_map_all.get((did, di))
            if not d or not d.get("draft_steps"):
                continue
            steps = d["draft_steps"]
            n = len(steps)
            if n < 2:
                continue

            # min_logprob: argmin over this draft's logprob
            lp_rec = lp_by_draft.get((did, di))
            if lp_rec:
                bounds = step_char_bounds(d["draft_text"], steps)
                sm = compute_step_metrics(
                    lp_rec["token_logprobs"], lp_rec["token_offsets"], bounds)
                worst_lp = min(range(n), key=lambda t: sm[t]["min_logprob"])
                rb_points[(did, di, "minlp")] = worst_lp

                # mean_nll: argmax NLL_t = argmin mean_logprob over steps
                worst_mean = min(range(n), key=lambda t: sm[t]["mean_logprob"])
                rb_points[(did, di, "mean_nll")] = worst_mean

            # PRM threshold: rollback to first step with score < PRM_THRESHOLD;
            # if none qualifies, fallback to the PRM_FALLBACK_ALPHA position
            se_scores = se_by_draft.get((did, di), [])
            if se_scores and len(se_scores) >= n and n >= 2:
                rb_step = None
                for si_idx in range(n):
                    if se_scores[si_idx] < PRM_THRESHOLD:
                        rb_step = si_idx
                        break
                if rb_step is None:
                    rb_step = max(1, math.ceil(PRM_FALLBACK_ALPHA * n)) - 1
                rb_points[(did, di, "prm_thr")] = rb_step

            # alpha-fixed baselines
            for a in ALPHA_VALUES:
                rb_points[(did, di, f"alpha_{a}")] = max(1, math.ceil(a * n)) - 1

            # random-uniform baselines: uniformly sample a rollback prefix length
            for p in RANDOM_PROBS:
                rb = _rng.randint(1, n - 1)
                rb_points[(did, di, f"rand_{p}")] = rb

    # ---- Phase 3: Suffix generation (deduplicated by rollback step) --------
    done_sfx = {(r["doc_id"], r["draft_idx"], r["rollback_step"], r["suffix_idx"])
                for r in existing if r.get("task_type") == "suffix"}
    # collect unique (doc_id, draft_idx, rollback_step) across all methods
    unique_rb = set()
    for (did, di, mk), rb in rb_points.items():
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

    # ---- Evaluate (two views) ---------------------------------------------
    print("\n--- Evaluation (with PRM tokens) ---")
    t0 = time.time()
    res_with = evaluate_all(questions, drafts, lp_recs, sfx_recs, sc_recs,
                            rb_points, args, se_recs, count_prm_tokens=True)
    print(f"Done in {time.time()-t0:.1f}s, {len(res_with)} configs")

    print("\n--- Evaluation (without PRM tokens) ---")
    t0 = time.time()
    res_no = evaluate_all(questions, drafts, lp_recs, sfx_recs, sc_recs,
                          rb_points, args, se_recs, count_prm_tokens=False)
    print(f"Done in {time.time()-t0:.1f}s, {len(res_no)} configs")

    # ---- Overhead wall-time summary ----------------------------------------
    print("\n--- Scoring overhead wall-time (ms per question) ---")
    for label, _, _, overhead in res_with:
        if overhead > 0:
            print(f"  {label:40s}  {overhead:10.1f} ms/q")
    lp_all_ms = [r.get("logprob_overhead_ms", 0.0) for r in lp_recs if r.get("logprob_overhead_ms")]
    prm_all_ms = [r.get("prm_overhead_ms", 0.0) for r in se_recs if r.get("prm_overhead_ms")]
    if lp_all_ms:
        print(f"  [logprob per-sample]  mean={np.mean(lp_all_ms):.1f}  "
              f"median={np.median(lp_all_ms):.1f}  p95={np.percentile(lp_all_ms,95):.1f} ms")
    if prm_all_ms:
        print(f"  [PRM per-sample]      mean={np.mean(prm_all_ms):.1f}  "
              f"median={np.median(prm_all_ms):.1f}  p95={np.percentile(prm_all_ms,95):.1f} ms")

    # ---- Save & Plot ------------------------------------------------------
    rows_with = [{"method": m, "tokens_per_q": t, "acc": a,
                  "overhead_ms_per_q": o} for m, t, a, o in res_with]
    rows_no   = [{"method": m, "tokens_per_q": t, "acc": a,
                  "overhead_ms_per_q": o} for m, t, a, o in res_no]

    (out_dir / "sweep_results_with_prm.json").write_text(
        json.dumps(rows_with, indent=2), encoding="utf-8")
    (out_dir / "sweep_results_no_prm.json").write_text(
        json.dumps(rows_no, indent=2), encoding="utf-8")
    # also keep the legacy filename pointing at the "with PRM" view
    (out_dir / "sweep_results.json").write_text(
        json.dumps(rows_with, indent=2), encoding="utf-8")
    print(f"Results -> {out_dir}/sweep_results_{{with_prm,no_prm}}.json")

    plot_pareto(rows_with, out_dir, dataset=DATASET,
                tag=suffix + "_with_prm",
                subtitle="PRM scoring tokens INCLUDED")
    plot_pareto(rows_no, out_dir, dataset=DATASET,
                tag=suffix + "_no_prm",
                subtitle="PRM scoring tokens EXCLUDED")
    print("Done.")


# ---- plotting -------------------------------------------------------------

def _sc_curve(rows):
    """Return sorted list of (tokens_per_q, acc) for SC@N points."""
    sc_pts = sorted(
        (r["tokens_per_q"], r["acc"]) for r in rows if r["method"].startswith("SC@")
    )
    return sc_pts


def _sc_acc_at(sc_pts, tokens):
    """Linearly interpolate SC accuracy at a given token budget; clamp to ends."""
    if not sc_pts:
        return None
    if tokens <= sc_pts[0][0]:
        return sc_pts[0][1]
    if tokens >= sc_pts[-1][0]:
        return sc_pts[-1][1]
    for i in range(len(sc_pts) - 1):
        x0, y0 = sc_pts[i]
        x1, y1 = sc_pts[i + 1]
        if x0 <= tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return sc_pts[-1][1]


def plot_pareto(rows, out_dir, dataset="gsm8k", tag="", subtitle=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ROOT / "figures" / f"entropy_triggered_sweep{tag}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- group classification (slimmed: only PRM score-drop kept) ---
    groups = {}
    for r in rows:
        m = r["method"]
        if m == "greedy":
            g = "Greedy"
        elif m.startswith("SC@"):
            g = "Self-Consistency"
        elif m.startswith("minlp_"):
            g = "min_logprob (argmin)"
        elif m.startswith("mean_nll_"):
            g = "mean_NLL (step-avg)"
        elif m.startswith("prm_thr"):
            g = "PRM threshold"
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
        "min_logprob (argmin)": "#2196F3",
        "mean_NLL (step-avg)": "#4CAF50",
        "PRM threshold": "#00BCD4",
        "alpha-fixed": "#795548",
        "random-repair": "#607D8B",
    }
    markers = {
        "Greedy": "*",
        "Self-Consistency": "D",
        "min_logprob (argmin)": "o",
        "mean_NLL (step-avg)": "s",
        "PRM threshold": "d",
        "alpha-fixed": "v",
        "random-repair": "<",
    }

    # =====================================================================
    # Plot 1: Pareto (acc vs tokens), with SC drawn as a connected curve
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    for g, pts in groups.items():
        xs = [p["tokens_per_q"] / 1000 for p in pts]
        ys = [p["acc"] * 100 for p in pts]
        ax.scatter(xs, ys, label=g, color=colors.get(g, "#666"),
                   marker=markers.get(g, "o"), s=40, alpha=0.7)
        if g == "Self-Consistency":
            paired = sorted(zip(xs, ys))
            ax.plot([x for x, _ in paired], [y for _, y in paired],
                    color=colors[g], linewidth=2.0, alpha=0.7,
                    linestyle="-", label="_nolegend_")
        else:
            paired = sorted(zip(xs, ys))
            front_x, front_y = [paired[0][0]], [paired[0][1]]
            for x, y in paired[1:]:
                if y >= front_y[-1]:
                    front_x.append(x)
                    front_y.append(y)
            if len(front_x) > 1:
                ax.plot(front_x, front_y, color=colors.get(g, "#666"),
                        linewidth=1.0, alpha=0.4, linestyle=":")

    ax.set_xlabel("Tokens per question (x1000)")
    ax.set_ylabel("Accuracy (%)")
    title = "Qwen2.5-3B-Instruct on GSM8K: Acc vs Compute"
    if subtitle:
        title += f"\n[{subtitle}]"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_pareto_{dataset}.{fmt}", dpi=200)
    plt.close(fig)
    print(f"Pareto figure -> {fig_dir}")

    # =====================================================================
    # Plot 2: Accuracy improvement vs Full SC at the SAME token budget
    #         (interpolated SC curve as the per-cost baseline)
    # =====================================================================
    sc_pts = _sc_curve(rows)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for g, pts in groups.items():
        if g in ("Self-Consistency",):
            continue
        xs = [p["tokens_per_q"] / 1000 for p in pts]
        gains = []
        for p in pts:
            sc_a = _sc_acc_at(sc_pts, p["tokens_per_q"])
            if sc_a is None:
                gains.append(0.0)
            else:
                gains.append((p["acc"] - sc_a) * 100)
        ax2.scatter(xs, gains, label=g, color=colors.get(g, "#666"),
                    marker=markers.get(g, "o"), s=40, alpha=0.8)
    ax2.axhline(0, color="gray", linewidth=1.0, linestyle="--",
                label="Full SC (interpolated)")
    ax2.set_xlabel("Tokens per question (x1000)")
    ax2.set_ylabel("Accuracy improvement vs Full SC (pp)")
    title2 = "Acc improvement over Full SC at same token budget"
    if subtitle:
        title2 += f"\n[{subtitle}]"
    ax2.set_title(title2)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    for fmt in ("png", "pdf"):
        fig2.savefig(fig_dir / f"fig_acc_gain_vs_sc_{dataset}.{fmt}", dpi=200)
    plt.close(fig2)
    print(f"Gain-vs-SC figure -> {fig_dir}")

    # =====================================================================
    # Plot 3: Acc gain (vs Full SC) per 1k tokens — efficiency view
    # =====================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for g, pts in groups.items():
        if g in ("Self-Consistency",):
            continue
        xs = [p["tokens_per_q"] / 1000 for p in pts]
        eff = []
        for p in pts:
            sc_a = _sc_acc_at(sc_pts, p["tokens_per_q"])
            if sc_a is None:
                eff.append(0.0)
            else:
                eff.append((p["acc"] - sc_a) * 100 /
                           max(p["tokens_per_q"] / 1000, 0.01))
        ax3.scatter(xs, eff, label=g, color=colors.get(g, "#666"),
                    marker=markers.get(g, "o"), s=40, alpha=0.8)
    ax3.axhline(0, color="gray", linewidth=1.0, linestyle="--",
                label="Full SC (interpolated)")
    ax3.set_xlabel("Tokens per question (x1000)")
    ax3.set_ylabel("Acc gain vs Full SC per 1k tokens (pp / 1k tok)")
    title3 = "Efficiency: acc gain over Full SC per 1k tokens"
    if subtitle:
        title3 += f"\n[{subtitle}]"
    ax3.set_title(title3)
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    for fmt in ("png", "pdf"):
        fig3.savefig(fig_dir / f"fig_efficiency_vs_sc_{dataset}.{fmt}", dpi=200)
    plt.close(fig3)
    print(f"Efficiency-vs-SC figure -> {fig_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()