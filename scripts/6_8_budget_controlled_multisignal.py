#!/usr/bin/env python3
"""
Budget-controlled rollback with pluggable error-step signals.

Extends 6_6 by supporting multiple rollback signal strategies, each with
two fallback modes when the signal does not trigger:
  - *_fb_last:  fallback to step n-1 (last step)
  - *_fb_start: fallback to step 0 (resample from scratch)

Signals:
  - gpt / gpt_fb_last / gpt_fb_start:
        Live GPT API call to locate first-error step (tau).
  - prm_drop / prm_drop_fb_last / prm_drop_fb_start:
        First i where score[i]-score[i+1] > threshold.
  - nll_drop / nll_drop_fb_last / nll_drop_fb_start:
        First i where nll[i]-nll[i-1] > threshold.

Each signal produces a rollback step per draft. Suffix generation and
majority-vote evaluation proceed identically to 6_6.

Usage:
    python scripts/6_8_budget_controlled_multisignal.py \\
        --budget 32 --gpus 0,1 \\
        --signals gpt gpt_fb_last gpt_fb_start \\
                  prm_drop prm_drop_fb_last prm_drop_fb_start \\
                  nll_drop nll_drop_fb_last nll_drop_fb_start
"""

import argparse, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
DATASET = "gsm8k"
PYTHON = sys.executable

PRM_DROP_THRESHOLD = 0.1
NLL_DROP_THRESHOLD = 0.2
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_MODEL_LEN = 4096
GPU_MEM = 0.90

ROLLBACK_CONFIGS = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2)]

ALL_SIGNALS = [
    "gpt_fb_last", "gpt_fb_start",
    "prm_drop", "prm_drop_fb_last", "prm_drop_fb_start",
    "nll_drop", "nll_drop_fb_last", "nll_drop_fb_start",
]

GPT_MODEL = "gpt-5.1"
GPT_MAX_WORKERS = 32

from src.prompt_templates import (
    build_prompt, get_stop_tokens, split_steps,
    extract_answer, check_answer,
)
from src.sweep_datasets import load_dataset_by_name
from src.step_judge import (
    build_first_error_prompt,
    call_gpt_first_error,
    load_env_file,
    make_client,
)


# SECTION: parse_args


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--model", default=MODEL_ID,
                    help="HuggingFace model ID for draft/suffix generation")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--n-sample", type=int, default=0,
                    help="Limit questions; 0=all")
    ap.add_argument("--signals", nargs="+",
                    default=ALL_SIGNALS,
                    choices=ALL_SIGNALS,
                    help="Which rollback signals to evaluate")
    ap.add_argument("--prm-drop-threshold", type=float,
                    default=PRM_DROP_THRESHOLD)
    ap.add_argument("--nll-drop-threshold", type=float,
                    default=NLL_DROP_THRESHOLD)
    ap.add_argument("--gpt-model", type=str, default=GPT_MODEL,
                    help="GPT model for first-error detection")
    ap.add_argument("--gpt-max-workers", type=int,
                    default=GPT_MAX_WORKERS)
    ap.add_argument("--gpt-temperature", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpt-cache", default="",
                    help="Path to gpt_first_error_cache.jsonl "
                         "(used as warm-start; new calls appended)")
    ap.add_argument("--nd", type=int, nargs="+", default=None,
                    help="Which nd values to evaluate "
                         "(default: all from ROLLBACK_CONFIGS)")
    ap.add_argument("--skip-phase", type=int, nargs="*", default=[],
                    help="Skip phases (1=drafts, 2=PRM, 25=GPT, "
                         "3=logprob, 4=suffix, 5=SC)")
    ap.add_argument("--_shard-id", type=int, default=-1)
    ap.add_argument("--_task-file", default="")
    ap.add_argument("--_gpu", default="0")
    ap.add_argument("--_prm", action="store_true")
    ap.add_argument("--_logprob", action="store_true")
    return ap.parse_args()


# SECTION: shard_worker


def run_shard(args):
    """Generate completions on one GPU."""
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
    sp_s = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P,
        max_tokens=MAX_TOKENS, stop=stop,
    )
    by_type = {}
    for i, t in enumerate(tasks):
        by_type.setdefault(t["task_type"], []).append(i)
    results = [None] * len(tasks)

    for ttype in ["draft", "suffix", "fullsc"]:
        idxs = by_type.get(ttype, [])
        if not idxs:
            continue
        for bi in range(0, len(idxs), 512):
            chunk = idxs[bi:bi + 512]
            pids = [tokenizer.encode(tasks[i]["prompt"],
                                     add_special_tokens=False)
                    for i in chunk]
            outs = llm.generate(
                [{"prompt_token_ids": p} for p in pids],
                sampling_params=sp_s)
            for idx, o in zip(chunk, outs):
                text = o.outputs[0].text
                toks = len(o.outputs[0].token_ids)
                t = tasks[idx]
                rec = {k: t[k] for k in t if k != "prompt"}
                if ttype == "draft":
                    steps = split_steps(text)
                    pred = extract_answer(DATASET, text)
                    rec.update(draft_text=text, draft_steps=steps,
                               draft_answer=pred, draft_tokens=toks,
                               n_steps=len(steps))
                elif ttype == "suffix":
                    pred = extract_answer(DATASET, text)
                    rec.update(suffix_text=text, suffix_answer=pred,
                               suffix_tokens=toks)
                else:
                    pred = extract_answer(DATASET, text)
                    rec.update(sc_text=text, sc_answer=pred,
                               sc_tokens=toks)
                rec["task_type"] = ttype
                results[idx] = rec
            print(f"[Shard {sid}] {ttype} batch "
                  f"{bi//512+1}/{(len(idxs)+511)//512}")

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Shard {sid}] Done: {sum(1 for r in results if r)}")


# SECTION: prm_shard


def run_prm_shard(args):
    """Score steps with PRM on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer as ATK
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, *a, **kw: self.get_seq_length())

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    sid = args._shard_id
    print(f"[PRM Shard {sid}] GPU {args._gpu}: {len(tasks)} tasks")

    prm_tok = ATK.from_pretrained(
        PRM_MODEL_ID, trust_remote_code=True)
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
        rec["prm_tokens"] = int(input_ids.shape[1])
        results[idx] = rec

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[PRM Shard {sid}] Done: "
          f"{sum(1 for r in results if r)}")


# SECTION: logprob_shard


def run_logprob_shard(args):
    """Collect per-token logprobs for NLL signal."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    sid = args._shard_id
    print(f"[LP Shard {sid}] GPU {args._gpu}: {len(tasks)} tasks")

    llm = LLM(
        model=MODEL_ID, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=0.60,
        max_model_len=2048,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=1)
    BATCH = 8
    results = [None] * len(tasks)
    for bi in range(0, len(tasks), BATCH):
        batch = tasks[bi:bi + BATCH]
        prompts = [t["full_prompt"] for t in batch]
        outputs = llm.generate(prompts, sp)
        for ti, (task, output) in enumerate(
                zip(batch, outputs)):
            rec = {k: v for k, v in task.items()
                   if k not in ("full_prompt",)}
            resp_off = task["resp_char_offset"]
            lps, offsets = [], []
            cum = 0
            if output.prompt_logprobs is not None:
                ptids = output.prompt_token_ids
                for pi, lp_dict in enumerate(
                        output.prompt_logprobs):
                    if lp_dict is None:
                        tok_id = ptids[pi]
                        decoded = tokenizer.decode([tok_id])
                        cum += len(decoded)
                        continue
                    tok_id = ptids[pi]
                    if tok_id in lp_dict:
                        lpo = lp_dict[tok_id]
                    else:
                        lpo = next(iter(lp_dict.values()))
                    decoded = lpo.decoded_token or ""
                    cpos = cum
                    cum += len(decoded)
                    if cpos >= resp_off:
                        lps.append(lpo.logprob)
                        offsets.append(cpos - resp_off)
            rec["token_logprobs"] = lps
            rec["token_offsets"] = offsets
            rec["task_type"] = "logprob"
            results[bi + ti] = rec
        print(f"[LP Shard {sid}] batch "
              f"{bi//BATCH+1}/{(len(tasks)+BATCH-1)//BATCH}")

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
    print(f"[LP Shard {sid}] Done: "
          f"{sum(1 for r in results if r)}")


# SECTION: gpu_pool

MIN_FREE_MEM_MIB_GEN = 6000
MIN_FREE_MEM_MIB_PRM = 10000
GPU_POLL_INTERVAL = 10


def _gpu_free_memory() -> Dict[str, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True)
    except Exception as e:
        print(f"[gpu_pool] nvidia-smi failed: {e}")
        return {}
    result = {}
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 2:
            gid = parts[0].strip()
            free = int(parts[1].strip())
            result[gid] = free
    return result


def _wait_for_free_gpu(
    gpu_ids: List[str],
    busy: Dict[str, subprocess.Popen],
    min_mem: int,
) -> str:
    while True:
        for gid in list(busy):
            if busy[gid].poll() is not None:
                del busy[gid]
        free_mem = _gpu_free_memory()
        for gid in gpu_ids:
            if gid in busy:
                continue
            mem = free_mem.get(gid, 0)
            if mem >= min_mem:
                return gid
        time.sleep(GPU_POLL_INTERVAL)


# SECTION: launch_shards


def launch_shards(tasks, gpu_ids, shard_dir, prm=False,
                  logprob=False):
    if not tasks:
        return []
    shard_dir.mkdir(parents=True, exist_ok=True)
    ns = len(gpu_ids)
    shards = [[] for _ in range(ns)]
    for i, t in enumerate(tasks):
        shards[i % ns].append(t)

    script = str(Path(__file__).resolve())
    if prm:
        min_mem = MIN_FREE_MEM_MIB_PRM
    else:
        min_mem = MIN_FREE_MEM_MIB_GEN

    busy: Dict[str, subprocess.Popen] = {}
    all_procs: List[Tuple[int, str, subprocess.Popen, Any]] = []
    out_files = []

    for si in range(ns):
        if not shards[si]:
            continue
        for t in shards[si]:
            t["_out"] = str(shard_dir / f"shard_{si}.jsonl")
        tf = shard_dir / f"tasks_{si}.json"
        tf.write_text(json.dumps(shards[si]), encoding="utf-8")
        out_files.append(shard_dir / f"shard_{si}.jsonl")

        gid = _wait_for_free_gpu(gpu_ids, busy, min_mem)
        cmd = [PYTHON, script,
               "--_shard-id", str(si), "--_task-file", str(tf),
               "--_gpu", gid, "--dataset", DATASET]
        if prm:
            cmd.append("--_prm")
        if logprob:
            cmd.append("--_logprob")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        env["TOKENIZERS_PARALLELISM"] = "false"
        lf = (shard_dir / f"log_{si}.txt").open("w")
        p = subprocess.Popen(
            cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        busy[gid] = p
        all_procs.append((si, gid, p, lf))
        print(f"  Shard {si} on GPU {gid}: "
              f"{len(shards[si])} tasks")

    for si, gid, p, lf in all_procs:
        p.wait()
        lf.close()
        log = (shard_dir / f"log_{si}.txt").read_text(
            errors="replace")
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


# SECTION: signal_computation


def step_char_bounds(response, steps):
    bounds, pos = [], 0
    for s in steps:
        idx = response.find(s, pos)
        if idx < 0:
            idx = pos
        bounds.append((idx, idx + len(s)))
        pos = idx + len(s)
    return bounds


def compute_step_mean_nll(lps, offs, bounds):
    nlls = []
    ti = 0
    for s0, s1 in bounds:
        while ti < len(offs) and offs[ti] < s0:
            ti += 1
        slps = []
        scan = ti
        while scan < len(offs) and offs[scan] < s1:
            slps.append(lps[scan])
            scan += 1
        nlls.append(-np.mean(slps) if slps else 0.0)
    return nlls


def compute_rollback_step_prm_drop(scores, n, threshold):
    """PRM-drop: first big drop. Returns dict with three variants."""
    result = {}
    triggered_step = None
    if n >= 2 and len(scores) >= 2:
        limit = min(len(scores), n)
        drops = [scores[i] - scores[i + 1]
                 for i in range(limit - 1)]
        for i, d in enumerate(drops):
            if d > threshold:
                triggered_step = i + 1
                break
    if triggered_step is not None:
        result["prm_drop"] = triggered_step
        result["prm_drop_fb_last"] = triggered_step
        result["prm_drop_fb_start"] = triggered_step
    else:
        result["prm_drop"] = max(n - 1, 0)
        result["prm_drop_fb_last"] = max(n - 1, 0)
        result["prm_drop_fb_start"] = 0
    return result


def compute_rollback_step_nll_drop(nlls, n, threshold):
    """NLL-drop: first big NLL increase. Returns dict with three variants."""
    result = {}
    triggered_step = None
    if n >= 2 and len(nlls) >= 2:
        deltas = [nlls[i] - nlls[i - 1]
                  for i in range(1, len(nlls))]
        for i, d in enumerate(deltas):
            if d > threshold:
                triggered_step = i + 1
                break
    if triggered_step is not None:
        result["nll_drop"] = triggered_step
        result["nll_drop_fb_last"] = triggered_step
        result["nll_drop_fb_start"] = triggered_step
    else:
        result["nll_drop"] = max(n - 1, 0)
        result["nll_drop_fb_last"] = max(n - 1, 0)
        result["nll_drop_fb_start"] = 0
    return result


def compute_rollback_step_gpt(tau, n):
    """GPT signal: tau is 1-indexed first-error step, -1 means no clear error.
    Returns dict with two fallback variants."""
    result = {}
    if tau is not None and tau >= 1:
        rb = max(0, min(int(tau) - 1, n - 1))
        result["gpt_fb_last"] = rb
        result["gpt_fb_start"] = rb
    else:
        # tau is None (API failure) or -1 (no clear error) -> fallback
        result["gpt_fb_last"] = max(n - 1, 0)
        result["gpt_fb_start"] = 0
    return result


# SECTION: evaluation


def _vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def evaluate_configs(
    questions, draft_map, prm_map, sfx_map,
    sc_recs, budget, active_signals,
):
    nq = len(questions)
    results = []

    # Greedy baseline: single draft (draft_idx=0), no voting
    greedy_correct = 0
    greedy_toks = 0
    for q in questions:
        d = draft_map.get((q["doc_id"], 0))
        if d and check_answer(DATASET, d.get("draft_answer", ""),
                              q["gold_answer"]):
            greedy_correct += 1
        greedy_toks += d.get("draft_tokens", 0) if d else 0
    results.append(dict(
        method="Greedy@1", nd=1, ns=0,
        acc=greedy_correct / nq,
        tokens_per_q=greedy_toks / nq,
        total_tokens=greedy_toks,
    ))

    sc_by_doc = defaultdict(list)
    for s in sc_recs:
        sc_by_doc[s["doc_id"]].append(s)
    for N in sorted(set([budget, 8, 16, 32, 40])):
        correct = 0
        total_toks = 0
        for q in questions:
            recs = sc_by_doc.get(q["doc_id"], [])[:N]
            answers = [r["sc_answer"] for r in recs]
            toks = sum(r["sc_tokens"] for r in recs)
            if check_answer(DATASET, _vote(answers),
                            q["gold_answer"]):
                correct += 1
            total_toks += toks
        results.append(dict(
            method=f"SC@{N}", nd=0, ns=N,
            acc=correct / nq,
            tokens_per_q=total_toks / nq,
            total_tokens=total_toks,
        ))

    all_strategies = set()
    for prm_rec in prm_map.values():
        for strat in prm_rec.get("_rb_steps", {}):
            all_strategies.add(strat)
    if not all_strategies:
        all_strategies = set(active_signals)

    for strat in sorted(all_strategies & set(active_signals)):
        for nd, ns in ROLLBACK_CONFIGS:
            if nd * ns != budget:
                continue
            ns_fair = ns - 1
            for variant, ns_use in [("full", ns),
                                    ("fair", ns_fair)]:
                correct = 0
                total_toks = 0
                total_prm_toks = 0
                for q in questions:
                    did = q["doc_id"]
                    answers = []
                    q_toks = 0
                    q_prm_toks = 0
                    for di in range(nd):
                        d = draft_map.get((did, di))
                        if not d:
                            continue
                        answers.append(d["draft_answer"])
                        q_toks += d["draft_tokens"]

                        prm_rec = prm_map.get((did, di))
                        if prm_rec:
                            q_prm_toks += prm_rec.get(
                                "prm_tokens", 0)
                        rb_steps = (prm_rec.get("_rb_steps", {})
                                    if prm_rec else {})
                        rb_step = rb_steps.get(strat)
                        if rb_step is not None:
                            for si in range(ns_use):
                                s = sfx_map.get(
                                    (did, di, strat,
                                     rb_step, si))
                                if s:
                                    answers.append(
                                        s["suffix_answer"])
                                    q_toks += s[
                                        "suffix_tokens"]

                    if check_answer(
                        DATASET, _vote(answers), q["gold_answer"]
                    ):
                        correct += 1
                    total_toks += q_toks
                    total_prm_toks += q_prm_toks

                n_answers = nd + nd * ns_use
                tag = (f"rollback_{strat}_nd{nd}_ns{ns}"
                       if variant == "full"
                       else f"rollback_{strat}_nd{nd}_ns{ns}_fair")
                results.append(dict(
                    method=tag,
                    strategy=strat,
                    variant=variant,
                    nd=nd, ns=ns,
                    n_answers=n_answers,
                    acc=correct / nq,
                    tokens_per_q=total_toks / nq,
                    total_tokens=total_toks,
                    prm_tokens=total_prm_toks,
                ))

    return results


# SECTION: figures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_figures(eval_results, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    sc_rows = [r for r in eval_results
               if r["method"].startswith("SC@")]
    rb_rows = [r for r in eval_results
               if r["method"].startswith("rollback_")]

    sc32 = next(
        (r for r in sc_rows if r["method"] == "SC@32"), None)
    if sc32 is None:
        return

    strat_colors = {
        "gpt_fb_last": "#27ae60",
        "gpt_fb_start": "#1abc9c",
        "prm_drop": "#e74c3c",
        "prm_drop_fb_last": "#c0392b",
        "prm_drop_fb_start": "#e67e22",
        "nll_drop": "#3498db",
        "nll_drop_fb_last": "#2980b9",
        "nll_drop_fb_start": "#8e44ad",
    }
    strat_markers = {
        "gpt_fb_last": "v",
        "gpt_fb_start": "<",
        "prm_drop": "o",
        "prm_drop_fb_last": "s",
        "prm_drop_fb_start": "p",
        "nll_drop": "D",
        "nll_drop_fb_last": "d",
        "nll_drop_fb_start": "h",
    }

    strategies = sorted(set(
        r.get("strategy", "prm_drop") for r in rb_rows))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    for r in sc_rows:
        ax.scatter(r["tokens_per_q"], r["acc"],
                   c="#888888", s=60, marker="s",
                   zorder=5, edgecolors="white", lw=0.5)
        ax.annotate(r["method"],
                    (r["tokens_per_q"], r["acc"]),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    plotted = set()
    for r in rb_rows:
        strat = r.get("strategy", "prm_drop")
        c = strat_colors.get(strat, "#e74c3c")
        m = strat_markers.get(strat, "o")
        label = strat if strat not in plotted else None
        plotted.add(strat)
        ax.scatter(r["tokens_per_q"], r["acc"], c=c, s=80,
                   marker=m, zorder=5, edgecolors="white",
                   lw=0.5, label=label)
        short = r["method"].replace("rollback_", "")
        ax.annotate(short, (r["tokens_per_q"], r["acc"]),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=7)

    for strat in strategies:
        s_rows = sorted(
            [r for r in rb_rows
             if r.get("strategy") == strat],
            key=lambda r: r["tokens_per_q"])
        if len(s_rows) > 1:
            rx = [r["tokens_per_q"] for r in s_rows]
            ry = [r["acc"] for r in s_rows]
            c = strat_colors.get(strat, "#e74c3c")
            ax.plot(rx, ry, "--", color=c, alpha=0.4, lw=1)

    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Tokens per Question")
    ax.set_ylabel("Majority-Vote Accuracy")
    ax.set_title("Efficiency Frontier: Multi-Signal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_efficiency_frontier.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures saved to {fig_dir}")


def save_summary_table(eval_results, out_dir, phase_times=None,
                       n_drafts=0, nq=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    sc32 = next((r for r in eval_results
                 if r["method"] == "SC@32"), None)
    if sc32 is None:
        return
    base_tpq = sc32["tokens_per_q"]
    rows = []
    for r in eval_results:
        savings = (1 - r["tokens_per_q"] / base_tpq) * 100
        rows.append({
            "method": r["method"],
            "strategy": r.get("strategy", ""),
            "nd": r.get("nd", ""),
            "ns": r.get("ns", ""),
            "acc": r["acc"],
            "tokens_per_q": r["tokens_per_q"],
            "savings_vs_sc32": savings,
        })

    md = [
        "| method | strategy | nd | ns | acc "
        "| tokens_per_q | savings_vs_sc32 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['strategy']} "
            f"| {r['nd']} | {r['ns']} "
            f"| {r['acc']:.4f} | {r['tokens_per_q']:.1f} "
            f"| {r['savings_vs_sc32']:.1f}% |"
        )

    if phase_times:
        md.append("")
        md.append("## Signal overhead (wall-clock)")
        md.append("")
        md.append("| signal | total_sec | ms_per_sample "
                   "| ms_per_question |")
        md.append("|---|---:|---:|---:|")
        for key, label in [("prm", "PRM"), ("nll", "NLL"),
                           ("gpt", "GPT")]:
            t = phase_times.get(key, 0.0)
            ps = t / n_drafts * 1000 if n_drafts else 0
            pq = t / nq * 1000 if nq else 0
            md.append(f"| {label} | {t:.1f} | {ps:.1f} "
                      f"| {pq:.1f} |")
        for key, label in [("draft", "Draft gen"),
                           ("suffix", "Suffix gen")]:
            t = phase_times.get(key, 0.0)
            md.append(f"| {label} | {t:.1f} | - | - |")

    (out_dir / "eval_summary_table.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"Saved table: {out_dir / 'eval_summary_table.md'}")


# SECTION: main


def main():
    global DATASET, MODEL_ID
    args = parse_args()
    DATASET = args.dataset
    MODEL_ID = args.model

    if args._shard_id >= 0:
        if args._prm:
            run_prm_shard(args)
        elif args._logprob:
            run_logprob_shard(args)
        else:
            run_shard(args)
        return

    signals = args.signals
    prm_thr = args.prm_drop_threshold
    nll_thr = args.nll_drop_threshold

    # Filter ROLLBACK_CONFIGS by --nd if specified
    global ROLLBACK_CONFIGS
    if args.nd:
        allowed = set(args.nd)
        ROLLBACK_CONFIGS = [(nd, ns) for nd, ns in ROLLBACK_CONFIGS
                            if nd in allowed]
        print(f"Filtered ROLLBACK_CONFIGS by --nd {args.nd}: "
              f"{ROLLBACK_CONFIGS}")

    gpu_ids = [g.strip() for g in args.gpus.split(",")
               if g.strip()]
    B = args.budget
    nd_max = max(nd for nd, ns in ROLLBACK_CONFIGS
                 if nd * ns == B)
    ns_max = max(ns for nd, ns in ROLLBACK_CONFIGS
                 if nd * ns == B)
    sc_n = B

    ns_for_draft = {}
    for di in range(nd_max):
        ns_for_draft[di] = max(
            ns for nd, ns in ROLLBACK_CONFIGS
            if nd * ns == B and di < nd
        )

    suffix = f"_{args.tag}" if args.tag else ""
    model_short = Path(MODEL_ID).name.lower().replace("-", "_")
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "results"
        / f"{model_short}_budget_multisignal{suffix}"
        / DATASET)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (PROJECT_ROOT / "figures"
               / f"budget_multisignal{suffix}")
    ckpt = out_dir / "checkpoint.jsonl"
    sd = out_dir / "_shards"

    questions = load_dataset_by_name(
        DATASET, args.n_sample, seed=42)
    nq = len(questions)
    q_map = {q["doc_id"]: q for q in questions}
    print(f"Dataset: {DATASET}, {nq} questions, "
          f"GPUs: {gpu_ids}")
    print(f"Budget B={B}, signals={signals}")

    existing = []
    if ckpt.exists():
        existing = [json.loads(l)
                    for l in ckpt.read_text().splitlines()
                    if l.strip()]
    print(f"Existing checkpoint: {len(existing)} records")

    # Timing bookkeeping — cumulative across runs
    timing_path = out_dir / "phase_times.json"
    if timing_path.exists():
        phase_times = json.loads(timing_path.read_text())
    else:
        phase_times = {}

    def _record_time(phase_name, elapsed):
        phase_times[phase_name] = phase_times.get(phase_name, 0.0) + elapsed
        timing_path.write_text(json.dumps(phase_times, indent=2))

    # ---- Phase 1: Sample drafts ----
    _t0 = time.time()
    if 1 not in args.skip_phase:
        done = {(r["doc_id"], r["draft_idx"])
                for r in existing
                if r.get("task_type") == "draft"}
        tasks = []
        for q in questions:
            p = build_prompt(MODEL_ID, DATASET,
                             q["question"],
                             context=q.get("context", ""))
            for di in range(nd_max):
                if (q["doc_id"], di) in done:
                    continue
                tasks.append(dict(
                    task_type="draft",
                    doc_id=q["doc_id"],
                    draft_idx=di,
                    gold_answer=q["gold_answer"],
                    prompt=p,
                ))
        if tasks:
            print(f"\n--- Phase 1: {len(tasks)} drafts ---")
            new = launch_shards(tasks, gpu_ids, sd / "p1")
            with ckpt.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(new)

    _record_time("draft", time.time() - _t0)

    drafts = [r for r in existing
              if r.get("task_type") == "draft"]
    draft_map = {(r["doc_id"], r["draft_idx"]): r
                 for r in drafts}
    print(f"Total drafts: {len(drafts)}")

    # ---- Phase 2: PRM scoring (if prm_drop in signals) ----
    _t0 = time.time()
    need_prm = any(s.startswith("prm_drop") for s in signals)
    if need_prm and 2 not in args.skip_phase:
        done = {(r["doc_id"], r.get("draft_idx", 0))
                for r in existing
                if r.get("task_type") == "prm"}
        prm_tasks = []
        for q in questions:
            for di in range(nd_max):
                if (q["doc_id"], di) in done:
                    continue
                d = draft_map.get((q["doc_id"], di))
                if not d or not d.get("draft_steps"):
                    continue
                steps = d["draft_steps"]
                step_text = ("<extra_0>".join(steps)
                             + "<extra_0>")
                messages = [
                    {"role": "system",
                     "content": "Please reason step by step, "
                     "and put your final answer within "
                     "\\boxed{}."},
                    {"role": "user",
                     "content": q["question"]},
                    {"role": "assistant",
                     "content": step_text},
                ]
                prm_tasks.append(dict(
                    task_type="prm",
                    doc_id=q["doc_id"],
                    draft_idx=di,
                    n_steps=len(steps),
                    prompt=messages,
                ))
        if prm_tasks:
            print(f"\n--- Phase 2: {len(prm_tasks)} PRM ---")
            new = launch_shards(
                prm_tasks, gpu_ids, sd / "p2", prm=True)
            with ckpt.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(new)

    _record_time("prm", time.time() - _t0)

    prm_recs = [r for r in existing
                if r.get("task_type") == "prm"]
    prm_map = {}
    for r in prm_recs:
        prm_map[(r["doc_id"], r.get("draft_idx", 0))] = r
    print(f"Total PRM records: {len(prm_recs)}")

    # ---- Phase 3: Logprob collection (if nll_drop) ----
    _t0 = time.time()
    need_lp = any(s.startswith("nll_drop") for s in signals)
    if need_lp and 3 not in args.skip_phase:
        done = {(r["doc_id"], r.get("draft_idx", 0))
                for r in existing
                if r.get("task_type") == "logprob"}
        lp_tasks = []
        for q in questions:
            for di in range(nd_max):
                if (q["doc_id"], di) in done:
                    continue
                d = draft_map.get((q["doc_id"], di))
                if not d or not d.get("draft_steps"):
                    continue
                p = build_prompt(MODEL_ID, DATASET,
                                 q["question"],
                                 context=q.get("context", ""))
                full = p + d["draft_text"]
                lp_tasks.append(dict(
                    task_type="logprob",
                    doc_id=q["doc_id"],
                    draft_idx=di,
                    full_prompt=full,
                    resp_char_offset=len(p),
                    n_steps=d.get("n_steps", 0),
                ))
        if lp_tasks:
            print(f"\n--- Phase 3: {len(lp_tasks)} "
                  f"logprob ---")
            new = launch_shards(
                lp_tasks, gpu_ids, sd / "p3",
                logprob=True)
            with ckpt.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(new)

    _record_time("nll", time.time() - _t0)

    lp_recs = [r for r in existing
               if r.get("task_type") == "logprob"]
    lp_map = {(r["doc_id"], r.get("draft_idx", 0)): r
              for r in lp_recs}
    print(f"Total logprob records: {len(lp_recs)}")

    # ---- Phase 2.5: GPT first-error detection (live API) ----
    _t0 = time.time()
    need_gpt = any(s.startswith("gpt") for s in signals)
    gpt_cache_path = out_dir / "gpt_first_error_cache.jsonl"
    gpt_tau_map: Dict[Tuple, int] = {}

    if need_gpt:
        load_env_file(PROJECT_ROOT / ".env")

        # Warm-start from existing cache (local or user-specified)
        warm_cache: Dict[str, dict] = {}
        for cp in [gpt_cache_path, Path(args.gpt_cache) if args.gpt_cache else None]:
            if cp and cp.exists():
                for line in cp.read_text("utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = f"{rec['doc_id']}|{rec.get('draft_idx', rec.get('sample_idx', 0))}"
                    warm_cache[key] = rec
        print(f"GPT warm cache: {len(warm_cache)} entries")

        # Build tasks for drafts not yet in cache
        gpt_tasks = []
        if 25 not in args.skip_phase:
            for q in questions:
                for di in range(nd_max):
                    d = draft_map.get((q["doc_id"], di))
                    if not d or not d.get("draft_steps"):
                        continue
                    key = f"{q['doc_id']}|{di}"
                    if key in warm_cache:
                        continue
                    gpt_tasks.append(dict(
                        doc_id=q["doc_id"],
                        draft_idx=di,
                        question=q["question"],
                        gold_answer=q["gold_answer"],
                        steps=d["draft_steps"],
                        n_steps=d.get("n_steps", len(d["draft_steps"])),
                    ))

        if gpt_tasks:
            print(f"\n--- Phase 2.5: {len(gpt_tasks)} GPT "
                  f"first-error calls ({args.gpt_model}) ---")
            gpt_client = make_client()

            def _gpt_process(task):
                prompt = build_first_error_prompt(
                    task["question"], task["gold_answer"],
                    task["steps"])
                parsed, raw = call_gpt_first_error(
                    gpt_client, args.gpt_model, prompt,
                    temperature=args.gpt_temperature)
                tau = None
                if (parsed and
                        isinstance(parsed.get("first_error_step"), int)):
                    tau = parsed["first_error_step"]
                return dict(
                    doc_id=task["doc_id"],
                    draft_idx=task["draft_idx"],
                    tau=tau,
                    gpt_parsed=parsed,
                    gpt_raw=raw,
                )

            from tqdm.auto import tqdm as _tqdm
            if args.gpt_max_workers <= 1:
                for t in _tqdm(gpt_tasks, desc="GPT first-error"):
                    rec = _gpt_process(t)
                    key = f"{rec['doc_id']}|{rec['draft_idx']}"
                    warm_cache[key] = rec
                    with gpt_cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                with ThreadPoolExecutor(
                        max_workers=args.gpt_max_workers) as pool:
                    futs = {pool.submit(_gpt_process, t): t
                            for t in gpt_tasks}
                    for fut in _tqdm(as_completed(futs),
                                     total=len(futs),
                                     desc="GPT first-error"):
                        rec = fut.result()
                        key = f"{rec['doc_id']}|{rec['draft_idx']}"
                        warm_cache[key] = rec
                        with gpt_cache_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"GPT cache now: {len(warm_cache)} entries")

        # Build gpt_tau_map from warm_cache
        for key, rec in warm_cache.items():
            parts = key.split("|")
            doc_id, di = parts[0], int(parts[1])
            gpt_tau_map[(doc_id, di)] = rec.get("tau")

    _record_time("gpt", time.time() - _t0)

    # ---- Compute rollback points per signal ----
    for q in questions:
        for di in range(nd_max):
            d = draft_map.get((q["doc_id"], di))
            prm_rec = prm_map.get((q["doc_id"], di))
            if not d:
                continue
            n = d.get("n_steps", 0)
            if prm_rec is None:
                prm_rec = {"doc_id": q["doc_id"],
                           "draft_idx": di}
                prm_map[(q["doc_id"], di)] = prm_rec

            rb_steps = {}

            if any(s.startswith("prm_drop") for s in signals):
                scores = prm_rec.get("step_scores", [])
                rb_steps.update(
                    compute_rollback_step_prm_drop(
                        scores, n, prm_thr))

            if any(s.startswith("nll_drop") for s in signals):
                lp_rec = lp_map.get((q["doc_id"], di))
                if lp_rec and d.get("draft_steps"):
                    bounds = step_char_bounds(
                        d["draft_text"], d["draft_steps"])
                    nlls = compute_step_mean_nll(
                        lp_rec["token_logprobs"],
                        lp_rec["token_offsets"], bounds)
                    rb_steps.update(
                        compute_rollback_step_nll_drop(
                            nlls, n, nll_thr))
                else:
                    rb_steps["nll_drop"] = max(n - 1, 0)
                    rb_steps["nll_drop_fb_last"] = max(n - 1, 0)
                    rb_steps["nll_drop_fb_start"] = 0

            if any(s.startswith("gpt") for s in signals):
                tau = gpt_tau_map.get((q["doc_id"], di))
                rb_steps.update(
                    compute_rollback_step_gpt(tau, n))

            prm_rec["_rb_steps"] = rb_steps

    # ---- Phase 4: Suffix generation ----
    _t0 = time.time()
    if 4 not in args.skip_phase:
        done_sfx = set()
        for r in existing:
            if r.get("task_type") == "suffix":
                done_sfx.add((
                    r["doc_id"], r["draft_idx"],
                    r.get("strategy", "prm_drop"),
                    r["rollback_step"],
                    r["suffix_idx"]))
        sfx_tasks = []
        for q in questions:
            did = q["doc_id"]
            for di in range(nd_max):
                d = draft_map.get((did, di))
                prm_rec = prm_map.get((did, di))
                if not d or not prm_rec:
                    continue
                rb_steps = prm_rec.get("_rb_steps", {})
                if not rb_steps:
                    continue
                steps = d["draft_steps"]
                for strat, rb in rb_steps.items():
                    if strat not in signals:
                        continue
                    b = min(rb, len(steps) - 1)
                    if b < 0:
                        b = 0
                    if b == 0:
                        prefix = ""
                    else:
                        prefix = ("\n\n".join(steps[:b])
                                  + "\n\n")
                    p = build_prompt(
                        MODEL_ID, DATASET, q["question"],
                        context=q.get("context", ""))
                    p += prefix
                    for si in range(ns_for_draft[di]):
                        key = (did, di, strat, b, si)
                        if key in done_sfx:
                            continue
                        sfx_tasks.append(dict(
                            task_type="suffix",
                            doc_id=did,
                            draft_idx=di,
                            strategy=strat,
                            rollback_step=b,
                            suffix_idx=si,
                            gold_answer=q["gold_answer"],
                            prompt=p,
                        ))
        if sfx_tasks:
            print(f"\n--- Phase 4: {len(sfx_tasks)} "
                  f"suffix generations ---")
            new = launch_shards(
                sfx_tasks, gpu_ids, sd / "p4")
            with ckpt.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(new)

    _record_time("suffix", time.time() - _t0)

    sfx_recs = [r for r in existing
                if r.get("task_type") == "suffix"]
    sfx_map = {}
    for s in sfx_recs:
        sfx_map[(s["doc_id"], s["draft_idx"],
                 s.get("strategy", "prm_drop"),
                 s["rollback_step"], s["suffix_idx"])] = s
    print(f"Total suffix records: {len(sfx_recs)}")

    # ---- Phase 5: Full SC baseline ----
    if 5 not in args.skip_phase:
        done_sc = {(r["doc_id"], r["sc_idx"])
                   for r in existing
                   if r.get("task_type") == "fullsc"}
        sc_tasks = []
        for q in questions:
            p = build_prompt(MODEL_ID, DATASET,
                             q["question"],
                             context=q.get("context", ""))
            for si in range(sc_n):
                if (q["doc_id"], si) in done_sc:
                    continue
                sc_tasks.append(dict(
                    task_type="fullsc",
                    doc_id=q["doc_id"],
                    sc_idx=si,
                    gold_answer=q["gold_answer"],
                    prompt=p,
                ))
        if sc_tasks:
            print(f"\n--- Phase 5: {len(sc_tasks)} SC ---")
            new = launch_shards(
                sc_tasks, gpu_ids, sd / "p5")
            with ckpt.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(new)

    sc_recs = [r for r in existing
               if r.get("task_type") == "fullsc"]
    print(f"Total SC records: {len(sc_recs)}")

    # ---- Evaluate ----
    print("\n--- Evaluation ---")

    # Signal overhead timing summary
    n_drafts = len(drafts)
    nq = len(questions)
    print(f"\n--- Signal overhead (wall-clock) ---")
    for phase_name, label in [
        ("prm", "PRM scoring"),
        ("nll", "NLL/logprob"),
        ("gpt", "GPT first-error"),
    ]:
        t = phase_times.get(phase_name, 0.0)
        per_sample = t / n_drafts * 1000 if n_drafts else 0
        per_q = t / nq * 1000 if nq else 0
        print(f"  {label:<20s}: {t:8.1f}s total | "
              f"{per_sample:.1f}ms/sample | {per_q:.1f}ms/question")
    for phase_name, label in [
        ("draft", "Draft generation"),
        ("suffix", "Suffix generation"),
    ]:
        t = phase_times.get(phase_name, 0.0)
        print(f"  {label:<20s}: {t:8.1f}s total")

    # Re-extract answers from raw text to pick up any
    # extract_answer improvements without re-generating.
    _n_re = 0
    for d in draft_map.values():
        if "draft_text" in d:
            d["draft_answer"] = extract_answer(DATASET, d["draft_text"])
            _n_re += 1
    for s in sfx_map.values():
        if "suffix_text" in s:
            s["suffix_answer"] = extract_answer(DATASET, s["suffix_text"])
            _n_re += 1
    for s in sc_recs:
        if "sc_text" in s:
            s["sc_answer"] = extract_answer(DATASET, s["sc_text"])
            _n_re += 1
    if _n_re:
        print(f"  Re-extracted answers for {_n_re} records")

    eval_results = evaluate_configs(
        questions, draft_map, prm_map, sfx_map,
        sc_recs, B, signals)

    print(f"\n{'Method':<35} {'Acc':>8} "
          f"{'Tok/Q':>10} {'TotalTok':>12}")
    print("-" * 68)
    for r in eval_results:
        print(f"{r['method']:<35} {r['acc']:>8.4f} "
              f"{r['tokens_per_q']:>10.0f} "
              f"{r['total_tokens']:>12,}")

    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(eval_results, indent=2,
                   ensure_ascii=False))
    print(f"\nSaved: {summary_path}")

    make_figures(eval_results, fig_dir)
    save_summary_table(eval_results, out_dir, phase_times,
                       n_drafts, nq)
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()