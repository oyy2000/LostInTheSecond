#!/usr/bin/env python3
"""
Budget-controlled rollback with pluggable error-step signals.

Extends 6_6 by supporting multiple rollback signal strategies:
  - gpt:       GPT-annotated first-error step (tau from cache)
  - prm_drop:  first i where score[i]-score[i+1] > threshold
               fallback to n-1 or 0 (not argmin)
  - nll_drop:  first i where nll[i]-nll[i-1] > threshold
               fallback to n-1 or 0 (not argmax)

Each signal produces a rollback step per draft. Suffix generation and
majority-vote evaluation proceed identically to 6_6.

Usage:
    python scripts/6_8_budget_controlled_multisignal.py \
        --budget 32 --gpus 0,1 \
        --signals gpt prm_drop nll_drop
"""

import argparse, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict
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
GPU_MEM = 0.85

ROLLBACK_CONFIGS = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2)]

ALL_SIGNALS = ["gpt", "prm_drop", "nll_drop"]

from src.prompt_templates import (
    build_prompt, get_stop_tokens, split_steps,
    extract_answer, check_answer,
)
from src.sweep_datasets import load_dataset_by_name


# SECTION: parse_args


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--n-sample", type=int, default=0,
                    help="Limit questions; 0=all")
    ap.add_argument("--signals", nargs="+",
                    default=["gpt", "prm_drop", "nll_drop"],
                    choices=ALL_SIGNALS,
                    help="Which rollback signals to evaluate")
    ap.add_argument("--prm-drop-threshold", type=float,
                    default=PRM_DROP_THRESHOLD)
    ap.add_argument("--nll-drop-threshold", type=float,
                    default=NLL_DROP_THRESHOLD)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpt-cache", default="",
                    help="Path to gpt_first_error_cache.jsonl")
    ap.add_argument("--skip-phase", type=int, nargs="*", default=[],
                    help="Skip phases (1=drafts, 2=PRM, 3=logprob, "
                         "4=suffix, 5=SC)")
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
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=1)
    BATCH = 32
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
    """PRM-drop signal: first big drop, fallback n-1 or 0."""
    if n < 2 or len(scores) < 2:
        return 0, "fallback_short"
    limit = min(len(scores), n)
    drops = [scores[i] - scores[i + 1]
             for i in range(limit - 1)]
    for i, d in enumerate(drops):
        if d > threshold:
            return i + 1, "prm_drop"
    return n - 1, "fallback_last"


def compute_rollback_step_nll_drop(nlls, n, threshold):
    """NLL-drop signal: first big NLL increase, fallback n-1 or 0."""
    if n < 2 or len(nlls) < 2:
        return 0, "fallback_short"
    deltas = [nlls[i] - nlls[i - 1] for i in range(1, len(nlls))]
    for i, d in enumerate(deltas):
        if d > threshold:
            return i + 1, "nll_drop"
    return n - 1, "fallback_last"


def compute_rollback_step_gpt(tau, n):
    """GPT signal: tau is 1-indexed first-error step."""
    if tau is None or tau < 1:
        return n - 1, "gpt_missing"
    rb = int(tau) - 1
    rb = max(0, min(rb, n - 1))
    return rb, "gpt"


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

    for strat in sorted(all_strategies):
        for nd, ns in ROLLBACK_CONFIGS:
            if nd * ns != budget:
                continue
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
                        for si in range(ns):
                            s = sfx_map.get(
                                (did, di, strat, rb_step, si))
                            if s:
                                answers.append(
                                    s["suffix_answer"])
                                q_toks += s["suffix_tokens"]

                if check_answer(
                    DATASET, _vote(answers), q["gold_answer"]
                ):
                    correct += 1
                total_toks += q_toks
                total_prm_toks += q_prm_toks

            results.append(dict(
                method=f"rollback_{strat}_nd{nd}_ns{ns}",
                strategy=strat,
                nd=nd, ns=ns,
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
        "gpt": "#2ecc71",
        "prm_drop": "#e74c3c",
        "nll_drop": "#3498db",
    }
    strat_markers = {
        "gpt": "^",
        "prm_drop": "o",
        "nll_drop": "D",
    }

    strategies = sorted(set(
        r.get("strategy", "prm_drop") for r in rb_rows))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
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


def save_summary_table(eval_results, out_dir):
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
    (out_dir / "eval_summary_table.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"Saved table: {out_dir / 'eval_summary_table.md'}")


# SECTION: main


def main():
    global DATASET
    args = parse_args()
    DATASET = args.dataset

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
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "results"
        / f"{DATASET}_budget_multisignal{suffix}")
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

    # ---- Phase 1: Sample drafts ----
    if 1 not in args.skip_phase:
        done = {(r["doc_id"], r["draft_idx"])
                for r in existing
                if r.get("task_type") == "draft"}
        tasks = []
        for q in questions:
            p = build_prompt(MODEL_ID, DATASET,
                             q["question"])
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

    drafts = [r for r in existing
              if r.get("task_type") == "draft"]
    draft_map = {(r["doc_id"], r["draft_idx"]): r
                 for r in drafts}
    print(f"Total drafts: {len(drafts)}")

    # ---- Phase 2: PRM scoring (if prm_drop in signals) ----
    need_prm = "prm_drop" in signals
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

    prm_recs = [r for r in existing
                if r.get("task_type") == "prm"]
    prm_map = {}
    for r in prm_recs:
        prm_map[(r["doc_id"], r.get("draft_idx", 0))] = r
    print(f"Total PRM records: {len(prm_recs)}")

    # ---- Phase 3: Logprob collection (if nll_drop) ----
    need_lp = "nll_drop" in signals
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
                                 q["question"])
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

    lp_recs = [r for r in existing
               if r.get("task_type") == "logprob"]
    lp_map = {(r["doc_id"], r.get("draft_idx", 0)): r
              for r in lp_recs}
    print(f"Total logprob records: {len(lp_recs)}")

    # ---- Load GPT cache (if gpt in signals) ----
    gpt_map = {}
    if "gpt" in signals:
        gpt_path = args.gpt_cache
        if not gpt_path:
            gpt_path = str(
                PROJECT_ROOT / "results"
                / f"{DATASET}_3b_multi_sample"
                / "first_error"
                / "gpt_first_error_cache.jsonl")
        gp = Path(gpt_path)
        if gp.exists():
            gpt_recs = [json.loads(l)
                        for l in gp.read_text().splitlines()
                        if l.strip()]
            for r in gpt_recs:
                gpt_map[(f"{DATASET}_{r['doc_id']}",
                         r["sample_idx"])] = r
            print(f"GPT cache: {len(gpt_recs)} entries")
        else:
            print(f"GPT cache not found: {gp}")

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

            if "prm_drop" in signals:
                scores = prm_rec.get("step_scores", [])
                rb, sig = compute_rollback_step_prm_drop(
                    scores, n, prm_thr)
                rb_steps["prm_drop"] = rb

            if "nll_drop" in signals:
                lp_rec = lp_map.get((q["doc_id"], di))
                if lp_rec and d.get("draft_steps"):
                    bounds = step_char_bounds(
                        d["draft_text"], d["draft_steps"])
                    nlls = compute_step_mean_nll(
                        lp_rec["token_logprobs"],
                        lp_rec["token_offsets"], bounds)
                    rb, sig = compute_rollback_step_nll_drop(
                        nlls, n, nll_thr)
                    rb_steps["nll_drop"] = rb
                else:
                    rb_steps["nll_drop"] = max(n - 1, 0)

            if "gpt" in signals:
                g = gpt_map.get((q["doc_id"], di))
                tau = g.get("tau") if g else None
                rb, sig = compute_rollback_step_gpt(tau, n)
                rb_steps["gpt"] = rb

            prm_rec["_rb_steps"] = rb_steps

    # ---- Phase 4: Suffix generation ----
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
                    b = min(rb, len(steps) - 1)
                    if b < 0:
                        b = 0
                    if b == 0:
                        prefix = ""
                    else:
                        prefix = ("\n\n".join(steps[:b])
                                  + "\n\n")
                    p = build_prompt(
                        MODEL_ID, DATASET, q["question"])
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
                             q["question"])
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
    save_summary_table(eval_results, out_dir)
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()