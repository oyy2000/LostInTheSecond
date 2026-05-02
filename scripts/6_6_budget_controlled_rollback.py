#!/usr/bin/env python3
"""
Budget-controlled rollback vs from-scratch comparison.

Fixed total sample budget B = n_draft * n_suffix (default 32).
Baseline: B from-scratch completions, majority vote.
Rollback: n_draft independent sampled drafts
          -> PRM judge finds the largest step-to-step score drop per draft
          -> rollback to that point -> resample n_suffix suffixes per draft
          -> majority vote over all n_draft answers + n_draft * n_suffix suffixes.

Configurations tested (B=32):
  n_draft=2,  n_suffix=16
  n_draft=4,  n_suffix=8
  n_draft=8,  n_suffix=4
  n_draft=16, n_suffix=2

Usage:
    python scripts/6_6_budget_controlled_rollback.py \
        --budget 32 --gpus 0,1
"""

import argparse, json, math, os, re, subprocess, sys, time
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
PRM_FALLBACK_FRAC = 0.8
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_MODEL_LEN = 4096
GPU_MEM = 0.85

ROLLBACK_CONFIGS = [(2, 16), (4, 8), (8, 4), (16, 2)]

from src.prompt_templates import (
    build_prompt, get_stop_tokens, split_steps,
    extract_answer, check_answer,
)
from src.sweep_datasets import load_dataset_by_name


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--n-sample", type=int, default=0,
                    help="Limit questions; 0=all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--skip-phase", type=int, nargs="*", default=[],
                    help="Skip phases (1=drafts, 2=PRM, 3=suffix, 4=SC)")
    # shard worker args
    ap.add_argument("--_shard-id", type=int, default=-1)
    ap.add_argument("--_task-file", default="")
    ap.add_argument("--_gpu", default="0")
    ap.add_argument("--_prm", action="store_true")
    return ap.parse_args()

# SECTION: shard_worker


def run_shard(args):
    """Generate completions on one GPU. Handles draft/suffix/fullsc tasks."""
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
            t["prompt"], tokenize=False, add_generation_prompt=False)
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
    print(f"[PRM Shard {sid}] Done: {sum(1 for r in results if r)}")

# SECTION: launch_shards


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
        p.wait()
        lf.close()
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

# SECTION: evaluation


def _vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def evaluate_configs(
    questions, draft_map, prm_map, sfx_map, sc_recs, budget,
):
    """Evaluate all rollback configs and SC baselines.

    prm_map: {(doc_id, draft_idx): {"_rb_step": int, ...}}
    """
    nq = len(questions)
    results = []

    # --- Full SC baselines ---
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
            if check_answer(DATASET, _vote(answers), q["gold_answer"]):
                correct += 1
            total_toks += toks
        results.append(dict(
            method=f"SC@{N}", nd=0, ns=N,
            acc=correct / nq, tokens_per_q=total_toks / nq,
            total_tokens=total_toks,
        ))

    # --- Rollback configs ---
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
                    q_prm_toks += prm_rec.get("prm_tokens", 0)

                rb_step = prm_rec.get("_rb_step") if prm_rec else None
                if rb_step is not None:
                    for si in range(ns):
                        s = sfx_map.get((did, di, rb_step, si))
                        if s:
                            answers.append(s["suffix_answer"])
                            q_toks += s["suffix_tokens"]

            if check_answer(DATASET, _vote(answers), q["gold_answer"]):
                correct += 1
            total_toks += q_toks
            total_prm_toks += q_prm_toks

        results.append(dict(
            method=f"rollback_nd{nd}_ns{ns}",
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

    sc_rows = [r for r in eval_results if r["method"].startswith("SC@")] 
    rb_rows = [r for r in eval_results if r["method"].startswith("rollback_")]
    rb_rows.sort(key=lambda r: r["nd"])

    sc32 = next((r for r in sc_rows if r["method"] == "SC@32"), None)
    if sc32 is None:
        return

    combined_rows = [sc32] + rb_rows
    labels = [r["method"].replace("rollback_", "")
              .replace("SC@32", "SC@32\n(baseline)")
              for r in combined_rows]
    accs = [r["acc"] for r in combined_rows]
    base_tpq = sc32["tokens_per_q"]
    savings = [0.0] + [(1 - r["tokens_per_q"] / base_tpq) * 100 for r in rb_rows]
    colors = ["#3498db"] + ["#e74c3c"] * len(rb_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Panel A: Accuracy vs config ---
    bars1 = ax1.bar(range(len(labels)), accs, color=colors, alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.axhline(sc32["acc"], color="#3498db", ls="--", lw=1, alpha=0.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Majority-Vote Accuracy")
    ax1.set_title("Accuracy: SC@32 vs Rollback Configs (B=32)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(0, max(accs) * 1.15)

    # --- Panel B: Token savings ---
    bars2 = ax2.bar(range(len(labels)), savings, color=colors, alpha=0.85)
    for bar, pct in zip(bars2, savings):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Token Savings vs SC@32 (%)")
    ax2.set_title("Token Savings (Generation Only)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(min(-5, min(savings) - 5), max(savings) * 1.15 if max(savings) > 0 else 5)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_accuracy_vs_config_and_token_savings.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure: Efficiency frontier ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for r in sc_rows:
        ax.scatter(r["tokens_per_q"], r["acc"], c="#3498db", s=60,
                   marker="s", zorder=5, edgecolors="white", lw=0.5)
        ax.annotate(r["method"], (r["tokens_per_q"], r["acc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    for r in rb_rows:
        ax.scatter(r["tokens_per_q"], r["acc"], c="#e74c3c", s=80,
                   marker="o", zorder=5, edgecolors="white", lw=0.5)
        short = r["method"].replace("rollback_", "")
        ax.annotate(short, (r["tokens_per_q"], r["acc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    if len(rb_rows) > 1:
        rx = [r["tokens_per_q"] for r in rb_rows]
        ry = [r["acc"] for r in rb_rows]
        ax.plot(rx, ry, "--", color="#e74c3c", alpha=0.4, lw=1)
    ax.set_xlabel("Tokens per Question")
    ax.set_ylabel("Majority-Vote Accuracy")
    ax.set_title("Efficiency Frontier: Accuracy vs Token Cost")
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
    sc32 = next((r for r in eval_results if r["method"] == "SC@32"), None)
    if sc32 is None:
        return
    base_tpq = sc32["tokens_per_q"]
    rows = []
    for r in eval_results:
        savings = (1 - r["tokens_per_q"] / base_tpq) * 100
        rows.append({
            "method": r["method"],
            "nd": r.get("nd", ""),
            "ns": r.get("ns", ""),
            "acc": r["acc"],
            "tokens_per_q": r["tokens_per_q"],
            "savings_vs_sc32": savings,
        })

    md_lines = [
        "| method | nd | ns | acc | tokens_per_q | savings_vs_sc32 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['method']} | {r['nd']} | {r['ns']} | "
            f"{r['acc']:.4f} | {r['tokens_per_q']:.1f} | {r['savings_vs_sc32']:.1f}% |"
        )
    (out_dir / "eval_summary_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved table: {out_dir / 'eval_summary_table.md'}")

# SECTION: main


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
    B = args.budget
    nd_max = max(nd for nd, ns in ROLLBACK_CONFIGS if nd * ns == B)
    ns_max = max(ns for nd, ns in ROLLBACK_CONFIGS if nd * ns == B)
    sc_n = B

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "results" / f"{DATASET}_budget_controlled{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = PROJECT_ROOT / "figures" / f"budget_controlled{suffix}"
    ckpt = out_dir / "checkpoint.jsonl"
    sd = out_dir / "_shards"

    questions = load_dataset_by_name(DATASET, args.n_sample, seed=42)
    nq = len(questions)
    q_map = {q["doc_id"]: q for q in questions}
    print(f"Dataset: {DATASET}, {nq} questions, GPUs: {gpu_ids}")
    print(f"Budget B={B}, nd_max={nd_max}, ns_max={ns_max}")

    existing = []
    if ckpt.exists():
        existing = [json.loads(l) for l in ckpt.read_text().splitlines()
                    if l.strip()]
    print(f"Existing checkpoint: {len(existing)} records")

    # ---- Phase 1: Sample nd_max independent drafts per question ----
    if 1 not in args.skip_phase:
        done_drafts = {(r["doc_id"], r["draft_idx"])
                       for r in existing if r.get("task_type") == "draft"}
        draft_tasks = []
        for q in questions:
            p = build_prompt(MODEL_ID, DATASET, q["question"])
            for di in range(nd_max):
                if (q["doc_id"], di) in done_drafts:
                    continue
                draft_tasks.append(dict(
                    task_type="draft", doc_id=q["doc_id"],
                    draft_idx=di, gold_answer=q["gold_answer"],
                    prompt=p,
                ))
        if draft_tasks:
            print(f"\n--- Phase 1: {len(draft_tasks)} drafts ---")
            new = launch_shards(draft_tasks, gpu_ids, sd / "p1")
            with ckpt.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            existing.extend(new)

    drafts = [r for r in existing if r.get("task_type") == "draft"]
    draft_map = {(r["doc_id"], r["draft_idx"]): r for r in drafts}
    print(f"Total drafts: {len(drafts)}")

    # ---- Phase 2: PRM scoring on all drafts ----
    if 2 not in args.skip_phase:
        done_prm = {(r["doc_id"], r.get("draft_idx", 0))
                    for r in existing if r.get("task_type") == "prm"}
        prm_tasks = []
        for q in questions:
            for di in range(nd_max):
                if (q["doc_id"], di) in done_prm:
                    continue
                d = draft_map.get((q["doc_id"], di))
                if not d or not d.get("draft_steps"):
                    continue
                steps = d["draft_steps"]
                step_text = "<extra_0>".join(steps) + "<extra_0>"
                messages = [
                    {"role": "system", "content":
                     "Please reason step by step, and put your final "
                     "answer within \\boxed{}."},
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": step_text},
                ]
                prm_tasks.append(dict(
                    task_type="prm", doc_id=q["doc_id"],
                    draft_idx=di, n_steps=len(steps),
                    prompt=messages,
                ))
        if prm_tasks:
            print(f"\n--- Phase 2: {len(prm_tasks)} PRM scorings ---")
            new = launch_shards(prm_tasks, gpu_ids, sd / "p2", prm=True)
            with ckpt.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            existing.extend(new)

    prm_recs = [r for r in existing if r.get("task_type") == "prm"]
    prm_map = {}
    for r in prm_recs:
        prm_map[(r["doc_id"], r.get("draft_idx", 0))] = r
    print(f"Total PRM records: {len(prm_recs)}")

    # ---- Compute rollback points from PRM score drops ----
    for q in questions:
        for di in range(nd_max):
            d = draft_map.get((q["doc_id"], di))
            prm_rec = prm_map.get((q["doc_id"], di))
            if not d or not prm_rec:
                continue
            scores = prm_rec.get("step_scores", [])
            n = d.get("n_steps", 0)
            if n < 2 or len(scores) < 2:
                prm_rec["_rb_step"] = max(
                    1, math.ceil(PRM_FALLBACK_FRAC * max(n, 2))) - 1
                prm_rec["_rb_signal"] = "fallback_short"
                prm_rec["_rb_drop"] = None
                continue

            limit = min(len(scores), n)
            drops = [scores[i] - scores[i + 1] for i in range(limit - 1)]
            if drops:
                md_idx = int(np.argmax(drops))
                max_drop = float(drops[md_idx])
                if max_drop > PRM_DROP_THRESHOLD:
                    prm_rec["_rb_step"] = md_idx + 1
                    prm_rec["_rb_signal"] = "prm_drop"
                    prm_rec["_rb_drop"] = max_drop
                    continue

            prm_rec["_rb_step"] = max(1, math.ceil(PRM_FALLBACK_FRAC * n)) - 1
            prm_rec["_rb_signal"] = "fallback"
            prm_rec["_rb_drop"] = float(max(drops)) if drops else None

    # ---- Phase 3: Suffix generation ----
    if 3 not in args.skip_phase:
        done_sfx = {(r["doc_id"], r["draft_idx"],
                     r["rollback_step"], r["suffix_idx"])
                    for r in existing if r.get("task_type") == "suffix"}
        sfx_tasks = []
        for q in questions:
            did = q["doc_id"]
            for di in range(nd_max):
                d = draft_map.get((did, di))
                prm_rec = prm_map.get((did, di))
                if not d or not prm_rec:
                    continue
                rb = prm_rec.get("_rb_step")
                if rb is None:
                    continue
                steps = d["draft_steps"]
                b = min(rb, len(steps) - 1)
                if b < 1:
                    b = 1
                prefix = "\n\n".join(steps[:b])
                p = build_prompt(MODEL_ID, DATASET, q["question"])
                p += prefix + "\n\n"
                for si in range(ns_max):
                    if (did, di, b, si) in done_sfx:
                        continue
                    sfx_tasks.append(dict(
                        task_type="suffix", doc_id=did,
                        draft_idx=di, rollback_step=b,
                        suffix_idx=si,
                        gold_answer=q["gold_answer"],
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
    sfx_map = {}
    for s in sfx_recs:
        sfx_map[(s["doc_id"], s["draft_idx"],
                 s["rollback_step"], s["suffix_idx"])] = s
    print(f"Total suffix records: {len(sfx_recs)}")

    # ---- Phase 4: Full SC baseline ----
    if 4 not in args.skip_phase:
        done_sc = {(r["doc_id"], r["sc_idx"])
                   for r in existing if r.get("task_type") == "fullsc"}
        sc_tasks = []
        for q in questions:
            p = build_prompt(MODEL_ID, DATASET, q["question"])
            for si in range(sc_n):
                if (q["doc_id"], si) in done_sc:
                    continue
                sc_tasks.append(dict(
                    task_type="fullsc", doc_id=q["doc_id"],
                    sc_idx=si, gold_answer=q["gold_answer"],
                    prompt=p,
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

    # ---- Evaluate ----
    print("\n--- Evaluation ---")
    eval_results = evaluate_configs(
        questions, draft_map, prm_map, sfx_map, sc_recs, B)

    print(f"\n{'Method':<25} {'Acc':>8} {'Tok/Q':>10} {'TotalTok':>12}")
    print("-" * 58)
    for r in eval_results:
        print(f"{r['method']:<25} {r['acc']:>8.4f} "
              f"{r['tokens_per_q']:>10.0f} {r['total_tokens']:>12,}")

    # ---- Save ----
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(eval_results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {summary_path}")

    # ---- Figures + table ----
    make_figures(eval_results, fig_dir)
    save_summary_table(eval_results, out_dir)
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
