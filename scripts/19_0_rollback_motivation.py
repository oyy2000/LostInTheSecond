#!/usr/bin/env python3
"""
Rollback motivation experiment: for each wrong greedy draft, resample from
every step boundary and measure recovery rate vs relative position.

For each dataset, the pipeline is:
  1. Generate one greedy draft per question.
  2. Filter to wrong drafts only.
  3. For each wrong draft with T steps, for each rollback point k in [0..T-1],
     generate N suffix continuations from steps[:k] (i.e. resample from the
     end of step k, before step k+1).
  4. Evaluate recovery rate at each relative position k/T.

This is the "resample one step before the bad prefix" experiment generalized
to all step boundaries, across multiple datasets.

Multi-GPU data-parallel via subprocess sharding.

Usage:
    # single dataset
    python scripts/19_0_rollback_motivation.py --dataset math500 --gpus 0,1,2,3,4,5,6,7

    # batch all motivation datasets
    python scripts/19_0_rollback_motivation.py --batch --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
BATCH_PER_GPU = 6000

DATASETS = ["math500", "hotpotqa", "gsm8k"]
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n-continuations", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--n-sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--batch", action="store_true")
    # shard worker args
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    return ap.parse_args()


# -- shard worker ----------------------------------------------------------

def run_shard(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams
    from src.prompt_templates import get_stop_tokens, extract_answer

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} tasks")

    stop = get_stop_tokens(args.model_id)
    llm = LLM(
        model=args.model_id, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    greedy_idx = [i for i, t in enumerate(tasks) if t.get("mode") == "greedy"]
    sample_idx = [i for i, t in enumerate(tasks) if t.get("mode") != "greedy"]

    sp_g = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=stop)
    sp_s = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop=stop,
    )

    results = [None] * len(tasks)
    if greedy_idx:
        outs = llm.generate([tasks[i]["prompt"] for i in greedy_idx], sp_g)
        for j, idx in enumerate(greedy_idx):
            results[idx] = outs[j]
    if sample_idx:
        outs = llm.generate([tasks[i]["prompt"] for i in sample_idx], sp_s)
        for j, idx in enumerate(sample_idx):
            results[idx] = outs[j]

    ds = args.dataset
    records = []
    for task, output in zip(tasks, results):
        text = output.outputs[0].text.strip()
        tt = task["task_type"]
        if tt == "draft":
            from src.prompt_templates import split_steps
            steps = split_steps(text)
            pred = extract_answer(ds, text)
            records.append({
                "task_type": "draft", "doc_id": task["doc_id"],
                "draft_text": text, "draft_steps": steps,
                "draft_answer": pred,
            })
        elif tt == "suffix":
            full = task["prefix_text"] + "\n\n" + text
            pred = extract_answer(ds, full)
            records.append({
                "task_type": "suffix", "doc_id": task["doc_id"],
                "rollback_k": task["rollback_k"],
                "n_steps": task["n_steps"],
                "suffix_idx": task["suffix_idx"],
                "pred_answer": pred,
            })

    Path(args._out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )
    print(f"[Shard {args._sid}] done -> {args._out}")


# -- launch helper ---------------------------------------------------------

def _load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def _append_jsonl(p: Path, recs: List[dict]):
    with p.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _launch(script: str, args, tasks: List[dict], sd: Path, gpus: List[str]) -> List[dict]:
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
        tf = sd / f"t_{si}.json"
        tf.write_text(json.dumps(shards[si], ensure_ascii=False), encoding="utf-8")
        of = sd / f"o_{si}.jsonl"
        outs.append(of)
        log = sd / f"log_{si}.txt"
        lf = open(log, "w")
        cmd = [
            sys.executable, script,
            "--model-id", args.model_id, "--dataset", args.dataset,
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--_sid", str(si), "--_gpu", gid,
            "--_tf", str(tf), "--_out", str(of),
        ]
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((si, gid, p, lf, log))

    failed = []
    for si, gid, p, lf, log in procs:
        p.wait()
        lf.close()
        rc = p.returncode
        txt = log.read_text("utf-8") if log.exists() else ""
        lines = txt.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"  Shard {si} (GPU {gid}) exit={rc}\n{tail}")
        if rc != 0:
            failed.append(si)
        log.unlink(missing_ok=True)

    if failed:
        print(f"WARNING: shards {failed} failed")

    recs = []
    for sf in outs:
        if sf.exists():
            recs.extend(_load_jsonl(sf))
            sf.unlink(missing_ok=True)
    for si in range(ns):
        (sd / f"t_{si}.json").unlink(missing_ok=True)
    return recs


def _batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# -- main pipeline ---------------------------------------------------------

def run_one_dataset(args):
    from src.sweep_datasets import load_dataset_by_name
    from src.prompt_templates import build_prompt, split_steps, check_answer

    ds = args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "results" / f"{ds}_rollback_motivation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = out_dir / "_shards"
    sd.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.jsonl"
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    script = str(Path(__file__).resolve())

    questions = load_dataset_by_name(ds, args.n_sample, args.seed)
    q_map = {q["doc_id"]: q for q in questions}
    nq = len(questions)
    print("=" * 60)
    print(f"Rollback motivation: {ds} ({nq} questions)")
    print(f"  model: {args.model_id}")
    print(f"  n_continuations: {args.n_continuations}")
    print(f"  GPUs: {gpus}")
    print("=" * 60)

    existing = _load_jsonl(ckpt)
    print(f"Checkpoint: {len(existing)} records")

    # Phase 1: greedy drafts
    done_drafts = {r["doc_id"] for r in existing if r["task_type"] == "draft"}
    draft_tasks = []
    for q in questions:
        if q["doc_id"] in done_drafts:
            continue
        draft_tasks.append({
            "task_type": "draft", "doc_id": q["doc_id"],
            "gold_answer": q["gold_answer"],
            "prompt": build_prompt(args.model_id, ds, q["question"]),
            "mode": "greedy",
        })
    print(f"\n[Phase 1] Drafts: {nq - len(draft_tasks)} cached, {len(draft_tasks)} pending")
    if draft_tasks:
        for bi, batch in enumerate(_batched(draft_tasks, BATCH_PER_GPU * len(gpus))):
            print(f"  batch {bi}: {len(batch)} tasks")
            new = _launch(script, args, batch, sd, gpus)
            _append_jsonl(ckpt, new)
            existing.extend(new)

    # Identify wrong drafts
    drafts = [r for r in existing if r["task_type"] == "draft"]
    wrong_drafts = []
    for d in drafts:
        q = q_map.get(d["doc_id"])
        if not q:
            continue
        kw = {}
        if "test" in q:
            kw["test"] = q["test"]
            kw["entry_point"] = q.get("entry_point", "")
        if not check_answer(ds, d.get("draft_answer", ""), q["gold_answer"], **kw):
            wrong_drafts.append(d)

    print(f"\nWrong drafts: {len(wrong_drafts)} / {len(drafts)}")

    # Phase 2: suffix continuations at every step boundary
    # For each wrong draft with T steps, rollback to k in [0..T-2]
    # (k=0 means resample from scratch after prompt, k=T-2 means keep all but last step)
    # We skip k=0 (no prefix at all) to keep it meaningful.
    done_sfx = {
        (r["doc_id"], r["rollback_k"], r["suffix_idx"])
        for r in existing if r["task_type"] == "suffix"
    }

    suffix_tasks = []
    for d in wrong_drafts:
        steps = d.get("draft_steps") or split_steps(d.get("draft_text", ""))
        T = len(steps)
        if T < 2:
            continue
        q = q_map[d["doc_id"]]
        base_prompt = build_prompt(args.model_id, ds, q["question"])
        for k in range(1, T):
            prefix = "\n\n".join(steps[:k])
            prompt = base_prompt + prefix + "\n\n"
            for si in range(args.n_continuations):
                if (d["doc_id"], k, si) in done_sfx:
                    continue
                suffix_tasks.append({
                    "task_type": "suffix", "doc_id": d["doc_id"],
                    "rollback_k": k, "n_steps": T,
                    "suffix_idx": si,
                    "gold_answer": q["gold_answer"],
                    "prefix_text": prefix, "prompt": prompt,
                    "mode": "sampled",
                })

    print(f"[Phase 2] Suffixes: {len(suffix_tasks)} pending")
    if suffix_tasks:
        for bi, batch in enumerate(_batched(suffix_tasks, BATCH_PER_GPU * len(gpus))):
            print(f"  batch {bi}: {len(batch)} tasks")
            new = _launch(script, args, batch, sd, gpus)
            _append_jsonl(ckpt, new)
            existing.extend(new)

    # Evaluate
    print("\n[Evaluation]")
    suffixes = [r for r in existing if r["task_type"] == "suffix"]
    by_pos = defaultdict(list)
    for s in suffixes:
        q = q_map.get(s["doc_id"])
        if not q:
            continue
        kw = {}
        if "test" in q:
            kw["test"] = q["test"]
            kw["entry_point"] = q.get("entry_point", "")
        correct = check_answer(ds, s.get("pred_answer", ""), q["gold_answer"], **kw)
        rel = s["rollback_k"] / s["n_steps"]
        by_pos[(s["doc_id"], s["rollback_k"], s["n_steps"])].append(float(correct))

    # Aggregate by relative position bins
    import numpy as np
    rel_positions = []
    recovery_rates = []
    for (doc_id, k, T), vals in by_pos.items():
        rel_positions.append(k / T)
        recovery_rates.append(np.mean(vals))

    rel_positions = np.array(rel_positions)
    recovery_rates = np.array(recovery_rates)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    summary_bins = []
    for i in range(n_bins):
        mask = (rel_positions >= bin_edges[i]) & (rel_positions < bin_edges[i + 1])
        if mask.sum() < 5:
            continue
        m = np.mean(recovery_rates[mask])
        se = np.std(recovery_rates[mask], ddof=1) / np.sqrt(mask.sum())
        center = (bin_edges[i] + bin_edges[i + 1]) / 2
        summary_bins.append({
            "bin_center": round(float(center), 3),
            "mean": round(float(m), 4),
            "se": round(float(se), 4),
            "n": int(mask.sum()),
        })
        print(f"  rel_pos=[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f}): "
              f"R={m:.4f} +/- {se:.4f} (n={mask.sum()})")

    greedy_acc = 1 - len(wrong_drafts) / len(drafts) if drafts else 0
    summary = {
        "dataset": ds, "model": args.model_id,
        "n_questions": nq, "n_wrong_drafts": len(wrong_drafts),
        "greedy_acc": round(greedy_acc, 4),
        "n_continuations": args.n_continuations,
        "bins": summary_bins,
    }
    (out_dir / "rollback_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSummary -> {out_dir / 'rollback_summary.json'}")

    try:
        sd.rmdir()
    except OSError:
        pass
    return summary


def main():
    args = parse_args()

    if args._sid >= 0:
        run_shard(args)
        return

    if args.batch:
        for ds in DATASETS:
            print("\n" + "=" * 70)
            args.dataset = ds
            args.out_dir = ""
            run_one_dataset(args)
        return

    if not args.dataset:
        print("ERROR: --dataset required (or use --batch)")
        sys.exit(1)
    run_one_dataset(args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
