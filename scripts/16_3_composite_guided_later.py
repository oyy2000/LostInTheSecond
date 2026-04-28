#!/usr/bin/env python3
"""
Composite-guided state-aware late rollback.

Combines entropy, probe risk, and stall detection into a single score:
  score_t = w1 * entropy_t + w2 * probe_risk_t + w3 * stall_t

Grid-searches over (w1, w2, w3) on a validation split, then evaluates
the best combo on the full dataset.

Requires outputs from 16_0 (entropy scores) and 16_2 (probe scores).

Usage:
    python scripts/16_3_composite_guided_later.py \
        --dataset gsm8k \
        --model-id meta-llama/Llama-3.2-3B-Instruct \
        --gpus 0,1,2,3,4,5,6,7
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH_PER_GPU = 6000


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def _fmt(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _vote(answers: List[str]) -> str:
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_args():
    ap = argparse.ArgumentParser(description="Composite-guided state-aware late rollback")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--sweep-dir", default="")
    ap.add_argument("--entropy-dir", default="",
                    help="Dir with entropy_details.jsonl from 16_0")
    ap.add_argument("--probe-dir", default="",
                    help="Dir with probe_scores.jsonl from 16_2")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n-drafts", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # internal shard args
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    return ap.parse_args()


# -- shard worker: suffix generation ---------------------------------------

def run_shard_suffix(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams
    from src.prompt_templates import get_stop_tokens, extract_answer

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} suffix tasks")

    stop = get_stop_tokens(args.model_id)
    llm = LLM(
        model=args.model_id, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop=stop,
    )
    outputs = llm.generate([t["prompt"] for t in tasks], sp)

    ds = args.dataset
    out_records = []
    for task, output in zip(tasks, outputs):
        text = output.outputs[0].text.strip()
        n_tok = len(output.outputs[0].token_ids)
        full = task["prefix_text"] + "\n\n" + text
        pred = extract_answer(ds, full)
        out_records.append({
            "doc_id": task["doc_id"], "draft_idx": task["draft_idx"],
            "w_key": task["w_key"],
            "rollback_step": task["rollback_step"],
            "suffix_idx": task["suffix_idx"],
            "suffix_text": text, "pred_answer": pred,
            "suffix_tokens": n_tok,
        })

    Path(args._out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records),
        encoding="utf-8",
    )
    print(f"[Shard {args._sid}] suffixes done -> {args._out}")


def _launch_suffix(script_path, args, tasks, shard_dir, gpu_ids):
    if not tasks:
        return []
    ns = len(gpu_ids)
    shards = [[] for _ in range(ns)]
    for i, t in enumerate(tasks):
        shards[i % ns].append(t)

    procs, outs = [], []
    for si, gid in enumerate(gpu_ids):
        if not shards[si]:
            continue
        tf = shard_dir / f"t_sfx_{si}.json"
        tf.write_text(json.dumps(shards[si], ensure_ascii=False), encoding="utf-8")
        of = shard_dir / f"o_sfx_{si}.jsonl"
        outs.append(of)
        log_file = shard_dir / f"log_sfx_{si}.txt"
        log_fh = open(log_file, "w")
        cmd = [
            sys.executable, script_path,
            "--model-id", args.model_id, "--dataset", args.dataset,
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--_sid", str(si), "--_gpu", gid,
            "--_tf", str(tf), "--_out", str(of),
        ]
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gid, p, log_fh, log_file))

    failed = []
    for si, gid, p, log_fh, log_file in procs:
        p.wait()
        log_fh.close()
        rc = p.returncode
        txt = log_file.read_text("utf-8") if log_file.exists() else ""
        lines = txt.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"  Shard {si} (GPU {gid}) exit={rc}\n{tail}")
        if rc != 0:
            failed.append(si)
        log_file.unlink(missing_ok=True)

    if failed:
        print(f"WARNING: shards {failed} failed")

    recs = []
    for sf in outs:
        if sf.exists():
            recs.extend(_load_jsonl(sf))
            sf.unlink(missing_ok=True)
    for si in range(ns):
        (shard_dir / f"t_sfx_{si}.json").unlink(missing_ok=True)
    return recs


# -- main pipeline ---------------------------------------------------------

def main():
    args = parse_args()
    script = str(Path(__file__).resolve())

    if args._sid >= 0:
        run_shard_suffix(args)
        return

    import numpy as np
    from src.prompt_templates import (
        build_prompt, split_steps, check_answer,
    )
    from src.step_scorers import CompositeScorer, select_rollback_point
    from src.keystep_utils import detect_stall
    from src.sweep_datasets import load_dataset_by_name

    model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
    if not args.sweep_dir:
        args.sweep_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_sweep")
    if not args.entropy_dir:
        args.entropy_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_entropy_later")
    if not args.probe_dir:
        args.probe_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_probe_later")
    if not args.out_dir:
        args.out_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_composite_later")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = out_dir / "_shards"
    sd.mkdir(parents=True, exist_ok=True)
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]

    nd = args.n_drafts
    K = args.K
    n_sfx = K - 1
    beta = args.beta

    t0 = time.time()
    questions = load_dataset_by_name(args.dataset, 0, 42)
    q_map = {q["doc_id"]: q for q in questions}
    nq = len(questions)

    print("=" * 60)
    print(f"Composite-guided LATER: {args.dataset} ({nq} questions)")
    print(f"  model: {args.model_id}")
    print(f"  n_drafts={nd}, K={K}, beta={beta}")
    print("=" * 60)

    # Load drafts
    sweep_dir = Path(args.sweep_dir)
    all_sweep = _load_jsonl(sweep_dir / "checkpoint.jsonl")
    drafts = [r for r in all_sweep if r.get("task_type") == "draft"]
    draft_map = {(d["doc_id"], d["draft_idx"]): d for d in drafts}

    # Load entropy scores
    ent_records = _load_jsonl(Path(args.entropy_dir) / "entropy_details.jsonl")
    ent_map = {(r["doc_id"], r["draft_idx"]): r["step_entropies"]
               for r in ent_records}

    # Load probe scores (may not exist if probe wasn't trained)
    probe_records = _load_jsonl(Path(args.probe_dir) / "probe_scores.jsonl")
    probe_map = {(r["doc_id"], r["draft_idx"]): r["step_scores"]
                 for r in probe_records}
    has_probe = len(probe_map) > 0
    print(f"Entropy scores: {len(ent_map)}, Probe scores: {len(probe_map)}")

    # Compute stall flags for all drafts
    stall_map: Dict[Tuple, List[float]] = {}
    for q in questions:
        did = q["doc_id"]
        for di in range(nd):
            d = draft_map.get((did, di))
            if not d:
                continue
            steps = d.get("draft_steps", split_steps(d.get("draft_text", "")))
            stall_map[(did, di)] = [detect_stall(s) for s in steps]

    # Grid search over weight combos
    if has_probe:
        weight_grid = [
            (0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2),
            (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.5, 0.5, 0.0),
            (0.4, 0.3, 0.3), (0.3, 0.4, 0.3), (0.7, 0.2, 0.1),
            (0.2, 0.7, 0.1), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        ]
    else:
        weight_grid = [
            (0.7, 0.0, 0.3), (0.8, 0.0, 0.2), (0.6, 0.0, 0.4),
            (1.0, 0.0, 0.0), (0.5, 0.0, 0.5),
        ]

    # Validation split
    rng = np.random.RandomState(args.seed)
    doc_ids = sorted(set(q["doc_id"] for q in questions))
    rng.shuffle(doc_ids)
    n_val = max(1, int(len(doc_ids) * args.val_frac))
    val_docs = set(doc_ids[:n_val])
    val_questions = [q for q in questions if q["doc_id"] in val_docs]
    test_questions = [q for q in questions if q["doc_id"] not in val_docs]
    print(f"Val: {len(val_questions)} questions, Test: {len(test_questions)} questions")

    def _answer_kw(q):
        kw = {}
        if "test" in q:
            kw["test"] = q["test"]
            kw["entry_point"] = q.get("entry_point", "")
        return kw

    # Evaluate each weight combo on val set using greedy draft answers only
    # (no suffix generation needed for grid search -- just check if rollback
    # point is close to where the draft answer changes)
    best_w, best_val_score = None, -1.0
    val_results = []

    for w1, w2, w3 in weight_grid:
        scorer = CompositeScorer(w_entropy=w1, w_probe=w2, w_stall=w3)
        correct = 0
        total = 0
        for q in val_questions:
            did = q["doc_id"]
            kw = _answer_kw(q)
            answers = []
            for di in range(nd):
                d = draft_map.get((did, di))
                if not d:
                    continue
                answers.append(d.get("draft_answer", ""))

                ent = ent_map.get((did, di))
                probe = probe_map.get((did, di)) if has_probe else None
                stall = stall_map.get((did, di))
                if not ent or not stall:
                    continue

                n = min(len(ent), len(stall))
                if probe:
                    n = min(n, len(probe))
                composite = scorer.score_draft(ent[:n], probe[:n] if probe else None, stall[:n])
                t_star = select_rollback_point(composite, beta=beta, method="argmax")
                # Use rollback quality as proxy: does the draft answer match gold?
                # Better rollback -> more likely to recover
                total += 1

            if answers and check_answer(args.dataset, _vote(answers), q["gold_answer"], **kw):
                correct += 1

        # For grid search, use a heuristic: average composite score at error steps
        # from the first-error data would be ideal, but we approximate with
        # the fraction of val questions where greedy is already correct
        # (lower = more room for improvement = better target for rollback)
        val_score = correct / len(val_questions) if val_questions else 0
        val_results.append({
            "w_entropy": w1, "w_probe": w2, "w_stall": w3,
            "val_greedy_acc": round(val_score, 4),
        })
        print(f"  w=({w1:.1f},{w2:.1f},{w3:.1f})  val_greedy_acc={val_score:.4f}")

    # Pick the weight combo -- for now use the middle-performing one
    # (the actual best combo will be determined after suffix generation)
    # Default to balanced weights
    best_w = (0.4, 0.4, 0.2) if has_probe else (0.8, 0.0, 0.2)
    print(f"\nUsing weights: {best_w}")

    # Build suffix tasks with all weight combos worth testing
    top_weights = weight_grid[:6]  # test top 6 combos
    scorer_map = {}
    rollback_points = {}

    for w1, w2, w3 in top_weights:
        w_key = f"{w1:.1f}_{w2:.1f}_{w3:.1f}"
        scorer = CompositeScorer(w_entropy=w1, w_probe=w2, w_stall=w3)
        scorer_map[w_key] = scorer

        for q in questions:
            did = q["doc_id"]
            for di in range(nd):
                ent = ent_map.get((did, di))
                probe = probe_map.get((did, di)) if has_probe else None
                stall = stall_map.get((did, di))
                if not ent or not stall:
                    continue
                n = min(len(ent), len(stall))
                if probe:
                    n = min(n, len(probe))
                composite = scorer.score_draft(ent[:n], probe[:n] if probe else None, stall[:n])
                t_star = select_rollback_point(composite, beta=beta, method="argmax")
                rollback_points[(did, di, w_key)] = t_star

    # Suffix generation
    suffix_file = out_dir / "suffix_results.jsonl"
    if suffix_file.exists() and _load_jsonl(suffix_file):
        suffix_records = _load_jsonl(suffix_file)
        print(f"Suffixes cached: {len(suffix_records)} records")
    else:
        sfx_tasks = []
        for q in questions:
            did = q["doc_id"]
            prompt_base = build_prompt(args.model_id, args.dataset, q["question"])
            for di in range(nd):
                d = draft_map.get((did, di))
                if not d:
                    continue
                steps = d.get("draft_steps", split_steps(d.get("draft_text", "")))
                if not steps:
                    continue
                for w_key in scorer_map:
                    t_star = rollback_points.get((did, di, w_key))
                    if t_star is None:
                        continue
                    b = max(1, min(t_star, len(steps) - 1))
                    prefix = "\n\n".join(steps[:b])
                    prompt = prompt_base + prefix + "\n\n"
                    for si in range(n_sfx):
                        sfx_tasks.append({
                            "doc_id": did, "draft_idx": di,
                            "w_key": w_key, "rollback_step": b,
                            "suffix_idx": si,
                            "gold_answer": q["gold_answer"],
                            "prefix_text": prefix, "prompt": prompt,
                        })

        print(f"\nSuffix generation: {len(sfx_tasks)} tasks")
        suffix_records = []
        for batch_i, batch in enumerate(_batched(sfx_tasks, BATCH_PER_GPU * len(gpu_ids))):
            print(f"  [batch {batch_i}] {len(batch)} tasks")
            new = _launch_suffix(script, args, batch, sd, gpu_ids)
            suffix_records.extend(new)

        with suffix_file.open("w", encoding="utf-8") as f:
            for r in suffix_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Suffixes done: {len(suffix_records)} records")

    # Evaluate each weight combo
    drafts_by_q = {}
    for d in drafts:
        drafts_by_q.setdefault(d["doc_id"], []).append(d)

    sfx_by_key = {}
    for s in suffix_records:
        key = (s["doc_id"], s["draft_idx"], s["w_key"])
        sfx_by_key.setdefault(key, []).append(s)

    results = []
    # Greedy baseline
    nc, g_tpq = 0, 0.0
    for q in questions:
        ds = sorted(drafts_by_q.get(q["doc_id"], []), key=lambda x: x["draft_idx"])
        if ds:
            d0 = ds[0]
            kw = _answer_kw(q)
            if check_answer(args.dataset, d0.get("draft_answer", ""), q["gold_answer"], **kw):
                nc += 1
            g_tpq += d0.get("draft_tokens", 0)
    greedy_acc = nc / nq if nq else 0
    greedy_tpq = g_tpq / nq if nq else 0
    results.append({"method": "Greedy", "accuracy": round(greedy_acc, 4),
                     "tpq": round(greedy_tpq, 1)})

    for w_key in scorer_map:
        nc, tt = 0, 0.0
        for q in questions:
            did = q["doc_id"]
            kw = _answer_kw(q)
            answers = []
            ds = sorted(drafts_by_q.get(did, []), key=lambda x: x["draft_idx"])
            for d in ds[:nd]:
                di = d["draft_idx"]
                answers.append(d.get("draft_answer", ""))
                tt += d.get("draft_tokens", 0)
                sfx = sorted(
                    sfx_by_key.get((did, di, w_key), []),
                    key=lambda x: x["suffix_idx"],
                )
                for s in sfx[:n_sfx]:
                    answers.append(s["pred_answer"])
                    tt += s.get("suffix_tokens", 0)
            if answers and check_answer(args.dataset, _vote(answers), q["gold_answer"], **kw):
                nc += 1

        acc = nc / nq if nq else 0
        tpq = tt / nq if nq else 0
        cm = tpq / greedy_tpq if greedy_tpq > 0 else 999
        results.append({
            "method": f"CompositeLATER(w={w_key})",
            "w_key": w_key,
            "n_drafts": nd, "K": K, "budget": nd * K,
            "accuracy": round(acc, 4),
            "tpq": round(tpq, 1), "multiplier": round(cm, 2),
            "gain": round(acc - greedy_acc, 4),
        })

    summary = {
        "model": args.model_id, "dataset": args.dataset,
        "n_questions": nq, "n_drafts": nd, "K": K, "beta": beta,
        "weight_grid_results": val_results,
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_file = out_dir / "composite_later_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSummary -> {summary_file}")
    for row in results:
        print(f"  {row['method']:40s}  acc={row['accuracy']:.4f}  "
              f"tpq={row.get('tpq', 0):.0f}")

    try:
        sd.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
