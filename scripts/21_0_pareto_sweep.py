#!/usr/bin/env python3
"""
Experiment 1: Iso-compute Pareto sweep.

For each (model, dataset) combo, generates data for:
  - Greedy (single point)
  - Self-Consistency at N in {1,2,4,8,16,32}
  - Random-Repair at p in {0.01..0.5}
  - Entropy-Triggered Lookback Repair (sweep threshold percentiles)

X-axis = total generation tokens per question (averaged).
Y-axis = accuracy (majority vote).

Phases:
  1. Load/generate greedy drafts + SC samples (reuse sweep checkpoint)
  2. Compute per-step logprobs + entropy metrics for greedy drafts
  3. Decide repair points for random + entropy triggers
  4. Generate suffix continuations for all repair configs
  5. Evaluate all methods, save results

Usage:
    python scripts/21_0_pareto_sweep.py --gpus 0,1,2,3,4,5,6,7
    python scripts/21_0_pareto_sweep.py --model-id Qwen/Qwen2.5-3B-Instruct --dataset gsm8k
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
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH_PER_GPU = 6000

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
DATASETS = ["gsm8k", "math500"]

SC_NS = [1, 2, 4, 8, 16, 32]
RANDOM_PS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
ENTROPY_PCTS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95]
REPAIR_K = 3
MAX_REPAIRS = 3


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def _append_jsonl(path: Path, records: List[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: List[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def _model_short(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-", "_")


# -- CLI -------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Iso-compute Pareto sweep")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--config", default=str(ROOT / "configs/pareto_sweep.yaml"))
    ap.add_argument("--batch", action="store_true",
                    help="Run all model x dataset combos")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=32)
    # internal shard args
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    ap.add_argument("--_phase", default="")
    return ap.parse_args()


# -- shard workers ---------------------------------------------------------

def run_shard_logprobs(args):
    """Compute prompt_logprobs for greedy drafts on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} logprob tasks")

    llm = LLM(
        model=args.model_id, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=0.70, max_model_len=2048,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    BATCH = min(args.batch_size, 4)
    out_path = Path(args._out)
    with out_path.open("w", encoding="utf-8") as fout:
        for bi in range(0, len(tasks), BATCH):
            batch = tasks[bi:bi + BATCH]
            prompts = [t["full_prompt"] for t in batch]
            outputs = llm.generate(prompts, sp)

            for task, output in zip(batch, outputs):
                rec = {k: v for k, v in task.items() if k != "full_prompt"}
                resp_offset = task["resp_char_offset"]
                lps, offsets = [], []
                cum_chars = 0
                if output.prompt_logprobs is not None:
                    ids = output.prompt_token_ids
                    for ti, lp_dict in enumerate(output.prompt_logprobs):
                        if lp_dict is None:
                            decoded = tokenizer.decode([ids[ti]])
                            cum_chars += len(decoded)
                            continue
                        tok_id = ids[ti]
                        if tok_id in lp_dict:
                            lp_obj = lp_dict[tok_id]
                        else:
                            lp_obj = next(iter(lp_dict.values()))
                        decoded = lp_obj.decoded_token or tokenizer.decode([tok_id])
                        char_pos = cum_chars
                        cum_chars += len(decoded)
                        if char_pos >= resp_offset:
                            lps.append(lp_obj.logprob)
                            offsets.append(char_pos - resp_offset)
                rec["token_logprobs"] = lps
                rec["token_offsets"] = offsets
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[Shard {args._sid}] logprobs done -> {out_path}")


def run_shard_generation(args):
    """Generate completions (SC samples or repair suffixes) on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams
    from src.prompt_templates import get_stop_tokens, extract_answer

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} gen tasks")

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
        tt = task.get("task_type", "")

        if tt == "fullsc":
            pred = extract_answer(ds, text)
            out_records.append({
                "task_type": "fullsc", "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "text": text, "pred_answer": pred, "tokens": n_tok,
            })
        elif tt == "repair_suffix":
            full = task["prefix_text"] + "\n\n" + text
            pred = extract_answer(ds, full)
            out_records.append({
                "task_type": "repair_suffix", "doc_id": task["doc_id"],
                "repair_step": task["repair_step"],
                "suffix_idx": task["suffix_idx"],
                "config_tag": task.get("config_tag", ""),
                "suffix_text": text, "pred_answer": pred,
                "suffix_tokens": n_tok,
            })
        elif tt == "draft":
            from src.prompt_templates import split_steps
            steps = split_steps(text)
            pred = extract_answer(ds, text)
            out_records.append({
                "task_type": "draft", "doc_id": task["doc_id"],
                "draft_idx": task.get("draft_idx", 0),
                "mode": task.get("mode", "greedy"),
                "draft_text": text, "draft_steps": steps,
                "draft_answer": pred, "draft_tokens": n_tok,
            })

    Path(args._out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records),
        encoding="utf-8",
    )
    print(f"[Shard {args._sid}] gen done -> {args._out}")


# -- launch helper ---------------------------------------------------------

def _launch(script_path: str, args, tasks: List[dict], shard_dir: Path,
            gpu_ids: List[str], phase: str) -> List[dict]:
    if not tasks:
        return []
    ns = len(gpu_ids)
    shards: List[List[dict]] = [[] for _ in range(ns)]
    for i, t in enumerate(tasks):
        shards[i % ns].append(t)

    procs, outs = [], []
    for si, gid in enumerate(gpu_ids):
        if not shards[si]:
            continue
        tf = shard_dir / f"t_{phase}_{si}.json"
        tf.write_text(json.dumps(shards[si], ensure_ascii=False), encoding="utf-8")
        of = shard_dir / f"o_{phase}_{si}.jsonl"
        outs.append(of)
        log_file = shard_dir / f"log_{phase}_{si}.txt"
        log_fh = open(log_file, "w")
        cmd = [
            sys.executable, script_path,
            "--model-id", args.model_id,
            "--dataset", args.dataset,
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--batch-size", str(args.batch_size),
            "--_sid", str(si), "--_gpu", gid,
            "--_tf", str(tf), "--_out", str(of),
            "--_phase", phase,
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
        (shard_dir / f"t_{phase}_{si}.json").unlink(missing_ok=True)
    return recs


# -- main pipeline ---------------------------------------------------------

def run_one_combo(model_id: str, dataset: str, args):
    """Run full Pareto sweep for one (model, dataset) combo."""
    from src.prompt_templates import (
        build_prompt, split_steps, extract_answer, check_answer,
    )
    from src.repair_engine import (
        compute_step_metrics_from_logprobs, should_trigger, METRIC_DIRECTION,
    )
    from src.step_scorers import step_char_boundaries
    from src.sweep_datasets import load_dataset_by_name
    from src.pareto_engine import (
        extract_pareto_front, sc_pareto_points, greedy_point,
        repair_evaluate, decide_random_repairs, decide_entropy_repairs,
        build_repair_suffix_tasks,
    )

    ms = _model_short(model_id)
    out_dir = ROOT / "results" / f"{dataset}_{ms}_pareto"
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = out_dir / "_shards"
    sd.mkdir(parents=True, exist_ok=True)
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    script = str(Path(__file__).resolve())

    args.model_id = model_id
    args.dataset = dataset

    t0 = time.time()
    questions = load_dataset_by_name(dataset, 0, 42)
    nq = len(questions)
    print("=" * 70)
    print(f"Pareto sweep: {model_id} x {dataset} ({nq} questions)")
    print(f"  GPUs: {gpu_ids}")
    print("=" * 70)

    # -- Phase 1: Load or generate greedy drafts --
    sweep_dir = ROOT / "results" / f"{dataset}_{ms}_sweep"
    draft_file = out_dir / "drafts.jsonl"

    if draft_file.exists():
        drafts = _load_jsonl(draft_file)
        print(f"Drafts cached: {len(drafts)} records")
    elif (sweep_dir / "checkpoint.jsonl").exists():
        all_sweep = _load_jsonl(sweep_dir / "checkpoint.jsonl")
        drafts = [r for r in all_sweep if r.get("task_type") == "draft"
                  and r.get("draft_idx", 0) == 0]
        _write_jsonl(draft_file, drafts)
        print(f"Loaded {len(drafts)} greedy drafts from sweep checkpoint")
    else:
        print("Generating greedy drafts...")
        draft_tasks = []
        for q in questions:
            p = build_prompt(model_id, dataset, q["question"])
            draft_tasks.append({
                "task_type": "draft", "doc_id": q["doc_id"],
                "draft_idx": 0, "mode": "greedy",
                "gold_answer": q["gold_answer"], "prompt": p,
            })
        drafts = []
        for batch in _batched(draft_tasks, BATCH_PER_GPU * len(gpu_ids)):
            new = _launch(script, args, batch, sd, gpu_ids, "draft")
            drafts.extend(new)
        _write_jsonl(draft_file, drafts)
        print(f"Generated {len(drafts)} greedy drafts")

    drafts_by_q: Dict[Any, List[dict]] = {}
    for d in drafts:
        drafts_by_q.setdefault(d["doc_id"], []).append(d)

    # -- Phase 2: Load or generate SC samples --
    sc_file = out_dir / "fullsc.jsonl"
    max_sc_n = max(SC_NS)

    if sc_file.exists():
        sc_records = _load_jsonl(sc_file)
        print(f"SC samples cached: {len(sc_records)} records")
    elif (sweep_dir / "checkpoint.jsonl").exists():
        all_sweep = _load_jsonl(sweep_dir / "checkpoint.jsonl")
        sc_records = [r for r in all_sweep if r.get("task_type") == "fullsc"]
        _write_jsonl(sc_file, sc_records)
        print(f"Loaded {len(sc_records)} SC samples from sweep checkpoint")
    else:
        print(f"Generating {max_sc_n} SC samples per question...")
        sc_tasks = []
        for q in questions:
            p = build_prompt(model_id, dataset, q["question"])
            for si in range(max_sc_n):
                sc_tasks.append({
                    "task_type": "fullsc", "doc_id": q["doc_id"],
                    "sample_idx": si, "gold_answer": q["gold_answer"],
                    "prompt": p, "mode": "sampled",
                })
        sc_records = []
        for batch in _batched(sc_tasks, BATCH_PER_GPU * len(gpu_ids)):
            new = _launch(script, args, batch, sd, gpu_ids, "sc")
            sc_records.extend(new)
        _write_jsonl(sc_file, sc_records)
        print(f"Generated {len(sc_records)} SC samples")

    # -- Phase 3: Logprob scoring for greedy drafts --
    logprob_file = out_dir / "draft_logprobs.jsonl"

    if logprob_file.exists():
        logprob_records = _load_jsonl(logprob_file)
        print(f"Logprobs cached: {len(logprob_records)} records")
    else:
        print("Computing logprobs for greedy drafts...")
        lp_tasks = []
        for q in questions:
            did = q["doc_id"]
            ds = drafts_by_q.get(did, [])
            if not ds:
                continue
            d0 = ds[0]
            prompt = build_prompt(model_id, dataset, q["question"])
            full = prompt + d0.get("draft_text", "")
            lp_tasks.append({
                "doc_id": did, "draft_idx": 0,
                "draft_text": d0.get("draft_text", ""),
                "draft_steps": d0.get("draft_steps", split_steps(d0.get("draft_text", ""))),
                "full_prompt": full,
                "resp_char_offset": len(prompt),
            })
        logprob_records = []
        for batch in _batched(lp_tasks, BATCH_PER_GPU * len(gpu_ids)):
            new = _launch(script, args, batch, sd, gpu_ids, "logprobs")
            logprob_records.extend(new)
        _write_jsonl(logprob_file, logprob_records)
        print(f"Logprobs done: {len(logprob_records)} records")

    # -- Phase 4: Compute step-level metrics --
    metrics_file = out_dir / "step_metrics.jsonl"

    if metrics_file.exists():
        step_metrics_all = _load_jsonl(metrics_file)
        print(f"Step metrics cached: {len(step_metrics_all)} records")
    else:
        print("Computing step-level metrics...")
        lp_map = {r["doc_id"]: r for r in logprob_records}
        step_metrics_all = []
        for q in questions:
            did = q["doc_id"]
            lp_rec = lp_map.get(did)
            if not lp_rec or not lp_rec.get("token_logprobs"):
                continue
            steps = lp_rec.get("draft_steps", [])
            if not steps:
                continue
            response = lp_rec.get("draft_text", "\n\n".join(steps))
            metrics = compute_step_metrics_from_logprobs(
                lp_rec["token_logprobs"], lp_rec.get("token_texts", []),
                steps, response,
            )
            step_metrics_all.append({
                "doc_id": did,
                "n_steps": len(steps),
                "step_metrics": [
                    {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in m.items()}
                    for m in metrics
                ],
            })
        _write_jsonl(metrics_file, step_metrics_all)
        print(f"Step metrics: {len(step_metrics_all)} trajectories")

    sm_map = {r["doc_id"]: r["step_metrics"] for r in step_metrics_all}

    # -- Phase 5: Decide repairs + generate suffixes --
    suffix_file = out_dir / "repair_suffixes.jsonl"

    if suffix_file.exists():
        suffix_records = _load_jsonl(suffix_file)
        print(f"Repair suffixes cached: {len(suffix_records)} records")
    else:
        import random as _random
        all_suffix_tasks = []

        # Random-repair configs
        for p_val in RANDOM_PS:
            repair_decisions: Dict[Any, List[int]] = {}
            for q in questions:
                did = q["doc_id"]
                ds = drafts_by_q.get(did, [])
                if not ds:
                    continue
                steps = ds[0].get("draft_steps", [])
                n = len(steps)
                if n < 2:
                    continue
                rng = _random.Random(42 + hash(did))
                repairs = []
                for t in range(n):
                    if len(repairs) >= MAX_REPAIRS:
                        break
                    if rng.random() < p_val:
                        repairs.append(t)
                if repairs:
                    repair_decisions[did] = repairs

            tag = f"random_p{p_val}"
            tasks = build_repair_suffix_tasks(
                questions, drafts_by_q, repair_decisions,
                model_id, dataset, REPAIR_K, tag,
            )
            all_suffix_tasks.extend(tasks)

        # Entropy lookback repair configs
        all_vals = []
        for sm in sm_map.values():
            for m in sm:
                v = m.get("logprob_drop")
                if v is not None:
                    all_vals.append(v)
        if all_vals:
            all_vals_arr = np.array(all_vals)
            for pct in ENTROPY_PCTS:
                thr = float(np.percentile(all_vals_arr, pct))
                repair_decisions = {}
                for q in questions:
                    did = q["doc_id"]
                    sm = sm_map.get(did, [])
                    if len(sm) < 2:
                        continue
                    repairs = decide_entropy_repairs(
                        sm, "lookback", "logprob_drop", thr, MAX_REPAIRS,
                    )
                    if repairs:
                        repair_decisions[did] = repairs
                tag = f"entropy_lookback_pct{pct}"
                tasks = build_repair_suffix_tasks(
                    questions, drafts_by_q, repair_decisions,
                    model_id, dataset, REPAIR_K, tag,
                )
                all_suffix_tasks.extend(tasks)

        # Entropy lookahead repair configs
        all_ent_vals = []
        for sm in sm_map.values():
            for m in sm:
                v = m.get("mean_entropy")
                if v is not None:
                    all_ent_vals.append(v)
        if all_ent_vals:
            all_ent_arr = np.array(all_ent_vals)
            for pct in ENTROPY_PCTS:
                thr = float(np.percentile(all_ent_arr, 100 - pct))
                repair_decisions = {}
                for q in questions:
                    did = q["doc_id"]
                    sm = sm_map.get(did, [])
                    if len(sm) < 2:
                        continue
                    repairs = decide_entropy_repairs(
                        sm, "lookahead", "mean_entropy", thr, MAX_REPAIRS,
                    )
                    if repairs:
                        repair_decisions[did] = repairs
                tag = f"entropy_lookahead_pct{pct}"
                tasks = build_repair_suffix_tasks(
                    questions, drafts_by_q, repair_decisions,
                    model_id, dataset, REPAIR_K, tag,
                )
                all_suffix_tasks.extend(tasks)

        print(f"\n[Phase 5] Generating {len(all_suffix_tasks)} repair suffixes...")
        suffix_records = []
        for batch in _batched(all_suffix_tasks, BATCH_PER_GPU * len(gpu_ids)):
            new = _launch(script, args, batch, sd, gpu_ids, "repair")
            suffix_records.extend(new)
        _write_jsonl(suffix_file, suffix_records)
        print(f"Repair suffixes done: {len(suffix_records)} records")

    # -- Phase 6: Evaluate all methods --
    print("\n[Phase 6] Evaluating...")
    all_results = []

    gp = greedy_point(questions, drafts, dataset)
    all_results.append(gp)
    print(f"  Greedy: acc={gp['accuracy']:.4f} tok={gp['mean_tokens']:.0f}")

    sc_pts = sc_pareto_points(questions, sc_records, dataset, SC_NS)
    all_results.extend(sc_pts)
    for sp in sc_pts:
        print(f"  SC(N={sp['N']}): acc={sp['accuracy']:.4f} tok={sp['mean_tokens']:.0f}")

    # Group suffix records by config_tag
    sfx_by_tag: Dict[str, List[dict]] = {}
    for s in suffix_records:
        tag = s.get("config_tag", "")
        sfx_by_tag.setdefault(tag, []).append(s)

    for tag, sfx_list in sorted(sfx_by_tag.items()):
        rp = repair_evaluate(questions, drafts_by_q, sfx_list, dataset, tag)
        all_results.append(rp)
        print(f"  {tag}: acc={rp['accuracy']:.4f} tok={rp['mean_tokens']:.0f}")

    summary = {
        "model": model_id, "dataset": dataset,
        "n_questions": nq,
        "results": all_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_file = out_dir / "pareto_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSummary -> {summary_file}")
    print(f"Elapsed: {_fmt(time.time() - t0)}")

    try:
        sd.rmdir()
    except OSError:
        pass
    return summary


def main():
    args = parse_args()
    script = str(Path(__file__).resolve())

    if args._sid >= 0:
        if args._phase == "logprobs":
            run_shard_logprobs(args)
        else:
            run_shard_generation(args)
        return

    from src.pareto_engine import (
        extract_pareto_front, sc_pareto_points, greedy_point,
        repair_evaluate, decide_entropy_repairs, build_repair_suffix_tasks,
    )

    if args.batch or (not args.model_id and not args.dataset):
        models = MODELS
        datasets = DATASETS
        if args.model_id:
            models = [args.model_id]
        if args.dataset:
            datasets = [args.dataset]
        for model in models:
            for ds in datasets:
                print("\n" + "=" * 70)
                print(f"  {model}  x  {ds}")
                print("=" * 70)
                run_one_combo(model, ds, args)
        return

    if not args.model_id or not args.dataset:
        print("ERROR: --model-id and --dataset required (or use --batch)")
        sys.exit(1)
    run_one_combo(args.model_id, args.dataset, args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
