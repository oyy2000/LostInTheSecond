#!/usr/bin/env python3
"""
Entropy-guided state-aware late rollback.

Pipeline:
  1. Load greedy drafts from an existing sweep checkpoint.
  2. Run vLLM prompt_logprobs forward pass to get per-token logprobs.
  3. Compute per-step entropy scores.
  4. Select dynamic rollback point t* per draft.
  5. Generate suffix continuations from t*.
  6. Evaluate accuracy + cost vs fixed-alpha baselines.

Multi-GPU via subprocess sharding (same pattern as sweep_engine).

Usage:
    python scripts/16_0_entropy_guided_later.py \
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


def _append_jsonl(path: Path, records: List[dict]):
    with path.open("a", encoding="utf-8") as f:
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


# -- CLI -------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Entropy-guided state-aware late rollback")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--sweep-dir", default="",
                    help="Directory with existing sweep checkpoint.jsonl")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n-drafts", type=int, default=4)
    ap.add_argument("--K", type=int, default=4, help="Per-draft budget (1 greedy + K-1 suffixes)")
    ap.add_argument("--betas", default="0.4,0.5,0.6")
    ap.add_argument("--methods", default="argmax,max_drop",
                    help="Rollback selection methods")
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


# -- shard worker: logprob scoring -----------------------------------------

def run_shard_logprobs(args):
    """Compute prompt_logprobs for each draft on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} logprob tasks")

    llm = LLM(
        model=args.model_id, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=0.70,
        max_model_len=2048,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    BATCH = min(args.batch_size, 4)  # small batches for logprob phase (prompt_logprobs is memory-heavy)
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
            print(f"[Shard {args._sid}] logprob batch "
                  f"{bi // BATCH + 1}/{(len(tasks) + BATCH - 1) // BATCH}")
    print(f"[Shard {args._sid}] logprobs done -> {out_path}")


# -- shard worker: suffix generation ---------------------------------------

def run_shard_suffix(args):
    """Generate suffix continuations on a single GPU."""
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
            "beta": task["beta"], "rb_method": task["rb_method"],
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
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                             env=env)
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

def main():
    args = parse_args()
    script = str(Path(__file__).resolve())

    # Dispatch to shard workers
    if args._sid >= 0:
        if args._phase == "logprobs":
            run_shard_logprobs(args)
        elif args._phase == "suffix":
            run_shard_suffix(args)
        return

    from src.prompt_templates import (
        build_prompt, split_steps, extract_answer, check_answer,
    )
    from src.step_scorers import (
        EntropyScorer, step_char_boundaries, select_rollback_point,
    )
    from src.sweep_datasets import load_dataset_by_name

    # Resolve directories
    model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
    if not args.sweep_dir:
        args.sweep_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_sweep")
    if not args.out_dir:
        args.out_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_entropy_later")

    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = out_dir / "_shards"
    sd.mkdir(parents=True, exist_ok=True)
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]

    betas = [float(b) for b in args.betas.split(",")]
    rb_methods = [m.strip() for m in args.methods.split(",")]
    nd = args.n_drafts
    K = args.K
    n_sfx = K - 1

    t0 = time.time()
    questions = load_dataset_by_name(args.dataset, 0, 42)
    q_map = {q["doc_id"]: q for q in questions}
    nq = len(questions)

    print("=" * 60)
    print(f"Entropy-guided LATER: {args.dataset} ({nq} questions)")
    print(f"  model: {args.model_id}")
    print(f"  n_drafts={nd}, K={K}, betas={betas}, methods={rb_methods}")
    print(f"  GPUs: {gpu_ids}")
    print("=" * 60)

    # Phase 1: Load existing drafts from sweep checkpoint
    ckpt_path = sweep_dir / "checkpoint.jsonl"
    if not ckpt_path.exists():
        print(f"ERROR: sweep checkpoint not found: {ckpt_path}")
        sys.exit(1)

    all_sweep = _load_jsonl(ckpt_path)
    drafts = [r for r in all_sweep if r.get("task_type") == "draft"]
    print(f"Loaded {len(drafts)} drafts from sweep checkpoint")

    draft_map: Dict[Tuple[Any, int], dict] = {}
    for d in drafts:
        draft_map[(d["doc_id"], d["draft_idx"])] = d

    # Phase 2: Forward scoring (logprobs)
    logprob_file = out_dir / "draft_logprobs.jsonl"
    if logprob_file.exists() and _load_jsonl(logprob_file):
        logprob_records = _load_jsonl(logprob_file)
        print(f"Logprobs cached: {len(logprob_records)} records")
    else:
        lp_tasks = []
        for q in questions:
            did = q["doc_id"]
            prompt = build_prompt(args.model_id, args.dataset, q["question"])
            for di in range(nd):
                d = draft_map.get((did, di))
                if not d:
                    continue
                full = prompt + d["draft_text"]
                lp_tasks.append({
                    "doc_id": did, "draft_idx": di,
                    "draft_text": d["draft_text"],
                    "draft_steps": d.get("draft_steps", split_steps(d["draft_text"])),
                    "full_prompt": full,
                    "resp_char_offset": len(prompt),
                })

        print(f"\n[Phase 2] Logprob scoring: {len(lp_tasks)} tasks")
        logprob_records = []
        for batch_i, batch in enumerate(_batched(lp_tasks, BATCH_PER_GPU * len(gpu_ids))):
            print(f"  [batch {batch_i}] {len(batch)} tasks")
            new = _launch(script, args, batch, sd, gpu_ids, "logprobs")
            logprob_records.extend(new)

        with logprob_file.open("w", encoding="utf-8") as f:
            for r in logprob_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Logprobs done: {len(logprob_records)} records -> {logprob_file}")

    # Phase 3: Compute entropy scores and select rollback points
    scorer = EntropyScorer(metric="entropy")
    lp_map: Dict[Tuple[Any, int], dict] = {}
    for r in logprob_records:
        lp_map[(r["doc_id"], r["draft_idx"])] = r

    rollback_points: Dict[Tuple[Any, int, float, str], int] = {}
    entropy_details = []

    for q in questions:
        did = q["doc_id"]
        for di in range(nd):
            lp_rec = lp_map.get((did, di))
            if not lp_rec:
                continue
            steps = lp_rec.get("draft_steps", [])
            if not steps or not lp_rec.get("token_logprobs"):
                continue

            response = lp_rec.get("draft_text", "\n\n".join(steps))
            boundaries = step_char_boundaries(response, steps)
            step_ent = scorer.score_draft(
                lp_rec["token_logprobs"], lp_rec["token_offsets"], boundaries,
            )

            for beta in betas:
                for method in rb_methods:
                    t_star = select_rollback_point(step_ent, beta=beta, method=method)
                    rollback_points[(did, di, beta, method)] = t_star

            entropy_details.append({
                "doc_id": did, "draft_idx": di,
                "n_steps": len(steps),
                "step_entropies": [round(e, 6) for e in step_ent],
            })

    ent_detail_file = out_dir / "entropy_details.jsonl"
    with ent_detail_file.open("w", encoding="utf-8") as f:
        for r in entropy_details:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Entropy details: {len(entropy_details)} drafts -> {ent_detail_file}")

    # Phase 4: Build suffix tasks from dynamic rollback points
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
                for beta in betas:
                    for method in rb_methods:
                        t_star = rollback_points.get((did, di, beta, method))
                        if t_star is None:
                            continue
                        b = max(1, min(t_star, len(steps) - 1))
                        prefix = "\n\n".join(steps[:b])
                        prompt = prompt_base + prefix + "\n\n"
                        for si in range(n_sfx):
                            sfx_tasks.append({
                                "doc_id": did, "draft_idx": di,
                                "beta": beta, "rb_method": method,
                                "rollback_step": b,
                                "suffix_idx": si,
                                "gold_answer": q["gold_answer"],
                                "prefix_text": prefix,
                                "prompt": prompt,
                            })

        print(f"\n[Phase 4] Suffix generation: {len(sfx_tasks)} tasks")
        suffix_records = []
        for batch_i, batch in enumerate(_batched(sfx_tasks, BATCH_PER_GPU * len(gpu_ids))):
            print(f"  [batch {batch_i}] {len(batch)} tasks")
            new = _launch(script, args, batch, sd, gpu_ids, "suffix")
            suffix_records.extend(new)

        with suffix_file.open("w", encoding="utf-8") as f:
            for r in suffix_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Suffixes done: {len(suffix_records)} records -> {suffix_file}")

    # Phase 5: Evaluate
    results = evaluate(args, questions, drafts, suffix_records, betas, rb_methods, nd, K)

    summary = {
        "model": args.model_id, "dataset": args.dataset,
        "n_questions": nq, "n_drafts": nd, "K": K,
        "betas": betas, "methods": rb_methods,
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_file = out_dir / "entropy_later_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSummary -> {summary_file}")
    print(f"Elapsed: {_fmt(time.time() - t0)}")

    for row in results:
        print(f"  {row['method']:30s}  acc={row['accuracy']:.4f}  "
              f"tpq={row.get('tpq', 0):.0f}")

    try:
        sd.rmdir()
    except OSError:
        pass


# -- evaluation ------------------------------------------------------------

def evaluate(args, questions, drafts, suffix_records, betas, rb_methods, nd, K):
    from src.prompt_templates import check_answer, extract_answer

    q_map = {q["doc_id"]: q for q in questions}
    nq = len(questions)

    drafts_by_q: Dict[Any, List[dict]] = {}
    for d in drafts:
        drafts_by_q.setdefault(d["doc_id"], []).append(d)

    sfx_by_key: Dict[Tuple, List[dict]] = {}
    for s in suffix_records:
        key = (s["doc_id"], s["draft_idx"], s["beta"], s["rb_method"])
        sfx_by_key.setdefault(key, []).append(s)

    def _answer_kw(q):
        kw = {}
        if "test" in q:
            kw["test"] = q["test"]
            kw["entry_point"] = q.get("entry_point", "")
        return kw

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
    results.append({
        "method": "Greedy", "accuracy": round(greedy_acc, 4),
        "tpq": round(greedy_tpq, 1),
    })

    # Entropy-guided LATER for each (beta, rb_method)
    n_sfx = K - 1
    for beta in betas:
        for method in rb_methods:
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
                        sfx_by_key.get((did, di, beta, method), []),
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
                "method": f"EntropyLATER(b={beta},m={method})",
                "beta": beta, "rb_method": method,
                "n_drafts": nd, "K": K, "budget": nd * K,
                "accuracy": round(acc, 4),
                "tpq": round(tpq, 1), "multiplier": round(cm, 2),
                "gain": round(acc - greedy_acc, 4),
            })

    return results


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
