#!/usr/bin/env python3
"""
Probe-guided state-aware late rollback.

Pipeline:
  1. Load greedy drafts from an existing sweep checkpoint.
  2. Run HF forward pass on 3B model to extract hidden states at step boundaries.
  3. Score each step with the frozen probe from 16_1.
  4. Select dynamic rollback point t* per draft.
  5. Generate suffix continuations from t*.
  6. Evaluate accuracy + cost vs fixed-alpha baselines.

Usage:
    python scripts/16_2_probe_guided_later.py \
        --dataset gsm8k \
        --model-id meta-llama/Llama-3.2-3B-Instruct \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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
    ap = argparse.ArgumentParser(description="Probe-guided state-aware late rollback")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--sweep-dir", default="")
    ap.add_argument("--probe-path", default="",
                    help="Path to best_probe.pkl from 16_1")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--scoring-gpu", default="auto",
                    help="GPU for HF hidden state extraction")
    ap.add_argument("--n-drafts", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--betas", default="0.4,0.5,0.6")
    ap.add_argument("--methods", default="argmax,max_drop")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    # internal shard args
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    return ap.parse_args()


def select_best_gpu(requested: str, min_free: int = 12000) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], encoding="utf-8",
        )
        free = {i: int(l.strip())
                for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return 0
    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        if usable:
            return max(usable, key=usable.get)
        return ids[0] if ids else 0
    candidates = {g: m for g, m in free.items() if m >= min_free}
    if candidates:
        return max(candidates, key=candidates.get)
    return max(free, key=free.get) if free else 0


# -- shard worker: suffix generation (reused from 16_0) --------------------

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


# -- hidden state extraction for drafts -----------------------------------

def extract_draft_hidden_states(
    model, tokenizer, drafts_info: List[dict],
    layer_idx: int, layer_idx_pos: int,
    max_seq_len: int,
) -> Dict[Tuple, np.ndarray]:
    """Extract hidden states at step boundaries for each draft.

    Returns dict mapping (doc_id, draft_idx) -> ndarray of shape (n_steps, hidden_dim).
    """
    import torch
    from src.prompt_templates import build_prompt, split_steps

    result = {}

    for info in drafts_info:
        did, di = info["doc_id"], info["draft_idx"]
        question = info["question"]
        draft_text = info["draft_text"]
        model_name = info.get("model_id", "")
        if not model_name:
            model_name = getattr(model.config, "_name_or_path",
                                 getattr(model.config, "name_or_path", ""))
        steps = info.get("draft_steps") or split_steps(draft_text)
        if not steps:
            continue

        prompt = build_prompt(model_name, info["dataset"], question)
        full_text = prompt + draft_text

        input_ids = tokenizer.encode(
            full_text, return_tensors="pt",
            add_special_tokens=False,
            truncation=True, max_length=max_seq_len,
        ).to(model.device)

        if input_ids.shape[1] < 10:
            continue

        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        boundary_positions = []
        current_text = prompt
        for step in steps:
            current_text += step
            step_end_ids = tokenizer.encode(current_text, add_special_tokens=False)
            pos = min(len(step_end_ids) - 1, input_ids.shape[1] - 1)
            if pos >= prompt_len:
                boundary_positions.append(pos)
            if "\n\n" in draft_text:
                current_text += "\n\n"

        if not boundary_positions:
            continue

        hidden_dim = hidden_states[0].shape[-1]
        hs = np.zeros((len(boundary_positions), hidden_dim), dtype=np.float32)
        for si, pos in enumerate(boundary_positions):
            if layer_idx < len(hidden_states):
                hs[si] = hidden_states[layer_idx][0, pos].float().cpu().numpy()

        result[(did, di)] = hs

    return result


# -- main pipeline ---------------------------------------------------------

def main():
    args = parse_args()
    script = str(Path(__file__).resolve())

    if args._sid >= 0:
        run_shard_suffix(args)
        return

    import torch
    from tqdm.auto import tqdm
    from src.prompt_templates import (
        build_prompt, split_steps, check_answer,
    )
    from src.step_scorers import ProbeScorer, select_rollback_point
    from src.sweep_datasets import load_dataset_by_name

    model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
    if not args.sweep_dir:
        args.sweep_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_sweep")
    if not args.probe_path:
        args.probe_path = str(
            ROOT / "results" / f"gsm8k_{model_short}_rollback_probe" / "best_probe.pkl"
        )
    if not args.out_dir:
        args.out_dir = str(ROOT / "results" / f"{args.dataset}_{model_short}_probe_later")

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
    print(f"Probe-guided LATER: {args.dataset} ({nq} questions)")
    print(f"  model: {args.model_id}")
    print(f"  probe: {args.probe_path}")
    print(f"  n_drafts={nd}, K={K}, betas={betas}")
    print("=" * 60)

    # Load probe
    probe_scorer = ProbeScorer(args.probe_path)
    layer_idx = probe_scorer.layer_idx
    with open(args.probe_path, "rb") as f:
        probe_ckpt = pickle.load(f)
    layer_idx_pos = probe_ckpt.get("layer_idx_pos", 0)
    print(f"Probe loaded: layer={layer_idx}, AUC={probe_ckpt.get('auc', '?')}")

    # Load drafts
    ckpt_path = sweep_dir / "checkpoint.jsonl"
    all_sweep = _load_jsonl(ckpt_path)
    drafts = [r for r in all_sweep if r.get("task_type") == "draft"]
    draft_map = {(d["doc_id"], d["draft_idx"]): d for d in drafts}
    print(f"Loaded {len(drafts)} drafts")

    # Phase 1: Extract hidden states and score with probe
    scores_file = out_dir / "probe_scores.jsonl"
    if scores_file.exists() and _load_jsonl(scores_file):
        score_records = _load_jsonl(scores_file)
        print(f"Probe scores cached: {len(score_records)} records")
        score_map = {(r["doc_id"], r["draft_idx"]): r["step_scores"]
                     for r in score_records}
    else:
        scoring_gpu = select_best_gpu(args.scoring_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(scoring_gpu)
        print(f"Scoring on GPU {scoring_gpu}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.float16
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=dtype, trust_remote_code=True,
            device_map="auto",
        )
        model.eval()

        drafts_info = []
        for q in questions:
            did = q["doc_id"]
            for di in range(nd):
                d = draft_map.get((did, di))
                if not d:
                    continue
                drafts_info.append({
                    "doc_id": did, "draft_idx": di,
                    "question": q["question"],
                    "dataset": args.dataset,
                    "model_id": args.model_id,
                    "draft_text": d["draft_text"],
                    "draft_steps": d.get("draft_steps"),
                })

        BATCH = 20
        all_hs = {}
        for bi in tqdm(range(0, len(drafts_info), BATCH), desc="Extracting hidden states"):
            batch = drafts_info[bi:bi + BATCH]
            batch_hs = extract_draft_hidden_states(
                model, tokenizer, batch,
                layer_idx, layer_idx_pos, args.max_seq_len,
            )
            all_hs.update(batch_hs)
            if (bi // BATCH) % 20 == 0:
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        score_map = {}
        score_records = []
        for (did, di), hs in all_hs.items():
            step_scores = probe_scorer.score_draft(hs)
            score_map[(did, di)] = step_scores
            score_records.append({
                "doc_id": did, "draft_idx": di,
                "n_steps": len(step_scores),
                "step_scores": [round(s, 6) for s in step_scores],
            })

        with scores_file.open("w", encoding="utf-8") as f:
            for r in score_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Probe scores: {len(score_records)} drafts -> {scores_file}")

    # Phase 2: Select rollback points
    rollback_points = {}
    for q in questions:
        did = q["doc_id"]
        for di in range(nd):
            step_scores = score_map.get((did, di))
            if not step_scores:
                continue
            for beta in betas:
                for method in rb_methods:
                    t_star = select_rollback_point(
                        step_scores, beta=beta, method=method,
                    )
                    rollback_points[(did, di, beta, method)] = t_star

    # Phase 3: Suffix generation
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
                                "rollback_step": b, "suffix_idx": si,
                                "gold_answer": q["gold_answer"],
                                "prefix_text": prefix, "prompt": prompt,
                            })

        print(f"\n[Phase 3] Suffix generation: {len(sfx_tasks)} tasks")
        suffix_records = []
        for batch_i, batch in enumerate(_batched(sfx_tasks, BATCH_PER_GPU * len(gpu_ids))):
            print(f"  [batch {batch_i}] {len(batch)} tasks")
            new = _launch_suffix(script, args, batch, sd, gpu_ids)
            suffix_records.extend(new)

        with suffix_file.open("w", encoding="utf-8") as f:
            for r in suffix_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Suffixes done: {len(suffix_records)} records")

    # Phase 4: Evaluate
    results = evaluate(args, questions, drafts, suffix_records, betas, rb_methods, nd, K)

    summary = {
        "model": args.model_id, "dataset": args.dataset,
        "n_questions": nq, "n_drafts": nd, "K": K,
        "probe_layer": layer_idx,
        "betas": betas, "methods": rb_methods,
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_file = out_dir / "probe_later_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSummary -> {summary_file}")
    for row in results:
        print(f"  {row['method']:30s}  acc={row['accuracy']:.4f}  "
              f"tpq={row.get('tpq', 0):.0f}")

    try:
        sd.rmdir()
    except OSError:
        pass


def evaluate(args, questions, drafts, suffix_records, betas, rb_methods, nd, K):
    from src.prompt_templates import check_answer

    nq = len(questions)
    drafts_by_q = {}
    for d in drafts:
        drafts_by_q.setdefault(d["doc_id"], []).append(d)

    sfx_by_key = {}
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
    n_sfx = K - 1

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
                "method": f"ProbeLATER(b={beta},m={method})",
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
