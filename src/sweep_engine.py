"""
Multi-GPU sweep engine for Late Rollback experiments.

Terminology
-----------
- n_drafts (nd): number of independent greedy drafts per question.
- K: per-draft sample budget. Each draft produces K total answers:
      1 greedy answer + (K-1) suffix continuations from the rollback point.
      K=2 means 1 extra suffix per draft.
- alpha: rollback fraction. Cut the draft at step ceil(alpha * T).
- Total answers per question = nd * K (all participate in majority vote).

Generation phases
-----------------
  1. Greedy drafts: nd_max greedy completions per question.
  2. Suffixes: for each (draft, alpha), generate (K_max - 1) suffix
     continuations from the prefix cut point.
  3. Full SC baseline: cfg.fullsc_n independent sampled completions
     (fixed, not tied to nd*K).

Public API
----------
- SweepConfig: dataclass with all hyperparameters
- run_shard(args): subprocess entry point (one GPU)
- run_sweep(cfg, script_path): full pipeline (parent process)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent

# Max tasks per GPU per launch round. Prevents OOM from sending millions of
# prompts to vLLM at once. Each GPU gets at most this many tasks; the rest
# are queued for the next round. Checkpoint is saved between rounds.
BATCH_PER_GPU = 8000


def _batched(lst: list, n: int):
    """Yield successive chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@dataclass
class SweepConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dataset: str = "gsm8k"
    out_dir: str = ""
    gpus: str = "0,1,2,3,4,5,6,7"
    n_sample: int = 0
    seed: int = 42
    n_drafts_list: List[int] = field(default_factory=lambda: [2, 4, 8])
    alphas: List[float] = field(default_factory=lambda: [0.6, 0.8])
    Ks: List[int] = field(default_factory=lambda: [2, 3, 4])
    fullsc_n: int = 40  # fixed number of Full SC independent samples
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    gpu_memory_utilization: float = 0.92
    max_model_len: int = 4096


# -- helpers ---------------------------------------------------------------

COMPETITION_DATASETS = {"aime2024", "amc2023", "olympiadbench"}


def _effective_tokens(cfg: SweepConfig) -> tuple:
    """Return (max_tokens, max_model_len) scaled up for competition datasets."""
    if cfg.dataset.lower() in COMPETITION_DATASETS:
        return max(cfg.max_tokens, 4096), max(cfg.max_model_len, 8192)
    return cfg.max_tokens, cfg.max_model_len

def _fmt(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _append_jsonl(path: Path, records: List[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


# -- shard worker (one per GPU, called via subprocess) ---------------------

def run_shard(args: argparse.Namespace):
    """Generate completions for a batch of tasks on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from src.prompt_templates import get_stop_tokens, split_steps, extract_answer

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    print(f"[Shard {args._sid}] GPU {args._gpu}: {len(tasks)} tasks")

    stop = get_stop_tokens(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    llm = LLM(
        model=args.model_id, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sp_g = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=stop)
    sp_s = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop=stop,
    )

    prompt_ids = [
        tokenizer.encode(t["prompt"], add_special_tokens=False) for t in tasks
    ]
    gi = [i for i, t in enumerate(tasks) if t.get("mode") == "greedy"]
    si = [i for i, t in enumerate(tasks) if t.get("mode") != "greedy"]

    res = [None] * len(tasks)
    if gi:
        o = llm.generate(prompt_token_ids=[prompt_ids[i] for i in gi],
                         sampling_params=sp_g)
        for j, idx in enumerate(gi):
            res[idx] = o[j]
    if si:
        o = llm.generate(prompt_token_ids=[prompt_ids[i] for i in si],
                         sampling_params=sp_s)
        for j, idx in enumerate(si):
            res[idx] = o[j]

    ds = args.dataset
    out_records = []
    for task, output in zip(tasks, res):
        text = output.outputs[0].text.strip()
        n_tok = len(output.outputs[0].token_ids)
        tt = task["task_type"]
        if tt == "draft":
            steps = split_steps(text)
            pred = extract_answer(ds, text)
            out_records.append({
                "task_type": "draft", "doc_id": task["doc_id"],
                "draft_idx": task["draft_idx"], "mode": task["mode"],
                "draft_text": text, "draft_steps": steps,
                "draft_answer": pred, "draft_tokens": n_tok,
            })
        elif tt == "suffix":
            full = task["prefix_text"] + "\n\n" + text
            pred = extract_answer(ds, full)
            out_records.append({
                "task_type": "suffix", "doc_id": task["doc_id"],
                "draft_idx": task["draft_idx"],
                "alpha": task["alpha"], "suffix_idx": task["suffix_idx"],
                "suffix_text": text, "pred_answer": pred,
                "suffix_tokens": n_tok,
            })
        elif tt == "fullsc":
            pred = extract_answer(ds, text)
            out_records.append({
                "task_type": "fullsc", "doc_id": task["doc_id"],
                "sample_idx": task["sample_idx"],
                "text": text, "pred_answer": pred, "tokens": n_tok,
            })

    Path(args._out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records),
        encoding="utf-8",
    )
    print(f"[Shard {args._sid}] done -> {args._out}")


# -- launch helper ---------------------------------------------------------

def _launch(
    script_path: str,
    args: argparse.Namespace,
    tasks: List[dict],
    shard_dir: Path,
    gpu_ids: List[str],
) -> List[dict]:
    """Split tasks across GPUs, run subprocesses, collect results."""
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
        tf = shard_dir / f"t_{si}.json"
        tf.write_text(json.dumps(shards[si], ensure_ascii=False), encoding="utf-8")
        of = shard_dir / f"o_{si}.jsonl"
        outs.append(of)
        log_file = shard_dir / f"log_{si}.txt"
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
        (shard_dir / f"t_{si}.json").unlink(missing_ok=True)
    return recs


# -- task builders ---------------------------------------------------------

def _build_draft_tasks(
    questions: List[dict], model_id: str, dataset: str,
    nd_max: int, existing: List[dict], tokenizer=None,
) -> List[dict]:
    from src.prompt_templates import build_prompt
    done = {(r["doc_id"], r["draft_idx"]) for r in existing if r["task_type"] == "draft"}
    tasks = []
    for q in questions:
        p = build_prompt(model_id, dataset, q["question"], tokenizer=tokenizer)
        for di in range(nd_max):
            if (q["doc_id"], di) in done:
                continue
            tasks.append({
                "task_type": "draft", "doc_id": q["doc_id"],
                "draft_idx": di,
                "mode": "greedy" if di == 0 else "sampled",
                "gold_answer": q["gold_answer"], "prompt": p,
            })
    return tasks


def _build_suffix_tasks(
    questions: List[dict], model_id: str, dataset: str,
    drafts: List[dict], nd_max: int, K_max: int,
    alphas: List[float], existing: List[dict],
    tokenizer=None,
) -> List[dict]:
    """K_max = max per-draft budget. Each draft needs (K_max - 1) suffixes."""
    from src.prompt_templates import build_prompt, split_steps
    n_sfx_per_draft = K_max - 1
    if n_sfx_per_draft < 1:
        return []

    done = {
        (r["doc_id"], r["draft_idx"], r["alpha"], r["suffix_idx"])
        for r in existing if r["task_type"] == "suffix"
    }
    draft_map: Dict[tuple, dict] = {}
    for d in drafts:
        if d["task_type"] == "draft":
            draft_map[(d["doc_id"], d["draft_idx"])] = d

    tasks = []
    for q in questions:
        did = q["doc_id"]
        p_base = build_prompt(model_id, dataset, q["question"], tokenizer=tokenizer)
        for di in range(nd_max):
            d = draft_map.get((did, di))
            if not d:
                continue
            steps = d.get("draft_steps") or split_steps(d.get("draft_text", ""))
            T = len(steps)
            if T == 0:
                continue
            for alpha in alphas:
                b = max(1, math.ceil(alpha * T))
                if b >= T:
                    b = T - 1
                if b < 1:
                    b = 1
                prefix = "\n\n".join(steps[:b])
                p = p_base + prefix + "\n\n"
                for si in range(n_sfx_per_draft):
                    if (did, di, alpha, si) in done:
                        continue
                    tasks.append({
                        "task_type": "suffix", "doc_id": did,
                        "alpha": alpha, "suffix_idx": si, "draft_idx": di,
                        "gold_answer": q["gold_answer"],
                        "prefix_text": prefix, "prompt": p,
                        "rollback_step": b, "draft_n_steps": T,
                        "mode": "sampled",
                    })
    return tasks


def _build_fullsc_tasks(
    questions: List[dict], model_id: str, dataset: str,
    n_total: int, existing: List[dict], tokenizer=None,
) -> List[dict]:
    """n_total = nd_max * K_max independent samples for fair Full SC comparison."""
    from src.prompt_templates import build_prompt
    done = {(r["doc_id"], r["sample_idx"]) for r in existing if r["task_type"] == "fullsc"}
    tasks = []
    for q in questions:
        p = build_prompt(model_id, dataset, q["question"], tokenizer=tokenizer)
        for si in range(n_total):
            if (q["doc_id"], si) in done:
                continue
            tasks.append({
                "task_type": "fullsc", "doc_id": q["doc_id"],
                "sample_idx": si, "gold_answer": q["gold_answer"],
                "prompt": p, "mode": "sampled",
            })
    return tasks


# -- evaluation ------------------------------------------------------------

def _answer_kw(q: dict) -> dict:
    kw = {}
    if "test" in q:
        kw["test"] = q["test"]
        kw["entry_point"] = q.get("entry_point", "")
    return kw


def _vote(answers: List[str]) -> str:
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def evaluate(cfg: SweepConfig, questions: List[dict], all_records: List[dict]) -> dict:
    """
    Evaluate all (nd, alpha, K) combos + Full SC baselines.

    Late Rollback with nd drafts and per-draft budget K:
      total answers = nd * K = nd greedy + nd * (K-1) suffixes.
    Full SC with same total budget = nd * K independent samples.
    """
    from src.prompt_templates import check_answer

    q_map = {q["doc_id"]: q for q in questions}
    nq = len(questions)

    drafts_by_q: Dict[str, List[dict]] = {}
    sfx_by_q: Dict[str, List[dict]] = {}
    fullsc_by_q: Dict[str, List[dict]] = {}
    for r in all_records:
        did = r["doc_id"]
        tt = r["task_type"]
        if tt == "draft":
            drafts_by_q.setdefault(did, []).append(r)
        elif tt == "suffix":
            sfx_by_q.setdefault(did, []).append(r)
        elif tt == "fullsc":
            fullsc_by_q.setdefault(did, []).append(r)

    results = []

    # greedy baseline (draft 0)
    nc = 0
    greedy_tpq = 0.0
    for q in questions:
        ds = drafts_by_q.get(q["doc_id"], [])
        if ds:
            d0 = sorted(ds, key=lambda x: x["draft_idx"])[0]
            kw = _answer_kw(q)
            if check_answer(cfg.dataset, d0.get("draft_answer", ""), q["gold_answer"], **kw):
                nc += 1
            greedy_tpq += d0.get("draft_tokens", 0)
    greedy_acc = nc / nq if nq else 0
    greedy_tpq = greedy_tpq / nq if nq else 0
    results.append({
        "method": "Greedy", "n_drafts": 0, "K": 1,
        "alpha": "-", "accuracy": round(greedy_acc, 4),
        "tpq": round(greedy_tpq, 1),
    })

    # Full SC baselines: evaluate at each unique LR budget level + fullsc_n
    budget_levels = sorted(set(
        [nd * K for nd in cfg.n_drafts_list for K in cfg.Ks]
        + [cfg.fullsc_n]
    ))
    for budget in budget_levels:
        nc, tt = 0, 0.0
        for q in questions:
            kw = _answer_kw(q)
            sc = fullsc_by_q.get(q["doc_id"], [])
            answers = [r["pred_answer"] for r in sc[:budget]]
            if not answers:
                continue
            if check_answer(cfg.dataset, _vote(answers), q["gold_answer"], **kw):
                nc += 1
            tt += sum(r.get("tokens", 0) for r in sc[:budget])
        acc = nc / nq if nq else 0
        tpq = tt / nq if nq else 0
        cm = tpq / greedy_tpq if greedy_tpq > 0 else 999
        results.append({
            "method": "FullSC", "n_drafts": 0, "K": budget,
            "budget": budget, "alpha": "-",
            "accuracy": round(acc, 4),
            "tpq": round(tpq, 1), "multiplier": round(cm, 2),
        })

    # Late Rollback: nd drafts, K per-draft budget -> (K-1) suffixes each
    for nd in cfg.n_drafts_list:
        for K in cfg.Ks:
            n_sfx = K - 1
            for alpha in cfg.alphas:
                nc, tt = 0, 0.0
                for q in questions:
                    did = q["doc_id"]
                    kw = _answer_kw(q)
                    answers = []
                    ds = sorted(
                        drafts_by_q.get(did, []),
                        key=lambda x: x["draft_idx"],
                    )
                    for d in ds[:nd]:
                        di = d["draft_idx"]
                        answers.append(d.get("draft_answer", ""))
                        tt += d.get("draft_tokens", 0)
                        sfx = sorted(
                            [r for r in sfx_by_q.get(did, [])
                             if r["draft_idx"] == di and r["alpha"] == alpha],
                            key=lambda x: x["suffix_idx"],
                        )
                        for s in sfx[:n_sfx]:
                            answers.append(s["pred_answer"])
                            tt += s.get("suffix_tokens", 0)

                    if answers and check_answer(
                        cfg.dataset, _vote(answers), q["gold_answer"], **kw,
                    ):
                        nc += 1

                acc = nc / nq if nq else 0
                tpq = tt / nq if nq else 0
                cm = tpq / greedy_tpq if greedy_tpq > 0 else 999
                gain = acc - greedy_acc
                results.append({
                    "method": "LateRollback",
                    "n_drafts": nd, "K": K, "budget": nd * K,
                    "alpha": alpha,
                    "accuracy": round(acc, 4),
                    "tpq": round(tpq, 1), "multiplier": round(cm, 2),
                    "gain": round(gain, 4),
                })

    return {
        "model": cfg.model_id,
        "dataset": cfg.dataset,
        "n_questions": nq,
        "greedy_acc": round(greedy_acc, 4),
        "results": results,
    }


# -- main entry point ------------------------------------------------------

def run_sweep(cfg: SweepConfig, script_path: str) -> dict:
    """Full pipeline: load -> 3-phase generation -> evaluate -> save."""
    from src.sweep_datasets import load_dataset_by_name

    if not cfg.out_dir:
        model_short = cfg.model_id.split("/")[-1].lower().replace("-", "_")
        cfg.out_dir = str(ROOT / "results" / f"{cfg.dataset}_{model_short}_sweep")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = out / "_shards"
    sd.mkdir(parents=True, exist_ok=True)
    gpu_ids = [g.strip() for g in cfg.gpus.split(",") if g.strip()]

    nd_max = max(cfg.n_drafts_list)
    K_max = max(cfg.Ks)
    fullsc_budget = cfg.fullsc_n

    t0 = time.time()
    questions = load_dataset_by_name(cfg.dataset, cfg.n_sample, cfg.seed)
    nq = len(questions)
    print("=" * 60)
    print(f"{cfg.dataset} sweep ({nq} questions)")
    print(f"  model: {cfg.model_id}")
    print(f"  n_drafts: {cfg.n_drafts_list}, Ks: {cfg.Ks}, alphas: {cfg.alphas}")
    print(f"  nd_max={nd_max}, K_max={K_max}, fullsc_n={fullsc_budget}")
    print(f"  GPUs: {gpu_ids}")

    eff_max_tokens, eff_max_model_len = _effective_tokens(cfg)
    print(f"  max_tokens={eff_max_tokens}, max_model_len={eff_max_model_len}"
          f"{' (competition upscaled)' if cfg.dataset.lower() in COMPETITION_DATASETS else ''}")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)

    ckpt = out / "checkpoint.jsonl"
    existing = _load_jsonl(ckpt)
    print(f"Checkpoint: {len(existing)} records loaded")

    ns = argparse.Namespace(
        model_id=cfg.model_id, dataset=cfg.dataset,
        temperature=cfg.temperature, top_p=cfg.top_p,
        max_tokens=eff_max_tokens,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=eff_max_model_len,
    )

    # Phase 1: greedy drafts (nd_max per question)
    draft_tasks = _build_draft_tasks(
        questions, cfg.model_id, cfg.dataset, nd_max, existing,
        tokenizer=tokenizer,
    )
    n_cached_drafts = nd_max * nq - len(draft_tasks)
    print(f"\n[Phase 1] Drafts: {n_cached_drafts} cached, {len(draft_tasks)} pending")
    if draft_tasks:
        for batch_i, batch in enumerate(_batched(draft_tasks, BATCH_PER_GPU * len(gpu_ids))):
            print(f"  [Phase 1 batch {batch_i}] {len(batch)} tasks")
            new = _launch(script_path, ns, batch, sd, gpu_ids)
            _append_jsonl(ckpt, new)
            existing.extend(new)

    # Phase 2: suffixes ((K_max - 1) per draft per alpha)
    sfx_tasks = _build_suffix_tasks(
        questions, cfg.model_id, cfg.dataset,
        existing, nd_max, K_max, cfg.alphas, existing,
        tokenizer=tokenizer,
    )
    print(f"[Phase 2] Suffixes: {len(sfx_tasks)} pending")
    if sfx_tasks:
        for batch_i, batch in enumerate(_batched(sfx_tasks, BATCH_PER_GPU * len(gpu_ids))):
            print(f"  [Phase 2 batch {batch_i}] {len(batch)} tasks")
            new = _launch(script_path, ns, batch, sd, gpu_ids)
            _append_jsonl(ckpt, new)
            existing.extend(new)

    # Phase 3: Full SC (nd_max * K_max independent samples)
    sc_tasks = _build_fullsc_tasks(
        questions, cfg.model_id, cfg.dataset, fullsc_budget, existing,
        tokenizer=tokenizer,
    )
    print(f"[Phase 3] Full SC: {len(sc_tasks)} pending")
    if sc_tasks:
        for batch_i, batch in enumerate(_batched(sc_tasks, BATCH_PER_GPU * len(gpu_ids))):
            print(f"  [Phase 3 batch {batch_i}] {len(batch)} tasks")
            new = _launch(script_path, ns, batch, sd, gpu_ids)
            _append_jsonl(ckpt, new)
            existing.extend(new)

    # Evaluate
    summary = evaluate(cfg, questions, existing)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    (out / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSummary -> {out / 'sweep_summary.json'}")
    print(f"Elapsed: {_fmt(time.time() - t0)}")

    try:
        sd.rmdir()
    except OSError:
        pass
    return summary
