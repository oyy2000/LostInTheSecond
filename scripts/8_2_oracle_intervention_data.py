#!/usr/bin/env python3
"""
Oracle intervention dataset for KeyStep-SD.

For each question where the small-model draft is wrong:
  Trajectory A: baseline small-model chain (from 8_0 Phase A)
  Trajectory B: for each step k, replace step k with a 7B-generated step,
                then let 3B continue. Record Delta_k.
  Trajectory C: full 7B solution as reference.

Produces a dataset of (step_features, Delta_k) for training a gain predictor.

Multi-GPU: shards (question, step_k) pairs across GPUs.
Each shard loads 7B for intervention, then 3B for continuation.

Usage:
    python scripts/8_2_oracle_intervention_data.py \
        --draft-file results/gsm8k_3b_multi_sample/keystep_sd/small_drafts.jsonl \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keystep_utils import (
    build_chat_prompt,
    build_prefix_prompt,
    extract_boxed_answer,
    load_config,
    load_jsonl,
    normalize_answer,
    split_steps,
    write_jsonl,
)

def parse_args():
    ap = argparse.ArgumentParser(description="Oracle intervention dataset")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/keystep_sd.yaml"))
    ap.add_argument("--draft-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/keystep_sd/small_drafts.jsonl"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/keystep_sd/oracle"))
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--n-continuations", type=int, default=0,
                    help="Override oracle.n_continuations from config")
    ap.add_argument("--limit", type=int, default=0)
    # internal shard args
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_phase", default="", help=argparse.SUPPRESS)
    return ap.parse_args()

def run_shard(args):
    """Worker: generate interventions or continuations on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    cfg = load_config(args.config)
    gcfg = cfg.get("generation", {})
    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    phase = args._phase
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: "
          f"{len(tasks)} tasks, phase={phase}")

    if phase == "intervene":
        model_id = cfg["large_model"]
        sp = SamplingParams(
            temperature=0.0, max_tokens=256,
            stop=gcfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>"]) + ["\n\n"],
        )
    elif phase == "continue":
        model_id = cfg["small_model"]
        n_cont = args.n_continuations or cfg.get("oracle", {}).get("n_continuations", 8)
        sp = SamplingParams(
            temperature=gcfg.get("temperature", 0.7),
            top_p=gcfg.get("top_p", 0.9),
            max_tokens=gcfg.get("max_tokens", 1024),
            stop=gcfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>"]),
            n=n_cont,
        )
    else:
        model_id = cfg["large_model"]
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=gcfg.get("max_tokens", 1024),
            stop=gcfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>"]),
        )

    llm = LLM(
        model=model_id, tensor_parallel_size=1, trust_remote_code=True,
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
        max_model_len=cfg.get("max_model_len", 2048), dtype="half",
    )
    prompts = [t["prompt"] for t in tasks]
    outputs = llm.generate(prompts, sp)

    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for task, output in zip(tasks, outputs):
            rec = {k: v for k, v in task.items() if k != "prompt"}
            if phase == "continue":
                completions = []
                for o in output.outputs:
                    gen = o.text.strip()
                    prefix_text = "\n\n".join(task.get("prefix_steps", []))
                    full = prefix_text + ("\n\n" + gen if gen else "")
                    pred = extract_boxed_answer(full)
                    em = float(normalize_answer(pred) == normalize_answer(task["gold_answer"]))
                    completions.append({
                        "text": gen,
                        "pred_answer": pred,
                        "exact_match": em,
                        "n_tokens": len(o.token_ids),
                    })
                rec["completions"] = completions
                rec["recovery_rate"] = (
                    sum(c["exact_match"] for c in completions) / len(completions)
                    if completions else 0.0
                )
            else:
                gen = output.outputs[0].text.strip()
                rec["generated_text"] = gen
                rec["n_tokens"] = len(output.outputs[0].token_ids)
                if phase == "full_solve":
                    rec["pred_answer"] = extract_boxed_answer(gen)
                    rec["exact_match"] = float(
                        normalize_answer(rec["pred_answer"])
                        == normalize_answer(task["gold_answer"]))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[Shard {args._shard_id}] Done -> {out_path}")

def launch_shards(args, tasks, phase, shard_dir, gpu_ids):
    if not tasks:
        print(f"  [{phase}] Nothing to do")
        return []
    n_shards = len(gpu_ids)
    shard_tasks = [[] for _ in range(n_shards)]
    for i, t in enumerate(tasks):
        shard_tasks[i % n_shards].append(t)

    script = str(Path(__file__).resolve())
    procs = []
    shard_outs = []
    for si, gid in enumerate(gpu_ids):
        if not shard_tasks[si]:
            continue
        tf = shard_dir / f"tasks_{phase}_{si}.json"
        tf.write_text(json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8")
        so = shard_dir / f"shard_{phase}_{si}.jsonl"
        shard_outs.append(so)
        cmd = [
            sys.executable, script,
            "--config", args.config,
            "--n-continuations", str(args.n_continuations),
            "--_shard-id", str(si),
            "--_shard-out", str(so),
            "--_gpu-id", gid,
            "--_task-file", str(tf),
            "--_phase", phase,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        env["TOKENIZERS_PARALLELISM"] = "false"
        print(f"  [{phase}] Shard {si} on GPU {gid} ({len(shard_tasks[si])} tasks)")
        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gid, p, log_fh))

    failed = []
    for si, gid, p, log_fh in procs:
        p.wait()
        log_fh.close()
        rc = p.returncode
        log_path = shard_dir / f"log_{si}.txt"
        text = log_path.read_text("utf-8", errors="replace")
        lines = text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"  [{phase}] Shard {si} (GPU {gid}) exit={rc}\n{tail}")
        if rc != 0:
            failed.append(si)
        else:
            log_path.unlink(missing_ok=True)
    if failed:
        print(f"ERROR: {phase} shards {failed} failed!")
        sys.exit(1)

    records = []
    for sf in shard_outs:
        if sf.exists():
            for line in sf.read_text("utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records

def main():
    args = parse_args()
    if args._shard_id >= 0:
        run_shard(args)
        return

    cfg = load_config(args.config)
    gpu_ids = (args.gpus or cfg.get("gpus", "0,1,2,3,4,5,6,7")).split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    drafts = load_jsonl(Path(args.draft_file))
    if not drafts:
        print(f"ERROR: no drafts in {args.draft_file}. Run 8_0 first.")
        sys.exit(1)

    wrong_drafts = [d for d in drafts if d.get("exact_match", 1.0) < 0.5]
    if args.limit > 0:
        wrong_drafts = wrong_drafts[:args.limit]
    print(f"Wrong drafts: {len(wrong_drafts)} / {len(drafts)}")

    t_start = time.time()

    # --- Phase 1: 7B generates one replacement step per (question, step_k) ---
    interv_file = out_dir / "interventions.jsonl"
    if interv_file.exists() and load_jsonl(interv_file):
        print("Phase 1 (intervene): cached, skipping.")
        interventions = load_jsonl(interv_file)
    else:
        print(f"\n{'='*60}\nPhase 1: 7B single-step interventions\n{'='*60}")
        tasks = []
        for d in wrong_drafts:
            steps = d.get("steps", [])
            for k in range(len(steps)):
                prefix_steps = steps[:k]
                prompt = build_chat_prompt(d["question"])
                if prefix_steps:
                    prompt += "\n\n".join(prefix_steps) + "\n\n"
                tasks.append({
                    "doc_id": d["doc_id"],
                    "step_idx": k,
                    "question": d["question"],
                    "gold_answer": d["gold_answer"],
                    "prefix_steps": prefix_steps,
                    "original_step": steps[k],
                    "prompt": prompt,
                })
        interventions = launch_shards(args, tasks, "intervene", shard_dir, gpu_ids)
        for iv in interventions:
            iv["large_step_text"] = iv.pop("generated_text", "").strip()
        write_jsonl(interv_file, interventions)
        print(f"Phase 1 done: {len(interventions)} interventions")

    # --- Phase 2: 3B continues from each modified prefix ---
    cont_file = out_dir / "continuations.jsonl"
    if cont_file.exists() and load_jsonl(cont_file):
        print("Phase 2 (continue): cached, skipping.")
        cont_records = load_jsonl(cont_file)
    else:
        print(f"\n{'='*60}\nPhase 2: 3B continuations from modified prefixes\n{'='*60}")
        draft_map = {d["doc_id"]: d for d in wrong_drafts}
        tasks = []
        for iv in interventions:
            d = draft_map.get(iv["doc_id"])
            if not d:
                continue
            steps = list(d["steps"])
            k = iv["step_idx"]
            steps[k] = iv["large_step_text"]
            prefix_steps = steps[: k + 1]
            prompt = build_prefix_prompt(d["question"], prefix_steps)
            tasks.append({
                "doc_id": iv["doc_id"],
                "step_idx": k,
                "question": d["question"],
                "gold_answer": d["gold_answer"],
                "prefix_steps": prefix_steps,
                "prompt": prompt,
            })
        cont_records = launch_shards(args, tasks, "continue", shard_dir, gpu_ids)
        write_jsonl(cont_file, cont_records)
        print(f"Phase 2 done: {len(cont_records)} continuation sets")

    # --- Phase 3: 7B full solve (reference) ---
    full_file = out_dir / "full_7b_solve.jsonl"
    if full_file.exists() and load_jsonl(full_file):
        print("Phase 3 (full_solve): cached, skipping.")
        full_records = load_jsonl(full_file)
    else:
        print(f"\n{'='*60}\nPhase 3: 7B full solutions\n{'='*60}")
        seen = set()
        tasks = []
        for d in wrong_drafts:
            if d["doc_id"] in seen:
                continue
            seen.add(d["doc_id"])
            tasks.append({
                "doc_id": d["doc_id"],
                "question": d["question"],
                "gold_answer": d["gold_answer"],
                "prompt": build_chat_prompt(d["question"]),
            })
        full_records = launch_shards(args, tasks, "full_solve", shard_dir, gpu_ids)
        write_jsonl(full_file, full_records)
        print(f"Phase 3 done: {len(full_records)} full solutions")

    # --- Phase 4: assemble oracle dataset ---
    print(f"\n{'='*60}\nPhase 4: Assemble oracle dataset\n{'='*60}")
    cont_map = {}
    for cr in cont_records:
        key = (cr["doc_id"], cr["step_idx"])
        cont_map[key] = cr

    oracle_records = []
    for iv in interventions:
        key = (iv["doc_id"], iv["step_idx"])
        cr = cont_map.get(key)
        if not cr:
            continue
        delta = cr.get("recovery_rate", 0.0)
        rec = {
            "doc_id": iv["doc_id"],
            "step_idx": iv["step_idx"],
            "question": iv.get("question", ""),
            "gold_answer": iv.get("gold_answer", ""),
            "original_step": iv.get("original_step", ""),
            "large_step_text": iv.get("large_step_text", ""),
            "recovery_rate": delta,
            "delta_k": delta,
            "n_tokens_intervention": iv.get("n_tokens", 0),
        }
        oracle_records.append(rec)

    oracle_file = out_dir / "oracle_dataset.jsonl"
    write_jsonl(oracle_file, oracle_records)

    total_time = time.time() - t_start
    print(f"\nOracle dataset: {len(oracle_records)} records -> {oracle_file}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
