#!/common/users/sl2148/anaconda3/envs/vllmdebug/bin/python
"""
Step 4a: Collect per-token logprobs for incorrect GSM8K trajectories.

Reuses the multi-GPU sharding pattern from 10_0_collect_delimiter_entropy.py.
Feeds [prompt + response] to Qwen2.5-3B-Instruct via vLLM prompt_logprobs
and stores per-token logprobs/offsets for downstream step-level analysis.

Usage:
    python scripts/4_0_collect_step_logprobs.py --gpus 0,1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keystep_utils import build_chat_prompt, load_jsonl, write_jsonl

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error/gpt_first_error_cache.jsonl"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/step_uncertainty"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


def run_shard(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: {len(tasks)} tasks")

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="half",
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    BATCH = args.batch_size
    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for bi in range(0, len(tasks), BATCH):
            batch_tasks = tasks[bi:bi + BATCH]
            prompts = [t["full_prompt"] for t in batch_tasks]
            outputs = llm.generate(prompts, sp)

            for task, output in zip(batch_tasks, outputs):
                rec = {k: v for k, v in task.items() if k != "full_prompt"}
                resp_offset = task["resp_char_offset"]
                lps, offsets, texts = [], [], []
                cum_chars = 0
                if output.prompt_logprobs is not None:
                    prompt_token_ids = output.prompt_token_ids
                    for ti, lp_dict in enumerate(output.prompt_logprobs):
                        if lp_dict is None:
                            tok_id = prompt_token_ids[ti]
                            decoded = tokenizer.decode([tok_id])
                            cum_chars += len(decoded)
                            continue
                        tok_id = prompt_token_ids[ti]
                        if tok_id in lp_dict:
                            logprob_obj = lp_dict[tok_id]
                        else:
                            logprob_obj = next(iter(lp_dict.values()))
                        decoded = logprob_obj.decoded_token or ""
                        char_pos = cum_chars
                        cum_chars += len(decoded)
                        if char_pos >= resp_offset:
                            lps.append(logprob_obj.logprob)
                            offsets.append(char_pos - resp_offset)
                            texts.append(decoded)
                rec["token_logprobs"] = lps
                rec["token_offsets"] = offsets
                rec["token_texts"] = texts
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[Shard {args._shard_id}] batch {bi // BATCH + 1}/"
                  f"{(len(tasks) + BATCH - 1) // BATCH} done")
    print(f"[Shard {args._shard_id}] Done -> {out_path}")


def launch_shards(args, tasks: List[Dict], gpu_ids: List[str], out_dir: Path) -> List[Dict]:
    if not tasks:
        print("No tasks to process.")
        return []
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = len(gpu_ids)
    shard_tasks: List[List[Dict]] = [[] for _ in range(n_shards)]
    for i, t in enumerate(tasks):
        shard_tasks[i % n_shards].append(t)

    script_path = str(Path(__file__).resolve())
    PYTHON = "/common/users/sl2148/anaconda3/envs/vllmdebug/bin/python"
    procs = []
    shard_out_files: List[Path] = []

    for si, gpu_id in enumerate(gpu_ids):
        if not shard_tasks[si]:
            continue
        task_file = shard_dir / f"tasks_{si}.json"
        task_file.write_text(
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8")
        shard_out = shard_dir / f"shard_{si}.jsonl"
        shard_out_files.append(shard_out)
        cmd = [
            PYTHON, script_path,
            "--model-id", args.model_id,
            "--gpu-mem", str(args.gpu_mem),
            "--max-model-len", str(args.max_model_len),
            "--batch-size", str(args.batch_size),
            "--_shard-id", str(si),
            "--_shard-out", str(shard_out),
            "--_gpu-id", gpu_id,
            "--_task-file", str(task_file),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"
        print(f"  Shard {si} on GPU {gpu_id} ({len(shard_tasks[si])} tasks)")
        log_file = shard_dir / f"log_{si}.txt"
        log_fh = log_file.open("w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p, log_fh))

    failed = []
    for si, gpu_id, p, log_fh in procs:
        p.wait()
        log_fh.close()
        rc = p.returncode
        log_path = shard_dir / f"log_{si}.txt"
        text = log_path.read_text("utf-8", errors="replace")
        lines = text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"  Shard {si} (GPU {gpu_id}) exit={rc}\n{tail}")
        if rc != 0:
            failed.append(si)
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))
        else:
            log_path.unlink(missing_ok=True)
    if failed:
        print(f"ERROR: shards {failed} failed!")
        sys.exit(1)

    records: List[Dict] = []
    for sf in shard_out_files:
        if sf.exists():
            for line in sf.read_text("utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


def build_tasks(cache_file: str, limit: int, max_chars: int) -> List[Dict]:
    cache = load_jsonl(Path(cache_file))
    records = [r for r in cache if r.get("tau") is not None]
    if limit > 0:
        records = records[:limit]
    tasks = []
    skipped = 0
    for r in records:
        prompt = build_chat_prompt(r["question"])
        response = r["response"]
        full = prompt + response
        if max_chars > 0 and len(full) > max_chars:
            skipped += 1
            continue
        tasks.append({
            "doc_id": r["doc_id"],
            "sample_idx": r["sample_idx"],
            "tau": r["tau"],
            "steps": r["steps"],
            "n_steps": r["n_steps"],
            "response": response,
            "full_prompt": full,
            "resp_char_offset": len(prompt),
        })
    if skipped:
        print(f"  Skipped {skipped} trajectories exceeding max_chars={max_chars}")
    return tasks


def main():
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "token_logprobs.jsonl"
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    max_chars = args.max_model_len * 4

    if out_file.exists():
        existing = load_jsonl(out_file)
        if existing:
            print(f"Logprobs already exist: {out_file} ({len(existing)} records)")
            return

    tasks = build_tasks(args.cache_file, args.limit, max_chars)
    print(f"Tasks: {len(tasks)}, GPUs: {gpu_ids}")
    t0 = time.time()
    results = launch_shards(args, tasks, gpu_ids, out_dir)
    write_jsonl(out_file, results)
    print(f"Done: {len(results)} records, time={time.time() - t0:.1f}s -> {out_file}")


if __name__ == "__main__":
    main()
