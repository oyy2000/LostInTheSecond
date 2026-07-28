#!/usr/bin/env python3
"""
Budget-controlled rollback with pluggable error-step signals.

Extends 6_6 by supporting multiple rollback signal strategies, each with
two fallback modes when the signal does not trigger:
  - *_fb_last:  fallback to step n-1 (last step)
  - *_fb_start: fallback to step 0 (resample from scratch)

Signals:
  - gpt / gpt_fb_last / gpt_fb_start:
        Live GPT API call to locate first-error step (tau).
  - prm_drop / prm_drop_fb_last / prm_drop_fb_start:
        First i where score[i]-score[i+1] > threshold.
  - prm_max_drop_fb_last:
        Step with the largest adjacent PRM decrease.
  - prm_below_threshold_fb_last:
        First step whose absolute PRM score is below a threshold.
  - nll_drop / nll_drop_fb_last / nll_drop_fb_start:
        First i where nll[i]-nll[i-1] > threshold.

Each signal produces a rollback step per draft. Suffix generation and
majority-vote evaluation proceed identically to 6_6.

Usage:
    python scripts/6_8_budget_controlled_multisignal.py \\
        --budget 32 --gpus 0,1 \\
        --signals gpt gpt_fb_last gpt_fb_start \\
                  prm_drop prm_drop_fb_last prm_drop_fb_start \\
                  nll_drop nll_drop_fb_last nll_drop_fb_start
"""

import argparse, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
SKYWORK_PRM_ID = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
DATASET = "gsm8k"
PYTHON = sys.executable

PRM_DROP_THRESHOLD = 0.1
PRM_ABSOLUTE_THRESHOLD = 0.8
NLL_DROP_THRESHOLD = 0.2
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_MODEL_LEN = 4096
GPU_MEM = 0.90

ROLLBACK_CONFIGS = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2)]

ALL_SIGNALS = [
    "gpt_fb_last", "gpt_fb_start",
    "prm_drop", "prm_drop_fb_last", "prm_drop_fb_start",
    "prm_max_drop_fb_last", "prm_below_threshold_fb_last",
    "nll_drop", "nll_drop_fb_last", "nll_drop_fb_start",
]

GPT_MODEL = "gpt-5.1"
GPT_MAX_WORKERS = 32

from src.prompt_templates import (
    build_prompt, get_stop_tokens, split_steps,
    extract_answer, check_answer,
)
from src.hotpotqa_answer_equiv import hotpotqa_f1
from src.sweep_datasets import load_dataset_by_name
from src.step_judge import (
    build_first_error_prompt,
    call_gpt_first_error,
    load_env_file,
    make_client,
)
from rebuttal.src.prm_control_strategies import (
    MAX_PRM_DROP_SIGNAL,
    PRM_BELOW_THRESHOLD_SIGNAL,
    max_prm_drop_assignment,
    prm_below_threshold_assignment,
)
from rebuttal.src.wallclock import (
    RunTimingRecorder,
    WorkerTiming,
    discover_task_event_store,
    discover_timing_manifest,
    load_manifest,
    publish_per_question_timings,
    task_event_store_for_manifest,
    write_generic_worker_timing,
)


# SECTION: parse_args


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--model", default=MODEL_ID,
                    help="HuggingFace model ID for draft/suffix generation")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--n-sample", type=int, default=0,
                    help="Limit questions; 0=all")
    ap.add_argument("--signals", nargs="+",
                    default=ALL_SIGNALS,
                    choices=ALL_SIGNALS,
                    help="Which rollback signals to evaluate")
    ap.add_argument("--prm-drop-threshold", type=float,
                    default=PRM_DROP_THRESHOLD)
    ap.add_argument(
        "--prm-absolute-threshold",
        type=float,
        default=PRM_ABSOLUTE_THRESHOLD,
        help=(
            "Absolute PRM score threshold used by "
            "prm_below_threshold_fb_last"
        ),
    )
    ap.add_argument("--prm-model", type=str, default=PRM_MODEL_ID,
                    help="PRM model ID (supports Qwen2.5-Math-PRM-7B "
                         "and Skywork-o1-Open-PRM-Qwen-2.5-1.5B)")
    ap.add_argument("--nll-drop-threshold", type=float,
                    default=NLL_DROP_THRESHOLD)
    ap.add_argument(
        "--logprob-batch-size",
        type=int,
        default=4,
        help=(
            "Per-GPU batch size for vLLM prompt-logprob scoring. "
            "Use a smaller value for long responses or constrained GPUs."
        ),
    )
    ap.add_argument("--gpt-model", type=str, default=GPT_MODEL,
                    help="GPT model for first-error detection")
    ap.add_argument("--gpt-max-workers", type=int,
                    default=GPT_MAX_WORKERS)
    ap.add_argument("--gpt-temperature", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--fig-dir", default="",
                    help="Directory for generated figures "
                         "(default: figures/budget_multisignal[_tag])")
    ap.add_argument("--gpt-cache", default="",
                    help="Path to gpt_first_error_cache.jsonl "
                         "(used as warm-start; new calls appended)")
    ap.add_argument("--nd", type=int, nargs="+", default=None,
                    help="Which nd values to evaluate "
                         "(default: all from ROLLBACK_CONFIGS)")
    ap.add_argument("--skip-phase", type=int, nargs="*", default=[],
                    help="Skip phases (1=drafts, 2=PRM, 25=GPT, "
                         "3=logprob, 4=suffix, 5=SC)")
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help=(
            "Prepare/cache checkpoint records without writing evaluation "
            "summaries or figures; stale evaluation artifacts are removed"
        ),
    )
    ap.add_argument("--draft-checkpoint", default="",
                    help="Path to an existing checkpoint.jsonl to "
                         "import draft records from (reuse drafts "
                         "without regenerating)")
    ap.add_argument(
        "--checkpoint-part",
        action="append",
        default=[],
        help=(
            "Read records from an additional immutable checkpoint JSONL. "
            "May be specified multiple times; newly generated records are "
            "written only to OUT_DIR/checkpoint.jsonl"
        ),
    )
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for vLLM sampling "
                         "(None = non-deterministic)")
    ap.add_argument(
        "--timing-mode",
        choices=("auto", "fresh", "eval-only"),
        default="auto",
        help=(
            "Wall-clock provenance mode. auto permits checkpoint reuse; "
            "fresh requires an empty checkpoint; eval-only labels an "
            "offline evaluation invocation"
        ),
    )
    ap.add_argument(
        "--reuse-timing-manifest",
        action="append",
        default=[],
        help=(
            "Timing manifest associated with a reused checkpoint. May be "
            "specified multiple times"
        ),
    )
    ap.add_argument("--_shard-id", type=int, default=-1)
    ap.add_argument("--_task-file", default="")
    ap.add_argument("--_gpu", default="0")
    ap.add_argument("--_prm", action="store_true")
    ap.add_argument("--_skywork-prm", action="store_true")
    ap.add_argument("--_prm-model-id", default=PRM_MODEL_ID)
    ap.add_argument("--_logprob", action="store_true")
    ap.add_argument("--_seed", type=int, default=-1)
    ap.add_argument("--_timing-file", default="")
    ap.add_argument("--_timing-phase", default="")
    ap.add_argument("--_timing-run-id", default="")
    return ap.parse_args()


def checkpoint_record_identity(record):
    """Return the identity used to merge immutable checkpoint parts."""
    task_type = record["task_type"]
    if task_type in {"draft", "prm", "nll", "logprob"}:
        return (
            task_type,
            record["doc_id"],
            int(record.get("draft_idx", 0)),
        )
    if task_type == "suffix":
        return (
            task_type,
            record["doc_id"],
            int(record["draft_idx"]),
            record.get("strategy", "prm_drop"),
            int(record["rollback_step"]),
            int(record["suffix_idx"]),
        )
    if task_type == "fullsc":
        return task_type, record["doc_id"], int(record["sc_idx"])
    raise ValueError(
        f"Unsupported checkpoint record type: {task_type!r}"
    )


def read_checkpoint_records(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [
            json.loads(line) for line in handle if line.strip()
        ]


def make_worker_timing(args, tasks, default_phase):
    timing_path = getattr(args, "_timing_file", "")
    if not timing_path:
        return None
    return WorkerTiming(
        path=Path(timing_path),
        phase=getattr(args, "_timing_phase", "") or default_phase,
        run_id=getattr(args, "_timing_run_id", "") or "untracked",
        shard_id=args._shard_id,
        gpu_id=args._gpu,
        tasks=tasks,
    )


# SECTION: shard_worker


def run_shard(args):
    """Generate completions on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    sid = args._shard_id
    timing = make_worker_timing(args, tasks, "generation")
    print(f"[Shard {sid}] GPU {args._gpu}: {len(tasks)} tasks")

    stop = get_stop_tokens(MODEL_ID)
    model_load_started_ns = time.perf_counter_ns()
    llm = LLM(
        model=MODEL_ID, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    if timing is not None:
        timing.record_model_load(model_load_started_ns)
    sp_s = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P,
        max_tokens=MAX_TOKENS, stop=stop,
        seed=args._seed if args._seed >= 0 else None,
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
            batch_started_ns = time.perf_counter_ns()
            batch_started_unix_sec = time.time()
            pids = [tokenizer.encode(tasks[i]["prompt"],
                                     add_special_tokens=False)
                    for i in chunk]
            outs = llm.generate(
                [{"prompt_token_ids": p} for p in pids],
                sampling_params=sp_s)
            batch_finished_unix_sec = time.time()
            batch_generated_tokens = 0
            for idx, o in zip(chunk, outs):
                text = o.outputs[0].text
                toks = len(o.outputs[0].token_ids)
                batch_generated_tokens += toks
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
                if timing is not None:
                    timing.record_vllm_request(
                        t,
                        o,
                        fallback_started_unix_sec=(
                            batch_started_unix_sec
                        ),
                        fallback_finished_unix_sec=(
                            batch_finished_unix_sec
                        ),
                    )
            if timing is not None:
                timing.record_batch(
                    [tasks[index] for index in chunk],
                    batch_started_ns,
                    generated_tokens=batch_generated_tokens,
                    input_tokens=sum(len(ids) for ids in pids),
                )
            print(f"[Shard {sid}] {ttype} batch "
                  f"{bi//512+1}/{(len(idxs)+511)//512}")

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if timing is not None:
        timing.finish()
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
    timing = make_worker_timing(args, tasks, "prm")
    print(f"[PRM Shard {sid}] GPU {args._gpu}: {len(tasks)} tasks")

    model_load_started_ns = time.perf_counter_ns()
    prm_tok = ATK.from_pretrained(
        PRM_MODEL_ID, trust_remote_code=True)
    prm_model = AutoModel.from_pretrained(
        PRM_MODEL_ID, device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    if timing is not None:
        timing.record_model_load(model_load_started_ns)
    step_sep_id = prm_tok.encode("<extra_0>")[0]

    results = [None] * len(tasks)
    for idx, t in enumerate(tasks):
        task_started_unix_sec = time.time()
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
        task_finished_unix_sec = time.time()
        rec = {k: t[k] for k in t if k != "prompt"}
        rec["step_scores"] = [round(s, 6) for s in scores]
        rec["prm_tokens"] = int(input_ids.shape[1])
        results[idx] = rec
        if timing is not None:
            timing.record_task(
                t,
                started_unix_sec=task_started_unix_sec,
                finished_unix_sec=task_finished_unix_sec,
                timing_source="single_task_wall",
            )

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if timing is not None:
        timing.finish()
    print(f"[PRM Shard {sid}] Done: "
          f"{sum(1 for r in results if r)}")


# SECTION: skywork_prm_shard


def run_skywork_prm_shard(args):
    """Score steps with Skywork-o1-Open-PRM-Qwen-2.5-1.5B on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    import torch

    # Skywork PRM only ships .bin weights; bypass the torch>=2.6
    # check added for CVE-2025-32434 (safe here: local trusted model).
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    sid = args._shard_id
    timing = make_worker_timing(args, tasks, "prm")
    print(f"[Skywork PRM Shard {sid}] GPU {args._gpu}: "
          f"{len(tasks)} tasks")

    skywork_cache = (PROJECT_ROOT
                     / "third_party/skywork-o1-prm-inference")
    sys.path.insert(0, str(skywork_cache))
    from model_utils.prm_model import PRM_MODEL
    from model_utils.io_utils import (
        prepare_batch_input_for_model,
        derive_step_rewards,
    )
    from transformers import AutoTokenizer as ATK

    prm_model_id = args._prm_model_id
    model_load_started_ns = time.perf_counter_ns()
    tokenizer = ATK.from_pretrained(
        prm_model_id, trust_remote_code=True,
        use_safetensors=True)
    model = PRM_MODEL.from_pretrained(
        prm_model_id, device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    ).eval()
    if timing is not None:
        timing.record_model_load(model_load_started_ns)

    BATCH = 8
    results = [None] * len(tasks)
    for bi in range(0, len(tasks), BATCH):
        batch = tasks[bi:bi + BATCH]
        batch_started_unix_sec = time.time()
        batch_input_ids = []
        batch_flags = []
        for t in batch:
            question = t["question"]
            steps = t["steps"]
            prompt_ids = tokenizer.encode(
                tokenizer.bos_token + question + "\n")
            reward_flags = [0] * len(prompt_ids)
            response_ids = []
            for i, step in enumerate(steps):
                sep = "\n\n" if i < len(steps) - 1 else "\n"
                chunk = step + sep
                chunk_ids = tokenizer.encode(
                    chunk, add_special_tokens=False)
                flags = [0] * len(chunk_ids)
                flags[-1] = 1
                response_ids.extend(chunk_ids)
                reward_flags.extend(flags)
            input_ids = prompt_ids + response_ids
            batch_input_ids.append(input_ids)
            batch_flags.append(reward_flags)

        padded_ids, attn_mask, padded_flags = (
            prepare_batch_input_for_model(
                batch_input_ids, batch_flags,
                tokenizer.pad_token_id))
        device = next(model.parameters()).device
        padded_ids = padded_ids.to(device)
        attn_mask = attn_mask.to(device)
        padded_flags = padded_flags.to(device)

        with torch.no_grad():
            _, _, rewards = model(
                input_ids=padded_ids,
                attention_mask=attn_mask,
                return_probs=True,
            )
        step_rewards = derive_step_rewards(rewards, padded_flags)
        torch.cuda.synchronize()
        batch_finished_unix_sec = time.time()

        for j, t in enumerate(batch):
            idx = bi + j
            rec = {k: t[k] for k in t
                   if k not in ("question", "steps", "prompt")}
            scores = step_rewards[j]
            rec["step_scores"] = [round(s, 6) for s in scores]
            rec["prm_tokens"] = len(batch_input_ids[j])
            results[idx] = rec
            if timing is not None:
                timing.record_task(
                    t,
                    started_unix_sec=batch_started_unix_sec,
                    finished_unix_sec=batch_finished_unix_sec,
                    timing_source="shared_batch_wall",
                )

        print(f"[Skywork PRM Shard {sid}] batch "
              f"{bi // BATCH + 1}/"
              f"{(len(tasks) + BATCH - 1) // BATCH}")

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
    if timing is not None:
        timing.finish()
    print(f"[Skywork PRM Shard {sid}] Done: "
          f"{sum(1 for r in results if r)}")


# SECTION: logprob_shard


def run_logprob_shard(args):
    """Collect per-token logprobs for NLL signal with vLLM."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    sid = args._shard_id
    timing = make_worker_timing(args, tasks, "nll")
    print(f"[LP Shard {sid}] GPU {args._gpu}: {len(tasks)} tasks")
    gpu_memory_utilization = float(
        os.environ.get("LOGPROB_GPU_MEMORY_UTILIZATION", "0.60")
    )
    if not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError(
            "LOGPROB_GPU_MEMORY_UTILIZATION must be between 0 and 1"
        )
    max_input_tokens = int(
        os.environ.get("LOGPROB_MAX_INPUT_TOKENS", str(MAX_MODEL_LEN))
    )
    if max_input_tokens <= 0:
        raise ValueError(
            "LOGPROB_MAX_INPUT_TOKENS must be positive"
        )
    print(
        f"[LP Shard {sid}] gpu_memory_utilization="
        f"{gpu_memory_utilization}, "
        f"max_input_tokens={max_input_tokens}"
    )

    model_load_started_ns = time.perf_counter_ns()
    llm = LLM(
        model=MODEL_ID, tensor_parallel_size=1,
        trust_remote_code=True, dtype="half",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_input_tokens + 1,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    if timing is not None:
        timing.record_model_load(model_load_started_ns)
    sp = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=1)
    batch_size = args.logprob_batch_size
    if batch_size <= 0:
        raise ValueError("--logprob-batch-size must be positive")
    results = [None] * len(tasks)
    for bi in range(0, len(tasks), batch_size):
        batch = tasks[bi:bi + batch_size]
        batch_started_ns = time.perf_counter_ns()
        batch_started_unix_sec = time.time()
        encodings = [
            tokenizer(
                task["full_prompt"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            for task in batch
        ]
        prompt_token_ids = [
            encoding["input_ids"] for encoding in encodings
        ]
        too_long = [
            len(ids) for ids in prompt_token_ids
            if len(ids) > max_input_tokens
        ]
        if too_long:
            raise ValueError(
                "NLL inputs exceed "
                f"max_input_tokens={max_input_tokens}: "
                f"{too_long[:3]}"
            )
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            sampling_params=sp,
        )
        batch_finished_unix_sec = time.time()
        for ti, (task, encoding, output) in enumerate(
                zip(batch, encodings, outputs)):
            rec = {k: v for k, v in task.items()
                   if k not in ("full_prompt",)}
            resp_off = task["resp_char_offset"]
            if output.prompt_logprobs is None:
                raise ValueError(
                    f"{task['doc_id']}|{task['draft_idx']}: "
                    "prompt_logprobs is missing"
                )
            lps, offsets = [], []
            for token_id, token_span, lp_dict in zip(
                    output.prompt_token_ids,
                    encoding["offset_mapping"],
                    output.prompt_logprobs):
                start_char, end_char = token_span
                if end_char <= resp_off:
                    continue
                if lp_dict is None or token_id not in lp_dict:
                    raise ValueError(
                        f"{task['doc_id']}|{task['draft_idx']}: "
                        "actual response token logprob is missing"
                    )
                lps.append(float(lp_dict[token_id].logprob))
                offsets.append(max(0, start_char - resp_off))
            rec["token_logprobs"] = lps
            rec["token_offsets"] = offsets
            rec["task_type"] = "logprob"
            rec["scoring_backend"] = "vllm"
            rec["input_tokens"] = len(encoding["input_ids"])
            rec["response_scored_tokens"] = len(lps)
            results[bi + ti] = rec
            if timing is not None:
                timing.record_vllm_request(
                    task,
                    output,
                    fallback_started_unix_sec=batch_started_unix_sec,
                    fallback_finished_unix_sec=batch_finished_unix_sec,
                )
        if timing is not None:
            timing.record_batch(
                batch,
                batch_started_ns,
                input_tokens=sum(
                    len(ids) for ids in prompt_token_ids
                ),
            )
        print(f"[LP Shard {sid}] batch "
              f"{bi // batch_size + 1}/"
              f"{(len(tasks) + batch_size - 1) // batch_size}")

    out_path = Path(tasks[0]["_out"]) if tasks else None
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
    if timing is not None:
        timing.finish()
    print(f"[LP Shard {sid}] Done: "
          f"{sum(1 for r in results if r)}")


# SECTION: gpu_pool

MIN_FREE_MEM_MIB_GEN = 6000
MIN_FREE_MEM_MIB_PRM = 10000
MIN_FREE_MEM_MIB_PRM_SMALL = 4000
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


def _vllm_port_for_worker(shard_id: int) -> int:
    """Return a process-specific base port for concurrent vLLM workers."""
    return 20000 + (
        (os.getpid() % 1000) * 32 + shard_id * 8
    ) % 40000


# SECTION: launch_shards


def launch_shards(tasks, gpu_ids, shard_dir, prm=False,
                  logprob=False, skywork_prm=False,
                  prm_model_id=None, seed=None, *,
                  logprob_batch_size=4,
                  phase_name="", timing_recorder=None,
                  return_stats=False):
    if not tasks:
        empty_stats = {
            "phase": phase_name,
            "tasks_executed": 0,
            "gpu_wait_sec": 0.0,
            "launcher_wall_sec": 0.0,
            "worker_wall_sec_max": 0.0,
            "gpu_seconds_sum": 0.0,
            "worker_vllm_ports": {},
            "worker_timing_files": [],
        }
        return ([], empty_stats) if return_stats else []
    launch_started_ns = time.perf_counter_ns()
    shard_dir.mkdir(parents=True, exist_ok=True)
    ns = len(gpu_ids)
    shards = [[] for _ in range(ns)]
    for i, t in enumerate(tasks):
        shards[i % ns].append(t)

    script = str(Path(__file__).resolve())
    if prm:
        min_mem = MIN_FREE_MEM_MIB_PRM
    elif skywork_prm:
        min_mem = MIN_FREE_MEM_MIB_PRM_SMALL
    else:
        min_mem = MIN_FREE_MEM_MIB_GEN

    busy: Dict[str, subprocess.Popen] = {}
    all_procs: List[Tuple[int, str, subprocess.Popen, Any]] = []
    out_files = []
    worker_timing_files = []
    worker_vllm_ports = {}
    gpu_wait_sec = 0.0

    for si in range(ns):
        if not shards[si]:
            continue
        for t in shards[si]:
            t["_out"] = str(shard_dir / f"shard_{si}.jsonl")
        tf = shard_dir / f"tasks_{si}.json"
        tf.write_text(json.dumps(shards[si]), encoding="utf-8")
        out_files.append(shard_dir / f"shard_{si}.jsonl")

        wait_started_ns = time.perf_counter_ns()
        gid = _wait_for_free_gpu(gpu_ids, busy, min_mem)
        gpu_wait_sec += (
            time.perf_counter_ns() - wait_started_ns
        ) / 1_000_000_000
        cmd = [PYTHON, script,
               "--_shard-id", str(si), "--_task-file", str(tf),
               "--_gpu", gid, "--dataset", DATASET,
               "--model", MODEL_ID]
        if prm:
            cmd.append("--_prm")
        if skywork_prm:
            cmd.append("--_skywork-prm")
            if prm_model_id:
                cmd += ["--_prm-model-id", prm_model_id]
        if logprob:
            cmd += [
                "--_logprob",
                "--logprob-batch-size",
                str(logprob_batch_size),
            ]
        if seed is not None:
            cmd += ["--_seed", str(seed)]
        if timing_recorder is not None:
            timing_file = timing_recorder.worker_timing_path(
                phase_name, si
            )
            worker_timing_files.append(timing_file)
            cmd += [
                "--_timing-file", str(timing_file),
                "--_timing-phase", phase_name,
                "--_timing-run-id", timing_recorder.run_id,
            ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        env["TOKENIZERS_PARALLELISM"] = "false"
        worker_vllm_port = _vllm_port_for_worker(si)
        env["VLLM_PORT"] = str(worker_vllm_port)
        worker_vllm_ports[str(si)] = worker_vllm_port
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
    worker_timings = []
    for timing_file in worker_timing_files:
        if not timing_file.is_file():
            raise RuntimeError(
                f"Worker did not write timing sidecar: {timing_file}"
            )
        worker_timings.append(load_manifest(timing_file))
    worker_walls = [
        float(value["worker_wall_sec"]) for value in worker_timings
    ]
    stats = {
        "phase": phase_name,
        "tasks_executed": len(tasks),
        "gpu_wait_sec": gpu_wait_sec,
        "launcher_wall_sec": (
            time.perf_counter_ns() - launch_started_ns
        ) / 1_000_000_000,
        "worker_wall_sec_max": max(worker_walls, default=0.0),
        "gpu_seconds_sum": sum(worker_walls),
        "worker_vllm_ports": worker_vllm_ports,
        "worker_timing_files": [
            str(path) for path in worker_timing_files
        ],
    }
    return (recs, stats) if return_stats else recs


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
    """PRM-drop: first big drop. Returns dict with three variants."""
    result = {}
    triggered_step = None
    if n >= 2 and len(scores) >= 2:
        limit = min(len(scores), n)
        drops = [scores[i] - scores[i + 1]
                 for i in range(limit - 1)]
        for i, d in enumerate(drops):
            if d > threshold:
                triggered_step = i + 1
                break
    if triggered_step is not None:
        result["prm_drop"] = triggered_step
        result["prm_drop_fb_last"] = triggered_step
        result["prm_drop_fb_start"] = triggered_step
    else:
        result["prm_drop"] = max(n - 1, 0)
        result["prm_drop_fb_last"] = max(n - 1, 0)
        result["prm_drop_fb_start"] = 0
    return result


def compute_rollback_step_nll_drop(nlls, n, threshold):
    """NLL-drop: first big NLL increase. Returns dict with three variants."""
    result = {}
    triggered_step = None
    if n >= 2 and len(nlls) >= 2:
        deltas = [nlls[i] - nlls[i - 1]
                  for i in range(1, len(nlls))]
        for i, d in enumerate(deltas):
            if d > threshold:
                triggered_step = i + 1
                break
    if triggered_step is not None:
        result["nll_drop"] = triggered_step
        result["nll_drop_fb_last"] = triggered_step
        result["nll_drop_fb_start"] = triggered_step
    else:
        result["nll_drop"] = max(n - 1, 0)
        result["nll_drop_fb_last"] = max(n - 1, 0)
        result["nll_drop_fb_start"] = 0
    return result


def compute_rollback_step_gpt(tau, n):
    """GPT signal: tau is 1-indexed first-error step, -1 means no clear error.
    Returns dict with two fallback variants."""
    result = {}
    if tau is not None and tau >= 1:
        rb = max(0, min(int(tau) - 1, n - 1))
        result["gpt_fb_last"] = rb
        result["gpt_fb_start"] = rb
    else:
        # tau is None (API failure) or -1 (no clear error) -> fallback
        result["gpt_fb_last"] = max(n - 1, 0)
        result["gpt_fb_start"] = 0
    return result


# SECTION: evaluation


def _vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def rollback_answer_allocation(variant, ns):
    """Return draft and suffix votes per draft for one result variant."""
    if variant == "full":
        return 1, ns
    if variant == "fair":
        return 1, max(ns - 1, 0)
    if variant == "fair_new":
        return 0, ns
    raise ValueError(f"Unknown rollback result variant: {variant}")


def evaluate_configs(
    questions, draft_map, prm_map, sfx_map,
    sc_recs, budget, active_signals,
    step_tokens_map=None,
):
    nq = len(questions)
    results = []
    report_f1 = DATASET in ("hotpotqa", "hotpotqa_open")

    # Greedy baseline: single draft (draft_idx=0), no voting
    greedy_correct = 0
    greedy_f1 = 0.0
    greedy_toks = 0
    for q in questions:
        d = draft_map.get((q["doc_id"], 0))
        answer = d.get("draft_answer", "") if d else ""
        if d and check_answer(DATASET, answer, q["gold_answer"]):
            greedy_correct += 1
        if report_f1:
            greedy_f1 += hotpotqa_f1(answer, q["gold_answer"])
        greedy_toks += d.get("draft_tokens", 0) if d else 0
    greedy_result = dict(
        method="Greedy@1", nd=1, ns=0,
        acc=greedy_correct / nq,
        tokens_per_q=greedy_toks / nq,
        total_tokens=greedy_toks,
        n_answers=1,
    )
    if report_f1:
        greedy_result["f1"] = greedy_f1 / nq
    results.append(greedy_result)

    sc_by_doc = defaultdict(list)
    for s in sc_recs:
        sc_by_doc[s["doc_id"]].append(s)
    for N in sorted({8, 16, budget}):
        correct = 0
        f1_total = 0.0
        total_toks = 0
        total_answers = 0
        for q in questions:
            recs = sc_by_doc.get(q["doc_id"], [])[:N]
            answers = [r["sc_answer"] for r in recs]
            toks = sum(r["sc_tokens"] for r in recs)
            voted = _vote(answers)
            if check_answer(DATASET, voted, q["gold_answer"]):
                correct += 1
            if report_f1:
                f1_total += hotpotqa_f1(voted, q["gold_answer"])
            total_toks += toks
            total_answers += len(answers)
        sc_result = dict(
            method=f"SC@{N}", nd=0, ns=N,
            acc=correct / nq,
            tokens_per_q=total_toks / nq,
            total_tokens=total_toks,
            n_answers=total_answers / nq,
        )
        if report_f1:
            sc_result["f1"] = f1_total / nq
        results.append(sc_result)

    all_strategies = set()
    for prm_rec in prm_map.values():
        for strat in prm_rec.get("_rb_steps", {}):
            all_strategies.add(strat)
    if not all_strategies:
        all_strategies = set(active_signals)

    for strat in sorted(all_strategies & set(active_signals)):
        for nd, ns in ROLLBACK_CONFIGS:
            if nd * ns != budget:
                continue
            for variant in ["full", "fair", "fair_new"]:
                correct = 0
                f1_total = 0.0
                total_toks = 0
                total_prm_toks = 0
                total_n_answers = 0
                total_draft_answers = 0
                total_suffix_answers = 0
                total_generated_drafts = 0
                for q in questions:
                    did = q["doc_id"]
                    answers = []
                    q_toks = 0
                    q_prm_toks = 0
                    for di in range(nd):
                        d = draft_map.get((did, di))
                        if not d:
                            continue
                        # Every rollback candidate requires generating its
                        # source draft, even when fair_new excludes that
                        # draft's answer from the vote.
                        q_toks += d["draft_tokens"]
                        total_generated_drafts += 1
                        prm_rec = prm_map.get((did, di))
                        if prm_rec:
                            q_prm_toks += prm_rec.get(
                                "prm_tokens", 0)
                        rb_steps = (prm_rec.get("_rb_steps", {})
                                    if prm_rec else {})
                        rb_step = rb_steps.get(strat)

                        if rb_step is not None:
                            draft_votes, suffix_votes = (
                                rollback_answer_allocation(variant, ns)
                            )
                            if draft_votes:
                                answers.append(d["draft_answer"])
                                total_draft_answers += 1
                            for si in range(suffix_votes):
                                s = sfx_map.get(
                                    (did, di, strat,
                                     rb_step, si))
                                if s:
                                    answers.append(
                                        s["suffix_answer"])
                                    q_toks += s[
                                        "suffix_tokens"]
                                    total_suffix_answers += 1
                        else:
                            answers.append(d["draft_answer"])
                            total_draft_answers += 1

                    total_n_answers += len(answers)
                    voted = _vote(answers)
                    if check_answer(DATASET, voted, q["gold_answer"]):
                        correct += 1
                    if report_f1:
                        f1_total += hotpotqa_f1(
                            voted, q["gold_answer"]
                        )
                    total_toks += q_toks
                    total_prm_toks += q_prm_toks

                avg_n_answers = total_n_answers / nq
                method_base = f"rollback_{strat}_nd{nd}_ns{ns}"
                tag = (
                    method_base
                    if variant == "full"
                    else f"{method_base}_{variant}"
                )
                result = dict(
                    method=tag,
                    strategy=strat,
                    variant=variant,
                    nd=nd, ns=ns,
                    n_answers=round(avg_n_answers, 1),
                    n_draft_answers=(
                        total_draft_answers / nq
                    ),
                    n_suffix_answers=(
                        total_suffix_answers / nq
                    ),
                    n_generated_drafts=(
                        total_generated_drafts / nq
                    ),
                    acc=correct / nq,
                    tokens_per_q=total_toks / nq,
                    total_tokens=total_toks,
                    prm_tokens=total_prm_toks,
                )
                if report_f1:
                    result["f1"] = f1_total / nq
                results.append(result)

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
        "gpt_fb_last": "#27ae60",
        "gpt_fb_start": "#1abc9c",
        "prm_drop": "#e74c3c",
        "prm_drop_fb_last": "#c0392b",
        "prm_drop_fb_start": "#e67e22",
        "prm_max_drop_fb_last": "#7c3aed",
        "prm_below_threshold_fb_last": "#db2777",
        "nll_drop": "#3498db",
        "nll_drop_fb_last": "#2980b9",
        "nll_drop_fb_start": "#8e44ad",
    }
    strat_markers = {
        "gpt_fb_last": "v",
        "gpt_fb_start": "<",
        "prm_drop": "o",
        "prm_drop_fb_last": "s",
        "prm_drop_fb_start": "p",
        "prm_max_drop_fb_last": "^",
        "prm_below_threshold_fb_last": "X",
        "nll_drop": "D",
        "nll_drop_fb_last": "d",
        "nll_drop_fb_start": "h",
    }

    strategies = sorted(set(
        r.get("strategy", "prm_drop") for r in rb_rows))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
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


def save_summary_table(
    eval_results,
    out_dir,
    phase_times=None,
    n_drafts=0,
    nq=0,
    timing_phases=None,
    per_question_timing=None,
):
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

    report_f1 = any("f1" in row for row in eval_results)
    md = [
        "| method | strategy | nd | ns | acc "
        + ("| F1 " if report_f1 else "")
        + "| tokens_per_q | savings_vs_sc32 |",
        "|---|---|---:|---:|---:|"
        + ("---:|" if report_f1 else "")
        + "---:|---:|",
    ]
    for r in rows:
        source = next(
            item for item in eval_results
            if item["method"] == r["method"]
        )
        md.append(
            f"| {r['method']} | {r['strategy']} "
            f"| {r['nd']} | {r['ns']} "
            f"| {r['acc']:.4f} "
            + (f"| {source['f1']:.4f} " if report_f1 else "")
            + f"| {r['tokens_per_q']:.1f} "
            f"| {r['savings_vs_sc32']:.1f}% |"
        )

    if per_question_timing is not None:
        md.append("")
        md.append(
            "## Per-question active wall-clock "
            "(model loading excluded)"
        )
        md.append("")
        md.append(
            "| signal | complete/questions | draft_gen_mean_sec "
            "| suffix_gen_mean_sec | total_gen_mean_sec "
            "| signal_mean_sec | generation/signal |"
        )
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for signal_summary in per_question_timing["signals"]:
            def _format(value):
                return (
                    f"{float(value):.4f}"
                    if value is not None
                    else "N/A"
                )

            md.append(
                f"| {signal_summary['signal']} "
                f"| {signal_summary['complete_questions']}/"
                f"{signal_summary['questions']} "
                f"| {_format(signal_summary['draft_generation_mean_sec'])} "
                f"| {_format(signal_summary['suffix_generation_mean_sec'])} "
                f"| {_format(signal_summary['total_generation_mean_sec'])} "
                f"| {_format(signal_summary['signal_computation_mean_sec'])} "
                f"| {_format(signal_summary['generation_to_signal_ratio_of_sums'])} "
                "|"
            )

    if timing_phases is not None:
        md.append("")
        md.append("## Wall-clock provenance (current invocation)")
        md.append("")
        md.append(
            "| phase | status | required | reused | executed "
            "| current_sec | from_scratch_sec | reconstruction |"
        )
        md.append("|---|---|---:|---:|---:|---:|---:|---|")
        for phase in timing_phases:
            reconstructed = phase[
                "reconstructed_from_scratch_wall_sec"
            ]
            reconstructed_text = (
                f"{reconstructed:.3f}"
                if reconstructed is not None
                else "N/A"
            )
            md.append(
                f"| {phase['phase']} | {phase['status']} "
                f"| {phase['tasks_required']} "
                f"| {phase['tasks_reused']} "
                f"| {phase['tasks_executed']} "
                f"| {phase['observed_current_wall_sec']:.3f} "
                f"| {reconstructed_text} "
                f"| {phase['reconstruction_kind'] or 'unavailable'} |"
            )
    elif phase_times:
        md.append("")
        md.append("## Signal overhead (wall-clock)")
        md.append("")
        md.append("| signal | total_sec | ms_per_sample "
                   "| ms_per_question |")
        md.append("|---|---:|---:|---:|")
        for key, label in [("prm", "PRM"), ("nll", "NLL"),
                           ("gpt", "GPT")]:
            t = phase_times.get(key, 0.0)
            ps = t / n_drafts * 1000 if n_drafts else 0
            pq = t / nq * 1000 if nq else 0
            md.append(f"| {label} | {t:.1f} | {ps:.1f} "
                      f"| {pq:.1f} |")
        for key, label in [("draft", "Draft gen"),
                           ("suffix", "Suffix gen")]:
            t = phase_times.get(key, 0.0)
            md.append(f"| {label} | {t:.1f} | - | - |")

    (out_dir / "eval_summary_table.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"Saved table: {out_dir / 'eval_summary_table.md'}")


# SECTION: main


def main():
    global DATASET, MODEL_ID
    main_started_ns = time.perf_counter_ns()
    args = parse_args()
    DATASET = args.dataset
    MODEL_ID = args.model

    if args._shard_id >= 0:
        worker_started_ns = time.perf_counter_ns()
        worker_tasks = json.loads(
            Path(args._task_file).read_text(encoding="utf-8")
        )
        if args._prm:
            run_prm_shard(args)
        elif args._skywork_prm:
            run_skywork_prm_shard(args)
        elif args._logprob:
            run_logprob_shard(args)
        else:
            run_shard(args)
        write_generic_worker_timing(
            Path(args._timing_file) if args._timing_file else None,
            phase=args._timing_phase or worker_tasks[0]["task_type"],
            run_id=args._timing_run_id or "untracked",
            shard_id=args._shard_id,
            gpu_id=args._gpu,
            tasks=worker_tasks,
            started_ns=worker_started_ns,
        )
        return

    signals = args.signals
    prm_thr = args.prm_drop_threshold
    prm_absolute_thr = args.prm_absolute_threshold
    nll_thr = args.nll_drop_threshold
    use_skywork = "skywork" in args.prm_model.lower()

    # Filter ROLLBACK_CONFIGS by --nd if specified
    global ROLLBACK_CONFIGS
    if args.nd:
        allowed = set(args.nd)
        ROLLBACK_CONFIGS = [(nd, ns) for nd, ns in ROLLBACK_CONFIGS
                            if nd in allowed]
        print(f"Filtered ROLLBACK_CONFIGS by --nd {args.nd}: "
              f"{ROLLBACK_CONFIGS}")

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
    prm_tag = ""
    if use_skywork:
        prm_short = Path(args.prm_model).name.lower().replace("-", "_")
        prm_tag = f"_prm_{prm_short}"
    model_short = Path(MODEL_ID).name.lower().replace("-", "_")
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "results"
        / f"{model_short}_budget_multisignal{prm_tag}{suffix}"
        / DATASET)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (Path(args.fig_dir) if args.fig_dir else
               PROJECT_ROOT / "figures"
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
    print(f"Sampling seed: {args.seed}")
    print(f"NLL logprob batch size: {args.logprob_batch_size}")
    print(f"PRM model: {args.prm_model}"
          f"{' (Skywork 1.5B)' if use_skywork else ''}")
    print(f"Absolute PRM threshold: {prm_absolute_thr}")

    checkpoint_parts = [Path(value) for value in args.checkpoint_part]
    local_existing = []
    if ckpt.exists():
        local_existing = read_checkpoint_records(ckpt)
    existing = []
    seen_checkpoint_records = {}
    for path in [*checkpoint_parts, ckpt]:
        records = (
            local_existing
            if path == ckpt
            else read_checkpoint_records(path)
        )
        for record in records:
            identity = checkpoint_record_identity(record)
            previous = seen_checkpoint_records.get(identity)
            if previous is not None:
                raise ValueError(
                    f"Duplicate checkpoint identity {identity}: "
                    f"{previous} and {path}"
                )
            seen_checkpoint_records[identity] = path
            existing.append(record)
        if path != ckpt:
            print(f"Checkpoint part {path}: {len(records)} records")
    print(
        f"Existing checkpoint union: {len(existing)} records "
        f"({len(local_existing)} local)"
    )
    if args.timing_mode == "fresh" and (
        local_existing or checkpoint_parts
    ):
        raise ValueError(
            "--timing-mode fresh does not permit existing local records "
            "or --checkpoint-part"
        )
    if args.timing_mode == "fresh" and args.draft_checkpoint:
        raise ValueError(
            "--timing-mode fresh cannot import --draft-checkpoint"
        )

    timing_source_candidates = [
        Path(value) for value in args.reuse_timing_manifest
    ]
    for path in checkpoint_parts:
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint part does not exist: {path}"
            )
        discovered = discover_timing_manifest(path)
        if discovered is not None:
            timing_source_candidates.append(discovered)
    current_timing = out_dir / "timing" / "timing_manifest.json"
    if current_timing.is_file():
        timing_source_candidates.append(current_timing)
    if args.draft_checkpoint:
        discovered = discover_timing_manifest(
            Path(args.draft_checkpoint)
        )
        if discovered is not None:
            timing_source_candidates.append(discovered)
    timing_sources = []
    seen_timing_sources = set()
    for path in timing_source_candidates:
        resolved = path.resolve()
        if resolved in seen_timing_sources:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Reused timing manifest does not exist: {resolved}"
            )
        seen_timing_sources.add(resolved)
        timing_sources.append(resolved)
    task_event_source_candidates = []
    for checkpoint_path in checkpoint_parts:
        discovered = discover_task_event_store(checkpoint_path)
        if discovered is not None:
            task_event_source_candidates.append(discovered.resolve())
    if args.draft_checkpoint:
        discovered = discover_task_event_store(
            Path(args.draft_checkpoint)
        )
        if discovered is not None:
            task_event_source_candidates.append(discovered.resolve())
    for manifest_path in timing_sources:
        discovered = task_event_store_for_manifest(manifest_path)
        if discovered is not None:
            task_event_source_candidates.append(discovered.resolve())
    task_event_sources = []
    seen_event_sources = set()
    for path in task_event_source_candidates:
        if path in seen_event_sources:
            continue
        seen_event_sources.add(path)
        task_event_sources.append(path)
    timing = RunTimingRecorder(
        out_dir,
        mode=args.timing_mode,
        context={
            "dataset": DATASET,
            "model": MODEL_ID,
            "prm_model": args.prm_model,
            "seed": args.seed,
            "budget": B,
            "signals": signals,
            "nd": [nd for nd, _ in ROLLBACK_CONFIGS],
            "logprob_batch_size": args.logprob_batch_size,
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_parts": [
                str(path.resolve()) for path in checkpoint_parts
            ],
            "task_event_sources": [
                str(path) for path in task_event_sources
            ],
        },
        source_manifests=timing_sources,
        started_ns=main_started_ns,
    )

    # Import drafts from external checkpoint if specified
    if args.draft_checkpoint:
        ext_path = Path(args.draft_checkpoint)
        if ext_path.exists():
            ext_recs = [json.loads(l)
                        for l in ext_path.read_text().splitlines()
                        if l.strip()]
            ext_drafts = [r for r in ext_recs
                          if r.get("task_type") == "draft"]
            done_ids = {(r["doc_id"], r["draft_idx"])
                        for r in existing
                        if r.get("task_type") == "draft"}
            imported = 0
            with ckpt.open("a") as f:
                for r in ext_drafts:
                    key = (r["doc_id"], r["draft_idx"])
                    if key not in done_ids:
                        f.write(json.dumps(r, ensure_ascii=False)
                                + "\n")
                        existing.append(r)
                        done_ids.add(key)
                        imported += 1
            print(f"Imported {imported} drafts from "
                  f"{ext_path}")
        else:
            print(f"WARNING: --draft-checkpoint not found: "
                  f"{ext_path}")

    # Legacy cumulative timings are retained for old consumers.  New code
    # should use timing/timing_manifest.json, which is invocation-scoped and
    # checkpoint-aware.
    timing_path = out_dir / "phase_times.json"
    if timing_path.exists():
        phase_times = json.loads(timing_path.read_text())
    else:
        phase_times = {}

    def _record_time(phase_name, elapsed, executed_count):
        if executed_count <= 0:
            return
        phase_times[phase_name] = phase_times.get(phase_name, 0.0) + elapsed
        timing_path.write_text(json.dumps(phase_times, indent=2))

    # ---- Phase 1: Sample drafts ----
    _t0 = time.perf_counter_ns()
    draft_new = []
    draft_launch = {}
    if 1 not in args.skip_phase:
        done = {(r["doc_id"], r["draft_idx"])
                for r in existing
                if r.get("task_type") == "draft"}
        tasks = []
        for q in questions:
            p = build_prompt(MODEL_ID, DATASET,
                             q["question"],
                             context=q.get("context", ""))
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
            draft_new, draft_launch = launch_shards(
                tasks,
                gpu_ids,
                sd / "p1",
                seed=args.seed,
                phase_name="draft",
                timing_recorder=timing,
                return_stats=True,
            )
            with ckpt.open("a") as f:
                for r in draft_new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(draft_new)

    drafts = [r for r in existing
              if r.get("task_type") == "draft"]
    draft_elapsed = (
        time.perf_counter_ns() - _t0
    ) / 1_000_000_000
    _record_time("draft", draft_elapsed, len(draft_new))
    timing.record_phase(
        "draft",
        required=drafts,
        executed=draft_new,
        current_wall_sec=draft_elapsed,
        launch_stats=draft_launch,
    )
    draft_map = {(r["doc_id"], r["draft_idx"]): r
                 for r in drafts}
    print(f"Total drafts: {len(drafts)}")

    # ---- Phase 2: PRM scoring (if prm_drop in signals) ----
    _t0 = time.perf_counter_ns()
    prm_new = []
    prm_launch = {}
    need_prm = any(s.startswith("prm_") for s in signals)
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
                if use_skywork:
                    prm_tasks.append(dict(
                        task_type="prm",
                        doc_id=q["doc_id"],
                        draft_idx=di,
                        n_steps=len(steps),
                        question=q["question"],
                        steps=steps,
                    ))
                else:
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
            print(f"\n--- Phase 2: {len(prm_tasks)} PRM "
                  f"({args.prm_model}) ---")
            if use_skywork:
                prm_new, prm_launch = launch_shards(
                    prm_tasks, gpu_ids, sd / "p2",
                    skywork_prm=True,
                    prm_model_id=args.prm_model,
                    phase_name="prm",
                    timing_recorder=timing,
                    return_stats=True,
                )
            else:
                prm_new, prm_launch = launch_shards(
                    prm_tasks,
                    gpu_ids,
                    sd / "p2",
                    prm=True,
                    phase_name="prm",
                    timing_recorder=timing,
                    return_stats=True,
                )
            with ckpt.open("a") as f:
                for r in prm_new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(prm_new)

    prm_recs = [r for r in existing
                if r.get("task_type") == "prm"]
    prm_elapsed = (
        time.perf_counter_ns() - _t0
    ) / 1_000_000_000
    _record_time("prm", prm_elapsed, len(prm_new))
    timing.record_phase(
        "prm",
        required=prm_recs if need_prm else [],
        executed=prm_new,
        current_wall_sec=prm_elapsed,
        launch_stats=prm_launch,
    )
    prm_map = {}
    for r in prm_recs:
        prm_map[(r["doc_id"], r.get("draft_idx", 0))] = r
    print(f"Total PRM records: {len(prm_recs)}")

    # ---- Phase 3: Logprob collection (if nll_drop) ----
    _t0 = time.perf_counter_ns()
    lp_new = []
    lp_launch = {}
    need_lp = any(s.startswith("nll_drop") for s in signals)
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
                                 q["question"],
                                 context=q.get("context", ""))
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
            lp_new, lp_launch = launch_shards(
                lp_tasks, gpu_ids, sd / "p3",
                logprob=True,
                logprob_batch_size=args.logprob_batch_size,
                phase_name="nll",
                timing_recorder=timing,
                return_stats=True,
            )
            with ckpt.open("a") as f:
                for r in lp_new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(lp_new)

    lp_recs = [r for r in existing
               if r.get("task_type") == "logprob"]
    lp_elapsed = (
        time.perf_counter_ns() - _t0
    ) / 1_000_000_000
    _record_time("nll", lp_elapsed, len(lp_new))
    timing.record_phase(
        "nll",
        required=lp_recs if need_lp else [],
        executed=lp_new,
        current_wall_sec=lp_elapsed,
        launch_stats=lp_launch,
    )
    lp_map = {(r["doc_id"], r.get("draft_idx", 0)): r
              for r in lp_recs}
    print(f"Total logprob records: {len(lp_recs)}")

    # ---- Phase 2.5: GPT first-error detection (live API) ----
    _t0 = time.perf_counter_ns()
    gpt_new = []
    need_gpt = any(s.startswith("gpt") for s in signals)
    gpt_cache_path = out_dir / "gpt_first_error_cache.jsonl"
    gpt_tau_map: Dict[Tuple, int] = {}

    if need_gpt:
        load_env_file(PROJECT_ROOT / ".env")

        # Warm-start from existing cache (local or user-specified)
        warm_cache: Dict[str, dict] = {}
        for cp in [gpt_cache_path, Path(args.gpt_cache) if args.gpt_cache else None]:
            if cp and cp.exists():
                for line in cp.read_text("utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = f"{rec['doc_id']}|{rec.get('draft_idx', rec.get('sample_idx', 0))}"
                    warm_cache[key] = rec
        print(f"GPT warm cache: {len(warm_cache)} entries")

        # Build tasks for drafts not yet in cache
        gpt_tasks = []
        if 25 not in args.skip_phase:
            for q in questions:
                for di in range(nd_max):
                    d = draft_map.get((q["doc_id"], di))
                    if not d or not d.get("draft_steps"):
                        continue
                    key = f"{q['doc_id']}|{di}"
                    if key in warm_cache:
                        continue
                    gpt_tasks.append(dict(
                        doc_id=q["doc_id"],
                        draft_idx=di,
                        question=q["question"],
                        gold_answer=q["gold_answer"],
                        steps=d["draft_steps"],
                        n_steps=d.get("n_steps", len(d["draft_steps"])),
                    ))

        if gpt_tasks:
            print(f"\n--- Phase 2.5: {len(gpt_tasks)} GPT "
                  f"first-error calls ({args.gpt_model}) ---")
            gpt_client = make_client()

            def _gpt_process(task):
                prompt = build_first_error_prompt(
                    task["question"], task["gold_answer"],
                    task["steps"])
                parsed, raw = call_gpt_first_error(
                    gpt_client, args.gpt_model, prompt,
                    temperature=args.gpt_temperature)
                tau = None
                if (parsed and
                        isinstance(parsed.get("first_error_step"), int)):
                    tau = parsed["first_error_step"]
                return dict(
                    doc_id=task["doc_id"],
                    draft_idx=task["draft_idx"],
                    tau=tau,
                    gpt_parsed=parsed,
                    gpt_raw=raw,
                )

            from tqdm.auto import tqdm as _tqdm
            if args.gpt_max_workers <= 1:
                for t in _tqdm(gpt_tasks, desc="GPT first-error"):
                    rec = _gpt_process(t)
                    gpt_new.append(rec)
                    key = f"{rec['doc_id']}|{rec['draft_idx']}"
                    warm_cache[key] = rec
                    with gpt_cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                with ThreadPoolExecutor(
                        max_workers=args.gpt_max_workers) as pool:
                    futs = {pool.submit(_gpt_process, t): t
                            for t in gpt_tasks}
                    for fut in _tqdm(as_completed(futs),
                                     total=len(futs),
                                     desc="GPT first-error"):
                        rec = fut.result()
                        gpt_new.append(rec)
                        key = f"{rec['doc_id']}|{rec['draft_idx']}"
                        warm_cache[key] = rec
                        with gpt_cache_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"GPT cache now: {len(warm_cache)} entries")

        # Build gpt_tau_map from warm_cache
        for key, rec in warm_cache.items():
            parts = key.split("|")
            doc_id, di = parts[0], int(parts[1])
            gpt_tau_map[(doc_id, di)] = rec.get("tau")

    gpt_required = [
        {
            "task_type": "gpt",
            "doc_id": draft["doc_id"],
            "draft_idx": draft["draft_idx"],
        }
        for draft in drafts
    ] if need_gpt else []
    gpt_executed = [
        {
            "task_type": "gpt",
            "doc_id": record["doc_id"],
            "draft_idx": record["draft_idx"],
        }
        for record in gpt_new
    ]
    gpt_elapsed = (
        time.perf_counter_ns() - _t0
    ) / 1_000_000_000
    _record_time("gpt", gpt_elapsed, len(gpt_new))
    timing.record_phase(
        "gpt",
        required=gpt_required,
        executed=gpt_executed,
        current_wall_sec=gpt_elapsed,
    )

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

            if any(s.startswith("prm_drop") for s in signals):
                scores = prm_rec.get("step_scores", [])
                rb_steps.update(
                    compute_rollback_step_prm_drop(
                        scores, n, prm_thr))

            scores = prm_rec.get("step_scores", [])
            if MAX_PRM_DROP_SIGNAL in signals:
                assignment = max_prm_drop_assignment(scores, n)
                rb_steps[MAX_PRM_DROP_SIGNAL] = int(
                    assignment["rollback_step"]
                )

            if PRM_BELOW_THRESHOLD_SIGNAL in signals:
                assignment = prm_below_threshold_assignment(
                    scores,
                    n,
                    prm_absolute_thr,
                )
                rb_steps[PRM_BELOW_THRESHOLD_SIGNAL] = int(
                    assignment["rollback_step"]
                )

            if any(s.startswith("nll_drop") for s in signals):
                lp_rec = lp_map.get((q["doc_id"], di))
                if lp_rec and d.get("draft_steps"):
                    bounds = step_char_bounds(
                        d["draft_text"], d["draft_steps"])
                    nlls = compute_step_mean_nll(
                        lp_rec["token_logprobs"],
                        lp_rec["token_offsets"], bounds)
                    rb_steps.update(
                        compute_rollback_step_nll_drop(
                            nlls, n, nll_thr))
                else:
                    rb_steps["nll_drop"] = max(n - 1, 0)
                    rb_steps["nll_drop_fb_last"] = max(n - 1, 0)
                    rb_steps["nll_drop_fb_start"] = 0

            if any(s.startswith("gpt") for s in signals):
                tau = gpt_tau_map.get((q["doc_id"], di))
                rb_steps.update(
                    compute_rollback_step_gpt(tau, n))

            prm_rec["_rb_steps"] = rb_steps

    # ---- Phase 4: Suffix generation ----
    _t0 = time.perf_counter_ns()
    suffix_new = []
    suffix_launch = {}
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
                    if strat not in signals:
                        continue
                    b = min(rb, len(steps) - 1)
                    if b < 0:
                        b = 0
                    if b == 0:
                        prefix = ""
                    else:
                        prefix = ("\n\n".join(steps[:b])
                                  + "\n\n")
                    p = build_prompt(
                        MODEL_ID, DATASET, q["question"],
                        context=q.get("context", ""))
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
            suffix_new, suffix_launch = launch_shards(
                sfx_tasks, gpu_ids, sd / "p4",
                seed=args.seed,
                phase_name="suffix",
                timing_recorder=timing,
                return_stats=True,
            )
            with ckpt.open("a") as f:
                for r in suffix_new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(suffix_new)

    sfx_recs = [r for r in existing
                if r.get("task_type") == "suffix"]
    suffix_elapsed = (
        time.perf_counter_ns() - _t0
    ) / 1_000_000_000
    _record_time("suffix", suffix_elapsed, len(suffix_new))
    timing.record_phase(
        "suffix",
        required=sfx_recs,
        executed=suffix_new,
        current_wall_sec=suffix_elapsed,
        launch_stats=suffix_launch,
    )
    sfx_map = {}
    for s in sfx_recs:
        sfx_map[(s["doc_id"], s["draft_idx"],
                 s.get("strategy", "prm_drop"),
                 s["rollback_step"], s["suffix_idx"])] = s
    print(f"Total suffix records: {len(sfx_recs)}")

    # ---- Phase 5: Full SC baseline ----
    # Drafts are i.i.d. samples from the same distribution, so we
    # reuse them as part of the SC pool and only supplement the rest.
    _t0 = time.perf_counter_ns()
    sc_new = []
    sc_launch = {}
    sc_supplement = max(0, sc_n - nd_max)
    if 5 not in args.skip_phase:
        done_sc = {(r["doc_id"], r["sc_idx"])
                   for r in existing
                   if r.get("task_type") == "fullsc"}
        sc_tasks = []
        for q in questions:
            p = build_prompt(MODEL_ID, DATASET,
                             q["question"],
                             context=q.get("context", ""))
            for si in range(sc_supplement):
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
            print(f"\n--- Phase 5: {len(sc_tasks)} SC "
                  f"(supplement; {nd_max} drafts reused) ---")
            sc_new, sc_launch = launch_shards(
                sc_tasks, gpu_ids, sd / "p5",
                seed=args.seed,
                phase_name="fullsc",
                timing_recorder=timing,
                return_stats=True,
            )
            with ckpt.open("a") as f:
                for r in sc_new:
                    f.write(json.dumps(r, ensure_ascii=False)
                            + "\n")
            existing.extend(sc_new)

    sc_recs = [r for r in existing
               if r.get("task_type") == "fullsc"]
    sc_elapsed = (
        time.perf_counter_ns() - _t0
    ) / 1_000_000_000
    _record_time("fullsc", sc_elapsed, len(sc_new))
    timing.record_phase(
        "fullsc",
        required=sc_recs,
        executed=sc_new,
        current_wall_sec=sc_elapsed,
        launch_stats=sc_launch,
    )
    # Merge drafts into SC pool (drafts are i.i.d. with fullsc)
    draft_as_sc = []
    for d in drafts:
        draft_as_sc.append(dict(
            doc_id=d["doc_id"],
            sc_answer=d.get("draft_answer", ""),
            sc_tokens=d.get("draft_tokens", 0),
        ))
    sc_pool = draft_as_sc + sc_recs
    print(f"Total SC pool: {len(draft_as_sc)} drafts + "
          f"{len(sc_recs)} fullsc = {len(sc_pool)}")

    per_question_timing = publish_per_question_timings(
        out_dir,
        checkpoint_records=existing,
        signals=signals,
        current_run_dir=timing.run_dir,
        source_event_stores=task_event_sources,
    )
    print(
        "Per-question timing: "
        f"{per_question_timing['task_events_recorded']}/"
        f"{per_question_timing['task_events_required']} task events"
    )
    for signal_summary in per_question_timing["signals"]:
        print(
            f"  {signal_summary['signal']}: "
            f"{signal_summary['complete_questions']}/"
            f"{signal_summary['questions']} complete questions, "
            "generation_mean="
            f"{signal_summary['total_generation_mean_sec']}, "
            "signal_mean="
            f"{signal_summary['signal_computation_mean_sec']}"
        )

    if args.skip_eval:
        stale_artifacts = [
            out_dir / "eval_summary.json",
            out_dir / "eval_summary_table.md",
            fig_dir / "fig_efficiency_frontier.json",
            fig_dir / "fig_efficiency_frontier.pdf",
            fig_dir / "fig_efficiency_frontier.png",
        ]
        removed = []
        for artifact in stale_artifacts:
            if artifact.is_file():
                artifact.unlink()
                removed.append(str(artifact))
        print("Evaluation skipped (checkpoint preparation only).")
        if removed:
            print("Removed stale evaluation artifacts:")
            for artifact in removed:
                print(f"  {artifact}")
        timing_manifest = timing.finalize()
        print(
            "Timing manifest: "
            f"{out_dir / 'timing' / 'timing_manifest.json'} "
            f"(current={timing_manifest['observed_current_end_to_end_wall_sec']:.3f}s, "
            "from_scratch="
            f"{timing_manifest['reconstructed_from_scratch_wall_sec']})"
        )
        return

    # ---- Evaluate ----
    evaluation_started_ns = time.perf_counter_ns()
    print("\n--- Evaluation ---")

    # Invocation-scoped wall-clock summary.  Historical time is printed only
    # when an exact task-set timing source was available.
    n_drafts = len(drafts)
    nq = len(questions)
    print("\n--- Wall-clock provenance (current invocation) ---")
    for phase in timing.phases:
        reconstructed = phase[
            "reconstructed_from_scratch_wall_sec"
        ]
        reconstructed_text = (
            f"{reconstructed:.3f}s"
            if reconstructed is not None
            else "N/A"
        )
        print(
            f"  {phase['phase']:<10s} "
            f"status={phase['status']:<12s} "
            f"tasks={phase['tasks_executed']}/"
            f"{phase['tasks_required']} "
            f"current={phase['observed_current_wall_sec']:.3f}s "
            f"from_scratch={reconstructed_text}"
        )

    # Re-extract answers from raw text to pick up any
    # extract_answer improvements without re-generating.
    _n_re = 0
    for d in draft_map.values():
        if "draft_text" in d:
            d["draft_answer"] = extract_answer(DATASET, d["draft_text"])
            _n_re += 1
    for s in sfx_map.values():
        if "suffix_text" in s:
            s["suffix_answer"] = extract_answer(DATASET, s["suffix_text"])
            _n_re += 1
    for s in sc_recs:
        if "sc_text" in s:
            s["sc_answer"] = extract_answer(DATASET, s["sc_text"])
            _n_re += 1
    if _n_re:
        print(f"  Re-extracted answers for {_n_re} records")

    step_tokens_path = out_dir / "draft_step_tokens.json"
    step_tokens_map = None
    if step_tokens_path.exists():
        step_tokens_map = json.loads(step_tokens_path.read_text())
        print(f"  Loaded step token counts: {len(step_tokens_map)} drafts")
    else:
        print("  WARNING: draft_step_tokens.json not found, "
              "using full draft_tokens for fair prefix cost")

    eval_results = evaluate_configs(
        questions, draft_map, prm_map, sfx_map,
        sc_pool, B, signals, step_tokens_map=step_tokens_map)

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
    evaluation_elapsed = (
        time.perf_counter_ns() - evaluation_started_ns
    ) / 1_000_000_000
    evaluation_task = [{
        "task_type": "evaluation",
        "doc_id": DATASET,
    }]
    timing.record_phase(
        "evaluation",
        required=evaluation_task,
        executed=evaluation_task,
        current_wall_sec=evaluation_elapsed,
    )
    save_summary_table(
        eval_results,
        out_dir,
        phase_times,
        n_drafts,
        nq,
        timing_phases=timing.phases,
        per_question_timing=per_question_timing,
    )
    timing_manifest = timing.finalize()
    print(
        "Timing manifest: "
        f"{out_dir / 'timing' / 'timing_manifest.json'} "
        f"(current={timing_manifest['observed_current_end_to_end_wall_sec']:.3f}s, "
        "from_scratch="
        f"{timing_manifest['reconstructed_from_scratch_wall_sec']})"
    )
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
