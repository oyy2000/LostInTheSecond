#!/usr/bin/env python3
"""Run same-draft Qwen NLL and PRM rollback controls.

The pipeline reuses the exact drafts produced by experiment 6_9. It scores
those drafts with the original experiment-6_8 NLL/PRM definitions, applies
the original fallback-before-last-step behavior, and generates only the new
suffix samples required for B=nd*ns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.src.nll_prm_controls import (  # noqa: E402
    ALL_SIGNALS,
    NLL_SIGNAL,
    PRM_SIGNAL,
    compute_step_mean_nll,
    nll_rollback_assignment,
    prm_rollback_assignment,
    retained_prefix,
    step_char_bounds,
)
from rebuttal.src.position_controls import (  # noqa: E402
    additional_suffixes_per_draft,
)
from rebuttal.src.stable_sampling import derive_sampling_seed  # noqa: E402
from src.hotpotqa_answer_equiv import hotpotqa_f1  # noqa: E402
from src.prompt_templates import (  # noqa: E402
    build_prompt,
    check_answer,
    extract_answer,
    get_stop_tokens,
)
from src.sweep_datasets import load_dataset_by_name  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=["math500", "hotpotqa", "hotpotqa_open"],
        required=True,
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="Comma-separated override for the configured GPU list",
    )
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--nll-threshold", type=float, default=None)
    parser.add_argument("--prm-threshold", type=float, default=None)
    parser.add_argument(
        "--reuse-scoring-checkpoint",
        type=Path,
        default=None,
        help=(
            "Reuse PRM/NLL records from a run on the exact same source "
            "draft checkpoint. Suffixes are also reused only when the "
            "new threshold selects the identical rollback step."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")

    parser.add_argument(
        "--_worker",
        choices=["prm", "nll", "suffix"],
        default="",
    )
    parser.add_argument("--_gpu", default="0")
    parser.add_argument("--_shard-id", type=int, default=-1)
    parser.add_argument("--_task-file", type=Path, default=None)
    parser.add_argument("--_seed", type=int, default=-1)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model",
        "prm_model",
        "signals",
        "thresholds",
        "budget",
        "nd",
        "dataset_seed",
        "generation_seed",
        "temperature",
        "top_p",
        "max_tokens",
        "max_model_len",
        "gpus",
        "source_results_root",
        "results_root",
        "datasets",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    if set(config["signals"]) != set(ALL_SIGNALS):
        raise ValueError(
            f"Expected exactly {ALL_SIGNALS}, found {config['signals']}"
        )
    return config


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
    return records


def _append_unique(
    path: Path,
    records: Iterable[dict],
    known_keys: set[str],
) -> List[dict]:
    added = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            task_key = record["task_key"]
            if task_key in known_keys:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            known_keys.add(task_key)
            added.append(record)
    return added


def _replace_jsonl(path: Path, records: Iterable[dict]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def _record_matches_shard_task(
    record: dict,
    task: dict,
    worker: str,
    config: dict,
) -> bool:
    if record.get("task_key") != task.get("task_key"):
        return False
    if worker == "nll":
        return (
            record.get("scoring_backend")
            == config.get("nll_backend", "vllm")
        )
    if worker == "suffix":
        fields = (
            "task_type",
            "doc_id",
            "draft_idx",
            "strategy",
            "rollback_step",
            "suffix_idx",
        )
        if any(record.get(field) != task.get(field) for field in fields):
            return False
        generation_seed = config.get("generation_seed")
        if generation_seed is not None:
            return record.get("sampling_seed") == derive_sampling_seed(
                int(generation_seed), task
            )
    return True


def _prune_shard_checkpoint(
    path: Path,
    tasks: List[dict],
    worker: str,
    config: dict,
) -> int:
    if not path.exists():
        return 0
    task_map = {task["task_key"]: task for task in tasks}
    retained = []
    retained_keys = set()
    records = _read_jsonl(path)
    for record in records:
        task_key = record.get("task_key")
        task = task_map.get(task_key)
        if (
            task is None
            or task_key in retained_keys
            or not _record_matches_shard_task(
                record, task, worker, config
            )
        ):
            continue
        retained.append(record)
        retained_keys.add(task_key)
    removed = len(records) - len(retained)
    if removed:
        _replace_jsonl(path, retained)
    return removed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_without_payload(task: dict, payload_keys: Iterable[str]) -> dict:
    excluded = set(payload_keys) | {"_out"}
    return {
        key: value for key, value in task.items() if key not in excluded
    }


def run_prm_worker(args, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer, DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, *unused_args, **unused_kwargs:
            self.get_seq_length()
        )

    tasks = json.loads(args._task_file.read_text(encoding="utf-8"))
    if not tasks:
        return
    output_path = Path(tasks[0]["_out"])
    completed = {
        record["task_key"]
        for record in _read_jsonl(output_path)
        if record.get("task_key")
    }
    pending = [
        task for task in tasks if task["task_key"] not in completed
    ]
    print(
        f"[PRM shard {args._shard_id}] GPU {args._gpu}: "
        f"{len(pending)}/{len(tasks)} pending"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["prm_model"], trust_remote_code=True
    )
    dataset_config = config["datasets"][args.dataset]
    tensor_parallel_size = int(
        dataset_config.get("prm_tensor_parallel_size", 1)
    )
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if tensor_parallel_size > 1:
        max_memory_gib = int(
            dataset_config["prm_max_memory_per_gpu_gib"]
        )
        model_kwargs.update(
            device_map="auto",
            max_memory={
                index: f"{max_memory_gib}GiB"
                for index in range(tensor_parallel_size)
            },
            low_cpu_mem_usage=True,
        )
    else:
        model_kwargs["device_map"] = {"": "cuda:0"}
    model = AutoModel.from_pretrained(
        config["prm_model"],
        **model_kwargs,
    ).eval()
    step_separator_id = tokenizer.encode("<extra_0>")[0]
    max_model_len = int(config["max_model_len"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for index, task in enumerate(pending, start=1):
            step_text = "<extra_0>".join(task["steps"]) + "<extra_0>"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Please reason step by step, and put your final "
                        "answer within \\boxed{}."
                    ),
                },
                {"role": "user", "content": task["question"]},
                {"role": "assistant", "content": step_text},
            ]
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            input_ids = torch.tensor(
                [tokenizer.encode(conversation)],
                dtype=torch.long,
                device=model.device,
            )
            if input_ids.shape[1] >= max_model_len:
                raise ValueError(
                    f"{task['task_key']} has {input_ids.shape[1]} PRM "
                    f"tokens, exceeding max_model_len={max_model_len}"
                )
            with torch.no_grad():
                logits = model(input_ids=input_ids)[0]
            separator_mask = input_ids == step_separator_id
            probabilities = functional.softmax(logits, dim=-1)
            scores = probabilities[
                0, separator_mask[0].to(probabilities.device), 1
            ]
            if len(scores) != task["n_steps"]:
                raise ValueError(
                    f"{task['task_key']}: expected {task['n_steps']} PRM "
                    f"scores, found {len(scores)}"
                )
            record = _record_without_payload(
                task, ("question", "steps")
            )
            record.update(
                task_type="prm",
                step_scores=[
                    round(float(score), 8)
                    for score in scores.detach().cpu()
                ],
                input_tokens=int(input_ids.shape[1]),
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if index % 100 == 0 or index == len(pending):
                print(
                    f"[PRM shard {args._shard_id}] "
                    f"{index}/{len(pending)}"
                )


def _actual_token_logprob(logprob_dict, token_id: int) -> float:
    if token_id not in logprob_dict:
        raise ValueError(
            f"Prompt logprobs omit actual token id {token_id}"
        )
    return float(logprob_dict[token_id].logprob)


def causal_token_logprobs(
    logits,
    input_ids,
    token_positions: List[int],
    chunk_size: int = 64,
) -> List[float]:
    """Return causal log-probabilities without a full float32 logits copy."""
    import torch

    if any(position <= 0 for position in token_positions):
        raise ValueError("Causal log-probabilities require position > 0")
    values = []
    for start in range(0, len(token_positions), chunk_size):
        positions = token_positions[start:start + chunk_size]
        row_indices = torch.tensor(
            [position - 1 for position in positions],
            device=logits.device,
            dtype=torch.long,
        )
        target_indices = torch.tensor(
            positions,
            device=input_ids.device,
            dtype=torch.long,
        )
        rows = logits.index_select(0, row_indices).float()
        targets = input_ids.index_select(0, target_indices)
        target_logits = rows.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        chunk_values = target_logits - torch.logsumexp(rows, dim=1)
        values.extend(
            float(value)
            for value in chunk_values.detach().cpu()
        )
    return values


def _wait_for_nll_gpu_memory(args, config) -> None:
    dataset_config = config["datasets"][args.dataset]
    min_free_mib = int(
        dataset_config.get("nll_min_free_memory_mib", 0)
    )
    if min_free_mib <= 0:
        return
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id", args._gpu,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        free_values = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        gpu_ids = [
            gpu_id.strip()
            for gpu_id in args._gpu.split(",")
            if gpu_id.strip()
        ]
        if len(free_values) != len(gpu_ids):
            raise RuntimeError(
                f"Expected {len(gpu_ids)} GPU memory values for "
                f"{args._gpu}, "
                f"found {free_values}"
            )
        free_mib = min(free_values)
        if free_mib >= min_free_mib:
            print(
                f"[NLL shard {args._shard_id}] GPU {args._gpu}: "
                f"{free_mib} MiB free, starting scorer"
            )
            return
        print(
            f"[NLL shard {args._shard_id}] GPU {args._gpu}: "
            f"waiting for {min_free_mib} MiB free "
            f"(currently {free_mib} MiB)"
        )
        time.sleep(60)


def run_nll_worker(args, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    _wait_for_nll_gpu_memory(args, config)

    tasks = json.loads(args._task_file.read_text(encoding="utf-8"))
    if not tasks:
        return
    output_path = Path(tasks[0]["_out"])
    completed = {
        record["task_key"]
        for record in _read_jsonl(output_path)
        if record.get("task_key")
    }
    pending = [
        task for task in tasks if task["task_key"] not in completed
    ]
    print(
        f"[NLL shard {args._shard_id}] GPU {args._gpu}: "
        f"{len(pending)}/{len(tasks)} pending"
    )

    backend = config.get("nll_backend", "vllm")
    if backend == "transformers":
        _run_transformers_nll_worker(
            args, config, pending, output_path
        )
        return
    if backend != "vllm":
        raise ValueError(f"Unsupported NLL backend: {backend}")

    from vllm import LLM, SamplingParams

    dataset_config = config["datasets"][args.dataset]
    tensor_parallel_size = int(
        dataset_config.get("nll_tensor_parallel_size", 1)
    )
    llm = LLM(
        model=config["model"],
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=float(
            dataset_config.get(
                "nll_gpu_memory_utilization",
                dataset_config["gpu_memory_utilization"],
            )
        ),
        # Prompt-logprob scoring asks vLLM for one dummy output token.
        # The scored prompt itself remains capped at the generation
        # environment's limit; the extra slot is only for that dummy token.
        max_model_len=int(config["max_model_len"]) + 1,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    batch_size = int(config.get("nll_batch_size", 8))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            encodings = [
                tokenizer(
                    task["full_prompt"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                for task in batch
            ]
            input_ids = [encoding["input_ids"] for encoding in encodings]
            if any(
                len(ids) > int(config["max_model_len"])
                for ids in input_ids
            ):
                too_long = [
                    len(ids) for ids in input_ids
                    if len(ids) > int(config["max_model_len"])
                ]
                raise ValueError(
                    f"NLL inputs exceed max_model_len: {too_long[:3]}"
                )
            generations = llm.generate(
                [{"prompt_token_ids": ids} for ids in input_ids],
                sampling_params=sampling,
            )
            for task, encoding, generation in zip(
                batch, encodings, generations
            ):
                prompt_logprobs = generation.prompt_logprobs
                if prompt_logprobs is None:
                    raise ValueError(
                        f"{task['task_key']}: prompt_logprobs is missing"
                    )
                token_logprobs = []
                token_offsets = []
                response_offset = task["resp_char_offset"]
                for token_id, token_span, logprob_dict in zip(
                    generation.prompt_token_ids,
                    encoding["offset_mapping"],
                    prompt_logprobs,
                ):
                    start_char, end_char = token_span
                    if end_char <= response_offset:
                        continue
                    if logprob_dict is None:
                        raise ValueError(
                            f"{task['task_key']}: response token has no "
                            "prompt logprob"
                        )
                    token_logprobs.append(
                        _actual_token_logprob(logprob_dict, token_id)
                    )
                    token_offsets.append(
                        max(0, start_char - response_offset)
                    )
                record = _record_without_payload(
                    task, ("full_prompt",)
                )
                record.update(
                    task_type="nll",
                    token_logprobs=token_logprobs,
                    token_offsets=token_offsets,
                    input_tokens=len(encoding["input_ids"]),
                    response_scored_tokens=len(token_logprobs),
                )
                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
            output.flush()
            done = min(start + len(batch), len(pending))
            print(
                f"[NLL shard {args._shard_id}] "
                f"{done}/{len(pending)}"
            )


def _run_transformers_nll_worker(
    args,
    config,
    pending: List[dict],
    output_path: Path,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    max_model_len = int(config["max_model_len"])
    chunk_size = int(config.get("nll_logprob_chunk_size", 64))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as output:
        for index, task in enumerate(pending, start=1):
            encoding = tokenizer(
                task["full_prompt"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids = encoding["input_ids"]
            if len(token_ids) > max_model_len:
                raise ValueError(
                    f"NLL input exceeds max_model_len: {len(token_ids)}"
                )
            response_offset = task["resp_char_offset"]
            positions = [
                position
                for position, (_, end_char) in enumerate(
                    encoding["offset_mapping"]
                )
                if end_char > response_offset
            ]
            if not positions:
                raise ValueError(
                    f"{task['task_key']}: response has no scored tokens"
                )
            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=model.device,
            )
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True,
                ).logits[0]
                token_logprobs = causal_token_logprobs(
                    logits,
                    input_ids[0],
                    positions,
                    chunk_size=chunk_size,
                )
            token_offsets = [
                max(
                    0,
                    encoding["offset_mapping"][position][0]
                    - response_offset,
                )
                for position in positions
            ]
            record = _record_without_payload(task, ("full_prompt",))
            record.update(
                task_type="nll",
                token_logprobs=token_logprobs,
                token_offsets=token_offsets,
                input_tokens=len(token_ids),
                response_scored_tokens=len(token_logprobs),
            )
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output.flush()
            del input_ids, logits
            if index % 10 == 0 or index == len(pending):
                print(
                    f"[NLL shard {args._shard_id}] "
                    f"{index}/{len(pending)}"
                )


def run_suffix_worker(args, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(args._task_file.read_text(encoding="utf-8"))
    if not tasks:
        return
    output_path = Path(tasks[0]["_out"])
    completed = {
        record["task_key"]
        for record in _read_jsonl(output_path)
        if record.get("task_key")
    }
    pending = [
        task for task in tasks if task["task_key"] not in completed
    ]
    print(
        f"[suffix shard {args._shard_id}] GPU {args._gpu}: "
        f"{len(pending)}/{len(tasks)} pending"
    )

    dataset_config = config["datasets"][args.dataset]
    llm = LLM(
        model=config["model"],
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=float(
            dataset_config["gpu_memory_utilization"]
        ),
        max_model_len=int(config["max_model_len"]),
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    stop = get_stop_tokens(config["model"])
    batch_size = int(config.get("generation_batch_size", 512))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            prompt_ids = [
                tokenizer.encode(
                    task["prompt"], add_special_tokens=False
                )
                for task in batch
            ]
            generations = llm.generate(
                [{"prompt_token_ids": ids} for ids in prompt_ids],
                sampling_params=[
                    SamplingParams(
                        temperature=float(config["temperature"]),
                        top_p=float(config["top_p"]),
                        max_tokens=int(config["max_tokens"]),
                        stop=stop,
                        seed=(
                            derive_sampling_seed(args._seed, task)
                            if args._seed >= 0 else None
                        ),
                    )
                    for task in batch
                ],
            )
            for task, generation in zip(batch, generations):
                text = generation.outputs[0].text
                record = _record_without_payload(
                    task, ("prompt", "retained_text")
                )
                record.update(
                    task_type="suffix",
                    generation_kind="suffix",
                    suffix_text=text,
                    suffix_answer=extract_answer(args.dataset, text),
                    suffix_tokens=len(generation.outputs[0].token_ids),
                    prefix_tokens=len(
                        tokenizer.encode(
                            task["retained_text"],
                            add_special_tokens=False,
                        )
                    ),
                )
                if args._seed >= 0:
                    record["sampling_seed"] = derive_sampling_seed(
                        args._seed, task
                    )
                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
            output.flush()
            done = min(start + len(batch), len(pending))
            print(
                f"[suffix shard {args._shard_id}] "
                f"{done}/{len(pending)}"
            )


def launch_shards(
    tasks: List[dict],
    worker: str,
    gpu_ids: List[str],
    shard_dir: Path,
    args,
    config: dict,
) -> List[dict]:
    if not tasks:
        return []
    shard_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = config["datasets"][args.dataset]
    if worker == "prm":
        tensor_parallel_size = int(
            dataset_config.get("prm_tensor_parallel_size", 1)
        )
    elif worker == "nll":
        tensor_parallel_size = int(
            dataset_config.get("nll_tensor_parallel_size", 1)
        )
    else:
        tensor_parallel_size = 1
    if tensor_parallel_size > len(gpu_ids):
        raise ValueError(
            f"{worker} tensor parallel size {tensor_parallel_size} exceeds "
            f"the {len(gpu_ids)} configured GPUs"
        )
    if tensor_parallel_size > 1:
        gpu_assignments = [
            ",".join(gpu_ids[:tensor_parallel_size])
        ]
    else:
        gpu_assignments = gpu_ids
    shards = [[] for _ in gpu_assignments]
    for index, task in enumerate(tasks):
        shards[index % len(gpu_assignments)].append(task)

    processes: List[Tuple[int, subprocess.Popen, Any]] = []
    output_paths = []
    task_keys = {task["task_key"] for task in tasks}
    for shard_id, (gpu_id, shard) in enumerate(
        zip(gpu_assignments, shards)
    ):
        if not shard:
            continue
        output_path = shard_dir / f"shard_{shard_id}.jsonl"
        removed = _prune_shard_checkpoint(
            output_path, shard, worker, config
        )
        if removed:
            print(
                f"  worker={worker} shard={shard_id} "
                f"removed_stale={removed}"
            )
        for task in shard:
            task["_out"] = str(output_path)
        task_file = shard_dir / f"tasks_{shard_id}.json"
        task_file.write_text(
            json.dumps(shard, ensure_ascii=False),
            encoding="utf-8",
        )
        log_handle = (
            shard_dir / f"log_{shard_id}.txt"
        ).open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config", str(args.config.resolve()),
            "--dataset", args.dataset,
            "--_worker", worker,
            "--_gpu", gpu_id,
            "--_shard-id", str(shard_id),
            "--_task-file", str(task_file),
        ]
        generation_seed = config["generation_seed"]
        if generation_seed is not None:
            command += ["--_seed", str(generation_seed)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["TOKENIZERS_PARALLELISM"] = "false"
        if worker == "nll" and tensor_parallel_size > 1:
            environment["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:False"
            )
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_id, process, log_handle))
        output_paths.append(output_path)
        print(
            f"  worker={worker} shard={shard_id} "
            f"gpu={gpu_id} tasks={len(shard)}"
        )

    for shard_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        log_path = shard_dir / f"log_{shard_id}.txt"
        tail = "\n".join(
            log_path.read_text(errors="replace").splitlines()[-8:]
        )
        print(f"  shard={shard_id} exit={return_code}\n{tail}")
        if return_code != 0:
            raise RuntimeError(
                f"{worker} shard {shard_id} failed; see {log_path}"
            )

    records = []
    for output_path in output_paths:
        for record in _read_jsonl(output_path):
            if record.get("task_key") in task_keys:
                records.append(record)
    return records


def build_rollback_configs(config: dict) -> List[Tuple[int, int]]:
    result = []
    budget = int(config["budget"])
    for nd in config["nd"]:
        nd = int(nd)
        if budget % nd:
            raise ValueError(f"Budget {budget} is not divisible by nd={nd}")
        result.append((nd, budget // nd))
    return result


def load_source_drafts(
    source_checkpoint: Path,
    questions: List[dict],
    nd_max: int,
) -> Dict[Tuple[str, int], dict]:
    expected_doc_ids = {question["doc_id"] for question in questions}
    drafts = {}
    with source_checkpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("task_type") != "draft"
                or record.get("doc_id") not in expected_doc_ids
            ):
                continue
            draft_idx = int(record["draft_idx"])
            if draft_idx >= nd_max:
                continue
            key = (record["doc_id"], draft_idx)
            if key in drafts:
                raise ValueError(f"Duplicate source draft: {key}")
            steps = record.get("draft_steps", [])
            if not steps:
                raise ValueError(f"Source draft has no steps: {key}")
            if int(record.get("n_steps", len(steps))) != len(steps):
                raise ValueError(f"Source draft step count mismatch: {key}")
            drafts[key] = record

    expected = len(questions) * nd_max
    if len(drafts) != expected:
        raise ValueError(
            f"Expected {expected} source drafts, found {len(drafts)}"
        )
    return drafts


def build_scoring_tasks(
    questions: List[dict],
    draft_map: Dict[Tuple[str, int], dict],
    nd_max: int,
    model: str,
    dataset: str,
    nll_backend: str = "vllm",
) -> Tuple[List[dict], List[dict]]:
    prm_tasks = []
    nll_tasks = []
    for question in questions:
        doc_id = question["doc_id"]
        base_prompt = build_prompt(
            model,
            dataset,
            question["question"],
            context=question.get("context", ""),
        )
        for draft_idx in range(nd_max):
            draft = draft_map[(doc_id, draft_idx)]
            steps = draft["draft_steps"]
            prm_tasks.append({
                "task_key": f"prm|{doc_id}|{draft_idx}",
                "task_type": "prm",
                "doc_id": doc_id,
                "draft_idx": draft_idx,
                "n_steps": len(steps),
                "question": question["question"],
                "steps": steps,
            })
            nll_tasks.append({
                "task_key": f"nll|{doc_id}|{draft_idx}",
                "task_type": "nll",
                "doc_id": doc_id,
                "draft_idx": draft_idx,
                "n_steps": len(steps),
                "scoring_backend": nll_backend,
                "full_prompt": base_prompt + draft["draft_text"],
                "resp_char_offset": len(base_prompt),
            })
    return prm_tasks, nll_tasks


def build_assignments(
    questions: List[dict],
    draft_map: Dict[Tuple[str, int], dict],
    scoring_records: List[dict],
    config: dict,
) -> List[dict]:
    score_map = {
        (
            record["doc_id"],
            record["draft_idx"],
            record["task_type"],
        ): record
        for record in scoring_records
        if record.get("task_type") in ("nll", "prm")
    }
    assignments = []
    nd_max = max(int(nd) for nd in config["nd"])
    for question in questions:
        doc_id = question["doc_id"]
        for draft_idx in range(nd_max):
            draft = draft_map[(doc_id, draft_idx)]
            n_steps = len(draft["draft_steps"])

            nll_record = score_map[(doc_id, draft_idx, "nll")]
            bounds = step_char_bounds(
                draft["draft_text"], draft["draft_steps"]
            )
            step_nlls = compute_step_mean_nll(
                nll_record["token_logprobs"],
                nll_record["token_offsets"],
                bounds,
            )
            nll_assignment = nll_rollback_assignment(
                step_nlls,
                n_steps,
                float(config["thresholds"][NLL_SIGNAL]),
            )

            prm_record = score_map[(doc_id, draft_idx, "prm")]
            prm_assignment = prm_rollback_assignment(
                prm_record["step_scores"],
                n_steps,
                float(config["thresholds"][PRM_SIGNAL]),
            )

            for assignment in (nll_assignment, prm_assignment):
                rollback_step = int(assignment["rollback_step"])
                assignment.update(
                    doc_id=doc_id,
                    draft_idx=draft_idx,
                    n_steps=n_steps,
                    target_step=rollback_step + 1,
                    target_fraction=(
                        (rollback_step + 1) / n_steps
                        if n_steps else 0.0
                    ),
                    fallback="last" if not assignment["triggered"] else None,
                )
                assignments.append(assignment)
    return assignments


def build_suffix_tasks(
    questions: List[dict],
    draft_map: Dict[Tuple[str, int], dict],
    assignments: List[dict],
    configs: List[Tuple[int, int]],
    config: dict,
) -> List[dict]:
    assignment_map = {
        (
            assignment["doc_id"],
            assignment["draft_idx"],
            assignment["signal"],
        ): assignment
        for assignment in assignments
    }
    counts = additional_suffixes_per_draft(configs)
    tasks = []
    nd_max = max(nd for nd, _ in configs)
    for question in questions:
        doc_id = question["doc_id"]
        base_prompt = build_prompt(
            config["model"],
            config["dataset"],
            question["question"],
            context=question.get("context", ""),
        )
        for draft_idx in range(nd_max):
            draft = draft_map[(doc_id, draft_idx)]
            for signal in config["signals"]:
                assignment = assignment_map[
                    (doc_id, draft_idx, signal)
                ]
                rollback_step = int(assignment["rollback_step"])
                retained = retained_prefix(
                    draft["draft_steps"], rollback_step
                )
                for suffix_idx in range(counts[draft_idx]):
                    tasks.append({
                        "task_key": (
                            f"suffix|{signal}|{doc_id}|{draft_idx}|"
                            f"{suffix_idx}"
                        ),
                        "task_type": "suffix",
                        "doc_id": doc_id,
                        "draft_idx": draft_idx,
                        "strategy": signal,
                        "rollback_step": rollback_step,
                        "suffix_idx": suffix_idx,
                        "gold_answer": question["gold_answer"],
                        "retained_text": retained,
                        "prompt": base_prompt + retained,
                    })
    return tasks


def _vote(answers: List[str]) -> str:
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def evaluate(
    questions: List[dict],
    draft_map: Dict[Tuple[str, int], dict],
    scoring_records: List[dict],
    suffix_records: List[dict],
    assignments: List[dict],
    configs: List[Tuple[int, int]],
    config: dict,
) -> List[dict]:
    suffix_map = {
        (
            record["doc_id"],
            record["draft_idx"],
            record["strategy"],
            record["suffix_idx"],
        ): record
        for record in suffix_records
    }
    assignment_map = {
        (
            record["doc_id"],
            record["draft_idx"],
            record["signal"],
        ): record
        for record in assignments
    }
    scoring_map = {
        (
            record["doc_id"],
            record["draft_idx"],
            record["task_type"],
        ): record
        for record in scoring_records
        if record.get("task_type") in ("nll", "prm")
    }
    dataset = config["dataset"]
    n_questions = len(questions)
    results = []

    greedy_correct = 0
    greedy_f1 = 0.0
    greedy_tokens = 0
    for question in questions:
        draft = draft_map[(question["doc_id"], 0)]
        answer = draft["draft_answer"]
        greedy_correct += int(
            check_answer(dataset, answer, question["gold_answer"])
        )
        if dataset in ("hotpotqa", "hotpotqa_open"):
            greedy_f1 += hotpotqa_f1(
                answer, question["gold_answer"]
            )
        greedy_tokens += draft["draft_tokens"]
    greedy = {
        "method": "Greedy@1",
        "nd": 1,
        "ns": 0,
        "n_answers": 1,
        "n_draft_answers": 1,
        "n_suffix_answers": 0,
        "acc": greedy_correct / n_questions,
        "tokens_per_q": greedy_tokens / n_questions,
        "signal_tokens_per_q": 0.0,
        "tokens_per_q_with_signal": greedy_tokens / n_questions,
        "total_tokens": greedy_tokens,
        "total_signal_tokens": 0,
    }
    if dataset in ("hotpotqa", "hotpotqa_open"):
        greedy["f1"] = greedy_f1 / n_questions
    results.append(greedy)

    task_type_for_signal = {
        NLL_SIGNAL: "nll",
        PRM_SIGNAL: "prm",
    }
    for signal in config["signals"]:
        for nd, ns in configs:
            correct = 0
            f1_total = 0.0
            total_tokens = 0
            total_signal_tokens = 0
            total_answers = 0
            total_draft_answers = 0
            total_suffix_answers = 0
            rollback_steps = []
            triggered = 0
            missing = []
            for question in questions:
                doc_id = question["doc_id"]
                answers = []
                question_tokens = 0
                question_signal_tokens = 0
                for draft_idx in range(nd):
                    draft = draft_map[(doc_id, draft_idx)]
                    answers.append(draft["draft_answer"])
                    question_tokens += draft["draft_tokens"]
                    total_draft_answers += 1
                    assignment = assignment_map[
                        (doc_id, draft_idx, signal)
                    ]
                    rollback_steps.append(
                        assignment["rollback_step"]
                    )
                    triggered += int(assignment["triggered"])
                    score_record = scoring_map[
                        (
                            doc_id,
                            draft_idx,
                            task_type_for_signal[signal],
                        )
                    ]
                    question_signal_tokens += score_record["input_tokens"]
                    for suffix_idx in range(ns - 1):
                        record = suffix_map.get(
                            (doc_id, draft_idx, signal, suffix_idx)
                        )
                        if record is None:
                            missing.append(
                                (
                                    doc_id,
                                    draft_idx,
                                    signal,
                                    suffix_idx,
                                )
                            )
                            continue
                        answers.append(record["suffix_answer"])
                        question_tokens += record["suffix_tokens"]
                        total_suffix_answers += 1
                voted = _vote(answers)
                correct += int(
                    check_answer(
                        dataset, voted, question["gold_answer"]
                    )
                )
                if dataset in ("hotpotqa", "hotpotqa_open"):
                    f1_total += hotpotqa_f1(
                        voted, question["gold_answer"]
                    )
                total_tokens += question_tokens
                total_signal_tokens += question_signal_tokens
                total_answers += len(answers)
            if missing:
                raise RuntimeError(
                    f"{len(missing)} suffix records missing; "
                    f"first={missing[:3]}"
                )
            tokens_per_q = total_tokens / n_questions
            signal_tokens_per_q = (
                total_signal_tokens / n_questions
            )
            result = {
                "method": f"{signal}_nd{nd}_ns{ns}",
                "strategy": signal,
                "variant": "draft_plus_suffix",
                "nd": nd,
                "ns": ns,
                "n_answers": total_answers / n_questions,
                "n_draft_answers": (
                    total_draft_answers / n_questions
                ),
                "n_suffix_answers": (
                    total_suffix_answers / n_questions
                ),
                "acc": correct / n_questions,
                "tokens_per_q": tokens_per_q,
                "signal_tokens_per_q": signal_tokens_per_q,
                "tokens_per_q_with_signal": (
                    tokens_per_q + signal_tokens_per_q
                ),
                "total_tokens": total_tokens,
                "total_signal_tokens": total_signal_tokens,
                "avg_rollback_step": (
                    sum(rollback_steps) / len(rollback_steps)
                ),
                "trigger_rate": triggered / len(rollback_steps),
            }
            if dataset in ("hotpotqa", "hotpotqa_open"):
                result["f1"] = f1_total / n_questions
            results.append(result)
    return results


def save_artifacts(
    results: List[dict],
    assignments: List[dict],
    out_dir: Path,
    config: dict,
):
    (out_dir / "eval_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "rollback_assignments.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for assignment in assignments:
            handle.write(
                json.dumps(assignment, ensure_ascii=False) + "\n"
            )

    hotpot = config["dataset"] in ("hotpotqa", "hotpotqa_open")
    lines = [
        "| method | nd | ns | accuracy | generated tokens/q "
        "| signal tokens/q | total tokens/q | trigger rate"
        + (" | F1 |" if hotpot else " |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|"
        + ("---:|" if hotpot else ""),
    ]
    for result in results:
        line = (
            f"| {result['method']} | {result.get('nd', '')} "
            f"| {result.get('ns', '')} | {result['acc']:.4f} "
            f"| {result['tokens_per_q']:.1f} "
            f"| {result['signal_tokens_per_q']:.1f} "
            f"| {result['tokens_per_q_with_signal']:.1f} "
            f"| {result.get('trigger_rate', 0.0):.4f} "
        )
        if hotpot:
            line += f"| {result['f1']:.4f} |"
        else:
            line += "|"
        lines.append(line)
    (out_dir / "eval_summary_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    colors = {
        NLL_SIGNAL: "#2563eb",
        PRM_SIGNAL: "#dc2626",
    }
    markers = {
        NLL_SIGNAL: "s",
        PRM_SIGNAL: "o",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for signal in config["signals"]:
        rows = sorted(
            [
                row for row in results
                if row.get("strategy") == signal
            ],
            key=lambda row: row["tokens_per_q"],
        )
        axis.plot(
            [row["tokens_per_q"] for row in rows],
            [row["acc"] for row in rows],
            color=colors[signal],
            marker=markers[signal],
            linewidth=1.7,
            markersize=6,
            label=signal.replace("_fb_last", "").replace("_", " "),
        )
        for row in rows:
            axis.annotate(
                f"nd={row['nd']}",
                (row["tokens_per_q"], row["acc"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Generated tokens per question")
    axis.set_ylabel("EM" if hotpot else "Accuracy")
    axis.set_title(
        f"{config['dataset'].upper()}: same-draft NLL/PRM controls"
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    axis.margins(x=0.08, y=0.12)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            out_dir / f"nll_prm_controls.{extension}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_run_config(path: Path, run_config: dict):
    normalized = json.loads(
        json.dumps(run_config, ensure_ascii=False, sort_keys=True)
    )
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != normalized:
            raise ValueError(
                f"Existing run config differs from requested config: {path}"
            )
        return
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_phase(
    phase_name: str,
    worker: str,
    all_tasks: List[dict],
    existing: List[dict],
    known_keys: set[str],
    checkpoint_path: Path,
    phase_times: dict,
    phase_times_path: Path,
    gpu_ids: List[str],
    shard_root: Path,
    args,
    config: dict,
):
    pending = [
        task for task in all_tasks if task["task_key"] not in known_keys
    ]
    print(f"{phase_name}: {len(pending)}/{len(all_tasks)} pending")
    if not pending:
        return
    start = time.time()
    generated = launch_shards(
        pending,
        worker,
        gpu_ids,
        shard_root / phase_name,
        args,
        config,
    )
    added = _append_unique(
        checkpoint_path, generated, known_keys
    )
    existing.extend(added)
    if len(added) != len(pending):
        raise RuntimeError(
            f"{phase_name}: expected {len(pending)} new records, "
            f"merged {len(added)}"
        )
    phase_times[phase_name] = (
        phase_times.get(phase_name, 0.0) + time.time() - start
    )
    phase_times_path.write_text(
        json.dumps(phase_times, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Dataset is not configured: {args.dataset}")
    config = dict(config)
    config["dataset"] = args.dataset
    if args.generation_seed is not None:
        config["generation_seed"] = args.generation_seed
    if args.nll_threshold is not None:
        config["thresholds"] = dict(config["thresholds"])
        config["thresholds"][NLL_SIGNAL] = args.nll_threshold
    if args.prm_threshold is not None:
        config["thresholds"] = dict(config["thresholds"])
        config["thresholds"][PRM_SIGNAL] = args.prm_threshold

    if args._worker:
        if args._task_file is None:
            raise ValueError("--_task-file is required for workers")
        {
            "prm": run_prm_worker,
            "nll": run_nll_worker,
            "suffix": run_suffix_worker,
        }[args._worker](args, config)
        return

    dataset_config = config["datasets"][args.dataset]
    n_sample = (
        args.n_sample
        if args.n_sample is not None
        else int(dataset_config["n_sample"])
    )
    configs = build_rollback_configs(config)
    nd_max = max(nd for nd, _ in configs)
    gpu_text = args.gpus or config["gpus"]
    gpu_ids = [gpu.strip() for gpu in gpu_text.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU must be configured")

    source_dir = (
        args.source_dir.resolve()
        if args.source_dir is not None
        else PROJECT_ROOT / config["source_results_root"] / args.dataset
    )
    source_checkpoint = source_dir / "checkpoint.jsonl"
    source_run_config_path = source_dir / "run_config.json"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if not source_run_config_path.is_file():
        raise FileNotFoundError(source_run_config_path)
    source_run_config = json.loads(
        source_run_config_path.read_text(encoding="utf-8")
    )
    if source_run_config["model"] != config["model"]:
        raise ValueError("Source and requested draft models differ")
    if source_run_config["dataset"] != args.dataset:
        raise ValueError("Source and requested datasets differ")
    if int(source_run_config["dataset_seed"]) != int(
        config["dataset_seed"]
    ):
        raise ValueError("Source and requested dataset seeds differ")
    if source_run_config.get("generation_seed") != config["generation_seed"]:
        raise ValueError(
            "Source and requested generation seeds differ"
        )
    if [
        list(pair) for pair in source_run_config["rollback_configs"]
    ] != [list(pair) for pair in configs]:
        raise ValueError("Source and requested rollback configs differ")

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else PROJECT_ROOT / config["results_root"] / args.dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.jsonl"
    phase_times_path = out_dir / "phase_times.json"
    shard_root = out_dir / "_shards"

    questions = load_dataset_by_name(
        args.dataset,
        n_sample,
        seed=int(config["dataset_seed"]),
    )
    if len(questions) != n_sample:
        raise ValueError(
            f"Expected {n_sample} questions, loaded {len(questions)}"
        )
    source_sha256 = _sha256(source_checkpoint)
    draft_map = load_source_drafts(
        source_checkpoint, questions, nd_max
    )
    run_config = {
        "model": config["model"],
        "prm_model": config["prm_model"],
        "dataset": args.dataset,
        "n_questions": len(questions),
        "dataset_seed": int(config["dataset_seed"]),
        "generation_seed": config["generation_seed"],
        "budget": int(config["budget"]),
        "rollback_configs": configs,
        "signals": config["signals"],
        "thresholds": config["thresholds"],
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "max_tokens": int(config["max_tokens"]),
        "max_model_len": int(config["max_model_len"]),
        "gpu_memory_utilization": float(
            dataset_config["gpu_memory_utilization"]
        ),
        "nll_gpu_memory_utilization": float(
            dataset_config.get(
                "nll_gpu_memory_utilization",
                dataset_config["gpu_memory_utilization"],
            )
        ),
        "nll_batch_size": int(config.get("nll_batch_size", 8)),
        "nll_backend": config.get("nll_backend", "vllm"),
        "nll_logprob_chunk_size": int(
            config.get("nll_logprob_chunk_size", 64)
        ),
        "nll_min_free_memory_mib": int(
            dataset_config.get("nll_min_free_memory_mib", 0)
        ),
        "nll_tensor_parallel_size": int(
            dataset_config.get("nll_tensor_parallel_size", 1)
        ),
        "prm_tensor_parallel_size": int(
            dataset_config.get("prm_tensor_parallel_size", 1)
        ),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "source_run_config": str(source_run_config_path.resolve()),
        "source_random_seed": source_run_config.get("random_seed"),
        "source_drafts": len(draft_map),
        "reuse_scoring_checkpoint": (
            str(args.reuse_scoring_checkpoint.resolve())
            if args.reuse_scoring_checkpoint is not None
            else None
        ),
        "semantics": (
            "Original 6_8 nll_drop_fb_last and prm_drop_fb_last; "
            "B=nd*ns voting uses each original draft plus ns-1 "
            "rollback suffixes"
        ),
        "answer_budget_semantics": (
            "nd original draft answers plus nd*(ns-1) suffix answers"
        ),
    }
    _write_run_config(out_dir / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2, ensure_ascii=False))

    prm_tasks, nll_tasks = build_scoring_tasks(
        questions,
        draft_map,
        nd_max,
        config["model"],
        args.dataset,
        config.get("nll_backend", "vllm"),
    )
    per_signal_suffixes = (
        len(questions)
        * sum(additional_suffixes_per_draft(configs).values())
    )
    if args.plan_only:
        print(
            f"PLAN: reused_drafts={len(draft_map)} "
            f"prm={len(prm_tasks)} nll={len(nll_tasks)} "
            f"suffixes={per_signal_suffixes * len(config['signals'])}"
        )
        return

    existing = _read_jsonl(checkpoint_path)
    task_keys = [
        record.get("task_key") for record in existing
        if record.get("task_key")
    ]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("Output checkpoint contains duplicate task keys")
    known_keys = set(task_keys)
    if args.reuse_scoring_checkpoint is not None:
        reuse_path = args.reuse_scoring_checkpoint.resolve()
        reuse_config_path = reuse_path.parent / "run_config.json"
        if not reuse_path.is_file() or not reuse_config_path.is_file():
            raise FileNotFoundError(
                f"Missing reusable scoring run: {reuse_path}"
            )
        reuse_config = json.loads(
            reuse_config_path.read_text(encoding="utf-8")
        )
        if (
            reuse_config.get("source_checkpoint_sha256")
            != source_sha256
        ):
            raise ValueError(
                "Reusable scores use a different source checkpoint"
            )
        for field in (
            "model",
            "prm_model",
            "dataset",
            "generation_seed",
            "temperature",
            "top_p",
            "max_tokens",
        ):
            if reuse_config.get(field) != run_config.get(field):
                raise ValueError(
                    f"Reusable run differs on {field}"
                )
        scoring_task_keys = {
            task["task_key"] for task in prm_tasks + nll_tasks
        }
        reusable_records = [
            record for record in _read_jsonl(reuse_path)
            if record.get("task_type") in ("prm", "nll")
            and record.get("task_key") in scoring_task_keys
        ]
        reusable_keys = {
            record["task_key"] for record in reusable_records
        }
        if reusable_keys != scoring_task_keys:
            raise ValueError(
                "Reusable checkpoint does not contain the exact scoring "
                f"task set: {len(reusable_keys)}/{len(scoring_task_keys)}"
            )
        added = _append_unique(
            checkpoint_path, reusable_records, known_keys
        )
        existing.extend(added)
        print(f"Reused {len(added)} PRM/NLL scoring records")
    phase_times = (
        json.loads(phase_times_path.read_text(encoding="utf-8"))
        if phase_times_path.exists() else {}
    )
    if args.reuse_scoring_checkpoint is not None:
        reuse_times_path = (
            args.reuse_scoring_checkpoint.resolve().parent
            / "phase_times.json"
        )
        reuse_times = json.loads(
            reuse_times_path.read_text(encoding="utf-8")
        )
        for phase in ("prm", "nll"):
            if float(reuse_times.get(phase, 0)) <= 0:
                raise ValueError(
                    f"Reusable run has no positive {phase} phase time"
                )
            phase_times.setdefault(phase, reuse_times[phase])
        phase_times_path.write_text(
            json.dumps(phase_times, indent=2) + "\n",
            encoding="utf-8",
        )

    _run_phase(
        "prm", "prm", prm_tasks, existing, known_keys,
        checkpoint_path, phase_times, phase_times_path,
        gpu_ids, shard_root, args, config,
    )
    _run_phase(
        "nll", "nll", nll_tasks, existing, known_keys,
        checkpoint_path, phase_times, phase_times_path,
        gpu_ids, shard_root, args, config,
    )
    scoring_records = [
        record for record in existing
        if record.get("task_type") in ("prm", "nll")
    ]
    expected_scoring = len(prm_tasks) + len(nll_tasks)
    if len(scoring_records) != expected_scoring:
        raise RuntimeError(
            f"Expected {expected_scoring} score records, "
            f"found {len(scoring_records)}"
        )

    assignments = build_assignments(
        questions, draft_map, scoring_records, config
    )
    suffix_tasks = build_suffix_tasks(
        questions, draft_map, assignments, configs, config
    )
    if args.reuse_scoring_checkpoint is not None:
        suffix_task_map = {
            task["task_key"]: task for task in suffix_tasks
        }
        reusable_suffixes = [
            record
            for record in _read_jsonl(
                args.reuse_scoring_checkpoint.resolve()
            )
            if record.get("task_type") == "suffix"
            and record.get("task_key") in suffix_task_map
            and int(record["rollback_step"])
            == int(
                suffix_task_map[record["task_key"]]["rollback_step"]
            )
        ]
        added = _append_unique(
            checkpoint_path, reusable_suffixes, known_keys
        )
        existing.extend(added)
        print(
            f"Reused {len(added)} threshold-compatible suffix records"
        )
    desired_suffix_keys = {
        task["task_key"] for task in suffix_tasks
    }
    obsolete_suffix_keys = {
        record["task_key"]
        for record in existing
        if record.get("task_type") == "suffix"
        and record["task_key"] not in desired_suffix_keys
    }
    if obsolete_suffix_keys:
        existing[:] = [
            record for record in existing
            if record.get("task_key") not in obsolete_suffix_keys
        ]
        known_keys.difference_update(obsolete_suffix_keys)
        _replace_jsonl(checkpoint_path, existing)
        print(
            "suffix: removed "
            f"{len(obsolete_suffix_keys)} obsolete records"
        )
    _run_phase(
        "suffix", "suffix", suffix_tasks, existing, known_keys,
        checkpoint_path, phase_times, phase_times_path,
        gpu_ids, shard_root, args, config,
    )
    suffix_records = [
        record for record in existing
        if record.get("task_type") == "suffix"
    ]
    if len(suffix_records) != len(suffix_tasks):
        raise RuntimeError(
            f"Expected {len(suffix_tasks)} suffix records, "
            f"found {len(suffix_records)}"
        )

    results = evaluate(
        questions,
        draft_map,
        scoring_records,
        suffix_records,
        assignments,
        configs,
        config,
    )
    save_artifacts(results, assignments, out_dir, config)
    print(
        f"Complete: source_drafts={len(draft_map)} "
        f"scoring={len(scoring_records)} "
        f"suffixes={len(suffix_records)}"
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
