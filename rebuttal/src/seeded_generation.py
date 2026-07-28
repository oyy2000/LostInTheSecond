"""Corrected vLLM worker with one deterministic seed per request."""

import json
import os
import time
from pathlib import Path
from typing import Any

from rebuttal.src.stable_sampling import derive_sampling_seed
from rebuttal.src.wallclock import WorkerTiming


def run_seeded_shard(pipeline: Any, args: Any) -> None:
    """Generate one shard using distinct SamplingParams for every task."""
    if args._seed < 0:
        raise ValueError(
            "The corrected rebuttal worker requires --seed/--_seed."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text(encoding="utf-8"))
    shard_id = args._shard_id
    timing = None
    timing_path = getattr(args, "_timing_file", "")
    if timing_path:
        timing = WorkerTiming(
            path=Path(timing_path),
            phase=getattr(args, "_timing_phase", "") or "generation",
            run_id=getattr(args, "_timing_run_id", "") or "untracked",
            shard_id=shard_id,
            gpu_id=args._gpu,
            tasks=tasks,
        )
    print(
        f"[Seeded shard {shard_id}] GPU {args._gpu}: "
        f"{len(tasks)} tasks; run_seed={args._seed}"
    )
    if not tasks:
        return

    task_seeds = [
        derive_sampling_seed(args._seed, task) for task in tasks
    ]
    if len(task_seeds) != len(set(task_seeds)):
        raise RuntimeError(
            "Per-request sampling seed collision detected in shard."
        )

    stop = pipeline.get_stop_tokens(pipeline.MODEL_ID)
    model_load_started_ns = time.perf_counter_ns()
    llm = LLM(
        model=pipeline.MODEL_ID,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=pipeline.GPU_MEM,
        max_model_len=pipeline.MAX_MODEL_LEN,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    if timing is not None:
        timing.record_model_load(model_load_started_ns)

    by_type = {}
    for index, task in enumerate(tasks):
        by_type.setdefault(task["task_type"], []).append(index)
    results = [None] * len(tasks)

    for task_type in ("draft", "suffix", "fullsc"):
        indices = by_type.get(task_type, [])
        if not indices:
            continue
        for batch_start in range(0, len(indices), 512):
            chunk = indices[batch_start:batch_start + 512]
            batch_started_ns = time.perf_counter_ns()
            batch_started_unix_sec = time.time()
            prompt_ids = [
                tokenizer.encode(
                    tasks[index]["prompt"],
                    add_special_tokens=False,
                )
                for index in chunk
            ]
            sampling_params = [
                SamplingParams(
                    temperature=pipeline.TEMPERATURE,
                    top_p=pipeline.TOP_P,
                    max_tokens=pipeline.MAX_TOKENS,
                    stop=stop,
                    seed=task_seeds[index],
                )
                for index in chunk
            ]
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in prompt_ids],
                sampling_params=sampling_params,
            )
            batch_finished_unix_sec = time.time()
            batch_generated_tokens = 0
            for index, output in zip(chunk, outputs):
                text = output.outputs[0].text
                token_count = len(output.outputs[0].token_ids)
                batch_generated_tokens += token_count
                task = tasks[index]
                record = {
                    key: value
                    for key, value in task.items()
                    if key != "prompt"
                }
                record["sampling_seed"] = task_seeds[index]
                if task_type == "draft":
                    steps = pipeline.split_steps(text)
                    record.update(
                        draft_text=text,
                        draft_steps=steps,
                        draft_answer=pipeline.extract_answer(
                            pipeline.DATASET, text
                        ),
                        draft_tokens=token_count,
                        n_steps=len(steps),
                    )
                elif task_type == "suffix":
                    record.update(
                        suffix_text=text,
                        suffix_answer=pipeline.extract_answer(
                            pipeline.DATASET, text
                        ),
                        suffix_tokens=token_count,
                    )
                else:
                    record.update(
                        sc_text=text,
                        sc_answer=pipeline.extract_answer(
                            pipeline.DATASET, text
                        ),
                        sc_tokens=token_count,
                    )
                record["task_type"] = task_type
                results[index] = record
                if timing is not None:
                    timing.record_vllm_request(
                        task,
                        output,
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
                    input_tokens=sum(len(ids) for ids in prompt_ids),
                )
            batch_number = batch_start // 512 + 1
            batch_total = (len(indices) + 511) // 512
            print(
                f"[Seeded shard {shard_id}] {task_type} batch "
                f"{batch_number}/{batch_total}"
            )

    output_path = Path(tasks[0]["_out"])
    with output_path.open("w", encoding="utf-8") as handle:
        for record in results:
            if record is not None:
                handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
    if timing is not None:
        timing.finish()
    print(
        f"[Seeded shard {shard_id}] Done: "
        f"{sum(record is not None for record in results)}"
    )
