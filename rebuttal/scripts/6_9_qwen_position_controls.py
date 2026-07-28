#!/usr/bin/env python3
"""Qwen 3B rebuttal controls for rollback position and final-step resampling.

The evaluated strategies are:

* random_uniform: uniformly sample the target reasoning step.
* fixed_25/fixed_50/fixed_75: rollback at a fixed step fraction.
* final_step_only: retain all but the final step and regenerate that step.

Each strategy uses the original nd draft answers plus nd*(ns-1) newly
generated suffix answers, keeping the total answer budget B=nd*ns. The
same sampled drafts and suffixes are reused across configurations.
"""

import argparse
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

from rebuttal.src.position_controls import (  # noqa: E402
    ALL_STRATEGIES,
    FINAL_STEP_STRATEGY,
    FIXED_FRACTIONS,
    POSITION_STRATEGIES,
    additional_suffixes_per_draft,
    final_step_prefix,
    final_step_rollback_step,
    rollback_steps_for_draft,
)
from rebuttal.src.stable_sampling import derive_sampling_seed  # noqa: E402
from src.hotpotqa_answer_equiv import hotpotqa_f1  # noqa: E402
from src.prompt_templates import (  # noqa: E402
    build_prompt,
    check_answer,
    extract_answer,
    get_stop_tokens,
    split_steps,
)
from src.sweep_datasets import load_dataset_by_name  # noqa: E402


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_CONFIGS = [(4, 8), (8, 4), (16, 2)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["math500", "hotpotqa", "hotpotqa_open"],
        required=True,
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--nd", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES)
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--reuse-checkpoint",
        default="",
        help=(
            "Read reusable draft/suffix records from another checkpoint. "
            "Only records required by the current answer allocation are "
            "used; the source file is not modified."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")

    parser.add_argument("--_worker", action="store_true")
    parser.add_argument("--_gpu", default="0")
    parser.add_argument("--_shard-id", type=int, default=-1)
    parser.add_argument("--_task-file", default="")
    parser.add_argument("--_seed", type=int, default=-1)
    return parser.parse_args()


def _task_key(record: dict) -> str:
    return record["task_key"]


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _append_unique(path: Path, records: Iterable[dict],
                   known_keys: set[str]) -> List[dict]:
    added = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            key = _task_key(record)
            if key in known_keys:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            known_keys.add(key)
            added.append(record)
    return added


def run_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text(encoding="utf-8"))
    if not tasks:
        return
    output_path = Path(tasks[0]["_out"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        record["task_key"]
        for record in _read_jsonl(output_path)
        if record.get("task_key")
    }

    print(
        f"[Shard {args._shard_id}] GPU {args._gpu}: "
        f"{len(tasks)} tasks, {len(completed)} checkpointed"
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    stop = get_stop_tokens(args.model)

    by_type: Dict[str, List[dict]] = {}
    for task in tasks:
        if task["task_key"] not in completed:
            by_type.setdefault(task["task_type"], []).append(task)

    with output_path.open("a", encoding="utf-8") as output:
        for task_type in ("draft", "suffix"):
            pending = by_type.get(task_type, [])
            for start in range(0, len(pending), 512):
                batch = pending[start:start + 512]
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
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=MAX_TOKENS,
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
                    n_tokens = len(generation.outputs[0].token_ids)
                    record = {
                        key: value
                        for key, value in task.items()
                        if key not in ("prompt", "retained_text", "_out")
                    }
                    if args._seed >= 0:
                        record["sampling_seed"] = derive_sampling_seed(
                            args._seed, task
                        )
                    if task_type == "draft":
                        steps = split_steps(text)
                        record.update(
                            task_type="draft",
                            draft_text=text,
                            draft_steps=steps,
                            draft_answer=extract_answer(
                                args.dataset, text
                            ),
                            draft_tokens=n_tokens,
                            n_steps=len(steps),
                        )
                    else:
                        retained = task.get("retained_text", "")
                        record.update(
                            task_type="suffix",
                            generation_kind=task_type,
                            suffix_text=text,
                            suffix_answer=extract_answer(
                                args.dataset, text
                            ),
                            suffix_tokens=n_tokens,
                            prefix_tokens=len(
                                tokenizer.encode(
                                    retained,
                                    add_special_tokens=False,
                                )
                            ),
                        )
                    output.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                output.flush()
                done = min(start + len(batch), len(pending))
                print(
                    f"[Shard {args._shard_id}] {task_type} "
                    f"{done}/{len(pending)}"
                )
    print(f"[Shard {args._shard_id}] Done")


def launch_shards(
    tasks: List[dict],
    gpu_ids: List[str],
    shard_dir: Path,
    args,
) -> List[dict]:
    if not tasks:
        return []
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in gpu_ids]
    for index, task in enumerate(tasks):
        shards[index % len(gpu_ids)].append(task)

    processes: List[Tuple[int, subprocess.Popen, Any]] = []
    task_keys = {task["task_key"] for task in tasks}
    output_paths = []
    for shard_id, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        if not shard:
            continue
        output_path = shard_dir / f"shard_{shard_id}.jsonl"
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
            "--_worker",
            "--_gpu", gpu_id,
            "--_shard-id", str(shard_id),
            "--_task-file", str(task_file),
            "--dataset", args.dataset,
            "--model", args.model,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        if args.generation_seed is not None:
            command += ["--_seed", str(args.generation_seed)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["TOKENIZERS_PARALLELISM"] = "false"
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_id, process, log_handle))
        output_paths.append(output_path)
        print(
            f"  shard={shard_id} gpu={gpu_id} tasks={len(shard)}"
        )

    for shard_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        log_path = shard_dir / f"log_{shard_id}.txt"
        tail = "\n".join(
            log_path.read_text(errors="replace").splitlines()[-5:]
        )
        print(f"  shard={shard_id} exit={return_code}\n{tail}")
        if return_code != 0:
            raise RuntimeError(
                f"Generation shard {shard_id} failed; see {log_path}"
            )

    records = []
    for output_path in output_paths:
        for record in _read_jsonl(output_path):
            if record.get("task_key") in task_keys:
                records.append(record)
    return records


def _vote(answers: List[str]) -> str:
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def build_rollback_configs(args) -> List[tuple[int, int]]:
    configs = []
    for nd in args.nd:
        if args.budget % nd:
            raise ValueError(
                f"Budget {args.budget} is not divisible by nd={nd}"
            )
        configs.append((nd, args.budget // nd))
    return configs


def build_suffix_tasks(
    questions: List[dict],
    draft_map: Dict[Tuple[str, int], dict],
    configs: List[tuple[int, int]],
    args,
) -> tuple[List[dict], List[dict]]:
    per_draft = additional_suffixes_per_draft(configs)
    tasks = []
    assignments = []
    for question in questions:
        doc_id = question["doc_id"]
        base_prompt = build_prompt(
            args.model,
            args.dataset,
            question["question"],
            context=question.get("context", ""),
        )
        for draft_idx in range(max(args.nd)):
            draft = draft_map[(doc_id, draft_idx)]
            steps = draft.get("draft_steps") or split_steps(
                draft.get("draft_text", "")
            )
            if not steps and draft.get("draft_text"):
                steps = [draft["draft_text"].strip()]
            n_steps = len(steps)
            positions = rollback_steps_for_draft(
                doc_id,
                draft_idx,
                n_steps,
                args.random_seed,
            )

            for strategy in POSITION_STRATEGIES:
                if strategy not in args.strategies:
                    continue
                rollback_step = positions[strategy]
                retained = (
                    "\n\n".join(steps[:rollback_step]) + "\n\n"
                    if rollback_step > 0 else ""
                )
                assignments.append({
                    "doc_id": doc_id,
                    "draft_idx": draft_idx,
                    "strategy": strategy,
                    "n_steps": n_steps,
                    "rollback_step": rollback_step,
                    "target_step": rollback_step + 1,
                    "target_fraction": (
                        (rollback_step + 1) / n_steps
                        if n_steps else 0.0
                    ),
                })
                for suffix_idx in range(per_draft[draft_idx]):
                    task_key = (
                        f"suffix|{strategy}|{doc_id}|{draft_idx}|"
                        f"{suffix_idx}"
                    )
                    tasks.append({
                        "task_key": task_key,
                        "task_type": "suffix",
                        "doc_id": doc_id,
                        "draft_idx": draft_idx,
                        "strategy": strategy,
                        "random_seed": (
                            args.random_seed
                            if strategy == "random_uniform" else None
                        ),
                        "rollback_step": rollback_step,
                        "suffix_idx": suffix_idx,
                        "gold_answer": question["gold_answer"],
                        "retained_text": retained,
                        "prompt": base_prompt + retained,
                    })

            if FINAL_STEP_STRATEGY in args.strategies:
                rollback_step = final_step_rollback_step(n_steps)
                retained = final_step_prefix(steps)
                assignments.append({
                    "doc_id": doc_id,
                    "draft_idx": draft_idx,
                    "strategy": FINAL_STEP_STRATEGY,
                    "n_steps": n_steps,
                    "rollback_step": rollback_step,
                    "target_step": n_steps,
                    "target_fraction": 1.0,
                })
                for suffix_idx in range(per_draft[draft_idx]):
                    task_key = (
                        f"suffix|{FINAL_STEP_STRATEGY}|{doc_id}|"
                        f"{draft_idx}|{suffix_idx}"
                    )
                    tasks.append({
                        "task_key": task_key,
                        "task_type": "suffix",
                        "doc_id": doc_id,
                        "draft_idx": draft_idx,
                        "strategy": FINAL_STEP_STRATEGY,
                        "random_seed": None,
                        "rollback_step": rollback_step,
                        "suffix_idx": suffix_idx,
                        "gold_answer": question["gold_answer"],
                        "retained_text": retained,
                        "prompt": base_prompt + retained,
                    })
    return tasks, assignments


def evaluate(
    questions: List[dict],
    draft_map: Dict[Tuple[str, int], dict],
    suffix_records: List[dict],
    assignments: List[dict],
    configs: List[tuple[int, int]],
    args,
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
            record["strategy"],
        ): record
        for record in assignments
    }
    n_questions = len(questions)
    results = []

    greedy_correct = 0
    greedy_f1 = 0.0
    greedy_tokens = 0
    for question in questions:
        draft = draft_map[(question["doc_id"], 0)]
        answer = draft["draft_answer"]
        greedy_correct += int(
            check_answer(
                args.dataset, answer, question["gold_answer"]
            )
        )
        if args.dataset in ("hotpotqa", "hotpotqa_open"):
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
        "total_tokens": greedy_tokens,
    }
    if args.dataset in ("hotpotqa", "hotpotqa_open"):
        greedy["f1"] = greedy_f1 / n_questions
    results.append(greedy)

    for strategy in args.strategies:
        for nd, ns in configs:
            correct = 0
            f1_total = 0.0
            total_tokens = 0
            total_answers = 0
            total_draft_answers = 0
            total_suffix_answers = 0
            missing = []
            rollback_steps = []
            for question in questions:
                doc_id = question["doc_id"]
                answers = []
                question_tokens = 0
                for draft_idx in range(nd):
                    draft = draft_map[(doc_id, draft_idx)]
                    answers.append(draft["draft_answer"])
                    question_tokens += draft["draft_tokens"]
                    total_draft_answers += 1
                    assignment = assignment_map[
                        (doc_id, draft_idx, strategy)
                    ]
                    rollback_steps.append(
                        assignment["rollback_step"]
                    )
                    for suffix_idx in range(ns - 1):
                        record = suffix_map.get(
                            (
                                doc_id,
                                draft_idx,
                                strategy,
                                suffix_idx,
                            )
                        )
                        if record is None:
                            missing.append(
                                (
                                    doc_id,
                                    draft_idx,
                                    strategy,
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
                        args.dataset,
                        voted,
                        question["gold_answer"],
                    )
                )
                if args.dataset in ("hotpotqa", "hotpotqa_open"):
                    f1_total += hotpotqa_f1(
                        voted, question["gold_answer"]
                    )
                total_tokens += question_tokens
                total_answers += len(answers)
            if missing:
                preview = ", ".join(map(str, missing[:3]))
                raise RuntimeError(
                    f"{len(missing)} suffix records missing; "
                    f"first: {preview}"
                )
            result = {
                "method": f"{strategy}_nd{nd}_ns{ns}",
                "strategy": strategy,
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
                "tokens_per_q": total_tokens / n_questions,
                "total_tokens": total_tokens,
                "avg_rollback_step": (
                    sum(rollback_steps) / len(rollback_steps)
                    if rollback_steps else None
                ),
            }
            if args.dataset in ("hotpotqa", "hotpotqa_open"):
                result["f1"] = f1_total / n_questions
            results.append(result)
    return results


def save_artifacts(
    results: List[dict],
    assignments: List[dict],
    out_dir: Path,
    args,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "rollback_assignments.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for assignment in assignments:
            handle.write(
                json.dumps(assignment, ensure_ascii=False) + "\n"
            )

    is_hotpot = args.dataset in ("hotpotqa", "hotpotqa_open")
    metric_name = "EM" if is_hotpot else "Accuracy"
    lines = [
        "| method | nd | ns | draft answers | suffix answers "
        "| total answers | accuracy | tokens/question"
        + (" | F1 |" if is_hotpot else " |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|"
        + ("---:|" if is_hotpot else ""),
    ]
    for result in results:
        line = (
            f"| {result['method']} | {result.get('nd', '')} "
            f"| {result.get('ns', '')} "
            f"| {result.get('n_draft_answers', '')} "
            f"| {result.get('n_suffix_answers', '')} "
            f"| {result.get('n_answers', '')} "
            f"| {result['acc']:.4f} "
            f"| {result['tokens_per_q']:.1f} "
        )
        if is_hotpot:
            line += f"| {result['f1']:.4f} |"
        else:
            line += "|"
        lines.append(line)
    (out_dir / "eval_summary_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    colors = {
        "random_uniform": "#525252",
        "fixed_25": "#2563eb",
        "fixed_50": "#059669",
        "fixed_75": "#dc2626",
        "final_step_only": "#7c3aed",
    }
    markers = {
        "random_uniform": "X",
        "fixed_25": "o",
        "fixed_50": "s",
        "fixed_75": "D",
        "final_step_only": "^",
    }
    annotation_offsets = {
        ("random_uniform", 8): (5, -14),
        ("fixed_75", 16): (5, 9),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for strategy in args.strategies:
        rows = sorted(
            [
                row for row in results
                if row.get("strategy") == strategy
            ],
            key=lambda row: row["tokens_per_q"],
        )
        axis.plot(
            [row["tokens_per_q"] for row in rows],
            [row["acc"] for row in rows],
            color=colors[strategy],
            marker=markers[strategy],
            linewidth=1.6,
            markersize=6,
            label=strategy.replace("_", " "),
        )
        for row in rows:
            offset = annotation_offsets.get(
                (strategy, row["nd"]), (5, 5)
            )
            axis.annotate(
                f"nd={row['nd']}",
                (row["tokens_per_q"], row["acc"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
    axis.set_xlabel("Generated tokens per question")
    axis.set_ylabel(metric_name)
    axis.set_title(
        f"{args.dataset.upper()}: Qwen2.5-3B position controls"
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    axis.margins(x=0.05, y=0.12)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            out_dir / f"position_controls.{extension}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(f"Saved results to {out_dir}")


def main():
    args = parse_args()
    if args._worker:
        run_worker(args)
        return

    configs = build_rollback_configs(args)
    nd_max = max(nd for nd, _ in configs)
    gpu_ids = [
        gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()
    ]
    model_short = Path(args.model).name.lower().replace("-", "_")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_ROOT / "rebuttal" / "results"
        / f"{model_short}_position_controls"
        / f"seed_{args.random_seed}"
        / args.dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.jsonl"
    shard_root = out_dir / "_shards"
    phase_times_path = out_dir / "phase_times.json"

    questions = load_dataset_by_name(
        args.dataset,
        args.n_sample,
        seed=args.dataset_seed,
    )
    if not questions:
        raise RuntimeError(f"No questions loaded for {args.dataset}")

    config = {
        "model": args.model,
        "dataset": args.dataset,
        "n_questions": len(questions),
        "dataset_seed": args.dataset_seed,
        "random_seed": args.random_seed,
        "generation_seed": args.generation_seed,
        "budget": args.budget,
        "rollback_configs": configs,
        "strategies": args.strategies,
        "fixed_position_semantics": (
            "target step = ceil(fraction * n_steps); retain earlier steps"
        ),
        "final_step_semantics": (
            "target the final split reasoning step; retain earlier steps"
        ),
        "answer_budget_semantics": (
            "nd original draft answers plus nd*(ns-1) suffix answers"
        ),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "reuse_checkpoint": (
            str(Path(args.reuse_checkpoint).resolve())
            if args.reuse_checkpoint else None
        ),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2))

    if args.plan_only:
        per_question = len(args.strategies) * sum(
            additional_suffixes_per_draft(configs).values()
        )
        print(
            f"PLAN: drafts={len(questions) * nd_max}, "
            f"suffixes={len(questions) * per_question}"
        )
        return

    phase_times = (
        json.loads(phase_times_path.read_text())
        if phase_times_path.exists() else {}
    )
    existing_by_key = {
        record["task_key"]: record
        for record in _read_jsonl(checkpoint_path)
        if record.get("task_key")
    }
    reuse_checkpoint = (
        Path(args.reuse_checkpoint).resolve()
        if args.reuse_checkpoint else None
    )
    source_record_count = 0
    if reuse_checkpoint:
        if not reuse_checkpoint.is_file():
            raise FileNotFoundError(reuse_checkpoint)
        source_records = _read_jsonl(reuse_checkpoint)
        source_record_count = len(source_records)
        for record in source_records:
            key = record.get("task_key")
            if key and key not in existing_by_key:
                existing_by_key[key] = record
        source_phase_times = reuse_checkpoint.parent / "phase_times.json"
        if (
            not phase_times
            and source_phase_times.is_file()
        ):
            source_times = json.loads(
                source_phase_times.read_text(encoding="utf-8")
            )
            phase_times = {
                **source_times,
                "reused_source": source_times,
            }
            phase_times_path.write_text(
                json.dumps(phase_times, indent=2),
                encoding="utf-8",
            )
    existing = list(existing_by_key.values())
    known_keys = {
        record["task_key"]
        for record in existing
        if record.get("task_key")
    }

    required_draft_keys = set()
    draft_tasks = []
    for question in questions:
        prompt = build_prompt(
            args.model,
            args.dataset,
            question["question"],
            context=question.get("context", ""),
        )
        for draft_idx in range(nd_max):
            task_key = f"draft|{question['doc_id']}|{draft_idx}"
            required_draft_keys.add(task_key)
            if task_key in known_keys:
                continue
            draft_tasks.append({
                "task_key": task_key,
                "task_type": "draft",
                "doc_id": question["doc_id"],
                "draft_idx": draft_idx,
                "gold_answer": question["gold_answer"],
                "prompt": prompt,
            })
    draft_start = time.time()
    if draft_tasks:
        print(f"Generating {len(draft_tasks)} drafts")
        generated = launch_shards(
            draft_tasks, gpu_ids, shard_root / "drafts", args
        )
        added = _append_unique(
            checkpoint_path, generated, known_keys
        )
        existing.extend(added)
        phase_times["draft"] = (
            phase_times.get("draft", 0.0) + time.time() - draft_start
        )
        phase_times_path.write_text(
            json.dumps(phase_times, indent=2),
            encoding="utf-8",
        )

    draft_map = {
        (record["doc_id"], record["draft_idx"]): record
        for record in existing
        if (
            record.get("task_type") == "draft"
            and record.get("task_key") in required_draft_keys
        )
    }
    expected_drafts = len(questions) * nd_max
    if len(draft_map) != expected_drafts:
        raise RuntimeError(
            f"Expected {expected_drafts} drafts, found {len(draft_map)}"
        )

    all_suffix_tasks, assignments = build_suffix_tasks(
        questions, draft_map, configs, args
    )
    suffix_tasks = [
        task for task in all_suffix_tasks
        if task["task_key"] not in known_keys
    ]
    counts = Counter(task["strategy"] for task in suffix_tasks)
    print(
        f"Suffix tasks pending: {len(suffix_tasks)} "
        f"by strategy={dict(counts)}"
    )
    suffix_start = time.time()
    if suffix_tasks:
        generated = launch_shards(
            suffix_tasks, gpu_ids, shard_root / "suffixes", args
        )
        added = _append_unique(
            checkpoint_path, generated, known_keys
        )
        existing.extend(added)
        phase_times["suffix"] = (
            phase_times.get("suffix", 0.0) + time.time() - suffix_start
        )
        phase_times_path.write_text(
            json.dumps(phase_times, indent=2),
            encoding="utf-8",
        )

    required_suffix_keys = {
        task["task_key"] for task in all_suffix_tasks
    }
    suffix_records = [
        record for record in existing
        if (
            record.get("task_type") == "suffix"
            and record.get("task_key") in required_suffix_keys
        )
    ]
    expected_suffixes = len(all_suffix_tasks)
    if len(suffix_records) != expected_suffixes:
        raise RuntimeError(
            f"Expected {expected_suffixes} suffix records, "
            f"found {len(suffix_records)}"
        )

    selected_records = [
        draft_map[key] for key in sorted(draft_map)
    ] + sorted(
        suffix_records,
        key=lambda record: (
            record["doc_id"],
            record["draft_idx"],
            record["strategy"],
            record["suffix_idx"],
        ),
    )
    checkpoint_tmp = checkpoint_path.with_suffix(".jsonl.tmp")
    with checkpoint_tmp.open("w", encoding="utf-8") as handle:
        for record in selected_records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
    checkpoint_tmp.replace(checkpoint_path)

    results = evaluate(
        questions,
        draft_map,
        suffix_records,
        assignments,
        configs,
        args,
    )
    save_artifacts(results, assignments, out_dir, args)
    provenance = {
        "answer_budget_semantics": config["answer_budget_semantics"],
        "reuse_checkpoint": config["reuse_checkpoint"],
        "source_checkpoint_records": source_record_count,
        "selected_draft_records": len(draft_map),
        "selected_suffix_records": len(suffix_records),
        "unused_source_suffix_records": max(
            0,
            sum(
                record.get("task_type") == "suffix"
                for record in existing
            ) - len(suffix_records),
        ),
    }
    (out_dir / "result_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Complete: drafts={len(draft_map)} "
        f"suffixes={len(suffix_records)}"
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
