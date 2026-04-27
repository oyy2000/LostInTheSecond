"""
Unified dataset loading for all sweep experiments.

Each loader returns List[dict] with at least:
    doc_id, question, gold_answer

Public API
----------
- load_dataset_by_name(name, n_sample, seed) -> List[dict]
- SUPPORTED_DATASETS: list of valid dataset names
"""

import json
import random
import re
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_DATASETS = [
    "gsm8k", "math500", "aime2024", "amc2023",
    "olympiadbench", "hotpotqa", "humaneval", "csqa",
]


def _load_gsm8k(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("openai/gsm8k", "main", split="test")
    items = []
    for i, row in enumerate(ds):
        ans_text = row["answer"].split("####")[-1].strip()
        items.append({
            "doc_id": f"gsm8k_{i}",
            "question": row["question"],
            "gold_answer": ans_text,
        })
    return _subsample(items, n_sample, seed)


def _load_math500(n_sample: int, seed: int) -> List[dict]:
    p = ROOT / "lm-evaluation-harness/math_eval_data/MATH-500/test.jsonl"
    items = []
    for i, line in enumerate(p.read_text("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        items.append({
            "doc_id": f"math500_{i}",
            "question": row["problem"],
            "gold_answer": row["answer"],
        })
    return _subsample(items, n_sample, seed)


def _load_aime2024(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("AI-MO/aimo-validation-aime", split="train")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"aime2024_{i}",
            "question": row["problem"],
            "gold_answer": str(row["answer"]),
        })
    return _subsample(items, n_sample, seed)


def _load_amc2023(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("AI-MO/aimo-validation-amc", split="train")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"amc2023_{i}",
            "question": row["problem"],
            "gold_answer": str(row["answer"]),
        })
    return _subsample(items, n_sample, seed)


def _load_olympiadbench(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("lmms-lab/OlympiadBench", split="test_en")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"olympiad_{i}",
            "question": row["question"],
            "gold_answer": str(row["final_answer"][0]) if row["final_answer"] else "",
        })
    return _subsample(items, n_sample, seed)


def _load_hotpotqa(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("hotpot_qa", "distractor", split="validation")
    items = []
    for row in ds:
        items.append({
            "doc_id": row["id"],
            "question": row["question"],
            "gold_answer": row["answer"],
        })
    return _subsample(items, n_sample, seed)


def _load_humaneval(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("openai/openai_humaneval", split="test")
    items = []
    for row in ds:
        items.append({
            "doc_id": row["task_id"],
            "question": row["prompt"],
            "gold_answer": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return _subsample(items, n_sample, seed)


def _load_csqa(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("tau/commonsense_qa", split="validation")
    items = []
    for i, row in enumerate(ds):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        items.append({
            "doc_id": f"csqa_{i}",
            "question": f"{row['question'].strip()}\n{choices_str}",
            "gold_answer": row["answerKey"],
        })
    return _subsample(items, n_sample, seed)


_LOADERS = {
    "gsm8k": _load_gsm8k,
    "math500": _load_math500,
    "aime2024": _load_aime2024,
    "amc2023": _load_amc2023,
    "olympiadbench": _load_olympiadbench,
    "hotpotqa": _load_hotpotqa,
    "humaneval": _load_humaneval,
    "csqa": _load_csqa,
}


def load_dataset_by_name(name: str, n_sample: int = 0, seed: int = 42) -> List[dict]:
    """Load a dataset by name. n_sample=0 means use all."""
    name = name.lower().replace("-", "").replace("_", "")
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {SUPPORTED_DATASETS}")
    return _LOADERS[name](n_sample, seed)


def _subsample(items: List[dict], n: int, seed: int) -> List[dict]:
    if n <= 0 or n >= len(items):
        return items
    random.seed(seed)
    return random.sample(items, n)
