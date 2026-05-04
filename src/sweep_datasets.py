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
    "humaneval", "csqa", "gsm8k", "math500", "aime2024", "aime2025", "amc2023",
    "olympiadbench", "hotpotqa", "hotpotqa_open",
    "2wikimultihopqa", "2wikimultihopqa_open",
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
            "gold_answer": str(int(row["answer"])) if float(row["answer"]) == int(float(row["answer"])) else str(row["answer"]),
        })
    return _subsample(items, n_sample, seed)


def _load_olympiadbench(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("math-ai/olympiadbench", split="test")
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


def _format_hotpotqa_context(ctx: dict) -> str:
    """Format HotpotQA context paragraphs into a readable string."""
    parts = []
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        parts.append(f"[{title}]\n" + " ".join(sents))
    return "\n\n".join(parts)


def _load_hotpotqa_open(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("hotpot_qa", "distractor", split="validation")
    items = []
    for row in ds:
        ctx_str = _format_hotpotqa_context(row["context"])
        items.append({
            "doc_id": row["id"],
            "question": row["question"],
            "gold_answer": row["answer"],
            "context": ctx_str,
        })
    return _subsample(items, n_sample, seed)


def _load_aime2025(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("yentinglin/aime_2025", split="train")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"aime2025_{i}",
            "question": row["problem"],
            "gold_answer": str(row["answer"]),
        })
    return _subsample(items, n_sample, seed)


def _load_2wikimultihopqa(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("scholarly-shadows-syndicate/2WikiMultiHopQA",
                 split="validation")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"2wiki_{i}",
            "question": row["question"],
            "gold_answer": row["answer"],
        })
    return _subsample(items, n_sample, seed)


def _format_2wiki_context(ctx_json: str) -> str:
    """Format 2WikiMultiHopQA context (JSON string) into readable text."""
    paragraphs = json.loads(ctx_json)
    parts = []
    for title, sents in paragraphs:
        parts.append(f"[{title}]\n" + " ".join(sents))
    return "\n\n".join(parts)


def _load_2wikimultihopqa_open(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("scholarly-shadows-syndicate/2WikiMultiHopQA",
                 split="validation")
    items = []
    for i, row in enumerate(ds):
        ctx_str = _format_2wiki_context(row["context"])
        items.append({
            "doc_id": f"2wiki_{i}",
            "question": row["question"],
            "gold_answer": row["answer"],
            "context": ctx_str,
        })
    return _subsample(items, n_sample, seed)


def _load_musique(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("bdsaglam/musique", split="validation")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"musique_{i}",
            "question": row["question"],
            "gold_answer": row["answer"],
        })
    return _subsample(items, n_sample, seed)


def _load_gpqa_diamond(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    # gpqa_diamond config may not be cached; fall back to gpqa_main
    try:
        ds = hf_load("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except (ValueError, Exception):
        ds = hf_load("Idavidrein/gpqa", "gpqa_main", split="train")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"gpqa_diamond_{i}",
            "question": row["Question"],
            "gold_answer": row["Correct Answer"],
        })
    return _subsample(items, n_sample, seed)


def _load_strategyqa(n_sample: int, seed: int) -> List[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("ChilleD/StrategyQA", split="test")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "doc_id": f"strategyqa_{i}",
            "question": row["question"],
            "gold_answer": "yes" if row["answer"] else "no",
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
    "aime2025": _load_aime2025,
    "amc2023": _load_amc2023,
    "olympiadbench": _load_olympiadbench,
    "hotpotqa": _load_hotpotqa,
    "hotpotqa_open": _load_hotpotqa_open,
    "2wikimultihopqa": _load_2wikimultihopqa,
    "2wikimultihopqa_open": _load_2wikimultihopqa_open,
    "musique": _load_musique,
    "strategyqa": _load_strategyqa,
    "gpqa_diamond": _load_gpqa_diamond,
    "humaneval": _load_humaneval,
    "csqa": _load_csqa,
}


def load_dataset_by_name(name: str, n_sample: int = 0, seed: int = 42) -> List[dict]:
    """Load a dataset by name. n_sample=0 means use all."""
    # Build a normalized lookup so names with hyphens/underscores still match
    norm = lambda s: s.lower().replace("-", "").replace("_", "")
    norm_loaders = {norm(k): v for k, v in _LOADERS.items()}
    key = norm(name)
    if key not in norm_loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {SUPPORTED_DATASETS}")
    return norm_loaders[key](n_sample, seed)


def _subsample(items: List[dict], n: int, seed: int) -> List[dict]:
    if n <= 0 or n >= len(items):
        return items
    random.seed(seed)
    return random.sample(items, n)
