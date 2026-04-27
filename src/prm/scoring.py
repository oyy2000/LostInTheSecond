#!/usr/bin/env python3

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

STEP_RE = re.compile(r"(?:^|\n)\s*Step\s*\d+\s*:\s*", re.IGNORECASE)
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass
class PRMSampleLite:
    exact_match: float
    n_tokens: int
    step2_score: float
    step_scores: List[float]


def split_steps(text: str, mode: str = "double_newline") -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if mode == "double_newline":
        steps = [x.strip() for x in text.split("\n\n") if x.strip()]
        if steps:
            return steps

    if mode == "single_newline":
        steps = [x.strip() for x in text.split("\n") if x.strip()]
        if steps:
            return steps

    if mode == "auto":
        if "\n\n" in text:
            steps = [x.strip() for x in text.split("\n\n") if x.strip()]
            if steps:
                return steps
        if "\n" in text:
            steps = [x.strip() for x in text.split("\n") if x.strip()]
            if steps:
                return steps

    hits = list(STEP_RE.finditer(text))
    if hits:
        spans = []
        for i, m in enumerate(hits):
            st = m.start()
            ed = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            spans.append(text[st:ed].strip())
        return [s for s in spans if s]

    chunks = [x.strip() for x in re.split(r"\n\s*\n+", text) if x.strip()]
    if len(chunks) >= 2:
        return chunks

    sents = [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+", text) if x.strip()]
    if len(sents) >= 2:
        return sents

    return [text]


class StepScorer:
    """Qwen PRM-style scorer using <extra_0> step markers."""

    def __init__(self, model_id: str, dtype: str):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype_map[dtype],
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.step_token_id = self.tok.encode("<extra_0>")[0]

    @torch.inference_mode()
    def score_steps(self, query: str, steps: List[str]) -> List[float]:
        if not steps:
            return []

        assistant_text = "<extra_0>".join(steps) + "<extra_0>"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
            {"role": "assistant", "content": assistant_text},
        ]
        if getattr(self.tok, "chat_template", None):
            conv = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        else:
            conv = (
                f"System: {SYSTEM_PROMPT}\n"
                f"User: {query.strip()}\n"
                f"Assistant: {assistant_text}"
            )

        input_ids = self.tok.encode(conv, return_tensors="pt").to(self.model.device)
        token_masks = input_ids == self.step_token_id
        out = self.model(input_ids=input_ids, use_cache=False)
        logits = out[0]

        probs = F.softmax(logits, dim=-1)
        probs = probs * token_masks.unsqueeze(-1)
        sample = probs[0]
        nz = sample[sample != 0]
        if nz.numel() == 0:
            return []

        step_scores = nz.view(-1, 2)[:, 1].detach().cpu().tolist()
        return step_scores[: len(steps)]

    @torch.inference_mode()
    def score_step(self, prefix: str, step_text: str) -> float:
        scores = self.score_steps(prefix, [step_text])
        return float(scores[0]) if scores else -1e9


def print_avg_score_per_step(samples: List[PRMSampleLite], title: str):
    if not samples:
        print(f"[{title}] empty")
        return

    max_steps = max((len(s.step_scores) for s in samples), default=0)
    print(f"\n[{title}] avg PRM score per step")
    for i in range(max_steps):
        vals = [s.step_scores[i] for s in samples if i < len(s.step_scores)]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        print(f"  Step {i + 1}: {avg:.6f} (n={len(vals)})")


def plot_avg_score_per_step(samples: List[PRMSampleLite], out_dir: Path, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not samples:
        return

    max_steps = max((len(s.step_scores) for s in samples), default=0)
    if max_steps <= 0:
        return

    xs, ys, ns = [], [], []
    for i in range(max_steps):
        vals = [s.step_scores[i] for s in samples if i < len(s.step_scores)]
        if not vals:
            continue
        xs.append(i + 1)
        ys.append(sum(vals) / len(vals))
        ns.append(len(vals))

    if not xs:
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o")
    for x, y, n in zip(xs, ys, ns):
        plt.text(x, y, f"n={n}", fontsize=8, ha="center", va="bottom")
    plt.xlabel("Step index")
    plt.ylabel("Average PRM score")
    plt.title(f"{title}: Avg PRM score per step")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", title)
    plt.savefig(out_dir / f"avg_prm_per_step_{safe_title}.png", dpi=180)
    plt.close()


def _avg_curve(samples: List[PRMSampleLite]):
    if not samples:
        return [], []
    max_steps = max((len(s.step_scores) for s in samples), default=0)
    xs, ys = [], []
    for i in range(max_steps):
        vals = [s.step_scores[i] for s in samples if i < len(s.step_scores)]
        if not vals:
            continue
        xs.append(i + 1)
        ys.append(sum(vals) / len(vals))
    return xs, ys


def plot_per_step_avg_correctness(
    samples: List[PRMSampleLite],
    out_dir: Path,
    title: str,
    threshold: float = 0.72,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not samples:
        return

    all_s = samples
    correct_s = [s for s in samples if s.exact_match >= 1.0]
    wrong_s = [s for s in samples if s.exact_match < 1.0]

    xa, ya = _avg_curve(all_s)
    xc, yc = _avg_curve(correct_s)
    xw, yw = _avg_curve(wrong_s)
    max_x = max([0] + xa + xc + xw)
    if max_x <= 0:
        return

    plt.figure(figsize=(10, 6))
    if xa:
        plt.plot(xa, ya, marker="o", linewidth=2, label=f"All (n@1={len(all_s)})")
    if xc:
        plt.plot(xc, yc, marker="s", linewidth=2, label=f"Correct (n@1={len(correct_s)})")
    if xw:
        plt.plot(xw, yw, marker="x", linewidth=2, label=f"Wrong (n@1={len(wrong_s)})")

    plt.axhline(y=threshold, linestyle="--", linewidth=1.5, label=f"thr={threshold:.2f}")
    plt.xlabel("Step k")
    plt.ylabel("Avg PRM Step Score (across samples)")
    plt.title(title)
    plt.xlim(1, max_x)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "per_step_avg_correctness.png", dpi=180)
    plt.close()


def plot_baseline(samples: List[PRMSampleLite], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    correct = [s.step2_score for s in samples if s.exact_match >= 1.0]
    wrong = [s.step2_score for s in samples if s.exact_match < 1.0]

    plt.figure(figsize=(7, 4))
    plt.hist(correct, bins=30, alpha=0.6, label="correct")
    plt.hist(wrong, bins=30, alpha=0.6, label="wrong")
    plt.xlabel("Step2 score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "prm_hist_correct_vs_wrong.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    data = [correct if correct else [0.0], wrong if wrong else [0.0]]
    plt.boxplot(data, tick_labels=["correct", "wrong"])
    plt.ylabel("Step2 score")
    plt.tight_layout()
    plt.savefig(out_dir / "prm_step2_boxplot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    xs = [s.n_tokens for s in samples]
    ys = [s.step2_score for s in samples]
    cs = ["tab:blue" if s.exact_match >= 1.0 else "tab:red" for s in samples]
    plt.scatter(xs, ys, c=cs, alpha=0.55, s=12)
    plt.xlabel("generation length (tokens)")
    plt.ylabel("Step2 score")
    plt.tight_layout()
    plt.savefig(out_dir / "prm_step2_vs_length.png", dpi=180)
    plt.close()
