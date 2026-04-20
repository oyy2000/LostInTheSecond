#!/usr/bin/env python3
"""
Experiment 7: Representation-Level Mode Injection

Instead of injecting correct TEXT prefix (Exp 1-2), inject at the hidden-state
level via activation addition (steering vectors).

Phase 1 — Extract "reasoning mode" steering vectors:
    For each model, run correct & wrong samples through the model,
    extract hidden states at step-1 boundary, compute:
        sv = mean(correct_hs) - mean(wrong_hs)  per probed layer

Phase 2 — Generate with steering vector injection:
    Hook into a transformer layer during generation, add α × sv to the
    hidden state at each decoding step.

Conditions:
    baseline      — no intervention
    steer_αX      — add α×sv at mid-layer during generation
    anti_steer    — add -1.0×sv at mid-layer (sanity check: should make dip worse)
    random_dir    — add random unit vector × ‖sv‖ (control for direction specificity)
    steer_early   — add α×sv only during first ~200 generated tokens

Usage:
    python 27_representation_injection.py --models 0.5B --gpus 2
    python 27_representation_injection.py --only-plot --models 0.5B,1.5B,3B
"""

import os
import sys

for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpus" and _i + 1 < len(sys.argv) and sys.argv[_i + 1] != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prm.scoring import StepScorer, split_steps

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_REGISTRY = {
    "0.5B": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "layers": 25},
    "1.5B": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "layers": 29},
    "3B":   {"id": "Qwen/Qwen2.5-3B-Instruct",   "layers": 37},
    "7B":   {"id": "Qwen/Qwen2.5-7B-Instruct",   "layers": 28},
}

PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# ── Steering Hook ────────────────────────────────────────────

class SteeringHook:
    """Forward hook that adds a steering vector to the residual stream."""

    def __init__(self, sv: torch.Tensor, alpha: float = 1.0, max_gen_tokens: int = -1):
        self.sv = sv
        self.alpha = alpha
        self.max_gen_tokens = max_gen_tokens
        self.active = False
        self.gen_step = 0

    def __call__(self, module, input, output):
        if not self.active:
            return output

        hs = output[0] if isinstance(output, tuple) else output

        # seq_len > 1 → prompt prefill; only inject during autoregressive decode
        if hs.shape[1] != 1:
            return output

        if 0 < self.max_gen_tokens <= self.gen_step:
            return output

        self.gen_step += 1
        sv_cast = self.sv.to(hs.device, dtype=hs.dtype)
        hs[:, -1:, :] = hs[:, -1:, :] + self.alpha * sv_cast

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return output

    def reset(self, active: bool = True, max_gen_tokens: int = -1):
        self.active = active
        self.gen_step = 0
        self.max_gen_tokens = max_gen_tokens


# ── Utilities ────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.5B,1.5B,3B,7B")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--prm-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_prm"))
    ap.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_baselines"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs" / "representation_injection"))
    ap.add_argument("--sv-samples", type=int, default=30,
                    help="Samples per correct/wrong group for steering vector extraction")
    ap.add_argument("--max-samples", type=int, default=100,
                    help="Max wrong samples to generate per condition")
    ap.add_argument("--alphas", default="1.0,3.0,5.0",
                    help="Steering strengths to sweep")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--early-token-limit", type=int, default=200,
                    help="Token limit for steer_early condition")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--only-plot", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def query_gpu_free_mem() -> Dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return {i: int(l.strip()) for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return {}


def select_best_gpu(requested: str, min_free: int = 15000) -> int:
    free = query_gpu_free_mem()
    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        return max(usable, key=usable.get) if usable else (ids[0] if ids else 0)
    candidates = {g: m for g, m in free.items() if m >= min_free}
    return max(candidates, key=candidates.get) if candidates else max(free, key=free.get)


def _find_samples_jsonl(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.rglob("samples_*.jsonl"))
    return cands[-1] if cands else None


def load_questions(baseline_dir: Path, tag: str) -> Dict[int, str]:
    run_dir = baseline_dir / f"baseline_{tag}"
    jsonl_path = _find_samples_jsonl(run_dir)
    if jsonl_path is None:
        return {}
    questions = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("filter") == "strict-match":
                continue
            questions[d.get("doc_id", -1)] = d.get("doc", {}).get("problem", "")
    return questions


def build_chat_prompt(question: str, tokenizer) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_full_input(question: str, response: str, tokenizer) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{response}"
    )


def get_probe_layers(n_layers: int) -> Dict[str, int]:
    return {
        "early": max(1, n_layers // 6),
        "mid": n_layers // 2,
        "late": n_layers - max(1, n_layers // 6),
    }


def _model_params(tag: str) -> float:
    for s, v in [("0.5b", 5e8), ("1.5b", 1.5e9), ("3b", 3e9), ("7b", 7e9), ("14b", 14e9)]:
        if s in tag.lower():
            return v
    return 1e9


# ── Phase 1: Extract Steering Vectors ────────────────────────

@torch.inference_mode()
def extract_steering_vectors(
    model,
    tokenizer,
    correct_samples: List[dict],
    wrong_samples: List[dict],
    questions: Dict[int, str],
    layer_indices: Dict[str, int],
    max_seq_len: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Compute sv = mean(correct_hs) - mean(wrong_hs) at step-1 boundary per layer."""

    layer_idx_to_name = {v: k for k, v in layer_indices.items()}
    all_layer_ids = sorted(layer_indices.values())

    correct_vecs = {li: [] for li in all_layer_ids}
    wrong_vecs = {li: [] for li in all_layer_ids}

    for label, group, store in [
        ("sv_correct", correct_samples, correct_vecs),
        ("sv_wrong", wrong_samples, wrong_vecs),
    ]:
        for s in tqdm(group, desc=label):
            steps = s.get("steps_text", [])
            if len(steps) < 2:
                continue
            doc_id = s.get("doc_id", -1)
            question = questions.get(doc_id, "")
            if not question:
                continue

            # Build input up to the end of step 1
            response_step1 = steps[0]
            text = build_full_input(question, response_step1, tokenizer)
            input_ids = tokenizer.encode(text, return_tensors="pt",
                                         add_special_tokens=False)
            if input_ids.shape[1] > max_seq_len:
                continue
            input_ids = input_ids.to(model.device)

            try:
                outputs = model(input_ids=input_ids, output_hidden_states=True,
                                use_cache=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

            for li in all_layer_ids:
                if li < len(outputs.hidden_states):
                    vec = outputs.hidden_states[li][0, -1, :].cpu().float()
                    store[li].append(vec)

    steering_vectors = {}
    for li in all_layer_ids:
        if correct_vecs[li] and wrong_vecs[li]:
            mean_c = torch.stack(correct_vecs[li]).mean(dim=0)
            mean_w = torch.stack(wrong_vecs[li]).mean(dim=0)
            sv = mean_c - mean_w
            name = layer_idx_to_name.get(li, str(li))
            steering_vectors[name] = sv
            norm = sv.norm().item()
            print(f"  SV layer {li} ({name}): ‖sv‖ = {norm:.4f}, "
                  f"n_correct={len(correct_vecs[li])}, n_wrong={len(wrong_vecs[li])}")

    return steering_vectors


# ── Phase 2: Generate with Steering ──────────────────────────

@torch.inference_mode()
def generate_with_steering(
    model,
    tokenizer,
    question: str,
    hook: Optional[SteeringHook],
    handle,
    max_new_tokens: int = 1024,
    max_gen_tokens: int = -1,
) -> str:
    """Generate a response, optionally with steering hook active."""
    prompt = build_chat_prompt(question, tokenizer)
    input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                 add_special_tokens=False).to(model.device)

    if hook is not None:
        hook.reset(active=True, max_gen_tokens=max_gen_tokens)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )

    if hook is not None:
        hook.reset(active=False)

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main per-model pipeline ─────────────────────────────────

def run_model(tag: str, args) -> Optional[Dict[str, Any]]:
    info = MODEL_REGISTRY[tag]
    model_id = info["id"]
    n_layers = info["layers"]

    prm_file = Path(args.prm_dir) / f"prm_{tag}.json"
    if not prm_file.exists():
        print(f"[SKIP] {tag}: no PRM data at {prm_file}")
        return None

    prm_data = json.loads(prm_file.read_text(encoding="utf-8"))
    samples = prm_data["samples"]
    correct = [s for s in samples if s["exact_match"] >= 1 and len(s.get("steps_text", [])) >= 3]
    wrong = [s for s in samples if s["exact_match"] < 1 and len(s.get("steps_text", [])) >= 3]

    random.seed(args.seed)

    questions = load_questions(Path(args.baseline_dir), tag)
    print(f"[DATA] {tag}: {len(correct)} correct, {len(wrong)} wrong, {len(questions)} questions")

    # ── Phase 1: Extract SVs ──
    sv_correct = random.sample(correct, min(args.sv_samples, len(correct)))
    sv_wrong = random.sample(wrong, min(args.sv_samples, len(wrong)))

    probe_layers = get_probe_layers(n_layers)
    print(f"  Probe layers: {probe_layers}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_map.get(args.dtype, torch.float16),
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    max_seq_len = 2048 if _model_params(tag) >= 7e9 else 4096
    print(f"  Extracting steering vectors (max_seq_len={max_seq_len})...")
    svs = extract_steering_vectors(
        model, tokenizer, sv_correct, sv_wrong, questions, probe_layers, max_seq_len,
    )

    if "mid" not in svs:
        print(f"[WARN] {tag}: no mid-layer SV extracted, skipping")
        del model
        torch.cuda.empty_cache()
        return None

    sv_mid = svs["mid"]
    mid_layer_idx = probe_layers["mid"]

    # Register hook on mid layer
    target_layer = model.model.layers[mid_layer_idx]
    hook_obj = SteeringHook(sv_mid, alpha=1.0)
    handle = target_layer.register_forward_hook(hook_obj)

    # ── Phase 2: Generate ──
    gen_wrong = random.sample(wrong, min(args.max_samples, len(wrong)))
    alphas = [float(a) for a in args.alphas.split(",")]

    conditions = []
    conditions.append(("baseline", 0.0, -1))
    for a in alphas:
        conditions.append((f"steer_a{a}", a, -1))
    conditions.append(("anti_steer", -alphas[0] if alphas else -1.0, -1))
    conditions.append((f"steer_early_a{alphas[-1] if alphas else 5.0}",
                        alphas[-1] if alphas else 5.0, args.early_token_limit))
    # random direction control
    random_sv = torch.randn_like(sv_mid)
    random_sv = random_sv / random_sv.norm() * sv_mid.norm()

    print(f"\n[GEN] {tag}: {len(gen_wrong)} samples × {len(conditions)+1} conditions")

    results = []
    for s in tqdm(gen_wrong, desc=f"{tag}/gen"):
        doc_id = s.get("doc_id", -1)
        question = questions.get(doc_id, "")
        if not question:
            continue

        entry = {"doc_id": doc_id, "question": question[:200], "generations": {}}

        for cond_name, alpha, max_tok in conditions:
            hook_obj.sv = sv_mid
            hook_obj.alpha = alpha
            is_baseline = (alpha == 0.0)

            response = generate_with_steering(
                model, tokenizer, question,
                hook=None if is_baseline else hook_obj,
                handle=handle,
                max_new_tokens=args.max_new_tokens,
                max_gen_tokens=max_tok,
            )
            entry["generations"][cond_name] = response

        # Random direction control
        hook_obj.sv = random_sv
        hook_obj.alpha = alphas[-1] if alphas else 5.0
        response = generate_with_steering(
            model, tokenizer, question,
            hook=hook_obj, handle=handle,
            max_new_tokens=args.max_new_tokens, max_gen_tokens=-1,
        )
        entry["generations"]["random_dir"] = response
        results.append(entry)

    handle.remove()
    del model
    torch.cuda.empty_cache()

    # ── Phase 3: PRM Score ──
    print(f"\n[PRM] Scoring {tag}...")
    scorer = StepScorer(PRM_MODEL, args.dtype)

    all_cond_names = [c[0] for c in conditions] + ["random_dir"]

    for entry in tqdm(results, desc=f"{tag}/prm"):
        q = entry["question"]
        full_q = questions.get(entry["doc_id"], q)
        entry["prm_scores"] = {}
        for cond_name in all_cond_names:
            text = entry["generations"].get(cond_name, "")
            steps = split_steps(text, mode="auto")
            if steps:
                try:
                    scores = scorer.score_steps(full_q, steps)
                except Exception:
                    scores = []
            else:
                scores = []
            entry["prm_scores"][cond_name] = scores

    del scorer
    torch.cuda.empty_cache()

    # ── Save ──
    sv_meta = {}
    for name, sv in svs.items():
        sv_meta[name] = {
            "layer_idx": probe_layers[name],
            "norm": float(sv.norm()),
            "dim": sv.shape[0],
        }

    output = {
        "model_tag": tag,
        "model_id": model_id,
        "n_layers": n_layers,
        "probe_layers": probe_layers,
        "steering_vectors": sv_meta,
        "conditions": all_cond_names,
        "alphas": alphas,
        "n_samples": len(results),
        "results": results,
    }
    return output


# ── Plotting ─────────────────────────────────────────────────

def _avg_prm_curve(results: List[dict], cond: str) -> Tuple[List[int], List[float]]:
    max_steps = 0
    for r in results:
        scores = r.get("prm_scores", {}).get(cond, [])
        if len(scores) > max_steps:
            max_steps = len(scores)
    xs, ys = [], []
    for i in range(max_steps):
        vals = [r["prm_scores"][cond][i] for r in results
                if i < len(r.get("prm_scores", {}).get(cond, []))]
        if len(vals) >= 5:
            xs.append(i + 1)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def plot_model_results(data: dict, out_dir: Path):
    tag = data["model_tag"]
    results = data["results"]
    conds = data["conditions"]

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    for idx, cond in enumerate(conds):
        xs, ys = _avg_prm_curve(results, cond)
        if xs:
            style = "--" if "anti" in cond or "random" in cond else "-"
            ax.plot(xs, ys, style, marker="o", markersize=3, linewidth=2,
                    color=cmap(idx % 10), label=cond)

    ax.set_xlabel("Step k", fontsize=12)
    ax.set_ylabel("Avg PRM Score", fontsize=12)
    ax.set_title(f"Representation Injection: {data['model_id'].split('/')[-1]}\n"
                 f"(n={data['n_samples']}, mid-layer={data['probe_layers']['mid']})",
                 fontsize=13)
    ax.axhline(y=0.72, ls=":", color="gray", alpha=0.5, label="threshold=0.72")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"rep_injection_{tag}.png", dpi=200)
    plt.close()

    # Dip depth comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    dip_depths = {}
    for cond in conds:
        xs, ys = _avg_prm_curve(results, cond)
        if ys:
            dip_depths[cond] = ys[0] - min(ys)
    if dip_depths:
        names = list(dip_depths.keys())
        vals = [dip_depths[n] for n in names]
        colors = ["#2196F3" if "steer" in n and "anti" not in n and "random" not in n
                  else "#F44336" if "anti" in n
                  else "#9E9E9E" for n in names]
        ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Dip Depth (step1_score - min_score)")
        ax.set_title(f"Dip Depth by Condition: {tag}")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"dip_depth_{tag}.png", dpi=200)
        plt.close()


def plot_cross_scale(all_data: Dict[str, dict], out_dir: Path):
    if len(all_data) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: baseline vs best steer for each model
    ax = axes[0]
    for idx, (tag, data) in enumerate(
        sorted(all_data.items(), key=lambda x: _model_params(x[0]))
    ):
        results = data["results"]
        xs_b, ys_b = _avg_prm_curve(results, "baseline")
        if xs_b:
            ax.plot(xs_b, ys_b, "--", marker="x", markersize=3, linewidth=1.5,
                    color=plt.cm.Set1(idx), alpha=0.5, label=f"{tag} baseline")
        # Find best steer condition (highest min score)
        best_cond, best_min = None, -1
        for c in data["conditions"]:
            if "steer_a" in c and "anti" not in c and "early" not in c:
                _, ys = _avg_prm_curve(results, c)
                if ys and min(ys) > best_min:
                    best_min = min(ys)
                    best_cond = c
        if best_cond:
            xs_s, ys_s = _avg_prm_curve(results, best_cond)
            if xs_s:
                ax.plot(xs_s, ys_s, "-", marker="o", markersize=3, linewidth=2,
                        color=plt.cm.Set1(idx), label=f"{tag} {best_cond}")

    ax.set_xlabel("Step k")
    ax.set_ylabel("Avg PRM Score")
    ax.set_title("Baseline vs Best Steering Across Scales")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    # Right: dip depth reduction
    ax = axes[1]
    tags_sorted = sorted(all_data.keys(), key=_model_params)
    params = [_model_params(t) / 1e9 for t in tags_sorted]
    baseline_dips, best_steer_dips = [], []
    for tag in tags_sorted:
        data = all_data[tag]
        results = data["results"]
        _, ys_b = _avg_prm_curve(results, "baseline")
        baseline_dip = (ys_b[0] - min(ys_b)) if ys_b else 0
        baseline_dips.append(baseline_dip)

        best_dip = baseline_dip
        for c in data["conditions"]:
            if "steer_a" in c and "anti" not in c:
                _, ys = _avg_prm_curve(results, c)
                if ys:
                    d = ys[0] - min(ys)
                    if d < best_dip:
                        best_dip = d
        best_steer_dips.append(best_dip)

    x_pos = np.arange(len(tags_sorted))
    ax.bar(x_pos - 0.15, baseline_dips, 0.3, label="Baseline", color="#9E9E9E")
    ax.bar(x_pos + 0.15, best_steer_dips, 0.3, label="Best Steer", color="#2196F3")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{t}\n({p:.1f}B)" for t, p in zip(tags_sorted, params)])
    ax.set_ylabel("Dip Depth")
    ax.set_title("Dip Depth: Baseline vs Steering")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_dir / "cross_scale_rep_injection.png", dpi=200)
    plt.close()


# ── Main ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    model_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}

    if not args.only_plot:
        gpu_id = select_best_gpu(args.gpus)
        print(f"Using GPU {gpu_id}\n")

        for tag in model_tags:
            if tag not in MODEL_REGISTRY:
                print(f"[SKIP] Unknown model: {tag}")
                continue

            result_file = out_dir / f"rep_injection_{tag}.json"
            if result_file.exists():
                print(f"[LOAD] {tag}: existing results found, skipping computation")
                all_data[tag] = json.loads(result_file.read_text(encoding="utf-8"))
                continue

            data = run_model(tag, args)
            if data is None:
                continue

            all_data[tag] = data
            result_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"[SAVE] {tag} -> {result_file}\n")
    else:
        for tag in model_tags:
            rf = out_dir / f"rep_injection_{tag}.json"
            if rf.exists():
                all_data[tag] = json.loads(rf.read_text(encoding="utf-8"))

    # ── Phase 4: Plot ──
    for tag, data in all_data.items():
        plot_model_results(data, plot_dir)
        print(f"[PLOT] {tag} done")

    plot_cross_scale(all_data, plot_dir)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
