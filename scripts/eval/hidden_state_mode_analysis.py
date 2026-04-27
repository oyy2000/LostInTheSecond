#!/usr/bin/env python3
"""
Hidden-state mode onset analysis across model scales.

For each model size, extract hidden states at each step boundary, project onto
the steering vector direction, and measure when the model "enters" a stable
reasoning mode. Compare mode onset latency across correct vs wrong samples
and across model sizes.

Usage:
    python 24_hidden_state_mode_analysis.py --models 1.5B,3B,7B --gpus auto
    python 24_hidden_state_mode_analysis.py --only-plot  # from existing data
"""

import os
import sys

# Set CUDA_VISIBLE_DEVICES before torch import to ensure proper GPU isolation
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpus" and _i + 1 < len(sys.argv) and sys.argv[_i + 1] != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prm.scoring import split_steps

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent

MODEL_REGISTRY = {
    "0.5B": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "layers": 25},
    "1.5B": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "layers": 29},
    "3B":   {"id": "Qwen/Qwen2.5-3B-Instruct",   "layers": 37},
    "7B":   {"id": "Qwen/Qwen2.5-7B-Instruct",   "layers": 28},
    "14B":  {"id": "Qwen/Qwen2.5-14B-Instruct",  "layers": 48},
}

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="1.5B,3B,7B")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--baseline-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_baselines"),
                    help="Directory with lm-eval baseline outputs (for loading questions)")
    ap.add_argument("--prm-dir", default=str(PROJECT_ROOT / "runs" / "multi_scale_prm"))
    ap.add_argument("--vector-dir", default=str(PROJECT_ROOT / "runs" / "vectors_first2steps"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "runs" / "hidden_state_mode"))
    ap.add_argument("--max-samples", type=int, default=100,
                    help="Max samples per correct/wrong group per model")
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="Max sequence length (0 = auto: 2048 for >=7B, 4096 otherwise)")
    ap.add_argument("--probe-layers", default="early,mid,late",
                    help="Which layers to probe: early,mid,late or specific indices like 4,12,20")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--min-free-mem-mb", type=int, default=15000)
    ap.add_argument("--only-plot", action="store_true")
    return ap.parse_args()


def select_best_gpu(requested: str, min_free: int) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        free = {i: int(l.strip()) for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return 0

    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        return max(usable, key=usable.get) if usable else (ids[0] if ids else 0)

    candidates = {g: m for g, m in free.items() if m >= min_free}
    return max(candidates, key=candidates.get) if candidates else max(free, key=free.get)


def _find_samples_jsonl(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.rglob("samples_*.jsonl"))
    return cands[-1] if cands else None


def load_questions_from_baseline(baseline_dir: Path, model_tag: str) -> Dict[int, str]:
    """Load original question text from lm-eval samples, keyed by doc_id."""
    run_dir = baseline_dir / f"baseline_{model_tag}"
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


def get_probe_layer_indices(n_layers: int, spec: str) -> List[int]:
    """Convert probe spec to actual layer indices."""
    if spec in ("early,mid,late", "auto"):
        early = max(1, n_layers // 6)
        mid = n_layers // 2
        late = n_layers - max(1, n_layers // 6)
        return sorted(set([early, mid, late]))

    try:
        return sorted(set(int(x.strip()) for x in spec.split(",") if x.strip()))
    except ValueError:
        return [n_layers // 4, n_layers // 2, 3 * n_layers // 4]


def build_chat_input(question: str, response: str, tokenizer) -> dict:
    """Build tokenized input for a question + response."""
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{response}"
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return inputs


def find_step_boundary_positions(
    response_text: str,
    steps: List[str],
    tokenizer,
    prompt_len: int,
) -> List[int]:
    """Find token positions corresponding to the end of each step."""
    positions = []
    accumulated = ""
    for step in steps:
        if accumulated:
            accumulated += "\n\n" + step
        else:
            accumulated = step
        token_ids = tokenizer.encode(accumulated, add_special_tokens=False)
        pos = prompt_len + len(token_ids) - 1
        positions.append(pos)
    return positions


@torch.inference_mode()
def extract_hidden_states_at_boundaries(
    model,
    tokenizer,
    question: str,
    response: str,
    steps: List[str],
    layer_indices: List[int],
    max_seq_len: int = 4096,
) -> Optional[Dict[str, Any]]:
    """Extract hidden states at step boundaries for specified layers."""
    inputs = build_chat_input(question, response, tokenizer)
    input_ids = inputs["input_ids"].to(model.device)

    if input_ids.shape[1] > max_seq_len:
        return None

    prompt_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    boundary_positions = find_step_boundary_positions(response, steps, tokenizer, prompt_len)
    valid_positions = [p for p in boundary_positions if p < input_ids.shape[1]]

    if not valid_positions:
        return None

    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

    result = {}
    for layer_idx in layer_indices:
        if layer_idx >= len(hidden_states):
            continue
        hs = hidden_states[layer_idx][0]  # (seq_len, hidden_dim)
        step_vectors = []
        for pos in valid_positions:
            vec = hs[pos].cpu().float().numpy()
            step_vectors.append(vec)
        result[f"layer_{layer_idx}"] = step_vectors

    return {
        "boundary_positions": valid_positions,
        "n_steps_extracted": len(valid_positions),
        "hidden_states": result,
    }


def compute_projections(
    hidden_states_by_layer: Dict[str, List[np.ndarray]],
    steering_vector: dict,
    layer_indices: List[int],
) -> Dict[str, List[float]]:
    """Project hidden states onto steering vector direction."""
    projections = {}

    for layer_idx in layer_indices:
        layer_key = f"layer_{layer_idx}"
        if layer_key not in hidden_states_by_layer:
            continue

        sv = None
        if hasattr(steering_vector, 'items'):
            for k, v in steering_vector.items():
                if isinstance(v, dict):
                    if layer_idx in v:
                        sv = v[layer_idx]
                        break
                elif hasattr(v, 'layer_activations'):
                    try:
                        sv = v.layer_activations.get(layer_idx)
                    except Exception:
                        pass

        if sv is None:
            # Try to extract from the steering vector object directly
            try:
                if hasattr(steering_vector, 'layer_activations'):
                    sv = steering_vector.layer_activations.get(layer_idx)
                elif isinstance(steering_vector, dict) and 'layer_activations' in steering_vector:
                    sv = steering_vector['layer_activations'].get(layer_idx)
            except Exception:
                pass

        if sv is None:
            continue

        if isinstance(sv, torch.Tensor):
            sv_np = sv.cpu().float().numpy().flatten()
        else:
            sv_np = np.array(sv).flatten()

        sv_norm = np.linalg.norm(sv_np)
        if sv_norm < 1e-8:
            continue

        sv_unit = sv_np / sv_norm

        step_projections = []
        for step_vec in hidden_states_by_layer[layer_key]:
            proj = float(np.dot(step_vec, sv_unit))
            step_projections.append(proj)

        projections[layer_key] = step_projections

    return projections


def analyze_model(
    model_tag: str,
    gpu_id: int,
    args,
) -> Optional[Dict[str, Any]]:
    """Full hidden-state analysis for one model."""

    model_info = MODEL_REGISTRY[model_tag]
    model_id = model_info["id"]
    n_layers = model_info["layers"]

    prm_file = Path(args.prm_dir) / f"prm_{model_tag}.json"
    if not prm_file.exists():
        print(f"[SKIP] {model_tag}: no PRM results")
        return None

    prm_data = json.loads(prm_file.read_text(encoding="utf-8"))
    samples = prm_data["samples"]
    correct = [s for s in samples if s["exact_match"] >= 1 and len(s.get("steps_text", [])) >= 3]
    wrong = [s for s in samples if s["exact_match"] < 1 and len(s.get("steps_text", [])) >= 3]

    import random
    random.seed(42)
    if len(correct) > args.max_samples:
        correct = random.sample(correct, args.max_samples)
    if len(wrong) > args.max_samples:
        wrong = random.sample(wrong, args.max_samples)

    print(f"[ANALYZE] {model_tag}: {len(correct)} correct, {len(wrong)} wrong samples")

    questions = load_questions_from_baseline(Path(args.baseline_dir), model_tag)
    print(f"  Loaded {len(questions)} questions from baseline samples")

    if args.max_seq_len > 0:
        max_seq_len = args.max_seq_len
    else:
        params = _model_params(model_tag)
        max_seq_len = 2048 if params >= 7e9 else 4096
    print(f"  Max sequence length: {max_seq_len}")

    layer_indices = get_probe_layer_indices(n_layers, args.probe_layers)
    print(f"  Probing layers: {layer_indices}")

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

    vec_dir = Path(args.vector_dir)
    model_vec_tag = model_id.replace("/", "_")
    vec_path = vec_dir / f"{model_vec_tag}_first2steps" / "steering_vector.pt"
    steering_vector = None
    if vec_path.exists():
        steering_vector = torch.load(vec_path, map_location="cpu")
        print(f"  Loaded steering vector from {vec_path}")
    else:
        print(f"  No steering vector at {vec_path}; will compute norms only")

    results = {"correct": [], "wrong": []}
    skipped_oom = 0

    for label, group in [("correct", correct), ("wrong", wrong)]:
        for s in tqdm(group, desc=f"{model_tag}/{label}"):
            steps = s.get("steps_text", [])
            doc_id = s.get("doc_id", -1)
            question = questions.get(doc_id, "")
            if not question:
                continue
            response = "\n\n".join(steps)

            try:
                hs_result = extract_hidden_states_at_boundaries(
                    model, tokenizer, question, response, steps,
                    layer_indices, max_seq_len=max_seq_len,
                )
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                torch.cuda.empty_cache()
                continue

            if hs_result is None:
                continue

            entry = {
                "doc_id": doc_id,
                "n_steps": len(steps),
                "n_extracted": hs_result["n_steps_extracted"],
                "step_scores": s.get("step_scores", []),
            }

            for lk, vecs in hs_result["hidden_states"].items():
                entry[f"{lk}_norms"] = [float(np.linalg.norm(v)) for v in vecs]

            if steering_vector is not None:
                projections = compute_projections(
                    hs_result["hidden_states"], steering_vector, layer_indices,
                )
                for lk, projs in projections.items():
                    entry[f"{lk}_projections"] = projs

            results[label].append(entry)

    if skipped_oom > 0:
        print(f"  [WARN] Skipped {skipped_oom} samples due to OOM")

    del model
    torch.cuda.empty_cache()

    return {
        "model_tag": model_tag,
        "model_id": model_id,
        "n_layers": n_layers,
        "layer_indices": layer_indices,
        "has_steering_vector": steering_vector is not None,
        "n_correct": len(results["correct"]),
        "n_wrong": len(results["wrong"]),
        "correct": results["correct"],
        "wrong": results["wrong"],
    }


def _avg_by_step(entries: list, key: str) -> Tuple[List[int], List[float]]:
    if not entries:
        return [], []
    max_steps = max((len(e.get(key, [])) for e in entries), default=0)
    xs, ys = [], []
    for i in range(max_steps):
        vals = [e[key][i] for e in entries if i < len(e.get(key, []))]
        if vals:
            xs.append(i + 1)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def plot_mode_analysis(
    all_model_data: Dict[str, dict],
    out_dir: Path,
):
    """Generate mode onset plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-model: norm and projection curves by step
    for model_tag, data in all_model_data.items():
        layers = data.get("layer_indices", [])

        for metric_type in ["norms", "projections"]:
            fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
            if len(layers) == 1:
                axes = [axes]

            for ax, layer_idx in zip(axes, layers):
                key = f"layer_{layer_idx}_{metric_type}"

                xc, yc = _avg_by_step(data["correct"], key)
                xw, yw = _avg_by_step(data["wrong"], key)

                if xc:
                    ax.plot(xc, yc, "g-o", linewidth=2, label=f"Correct (n={data['n_correct']})")
                if xw:
                    ax.plot(xw, yw, "r-x", linewidth=2, label=f"Wrong (n={data['n_wrong']})")

                ax.set_xlabel("Step k")
                ax.set_ylabel(f"Avg {metric_type.rstrip('s')}")
                ax.set_title(f"Layer {layer_idx}")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.25)

            plt.suptitle(
                f"Hidden State {metric_type.title()} per Step: {data.get('model_id', model_tag)}",
                fontsize=13,
            )
            plt.tight_layout()
            plt.savefig(out_dir / f"mode_{metric_type}_{model_tag}.png", dpi=180)
            plt.close()

    # Cross-scale: mode onset comparison (projections)
    if len(all_model_data) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_model_data)))

        for idx, (tag, data) in enumerate(
            sorted(all_model_data.items(), key=lambda x: _model_params(x[0]))
        ):
            layers = data.get("layer_indices", [])
            if not layers:
                continue
            mid_layer = layers[len(layers) // 2]
            key = f"layer_{mid_layer}_projections"

            xw, yw = _avg_by_step(data["wrong"], key)
            if xw:
                short = data.get("model_id", tag).split("/")[-1]
                ax.plot(xw, yw, marker="o", linewidth=2, color=colors[idx],
                        label=f"{short} (L{mid_layer}, n={data['n_wrong']})")

        ax.set_xlabel("Step k", fontsize=12)
        ax.set_ylabel("Avg Steering Vector Projection (wrong samples)", fontsize=12)
        ax.set_title("Reasoning Mode Onset: Wrong Answers Across Scales", fontsize=13)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "cross_scale_mode_onset.png", dpi=200)
        plt.close()


def _model_params(tag: str) -> float:
    for size, val in [("0.5b", 5e8), ("1.5b", 1.5e9), ("3b", 3e9),
                      ("7b", 7e9), ("14b", 14e9)]:
        if size in tag.lower():
            return val
    return 1e9


def main():
    args = parse_args()
    model_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_model_data = {}

    if not args.only_plot:
        gpu_id = select_best_gpu(args.gpus, args.min_free_mem_mb)
        print(f"Using GPU {gpu_id}\n")

        for tag in model_tags:
            if tag not in MODEL_REGISTRY:
                print(f"[SKIP] Unknown: {tag}")
                continue

            result_file = out_dir / f"hidden_state_{tag}.json"
            if result_file.exists():
                print(f"[LOAD] {tag}: loading existing results")
                all_model_data[tag] = json.loads(result_file.read_text(encoding="utf-8"))
                continue

            data = analyze_model(tag, gpu_id, args)
            if data is None:
                continue

            # Save without the large hidden state arrays (keep only metrics)
            all_model_data[tag] = data
            result_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"[SAVE] {tag} -> {result_file}")
    else:
        for tag in model_tags:
            rf = out_dir / f"hidden_state_{tag}.json"
            if rf.exists():
                all_model_data[tag] = json.loads(rf.read_text(encoding="utf-8"))

    plot_mode_analysis(all_model_data, out_dir / "plots")
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
