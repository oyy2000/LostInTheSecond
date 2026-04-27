#!/usr/bin/env python3
"""
Train a linear probe on 3B hidden states to predict first-error steps.

Pipeline:
  1. Load wrong trajectories with tau labels from first_error cache.
  2. Run HF forward pass on the 3B model to extract hidden states at step
     boundaries (last token of each step).
  3. Label: y_t=1 if t==tau (first error), y_t=0 if t<tau.
     Steps after tau are excluded (contaminated by error propagation).
  4. Train sklearn LogisticRegression, sweep layers.
  5. Evaluate AUROC on held-out doc_ids.
  6. Save best probe checkpoint as pickle.

Usage:
    python scripts/16_1_train_rollback_probe.py \
        --model-id meta-llama/Llama-3.2-3B-Instruct \
        --gpus auto
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser(description="Train rollback probe on 3B hidden states")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--cache-file", default="",
                    help="Path to gpt_first_error_cache.jsonl")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--layers", default="8,16,24,28",
                    help="Layer indices to probe")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def select_best_gpu(requested: str, min_free: int = 12000) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        free = {i: int(l.strip())
                for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return 0
    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        if usable:
            return max(usable, key=usable.get)
        return ids[0] if ids else 0
    candidates = {g: m for g, m in free.items() if m >= min_free}
    if candidates:
        return max(candidates, key=candidates.get)
    return max(free, key=free.get) if free else 0


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -- hidden state extraction -----------------------------------------------

def extract_hidden_states_batch(
    model, tokenizer, items: List[dict],
    layer_indices: List[int], max_seq_len: int,
    model_id: str = "",
) -> Tuple[List[np.ndarray], List[dict]]:
    """Extract hidden states at step boundaries for a batch of trajectories.

    Returns (hs_list, meta_list) where each hs is (n_steps, n_layers, hidden_dim).
    """
    import torch
    from src.prompt_templates import build_prompt

    if not model_id:
        model_id = getattr(model.config, "_name_or_path",
                           getattr(model.config, "name_or_path", ""))

    all_hs, all_meta = [], []

    for item in items:
        question = item["question"]
        response = item["response"]
        steps = item["steps"]
        tau = item["tau"]

        prompt = build_prompt(model_id, "gsm8k", question)
        full_text = prompt + response

        input_ids = tokenizer.encode(
            full_text, return_tensors="pt",
            add_special_tokens=False,
            truncation=True, max_length=max_seq_len,
        ).to(model.device)

        if input_ids.shape[1] < 10:
            continue

        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        boundary_positions = []
        current_text = prompt
        for step in steps:
            current_text += step
            step_end_ids = tokenizer.encode(
                current_text, add_special_tokens=False,
            )
            pos = min(len(step_end_ids) - 1, input_ids.shape[1] - 1)
            if pos >= prompt_len:
                boundary_positions.append(pos)
            if "\n\n" in response:
                current_text += "\n\n"

        if not boundary_positions:
            continue

        n_steps = len(boundary_positions)
        n_layers = len(layer_indices)
        hidden_dim = hidden_states[0].shape[-1]
        hs = np.zeros((n_steps, n_layers, hidden_dim), dtype=np.float32)

        for si, pos in enumerate(boundary_positions):
            for li, layer_idx in enumerate(layer_indices):
                if layer_idx < len(hidden_states):
                    hs[si, li] = hidden_states[layer_idx][0, pos].float().cpu().numpy()

        all_hs.append(hs)
        all_meta.append({
            "doc_id": item["doc_id"],
            "sample_idx": item["sample_idx"],
            "tau": tau,
            "n_steps": n_steps,
            "n_steps_original": len(steps),
        })

    return all_hs, all_meta


# -- probe training --------------------------------------------------------

def build_dataset(
    hs_list: List[np.ndarray],
    meta_list: List[dict],
    layer_idx_pos: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Build (X, y, doc_ids) for a single layer.

    Only includes steps t < tau (label=0) and t == tau (label=1).
    Steps after tau are excluded.
    """
    X_rows, y_rows, doc_ids = [], [], []
    for hs, meta in zip(hs_list, meta_list):
        tau = meta["tau"]  # 1-indexed
        error_idx = tau - 1  # 0-indexed
        n_steps = hs.shape[0]
        for t in range(min(n_steps, error_idx + 1)):
            X_rows.append(hs[t, layer_idx_pos])
            y_rows.append(1.0 if t == error_idx else 0.0)
            doc_ids.append(meta["doc_id"])
    return np.array(X_rows), np.array(y_rows), doc_ids


def train_and_eval(
    X: np.ndarray, y: np.ndarray, doc_ids: List[int],
    test_frac: float, seed: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    unique_docs = sorted(set(doc_ids))
    rng.shuffle(unique_docs)
    n_test = max(1, int(len(unique_docs) * test_frac))
    test_docs = set(unique_docs[:n_test])

    doc_arr = np.array(doc_ids)
    train_mask = np.array([d not in test_docs for d in doc_ids])
    test_mask = ~train_mask

    if train_mask.sum() < 10 or test_mask.sum() < 5:
        return {"auc": 0.5, "accuracy": 0.0, "n_train": 0, "n_test": 0}
    if len(np.unique(y[train_mask])) < 2:
        return {"auc": 0.5, "accuracy": 0.0, "n_train": 0, "n_test": 0}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_mask])
    X_te = scaler.transform(X[test_mask])
    y_tr, y_te = y[train_mask], y[test_mask]

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(float)
    acc = float(np.mean(preds == y_te))

    try:
        auc = float(roc_auc_score(y_te, probs))
    except Exception:
        auc = 0.5

    return {
        "auc": auc, "accuracy": acc,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_test": float(y_te.mean()),
        "probe": clf, "scaler": scaler,
    }


# -- main ------------------------------------------------------------------

def main():
    import torch
    from tqdm.auto import tqdm

    args = parse_args()

    if not args.cache_file:
        args.cache_file = str(
            PROJECT_ROOT / "results/gsm8k_3b_multi_sample/first_error"
            / "gpt_first_error_cache.jsonl"
        )
    if not args.out_dir:
        model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
        args.out_dir = str(PROJECT_ROOT / "results" / f"gsm8k_{model_short}_rollback_probe")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = [int(x) for x in args.layers.split(",")]

    # Load wrong trajectories with tau
    cache = read_jsonl(Path(args.cache_file))
    items = [r for r in cache if r.get("tau") is not None]
    if args.max_samples > 0:
        items = items[:args.max_samples]
    print(f"Loaded {len(items)} wrong trajectories with tau labels")

    # Check for cached hidden states
    hs_file = out_dir / "hidden_states.npz"
    meta_file = out_dir / "meta.jsonl"

    if hs_file.exists() and meta_file.exists():
        print(f"Loading cached hidden states from {hs_file}")
        data = np.load(hs_file, allow_pickle=True)
        hs_list = list(data["hs_list"])
        meta_list = read_jsonl(meta_file)
    else:
        # Select GPU and load model
        gpu_id = select_best_gpu(args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Using GPU {gpu_id}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, args.dtype, torch.float16)
        print(f"Loading {args.model_id} ({args.dtype})...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=dtype, trust_remote_code=True,
            device_map="auto",
        )
        model.eval()

        hs_list, meta_list = [], []
        BATCH = 10
        for bi in tqdm(range(0, len(items), BATCH), desc="Extracting hidden states"):
            batch = items[bi:bi + BATCH]
            batch_hs, batch_meta = extract_hidden_states_batch(
                model, tokenizer, batch, layer_indices, args.max_seq_len,
                model_id=args.model_id,
            )
            hs_list.extend(batch_hs)
            meta_list.extend(batch_meta)
            if len(hs_list) % 100 == 0:
                torch.cuda.empty_cache()

        np.savez_compressed(hs_file, hs_list=np.array(hs_list, dtype=object))
        with meta_file.open("w", encoding="utf-8") as f:
            for m in meta_list:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"Saved {len(hs_list)} hidden states -> {hs_file}")

        del model
        torch.cuda.empty_cache()

    # Train probes for each layer
    print(f"\nTraining probes: {len(hs_list)} samples, layers={layer_indices}")
    all_results = []
    best_auc, best_layer, best_probe_data = 0.0, None, None

    for li, layer in enumerate(layer_indices):
        X, y, doc_ids = build_dataset(hs_list, meta_list, li)
        if len(X) < 20:
            print(f"  Layer {layer}: too few samples ({len(X)}), skipping")
            continue

        metrics = train_and_eval(X, y, doc_ids, args.test_frac, args.seed)
        result = {
            "layer": layer, "layer_idx_pos": li,
            "auc": metrics["auc"], "accuracy": metrics["accuracy"],
            "n_train": metrics["n_train"], "n_test": metrics["n_test"],
            "pos_rate_train": metrics.get("pos_rate_train", 0),
            "pos_rate_test": metrics.get("pos_rate_test", 0),
        }
        all_results.append(result)
        print(f"  Layer {layer}: AUC={metrics['auc']:.4f}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"n_train={metrics['n_train']}  n_test={metrics['n_test']}")

        if metrics["auc"] > best_auc and "probe" in metrics:
            best_auc = metrics["auc"]
            best_layer = layer
            best_probe_data = {
                "probe": metrics["probe"],
                "scaler": metrics["scaler"],
                "layer_idx": layer,
                "layer_idx_pos": li,
                "auc": metrics["auc"],
                "model_id": args.model_id,
            }

    # Save results
    results_file = out_dir / "probe_results.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults -> {results_file}")

    # Save best probe
    if best_probe_data:
        probe_file = out_dir / "best_probe.pkl"
        with open(probe_file, "wb") as f:
            pickle.dump(best_probe_data, f)
        print(f"Best probe: layer={best_layer}, AUC={best_auc:.4f} -> {probe_file}")
    else:
        print("WARNING: no probe trained successfully")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
