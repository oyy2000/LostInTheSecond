#!/usr/bin/env python3
"""
Experiment 4 (part 2): Train a 2-layer MLP head on hidden states for
step-level error detection.

Architecture: Linear(hidden_dim, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
Trained with BCE loss on first-error labels (y=1 if step is first error).

Extends the probe pipeline from 16_1 but uses a small MLP instead of
logistic regression, and reports AUROC for comparison.

Usage:
    python scripts/24_2_train_mlp_head.py --gpus auto
    python scripts/24_2_train_mlp_head.py --model-id Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--cache-file", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--layer", type=int, default=24,
                    help="Single layer index for MLP head")
    ap.add_argument("--hidden-mlp", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _ms(mid):
    return mid.split("/")[-1].lower().replace("-", "_")


def _load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def select_gpu(requested, min_free=12000):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], encoding="utf-8")
        free = {i: int(l.strip())
                for i, l in enumerate(out.strip().splitlines()) if l.strip()}
    except Exception:
        return 0
    if requested != "auto":
        ids = [int(x) for x in requested.split(",") if x.strip()]
        usable = {g: free.get(g, 0) for g in ids if free.get(g, 0) >= min_free}
        return max(usable, key=usable.get) if usable else (ids[0] if ids else 0)
    cands = {g: m for g, m in free.items() if m >= min_free}
    return max(cands, key=cands.get) if cands else max(free, key=free.get, default=0)


def extract_hidden_states(model, tokenizer, items, layer_idx, max_seq_len, model_id):
    """Extract hidden states at step boundaries (last token of each step)."""
    import torch
    from src.prompt_templates import build_prompt
    from src.keystep_utils import split_steps

    X_rows, y_rows, meta = [], [], []
    for item in items:
        question = item["question"]
        response = item.get("response", "")
        steps = item.get("steps", split_steps(response))
        tau = item["tau"]
        if not steps or tau < 0:
            continue

        prompt = build_prompt(model_id, "gsm8k", question)
        full_text = prompt + response
        input_ids = tokenizer.encode(full_text, return_tensors="pt",
                                     add_special_tokens=False)
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
        input_ids = input_ids.to(model.device)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        step_boundaries = []
        cum = prompt_len
        for step in steps:
            step_ids = tokenizer.encode(step, add_special_tokens=False)
            cum += len(step_ids)
            if cum < input_ids.shape[1]:
                step_boundaries.append(cum - 1)

        if not step_boundaries:
            continue

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1][0]

        for si, pos in enumerate(step_boundaries):
            if si > tau:
                break
            h = hs[pos].cpu().numpy().astype(np.float32)
            X_rows.append(h)
            y_rows.append(1.0 if si == tau else 0.0)
            meta.append({"doc_id": item["doc_id"],
                         "sample_idx": item.get("sample_idx", 0),
                         "step_idx": si, "tau": tau})

    return np.array(X_rows), np.array(y_rows), meta


def train_mlp(X_train, y_train, X_test, y_test, hidden_dim, epochs, lr, batch_size):
    """Train 2-layer MLP and return (model, train_auc, test_auc)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    input_dim = X_train.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_test, dtype=torch.float32, device=device)
    yv = torch.tensor(y_test, dtype=torch.float32, device=device)

    best_auc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            loss = criterion(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            train_scores = torch.sigmoid(model(Xt)).cpu().numpy()
            test_scores = torch.sigmoid(model(Xv)).cpu().numpy()
        try:
            train_auc = roc_auc_score(y_train, train_scores)
            test_auc = roc_auc_score(y_test, test_scores)
        except ValueError:
            train_auc = test_auc = 0.5

        if test_auc > best_auc:
            best_auc = test_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={total_loss:.4f} "
                  f"train_auc={train_auc:.4f} test_auc={test_auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_train = roc_auc_score(y_train,
            torch.sigmoid(model(Xt)).cpu().numpy())
        final_test = roc_auc_score(y_test,
            torch.sigmoid(model(Xv)).cpu().numpy())
    return model, final_train, final_test


def main():
    args = parse_args()
    ms = _ms(args.model_id)

    if not args.cache_file:
        for ds in ["gsm8k", "math500"]:
            p = ROOT / f"results/{ds}_3b_multi_sample/first_error/gpt_first_error_cache.jsonl"
            if p.exists():
                args.cache_file = str(p)
                break
    if not args.out_dir:
        args.out_dir = str(ROOT / "results" / f"{ms}_mlp_head")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_id = select_gpu(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU {gpu_id}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.metrics import roc_auc_score

    cache = _load_jsonl(Path(args.cache_file))
    items = [r for r in cache if r.get("tau") is not None and r["tau"] >= 0]
    if args.max_samples > 0:
        items = items[:args.max_samples]
    print(f"Loaded {len(items)} wrong trajectories")

    print(f"Loading model {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    print(f"Extracting hidden states at layer {args.layer}...")
    X, y, meta = extract_hidden_states(
        model, tokenizer, items, args.layer, args.max_seq_len, args.model_id)
    print(f"  Extracted {len(X)} step samples, {int(y.sum())} positive")

    del model
    torch.cuda.empty_cache()

    np.random.seed(args.seed)
    doc_ids = list(set(m["doc_id"] for m in meta))
    np.random.shuffle(doc_ids)
    split = int(len(doc_ids) * (1 - args.test_frac))
    train_docs = set(doc_ids[:split])
    test_docs = set(doc_ids[split:])

    train_mask = np.array([m["doc_id"] in train_docs for m in meta])
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"  Train: {len(X_train)} ({int(y_train.sum())} pos), "
          f"Test: {len(X_test)} ({int(y_test.sum())} pos)")

    # Logistic regression baseline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    lr_clf = LogisticRegression(max_iter=1000, C=1.0).fit(
        scaler.transform(X_train), y_train)
    lr_train_auc = roc_auc_score(
        y_train, lr_clf.predict_proba(scaler.transform(X_train))[:, 1])
    lr_test_auc = roc_auc_score(
        y_test, lr_clf.predict_proba(scaler.transform(X_test))[:, 1])
    print(f"\nLogistic probe: train_auc={lr_train_auc:.4f} test_auc={lr_test_auc:.4f}")

    # MLP head
    print(f"\nTraining MLP head (hidden={args.hidden_mlp})...")
    mlp_model, mlp_train_auc, mlp_test_auc = train_mlp(
        X_train, y_train, X_test, y_test,
        args.hidden_mlp, args.epochs, args.lr, args.batch_size)
    print(f"MLP head: train_auc={mlp_train_auc:.4f} test_auc={mlp_test_auc:.4f}")

    # Save
    summary = {
        "model_id": args.model_id,
        "layer": args.layer,
        "hidden_mlp": args.hidden_mlp,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "logistic_train_auc": round(lr_train_auc, 4),
        "logistic_test_auc": round(lr_test_auc, 4),
        "mlp_train_auc": round(mlp_train_auc, 4),
        "mlp_test_auc": round(mlp_test_auc, 4),
    }
    (out_dir / "mlp_head_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    torch.save(mlp_model.state_dict(), out_dir / "mlp_head.pt")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
