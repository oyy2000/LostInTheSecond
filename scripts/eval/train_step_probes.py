#!/usr/bin/env python3
"""
Train linear probes on hidden states to predict trajectory correctness (v2).

GPU-accelerated PyTorch training. Supports corrupted conditions.
For each (condition, layer, step), trains a linear probe and shallow MLP.

Usage:
    python scripts/eval/train_step_probes.py \
        --hs-dir results/gsm8k_7b_v2/hidden_states \
        --out-dir results/gsm8k_7b_v2/probes \
        --conditions original,corrected_k2,corrupted_k2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train step-level probes (GPU, v2)")
    ap.add_argument("--hs-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/hidden_states"))
    ap.add_argument("--out-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/probes"))
    ap.add_argument("--conditions",
                    default="original,corrected_k1,corrected_k2,corrected_k3,corrected_k4,"
                            "corrupted_k1,corrupted_k2,corrupted_k3,corrupted_k4")
    ap.add_argument("--max-step", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--unified", action="store_true",
                    help="Train probe on 'original' only, then score all conditions with frozen probe")
    return ap.parse_args()


def load_hs_and_meta(hs_dir: Path, condition: str):
    hs_path = hs_dir / f"hs_{condition}.pt"
    meta_path = hs_dir / f"meta_{condition}.jsonl"
    if not hs_path.exists():
        print(f"  [SKIP] {hs_path} not found")
        return None, None, None
    data = torch.load(hs_path, map_location="cpu", weights_only=False)
    meta = []
    if meta_path.exists():
        with meta_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta.append(json.loads(line))
    return data["hidden_states"], meta, data["layer_indices"]


class LinearProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(
    model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device,
    epochs: int = 30, lr: float = 1e-3, batch_size: int = 512,
) -> Dict[str, float]:
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(float)
        yt = y_test.cpu().numpy()

    metrics = {"accuracy": float(np.mean(preds == yt))}
    if len(np.unique(yt)) >= 2:
        try:
            metrics["auc"] = float(roc_auc_score(yt, probs))
        except Exception:
            metrics["auc"] = 0.5
    else:
        metrics["auc"] = 0.5
    return metrics


def score_with_frozen_probe(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Score samples with a frozen probe, return P(correct)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def run_unified(args, device) -> List[Dict]:
    """Train probe on 'original', freeze, score all conditions."""
    hs_dir = Path(args.hs_dir)
    out_dir = Path(args.out_dir)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # Must include original
    if "original" not in conditions:
        conditions = ["original"] + conditions

    # Load all data
    all_data = {}
    for cond in conditions:
        hs_list, meta, layer_indices_cond = load_hs_and_meta(hs_dir, cond)
        if hs_list is None:
            continue
        all_data[cond] = (hs_list, meta, layer_indices_cond)

    if "original" not in all_data:
        print("ERROR: original hidden states not found")
        return []

    orig_hs, orig_meta, layer_indices = all_data["original"]

    # Labels for original
    orig_labels = np.array([m.get("exact_match", 0.0) for m in orig_meta], dtype=np.float32)
    orig_labels = (orig_labels >= 1.0).astype(np.float32)

    # Split original by doc_id
    rng = np.random.RandomState(args.seed)
    doc_ids_orig = [m["doc_id"] for m in orig_meta]
    unique_docs = sorted(set(doc_ids_orig))
    rng.shuffle(unique_docs)
    n_test = max(1, int(len(unique_docs) * args.test_frac))
    test_docs = set(unique_docs[:n_test])
    orig_train_mask = np.array([d not in test_docs for d in doc_ids_orig])
    orig_test_mask = ~orig_train_mask

    print(f"Original: {len(orig_meta)} samples, "
          f"train={orig_train_mask.sum()}, test={orig_test_mask.sum()}, "
          f"pos_rate={orig_labels.mean():.3f}")

    all_results = []

    for li, layer in enumerate(layer_indices):
        max_steps = orig_hs[0].shape[0]

        for step in range(1, min(args.max_step + 1, max_steps + 1)):
            si = step - 1

            # Extract original vectors for this layer/step
            X_orig = np.array([orig_hs[i][si, li].float().numpy()
                               for i in range(len(orig_hs))
                               if si < orig_hs[i].shape[0]])
            if len(X_orig) < len(orig_meta):
                # Some samples don't have this step — skip
                continue

            y_orig = orig_labels
            if orig_train_mask.sum() < 10 or orig_test_mask.sum() < 5:
                continue
            if len(np.unique(y_orig[orig_train_mask])) < 2:
                continue

            X_tr = torch.tensor(X_orig[orig_train_mask], dtype=torch.float32)
            y_tr = torch.tensor(y_orig[orig_train_mask], dtype=torch.float32)
            X_te = torch.tensor(X_orig[orig_test_mask], dtype=torch.float32)
            y_te = torch.tensor(y_orig[orig_test_mask], dtype=torch.float32)

            # Train MLP probe on original
            probe = MLPProbe(X_orig.shape[1])
            orig_metrics = train_probe(probe, X_tr, y_tr, X_te, y_te, device,
                                       args.epochs, args.lr, args.batch_size)

            result_orig = {
                "condition": "original",
                "layer": layer,
                "step": step,
                "unified_auc": orig_metrics["auc"],
                "unified_acc": orig_metrics["accuracy"],
            }
            all_results.append(result_orig)

            # Freeze probe, score all other conditions
            probe.eval()
            for cond in conditions:
                if cond == "original":
                    continue
                if cond not in all_data:
                    continue

                cond_hs, cond_meta, _ = all_data[cond]
                cond_labels = np.array([m.get("exact_match", 0.0) for m in cond_meta],
                                       dtype=np.float32)
                cond_labels = (cond_labels >= 1.0).astype(np.float32)

                # Extract vectors
                valid = [i for i in range(len(cond_hs)) if si < cond_hs[i].shape[0]]
                if len(valid) < 20:
                    continue

                X_cond = np.array([cond_hs[i][si, li].float().numpy() for i in valid])
                y_cond = cond_labels[valid]

                if len(np.unique(y_cond)) < 2:
                    # Can't compute AUC, but can compute mean P(correct)
                    probs = score_with_frozen_probe(probe, X_cond, device)
                    result = {
                        "condition": cond,
                        "layer": layer,
                        "step": step,
                        "unified_auc": 0.5,
                        "unified_acc": float(np.mean((probs >= 0.5) == y_cond)),
                        "mean_p_correct": float(probs.mean()),
                    }
                else:
                    probs = score_with_frozen_probe(probe, X_cond, device)
                    try:
                        auc = float(roc_auc_score(y_cond, probs))
                    except Exception:
                        auc = 0.5
                    result = {
                        "condition": cond,
                        "layer": layer,
                        "step": step,
                        "unified_auc": auc,
                        "unified_acc": float(np.mean((probs >= 0.5) == y_cond)),
                        "mean_p_correct": float(probs.mean()),
                    }
                all_results.append(result)

            print(f"  L{layer} S{step}: orig_auc={orig_metrics['auc']:.3f}  "
                  + "  ".join(f"{c}={[r for r in all_results if r['condition']==c and r['layer']==layer and r['step']==step][0]['unified_auc']:.3f}"
                             for c in conditions if c != "original" and c in all_data
                             if any(r['condition']==c and r['layer']==layer and r['step']==step for r in all_results)))

    return all_results


def main() -> None:
    args = parse_args()
    hs_dir = Path(args.hs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.unified:
        all_results = run_unified(args, device)
        out_file = out_dir / "unified_probe_results.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(all_results)} unified results -> {out_file}")
        _plot_unified_comparison(all_results, out_dir)
        return

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rng = np.random.RandomState(args.seed)
    all_results = []

    for cond in conditions:
        hs_list, meta, layer_indices = load_hs_and_meta(hs_dir, cond)
        if hs_list is None:
            continue

        print(f"\n=== {cond}: {len(hs_list)} samples, layers={layer_indices} ===")

        for li, layer in enumerate(layer_indices):
            for step in range(1, args.max_step + 1):
                step_idx = step - 1
                X_list, y_list, doc_ids = [], [], []

                for i, (hs, m) in enumerate(zip(hs_list, meta)):
                    if step_idx >= hs.shape[0]:
                        continue
                    X_list.append(hs[step_idx, li].numpy())
                    y_list.append(float(m.get("exact_match", 0.0) >= 1.0))
                    doc_ids.append(m.get("doc_id", i))

                if len(X_list) < 20:
                    continue

                X = np.array(X_list)
                y = np.array(y_list)

                # Split by doc_id
                unique_docs = list(set(doc_ids))
                rng.shuffle(unique_docs)
                n_test = max(1, int(len(unique_docs) * args.test_frac))
                test_docs = set(unique_docs[:n_test])

                train_mask = np.array([d not in test_docs for d in doc_ids])
                test_mask = ~train_mask

                if train_mask.sum() < 10 or test_mask.sum() < 5:
                    continue
                if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
                    continue

                X_tr = torch.tensor(X[train_mask], dtype=torch.float32)
                y_tr = torch.tensor(y[train_mask], dtype=torch.float32)
                X_te = torch.tensor(X[test_mask], dtype=torch.float32)
                y_te = torch.tensor(y[test_mask], dtype=torch.float32)

                # Linear probe
                linear = LinearProbe(X.shape[1])
                lin_metrics = train_probe(linear, X_tr, y_tr, X_te, y_te, device,
                                          args.epochs, args.lr, args.batch_size)

                # MLP probe
                mlp = MLPProbe(X.shape[1])
                mlp_metrics = train_probe(mlp, X_tr, y_tr, X_te, y_te, device,
                                          args.epochs, args.lr, args.batch_size)

                result = {
                    "condition": cond,
                    "layer": layer,
                    "step": step,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "linear_auc": lin_metrics["auc"],
                    "linear_acc": lin_metrics["accuracy"],
                    "mlp_auc": mlp_metrics["auc"],
                    "mlp_acc": mlp_metrics["accuracy"],
                }
                all_results.append(result)
                print(f"  {cond} L{layer} S{step}: "
                      f"linear={lin_metrics['auc']:.3f} mlp={mlp_metrics['auc']:.3f}")

    # Save results
    out_file = out_dir / "probe_results.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_results)} results -> {out_file}")

    # Plot
    _plot_probe_comparison(all_results, out_dir)


def _plot_probe_comparison(results: List[Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not results:
        return

    conditions = sorted(set(r["condition"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(conditions), 2)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for ci, cond in enumerate(conditions):
        cond_results = [r for r in results if r["condition"] == cond]
        steps = sorted(set(r["step"] for r in cond_results))
        step_aucs = []
        for s in steps:
            step_r = [r for r in cond_results if r["step"] == s]
            best_auc = max(r["mlp_auc"] for r in step_r) if step_r else 0.5
            step_aucs.append(best_auc)
        ax.plot(steps, step_aucs, "o-", color=colors[ci], label=cond, linewidth=2)

    ax.axhline(y=0.5, linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Probe AUC (best layer)", fontsize=12)
    ax.set_title("Probe AUC by Condition", fontsize=13)
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "probe_auc_comparison.png", dpi=180)
    plt.close()


def _plot_unified_comparison(results: List[Dict], out_dir: Path) -> None:
    """Plot unified probe AUC: original vs corrected vs control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    conditions = sorted(set(r["condition"] for r in results))
    cmap = {
        "original": "#2c3e50",
        "corrected_k1": "#3498db", "corrected_k2": "#2980b9",
        "corrected_k3": "#1abc9c", "corrected_k4": "#16a085",
        "control_k1": "#e74c3c", "control_k2": "#c0392b",
        "control_k3": "#e67e22", "control_k4": "#d35400",
    }

    # Find best layer (highest original AUC)
    orig_by_layer = {}
    for r in results:
        if r["condition"] == "original":
            orig_by_layer.setdefault(r["layer"], []).append(r["unified_auc"])
    best_layer = max(orig_by_layer, key=lambda l: np.mean(orig_by_layer[l]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in conditions:
        cr = sorted([r for r in results if r["condition"] == cond and r["layer"] == best_layer],
                     key=lambda x: x["step"])
        if not cr:
            continue
        xs = [r["step"] for r in cr]
        ys = [r["unified_auc"] for r in cr]
        ax.plot(xs, ys, "o-", color=cmap.get(cond, "gray"), label=cond, linewidth=2,
                markersize=5, alpha=0.85 if "control" not in cond else 0.5)

    ax.axhline(y=0.5, linestyle="--", color="gray", alpha=0.4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Unified Probe AUC (frozen from original)", fontsize=12)
    ax.set_title(f"Unified Probe Transfer — Layer {best_layer}", fontsize=13)
    ax.set_ylim(0.35, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_unified_auc_vs_step.png", dpi=180)
    plt.savefig(out_dir / "fig5_unified_auc_vs_step.pdf")
    plt.close()
    print("  fig5_unified_auc_vs_step done")


if __name__ == "__main__":
    main()
