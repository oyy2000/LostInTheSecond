#!/usr/bin/env python3
"""
Diff-in-diff probe analysis (Route B) — GPU-accelerated, v2.

Supports both corrected (expect Delta > 0) and corrupted (expect Delta < 0).

For paired (original, modified) trajectories, compute:
  Delta = (AUC_late|modified - AUC_late|original)
        - (AUC_early|modified - AUC_early|original)

Usage:
    python scripts/eval/diff_in_diff_probe.py \
        --hs-dir results/gsm8k_7b_v2/hidden_states \
        --out-dir results/gsm8k_7b_v2/diff_in_diff \
        --modified-condition corrected_k2

    python scripts/eval/diff_in_diff_probe.py \
        --hs-dir results/gsm8k_7b_v2/hidden_states \
        --out-dir results/gsm8k_7b_v2/diff_in_diff \
        --modified-condition corrupted_k2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs-dir", default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/hidden_states"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/diff_in_diff"))
    ap.add_argument("--modified-condition", default="corrected_k2",
                    help="corrected_k{1-4} or corrupted_k{1-4}")
    ap.add_argument("--early-steps", default="1,2")
    ap.add_argument("--late-steps", default="3,4,5,6")
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_hs_and_meta(hs_dir: Path, condition: str):
    hs_file = hs_dir / f"hs_{condition}.pt"
    meta_file = hs_dir / f"meta_{condition}.jsonl"
    if not hs_file.exists():
        return None, None, None
    data = torch.load(hs_file, map_location="cpu", weights_only=False)
    meta = []
    if meta_file.exists():
        with meta_file.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta.append(json.loads(line))
    return data["hidden_states"], meta, data["layer_indices"]


def gpu_probe_auc(X: torch.Tensor, y: torch.Tensor, device: torch.device,
                  epochs: int = 20, lr: float = 1e-3) -> float:
    if len(X) < 10 or y.sum() < 2 or (1 - y).sum() < 2:
        return 0.5
    dim = X.shape[1]
    model = nn.Linear(dim, 1).to(device)
    X, y = X.to(device), y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    n = len(X)
    perm = torch.randperm(n)
    split = int(n * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]
    dataset = TensorDataset(X[train_idx], y[train_idx])
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb).squeeze(-1), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X[test_idx]).squeeze(-1)).cpu().numpy()
        yt = y[test_idx].cpu().numpy()
    if len(np.unique(yt)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(yt, probs))
    except Exception:
        return 0.5


def main():
    args = parse_args()
    hs_dir = Path(args.hs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    early_steps = [int(x) - 1 for x in args.early_steps.split(",")]
    late_steps = [int(x) - 1 for x in args.late_steps.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hs_orig, meta_orig, layer_indices = load_hs_and_meta(hs_dir, "original")
    hs_mod, meta_mod, _ = load_hs_and_meta(hs_dir, args.modified_condition)

    if hs_orig is None or hs_mod is None:
        print("Missing data"); sys.exit(1)

    # Build paired index
    mod_map = {}
    for i, m in enumerate(meta_mod):
        mod_map[(m["doc_id"], m.get("sample_idx", 0))] = i

    pairs = []
    for i, m in enumerate(meta_orig):
        key = (m["doc_id"], m.get("sample_idx", 0))
        if key in mod_map:
            pairs.append((i, mod_map[key]))

    print(f"Paired samples: {len(pairs)}")
    if len(pairs) < 20:
        print("Too few pairs"); sys.exit(1)

    rng = np.random.RandomState(args.seed)
    results = []

    for li, layer in enumerate(layer_indices):
        print(f"\n--- Layer {layer} ---")

        def collect_data(hs_list, meta_list, indices, step_indices):
            X_list, y_list = [], []
            for idx in indices:
                hs = hs_list[idx]
                label = float(meta_list[idx].get("exact_match", 0.0) >= 1.0)
                for si in step_indices:
                    if si < hs.shape[0]:
                        X_list.append(hs[si, li].numpy())
                        y_list.append(label)
            if not X_list:
                return None, None
            return torch.tensor(np.array(X_list), dtype=torch.float32), \
                   torch.tensor(np.array(y_list), dtype=torch.float32)

        orig_indices = [p[0] for p in pairs]
        mod_indices = [p[1] for p in pairs]

        # Observed AUCs
        X_early_orig, y_early_orig = collect_data(hs_orig, meta_orig, orig_indices, early_steps)
        X_late_orig, y_late_orig = collect_data(hs_orig, meta_orig, orig_indices, late_steps)
        X_early_mod, y_early_mod = collect_data(hs_mod, meta_mod, mod_indices, early_steps)
        X_late_mod, y_late_mod = collect_data(hs_mod, meta_mod, mod_indices, late_steps)

        if any(x is None for x in [X_early_orig, X_late_orig, X_early_mod, X_late_mod]):
            continue

        auc_early_orig = gpu_probe_auc(X_early_orig, y_early_orig, device)
        auc_late_orig = gpu_probe_auc(X_late_orig, y_late_orig, device)
        auc_early_mod = gpu_probe_auc(X_early_mod, y_early_mod, device)
        auc_late_mod = gpu_probe_auc(X_late_mod, y_late_mod, device)

        observed_delta = (auc_late_mod - auc_late_orig) - (auc_early_mod - auc_early_orig)

        # Permutation test
        n_pairs = len(pairs)
        perm_deltas = []
        for _ in range(args.n_permutations):
            swap = rng.random(n_pairs) < 0.5
            perm_orig_idx = []
            perm_mod_idx = []
            for pi, (oi, mi) in enumerate(pairs):
                if swap[pi]:
                    perm_orig_idx.append(mi)
                    perm_mod_idx.append(oi)
                else:
                    perm_orig_idx.append(oi)
                    perm_mod_idx.append(mi)

            # For swapped pairs, we need to use the right hs_list
            # Simpler: pool all data and permute labels
            X_e_o, y_e_o = collect_data(hs_orig, meta_orig,
                                         [pairs[i][0] if not swap[i] else pairs[i][0] for i in range(n_pairs)],
                                         early_steps)
            X_l_o, y_l_o = collect_data(hs_orig, meta_orig,
                                         [pairs[i][0] if not swap[i] else pairs[i][0] for i in range(n_pairs)],
                                         late_steps)
            X_e_m, y_e_m = collect_data(hs_mod, meta_mod,
                                         [pairs[i][1] if not swap[i] else pairs[i][1] for i in range(n_pairs)],
                                         early_steps)
            X_l_m, y_l_m = collect_data(hs_mod, meta_mod,
                                         [pairs[i][1] if not swap[i] else pairs[i][1] for i in range(n_pairs)],
                                         late_steps)

            if any(x is None for x in [X_e_o, X_l_o, X_e_m, X_l_m]):
                continue

            a1 = gpu_probe_auc(X_e_o, y_e_o, device)
            a2 = gpu_probe_auc(X_l_o, y_l_o, device)
            a3 = gpu_probe_auc(X_e_m, y_e_m, device)
            a4 = gpu_probe_auc(X_l_m, y_l_m, device)
            perm_deltas.append((a4 - a2) - (a3 - a1))

        if perm_deltas:
            p_value = float(np.mean(np.array(perm_deltas) >= observed_delta))
        else:
            p_value = 1.0

        result = {
            "layer": layer,
            "modified_condition": args.modified_condition,
            "early_steps": [s + 1 for s in early_steps],
            "late_steps": [s + 1 for s in late_steps],
            "auc_early_orig": auc_early_orig,
            "auc_late_orig": auc_late_orig,
            "auc_early_mod": auc_early_mod,
            "auc_late_mod": auc_late_mod,
            "observed_delta": observed_delta,
            "p_value": p_value,
            "n_pairs": n_pairs,
        }
        results.append(result)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  Layer {layer}: Delta={observed_delta:.4f}, p={p_value:.3f} {sig}")

    # Save
    out_file = out_dir / f"diff_in_diff_{args.modified_condition}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults -> {out_file}")

    # Heatmaps
    _plot_heatmaps(hs_dir, out_dir, args.modified_condition)


def _plot_heatmaps(hs_dir: Path, out_dir: Path, modified_condition: str) -> None:
    """Plot separability heatmaps from probe results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    probe_file = hs_dir.parent / "probes" / "probe_results.jsonl"
    if not probe_file.exists():
        return

    results = []
    with probe_file.open() as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    for cond in ["original", modified_condition]:
        cond_r = [r for r in results if r["condition"] == cond]
        if not cond_r:
            continue

        layers = sorted(set(r["layer"] for r in cond_r))
        steps = sorted(set(r["step"] for r in cond_r))
        mat = np.full((len(layers), len(steps)), np.nan)

        for r in cond_r:
            li = layers.index(r["layer"])
            si = steps.index(r["step"])
            mat[li, si] = r["mlp_auc"]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0, origin="lower")
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_title(f"Probe AUC — {cond}")
        plt.colorbar(im, ax=ax, label="AUC")
        plt.tight_layout()
        plt.savefig(out_dir / f"separability_heatmap_{cond}.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    main()
