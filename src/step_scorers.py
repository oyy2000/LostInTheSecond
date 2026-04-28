"""
Step-level scoring for state-aware late rollback.

Three scorer backends:
  - EntropyScorer: per-step token entropy from vLLM prompt_logprobs
  - ProbeScorer: linear probe on hidden states at step boundaries
  - CompositeScorer: weighted combination of entropy + probe + stall

Public API:
  - select_rollback_point(scores, beta, method) -> int
  - entropy_from_logprobs(token_logprobs, token_offsets, step_boundaries) -> List[float]
"""

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Rollback point selection
# ---------------------------------------------------------------------------

def select_rollback_point(
    step_scores: List[float],
    beta: float = 0.5,
    method: str = "argmax",
    tau_threshold: float = 0.0,
    L_consecutive: int = 2,
) -> int:
    """Pick the rollback step in the late region of a draft.

    Parameters
    ----------
    step_scores : list of float
        Per-step suspicion scores (higher = more suspicious).
    beta : float
        Only consider steps at position t/T >= beta.
    method : str
        "argmax"  -- step with highest score in late region.
        "max_drop" -- step with largest score increase from t-1 to t.
        "first_consecutive" -- first run of L_consecutive steps above tau_threshold.
    tau_threshold : float
        Threshold for the "first_consecutive" method.
    L_consecutive : int
        Run length for the "first_consecutive" method.

    Returns
    -------
    int
        0-indexed step index. Rollback keeps steps[:t*] as prefix.
        Falls back to ceil(0.8 * T) if no candidate is found.
    """
    T = len(step_scores)
    if T <= 1:
        return 0

    start = max(1, int(math.ceil(beta * T)))
    candidates = list(range(start, T))
    if not candidates:
        return max(1, T - 1)

    if method == "argmax":
        best = max(candidates, key=lambda t: step_scores[t])
        return best

    if method == "max_drop":
        best, best_drop = candidates[0], -float("inf")
        for t in candidates:
            drop = step_scores[t] - step_scores[t - 1]
            if drop > best_drop:
                best_drop = drop
                best = t
        return best

    if method == "first_consecutive":
        run = 0
        for t in candidates:
            if step_scores[t] >= tau_threshold:
                run += 1
                if run >= L_consecutive:
                    return t - L_consecutive + 1
            else:
                run = 0
        # fallback: argmax
        return max(candidates, key=lambda t: step_scores[t])

    raise ValueError(f"Unknown rollback method: {method}")


# ---------------------------------------------------------------------------
# Entropy scorer (operates on pre-computed logprobs)
# ---------------------------------------------------------------------------

def entropy_from_logprobs(
    token_logprobs: List[float],
    token_offsets: List[int],
    step_boundaries: List[Tuple[int, int]],
) -> List[float]:
    """Compute mean token entropy per step from vLLM prompt_logprobs.

    Uses H(token) = -p * log(p) where p = exp(logprob).
    Returns one entropy value per step.
    """
    scores: List[float] = []
    ti = 0
    for s_start, s_end in step_boundaries:
        while ti < len(token_offsets) and token_offsets[ti] < s_start:
            ti += 1
        step_hs: List[float] = []
        scan = ti
        while scan < len(token_offsets) and token_offsets[scan] < s_end:
            lp = token_logprobs[scan]
            if lp < 0:
                p = math.exp(lp)
                step_hs.append(-p * lp)
            else:
                step_hs.append(0.0)
            scan += 1
        scores.append(sum(step_hs) / len(step_hs) if step_hs else 0.0)
    return scores


def neglogprob_from_logprobs(
    token_logprobs: List[float],
    token_offsets: List[int],
    step_boundaries: List[Tuple[int, int]],
) -> List[float]:
    """Mean negative log-probability per step."""
    scores: List[float] = []
    ti = 0
    for s_start, s_end in step_boundaries:
        while ti < len(token_offsets) and token_offsets[ti] < s_start:
            ti += 1
        step_lps: List[float] = []
        scan = ti
        while scan < len(token_offsets) and token_offsets[scan] < s_end:
            step_lps.append(token_logprobs[scan])
            scan += 1
        scores.append(-sum(step_lps) / len(step_lps) if step_lps else 0.0)
    return scores


class EntropyScorer:
    """Score steps by token entropy from pre-computed logprobs."""

    def __init__(self, metric: str = "entropy"):
        self.metric = metric  # "entropy" or "neglogprob"

    def score_draft(
        self,
        token_logprobs: List[float],
        token_offsets: List[int],
        step_boundaries: List[Tuple[int, int]],
    ) -> List[float]:
        if self.metric == "neglogprob":
            return neglogprob_from_logprobs(
                token_logprobs, token_offsets, step_boundaries,
            )
        return entropy_from_logprobs(
            token_logprobs, token_offsets, step_boundaries,
        )


# ---------------------------------------------------------------------------
# Probe scorer (operates on pre-extracted hidden states)
# ---------------------------------------------------------------------------

class ProbeScorer:
    """Score steps using a pre-trained linear probe on hidden states."""

    def __init__(self, probe_path: str):
        with open(probe_path, "rb") as f:
            ckpt = pickle.load(f)
        self.probe = ckpt["probe"]       # sklearn LogisticRegression
        self.layer_idx = ckpt["layer_idx"]
        self.scaler = ckpt.get("scaler")  # optional StandardScaler

    def score_draft(
        self,
        hidden_states: "np.ndarray",
    ) -> List[float]:
        """Score each step from hidden states.

        Parameters
        ----------
        hidden_states : ndarray of shape (n_steps, hidden_dim)
            Hidden states at step boundaries for the selected layer.

        Returns
        -------
        list of float
            Risk probability per step (higher = more likely error).
        """
        X = hidden_states
        if self.scaler is not None:
            X = self.scaler.transform(X)
        probs = self.probe.predict_proba(X)[:, 1]
        return probs.tolist()


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

class CompositeScorer:
    """Weighted combination of entropy, probe risk, and stall signal."""

    def __init__(
        self,
        w_entropy: float = 0.4,
        w_probe: float = 0.4,
        w_stall: float = 0.2,
    ):
        self.w_entropy = w_entropy
        self.w_probe = w_probe
        self.w_stall = w_stall

    @staticmethod
    def _normalize(vals: List[float]) -> List[float]:
        if not vals:
            return vals
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng < 1e-12:
            return [0.5] * len(vals)
        return [(v - lo) / rng for v in vals]

    def score_draft(
        self,
        entropy_scores: List[float],
        probe_scores: Optional[List[float]],
        stall_flags: List[float],
    ) -> List[float]:
        n = len(entropy_scores)
        e_norm = self._normalize(entropy_scores)
        p_norm = self._normalize(probe_scores) if probe_scores else [0.0] * n
        combined = []
        for i in range(n):
            s = (self.w_entropy * e_norm[i]
                 + self.w_probe * p_norm[i]
                 + self.w_stall * stall_flags[i])
            combined.append(s)
        return combined


# ---------------------------------------------------------------------------
# Utility: step char boundaries (shared with delimiter_entropy.py)
# ---------------------------------------------------------------------------

def step_char_boundaries(
    response: str, steps: List[str],
) -> List[Tuple[int, int]]:
    """Return (char_start, char_end) for each step in the response text."""
    boundaries: List[Tuple[int, int]] = []
    pos = 0
    for step in steps:
        idx = response.find(step, pos)
        if idx < 0:
            idx = pos
        boundaries.append((idx, idx + len(step)))
        pos = idx + len(step)
    return boundaries
