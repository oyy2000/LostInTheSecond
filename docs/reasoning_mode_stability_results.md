# Reasoning Mode Stability — Experimental Results

**Date**: 2026-03-24
**Status**: Exp 0, 3, 4, 5 complete; Exp 1-2 partial (0.5B, 1.5B done; 3B running; 7B pending); Exp 6 pending

---

## 1. Exp 0: Multi-Scale Baseline + PRM Profiling

**Setup**: MATH-500 benchmark, 500 samples per model, PRM = Qwen2.5-Math-PRM-7B, greedy decoding, step splitting = auto.

### 1.1 Instruct Models

| Model | Accuracy | N_correct | N_wrong | Dip Depth (wrong) | Min Score (wrong) | Min Step | Recovery Step | Stability Ratio | Step-1 Score |
|-------|----------|-----------|---------|--------------------|--------------------|----------|---------------|-----------------|--------------|
| 0.5B  | 24.6%    | 123       | 377     | 0.3592             | 0.5646             | 6        | 26            | 0.8686          | 0.9238       |
| 1.5B  | 40.2%    | 201       | 299     | 0.2525             | 0.6811             | 7        | 18            | 0.8754          | 0.9336       |
| 3B    | 63.6%    | 318       | 182     | 0.1959             | 0.7299             | 4        | 12            | 0.9890          | 0.9258       |
| 7B    | 72.8%    | 364       | 136     | 0.1996             | 0.7470             | 8        | 12            | 0.9458          | 0.9467       |

**Correct samples** (for reference):

| Model | Dip Depth | Min Score | Min Step | Recovery Step | Stability Ratio | Step-1 Score |
|-------|-----------|-----------|----------|---------------|-----------------|--------------|
| 0.5B  | 0.0622    | 0.9140    | 4        | 5             | 1.0076          | 0.9763       |
| 1.5B  | 0.0378    | 0.9449    | 5        | 6             | 1.0041          | 0.9826       |
| 3B    | 0.0209    | 0.9648    | 4        | 5             | 0.9843          | 0.9857       |
| 7B    | 0.0130    | 0.9773    | 5        | 6             | 0.9975          | 0.9903       |

### 1.2 Key Finding: "Dip-then-Recovery" Pattern Confirmed

The data strongly supports the core hypothesis:

1. **Wrong answers exhibit a clear dip**: PRM scores start high (step-1 ≈ 0.92-0.95), drop sharply in early steps, then partially recover. Dip depth ranges from 0.36 (0.5B) to 0.20 (7B).

2. **Correct answers show almost no dip**: Dip depth is 0.01-0.06 for correct samples — the model "locks in" to the right mode immediately.

3. **Dip decreases with scale**: 0.359 → 0.253 → 0.196 → 0.200. The trend is monotonically decreasing from 0.5B to 3B, with a plateau from 3B to 7B.

4. **Recovery gets faster**: Recovery step drops from 26 (0.5B) to 12 (3B, 7B) — larger models stabilize in fewer steps.

### 1.3 Base Models (Exp 0 / Exp 6 data)

PRM data generated for 0.5B-base, 1.5B-base, 3B-base, 7B-base. Stored in `runs/multi_scale_prm/prm_{tag}.json`. Combined analysis pending (scheduled in `run_overnight.sh` Phase 4).

---

## 2. Exp 5: Scaling Law for Mode Stability

### 2.1 Power-Law Fits

| Metric | Formula | R² | Exponent |
|--------|---------|-----|----------|
| Dip Depth (wrong) | `69.70 × N^(-0.264)` | 0.925 | -0.264 |
| Accuracy | `1.34×10⁻⁴ × N^(0.381)` | 0.932 | 0.381 |
| Min Score (wrong) | `0.077 × N^(0.102)` | 0.893 | 0.102 |
| Stability Ratio | `0.375 × N^(0.042)` | 0.550 | 0.042 |

### 2.2 Key Finding: Distinct Scaling Exponent

**Mode stability scales differently from accuracy.** The dip depth scaling exponent (α = 0.264) is meaningfully different from the accuracy exponent (α = 0.381). This suggests reasoning mode stability is not simply a byproduct of accuracy improvement but a distinct phenomenon with its own scaling behavior.

### 2.3 Extrapolation

Using the power-law fit for dip depth:
- **Dip disappearance threshold**: 0.02
- **Estimated parameters needed**: ~25,445B (25.4 trillion)
- This implies the reasoning mode instability will persist even for very large models, though diminished.

---

## 3. Exp 3: Hidden-State Mode Analysis

### 3.1 Setup
- Extract hidden states at step boundaries using `output_hidden_states=True`
- Probe at early/mid/late layers per model
- No steering vectors available → norms-only analysis

### 3.2 Data Collected

| Model | Correct | Wrong | Probe Layers | OOM Skips | Notes |
|-------|---------|-------|--------------|-----------|-------|
| 0.5B  | 100     | 100   | [4, 12, 21]  | 0         | Full data |
| 1.5B  | 97      | 60    | [4, 14, 25]  | 43        | Partial due to OOM |
| 3B    | 100     | 92    | [6, 18, 31]  | 8         | Near-complete |
| 7B    | 100     | 98    | [4, 14, 24]  | Uncertain | Will be re-run with 2-GPU + max_seq_len=512 |

### 3.3 Plots Generated
- `mode_norms_{model}.png` — Hidden state L2 norms at step boundaries (correct vs wrong)
- `mode_projections_{model}.png` — Steering vector projections (empty without vectors)
- `cross_scale_mode_onset.png` — Cross-scale comparison of mode onset

---

## 4. Exp 4: Conditioned PRM Analysis

### 4.1 Analyses Completed

1. **Chain Length Conditioning**: PRM curves split by short/medium/long chains. Plots in `runs/conditioned_prm/chain_length/`.

2. **Correct-Wrong Divergence**: Measures at which step the PRM curves for correct and wrong answers diverge most. Results in `divergence/divergence_summary.json`.

3. **Step Statistics**: Per-step mean, std, and distribution of PRM scores. Results in `statistics/step_statistics.json`.

4. **Step-1 Predictiveness**: How well the step-1 PRM score predicts final correctness (binary classification). Results in `step1_predict/step1_predictiveness.json`.

---

## 5. Exp 1-2: Good-Prefix Injection (Partial)

### 5.1 Setup
- Take wrong-answer samples from Exp 0 (up to 200 per model)
- Three conditions:
  - `same_problem`: Inject first K correct steps from the model's own correct solution to the same problem (matched by doc_id, fallback to any correct)
  - `cross_problem`: Inject first K correct steps from a different problem
  - `no_prefix`: Generate from scratch (baseline)
- K ∈ {1, 2, 3} prefix steps
- Generate continuation with greedy decoding (max_new_tokens=1024)
- PRM-score the full chain

### 5.2 Completed Models

| Model | Status | File | Wrong Samples | Correct Available |
|-------|--------|------|---------------|-------------------|
| 0.5B  | Done   | `injection_0.5B.json` (6.4MB) | 200 | 123 |
| 1.5B  | Done   | `injection_1.5B.json` (6.2MB) | 200 | 201 |
| 3B    | Running | — | 182 | 318 |
| 7B    | Pending | — | 136 | 364 |

### 5.3 Pending
- 3B: Currently running via `run_parallel_exp1_3.sh` on GPU 7
- 7B: Will be run by `run_overnight.sh` with `--gpus 0,7` (2-GPU split to avoid OOM)
- Final cross-model aggregation plots will be generated after all models complete

---

## 6. Output File Locations

```
runs/multi_scale_prm/
  multi_scale_summary.json              # ← Main Exp 0 results
  prm_{model}.json                      # Per-model detailed PRM scores

runs/scaling_law_analysis/
  scaling_law_results.json              # ← Main Exp 5 results (fits + extrapolation)
  scaling_laws.png                      # Scaling law fit plots
  dip_vs_accuracy.png                   # Dip depth vs accuracy scatter

runs/hidden_state_mode/
  hidden_state_{model}.json             # Exp 3 per-model data
  plots/mode_norms_{model}.png          # Norm plots
  plots/cross_scale_mode_onset.png      # Cross-scale comparison

runs/conditioned_prm/
  divergence/divergence_summary.json    # Exp 4 divergence analysis
  step1_predict/step1_predictiveness.json  # Step-1 as predictor
  statistics/step_statistics.json       # Per-step statistics

runs/good_prefix_exp/
  injection_{model}.json                # Exp 1-2 per-model results
  plots/injection_{model}.png           # Per-model PRM curve comparison
  plots/injection_benefit_vs_scale.png  # Cross-scale injection benefit
```

---

## 7. Autonomous Runner Status

`run_overnight.sh` is running via `nohup` and will complete all remaining work:

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Wait for current Exp 1-2 (3B on GPU 7) | RUNNING |
| 2 | 7B Exp 1-2 with 2-GPU split (GPUs 0,7) | PENDING |
| 3 | Re-run Exp 3 7B (2 GPUs, max_seq_len=512) | PENDING |
| 4 | Exp 6: Base vs Instruct comparison plots | PENDING |
| 5 | Final aggregation (all Exp 1-5 plots) | PENDING |

**Monitor**: `tail -f runs/parallel_logs/overnight_runner.log`

---

## 8. Preliminary Conclusions

1. **Hypothesis supported**: The "dip-then-recovery" pattern in PRM scores for wrong answers is real and robust across model scales.

2. **Mode stability scales with model size** with a power-law exponent of -0.264 (R²=0.925), distinct from accuracy scaling (exponent 0.381).

3. **The dip is primarily a wrong-answer phenomenon**: Correct answers show negligible dip (0.01-0.06), suggesting the model's mode is stable when it "gets it right."

4. **Recovery improves with scale**: Larger models recover from the early instability in fewer steps (26 → 12).

5. **Pending validation**: Exp 1-2 results (prefix injection) will test the causal mechanism — whether injecting a "stable mode" via correct first steps eliminates the dip.
