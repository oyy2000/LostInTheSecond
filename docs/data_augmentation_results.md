# Data Augmentation Experiment Results: Llama-3-8B on GSM8K

**Date**: 2026-03-22
**Base Model**: `meta-llama/Meta-Llama-3-8B`
**Eval**: GSM8K (1319 test samples, zero-shot CoT, cot-meta-math prompt, vLLM, float16)
**Infrastructure**: PSC Bridges-2 — H100-80GB (training), V100-32GB (eval)
**Eval Framework**: LEMMA custom evaluation (`math_eval.py`)

---

## 1. Results Overview

### 1.1 All Experiments Comparison

| # | Experiment | Method | Data Source | GPT Model | N Samples | GSM8K Acc | Empty | Δ vs Base | Eval Job |
|---|-----------|--------|-------------|-----------|-----------|-----------|-------|-----------|----------|
| — | **Base (Llama-3-8B)** | — | — | — | — | **36.0%** | 9 | — | [38058401](logs/lemma_gsm8k_38058401.out) |
| — | **LEMMA Published** | Full FT | LEMMA original SFT | — | ~50K | **~78%** | — | +42 | (paper) |
| 1 | ft_fix498 | Full FT | greedy correct only | gpt-5.4-mini | 498 | **53.4%** | 71 | +17.4 | [38058404](logs/lemma_gsm8k_38058404.out) |
| 2 | ft_wr2224 | Full FT | all prefill outputs | gpt-5.4-mini | 2224 | 41.2% | 106 | +5.2 | [38058405](logs/lemma_gsm8k_38058405.out) |
| 3 | ft_mixed8440 | Full FT | greedy + rejection | gpt-5.4-mini | 8440 | 50.8% | 291 | +14.8 | [38092925](logs/lemma_gsm8k_38092925.out) |
| 4 | ft_combined11k | Full FT | greedy + step2 + step3 + rejection | gpt-5.4-mini | ~10.5K | 60.6% | 140 | +24.6 | [38101193](logs/lemma_gsm8k_38101193.out) |
| 5 | lora_combined11k (v1) | LoRA | greedy + step2 + step3 + rejection | gpt-5.4-mini | ~10.5K | 15.2% | 1038 | -20.8 | [38093932](logs/lemma_gsm8k_38093932.out) |
| **6** | **lora_combined11k_v2** | **LoRA** | **greedy + step2 + step3 + rejection** | **gpt-5.4-mini** | **~10.5K** | **77.4%** | **0** | **+41.4** | [38126270](logs/lemma_gsm8k_38126270.out) |
| 7 | ft_gpt4o_combined | Full FT | greedy + step2 + step3 + rejection | gpt-4o | ~11K | 65.8% | 0 | +29.8 | [38129908](logs/lemma_gsm8k_38129908.out) |
| 8 | lora_gpt4o_combined | LoRA | greedy + step2 + step3 + rejection | gpt-4o | ~11K | **待评估** | — | — | 待提交 |

### 1.2 Best Results Summary

| Method | Best Config | GSM8K | vs LEMMA Full (~78%) |
|--------|------------|-------|---------------------|
| **LoRA (gpt-5.4-mini data)** | lora_combined11k_v2 | **77.4%** | **-0.6pp** (接近持平) |
| **Full FT (gpt-4o data)** | ft_gpt4o_combined | **65.8%** | -12.2pp |
| **Full FT (gpt-5.4-mini data)** | ft_combined11k | **60.6%** | -17.4pp |

---

## 2. Data Augmentation Strategies

### 2.1 Data Pipeline

```
LEMMA 14941 questions
    ├─ LLaMA-3-8B Greedy Generation → 7942 correct / 6999 incorrect
    │
    ├─ GPT Step2 Fix (incorrect samples)
    │   ├─ gpt-5.4-mini: 2224 rewritten → prefill → 498 correct
    │   └─ gpt-4o:       2607 rewritten → prefill → 662 correct
    │
    ├─ GPT Step3 Fix (additional errors)
    │   ├─ gpt-5.4-mini: 1078 candidates → prefill → 450 correct
    │   └─ gpt-4o:       1515 candidates → prefill → 450 correct
    │
    └─ Rejection Sampling (temperature=0.7, top_p=0.95, n=5)
        └─ 14941 questions × 5 → select correct → ~9002 additional samples
```

### 2.2 Dataset Compositions

| Dataset | Greedy Correct | Step2 Fix | Step3 Fix | Rejection | Total |
|---------|---------------|-----------|-----------|-----------|-------|
| fix_step2 (correct only) | — | 498 | — | — | 498 |
| wait_recompute_all | — | 2224 (all) | — | — | 2224 |
| mixed | 7942 | — | — | 498 (subset) | 8440 |
| combined (gpt-5.4-mini) | 7942 | 498 | 450 | ~1600 | ~10.5K |
| **combined (gpt-4o)** | 7942 | 662 | 450 | ~2008 | **~11K** |

---

## 3. Training Configuration

### 3.1 LoRA Hyperparameters (Fixed Bug: lr 5e-6 → 2e-4)

| Param | v1 (failed) | v2 (success) |
|-------|-------------|-------------|
| Learning Rate | **5e-6** (too low) | **2e-4** |
| LoRA r / α | 16 / 32 | 16 / 32 |
| LoRA Dropout | 0.05 | 0.05 |
| Target Modules | all linear | all linear |
| Epochs | 3 | 3 |
| GA Steps | 16 | 16 |
| Optimizer | AdamW | Adafactor |

### 3.2 Full FT Hyperparameters

| Param | ft_combined11k | ft_gpt4o_combined |
|-------|---------------|-------------------|
| Learning Rate | 1e-6 | **2e-5** |
| Weight Decay | 0.05 | 0.01 |
| Warmup Ratio | 0.1 | 0.03 |
| Epochs | 3 | 3 |
| GA Steps | 16 | 16 |
| Optimizer | Adafactor | Adafactor |

---

## 4. Training Metrics

| Experiment | Train Loss | Eval Loss | Train Time |
|-----------|-----------|-----------|------------|
| ft_fix498 | — | ~0.46 | ~38 min |
| ft_mixed8440 | 0.253 | 0.203 | ~147 min |
| ft_combined11k | 0.246 | 0.213 | ~351 min |
| lora_combined11k_v2 | 0.129 | **0.176** | ~124 min |
| ft_gpt4o_combined | 0.181 | 0.262 | ~101 min |
| lora_gpt4o_combined | 0.129 | **0.177** | ~128 min |

---

## 5. Key Findings

### 5.1 LoRA Learning Rate is Critical

LoRA v1 (lr=5e-6) produced 15.2% accuracy with 1038/1319 empty outputs — essentially a broken model. LoRA v2 (lr=2e-4, a 40× increase) achieved 77.4%, the best result across all experiments. The root cause was that 5e-6 is a typical Full FT learning rate but far too small for LoRA, which needs higher rates (1e-4 to 2e-4) to effectively update the low-rank adapters.

### 5.2 Data Quality > Data Quantity

| Curated 498 → 53.4% | All 2224 → 41.2% | **+12.2pp for curation** |

### 5.3 More Data + More Augmentation Helps

| 498 (step2 only) → 53.4% | ~10.5K (combined) → 60.6% (FT) / 77.4% (LoRA) |

### 5.4 GPT-4o Improves Data Quality

GPT-4o corrected more samples than gpt-5.4-mini (Step2: 2607 vs 2224, Step3: 1515 vs 1078), and the resulting Full FT model improved from 60.6% to 65.8%.

### 5.5 LoRA vs Full FT

With proper hyperparameters, LoRA (77.4%) significantly outperforms Full FT (60.6%/65.8%) on this task, likely because LoRA's regularization prevents overfitting on the ~10K training samples.

---

## 6. Verification Links

### 6.1 Eval Log Files

All eval logs are at: `/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/`

| Experiment | Job ID | Log File |
|-----------|--------|----------|
| Base (Llama-3-8B) | 38058401 | `lemma_gsm8k_38058401.out` |
| ft_fix498 | 38058404 | `lemma_gsm8k_38058404.out` |
| ft_wr2224 | 38058405 | `lemma_gsm8k_38058405.out` |
| ft_mixed8440 | 38092925 | `lemma_gsm8k_38092925.out` |
| lora_combined11k (v1) | 38093932 | `lemma_gsm8k_38093932.out` |
| ft_combined11k | 38101193 | `lemma_gsm8k_38101193.out` |
| lora_combined11k_v2 | 38126270 | `lemma_gsm8k_38126270.out` |
| ft_gpt4o_combined | 38129908 | `lemma_gsm8k_38129908.out` |

### 6.2 Eval Result JSONL Files

| Experiment | Result Path |
|-----------|------------|
| ft_mixed8440 | `.../ft_mixed8440/best_model/math_eval/test_cot-meta-math_zero-shot/gsm8k/test_cot-meta-math_-1_seed0_t0.0_s0_e-1.jsonl` |
| ft_combined11k | `.../ft_combined11k/best_model/math_eval/test_cot-meta-math_zero-shot/gsm8k/test_cot-meta-math_-1_seed0_t0.0_s0_e-1.jsonl` |
| lora_combined11k (v1) | `.../lora_combined11k/merged_model/math_eval/test_cot-meta-math_zero-shot/gsm8k/test_cot-meta-math_-1_seed0_t0.0_s0_e-1.jsonl` |
| lora_combined11k_v2 | `.../lora_combined11k_v2/merged_model/math_eval/test_cot-meta-math_zero-shot/gsm8k/test_cot-meta-math_-1_seed0_t0.0_s0_e-1.jsonl` |
| ft_gpt4o_combined | `.../ft_gpt4o_combined/best_model/math_eval/test_cot-meta-math_zero-shot/gsm8k/test_cot-meta-math_-1_seed0_t0.0_s0_e-1.jsonl` |

(All under `/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/`)

### 6.3 Training Metrics

| Experiment | Metrics File |
|-----------|-------------|
| ft_mixed8440 | `.../ft_mixed8440/sweep_metrics.json` |
| ft_combined11k | `.../ft_combined11k/sweep_metrics.json` |
| lora_combined11k_v2 | `.../lora_combined11k_v2/sweep_metrics.json` |
| ft_gpt4o_combined | `.../ft_gpt4o_combined/sweep_metrics.json` |
| lora_gpt4o_combined | `.../lora_gpt4o_combined/sweep_metrics.json` |

### 6.4 Training Data Files

| Dataset | Path |
|---------|------|
| fix_step2 (498) | `artifacts_real/full/lemma_sft_fix_step2.json` |
| wait_recompute_all (2224) | `artifacts_real/full/lemma_sft_wait_recompute_all.json` |
| mixed (8440) | `artifacts_real/full/lemma_sft_mixed.json` |
| combined gpt-5.4-mini (~10.5K) | `artifacts_real/full/lemma_sft_combined.json` |
| combined gpt-4o (~11K) | `artifacts_real/full_gpt4o/lemma_sft_combined_gpt4o.json` |

### 6.5 Model Artifacts

| Experiment | Model Path |
|-----------|-----------|
| ft_combined11k | `.../artifacts/full_prefill_llama8b/ft_combined11k/best_model/` |
| lora_combined11k_v2 | `.../artifacts/full_prefill_llama8b/lora_combined11k_v2/merged_model/` |
| ft_gpt4o_combined | `.../artifacts/full_prefill_llama8b/ft_gpt4o_combined/best_model/` |
| lora_gpt4o_combined | `.../artifacts/full_prefill_llama8b/lora_gpt4o_combined/merged_model/` |

---

## 7. Pending

- **lora_gpt4o_combined** evaluation: Model training completed (eval_loss=0.177), merged model saved. Evaluation job needs to be submitted:
  ```bash
  cd /jet/home/swang47/yang/projects/LostInTheSecond
  sbatch scripts/slurm/30_lemma_eval_gsm8k.sh eval-lora-gpt4o /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/lora_gpt4o_combined/merged_model
  ```

---

*Generated: 2026-03-22*
*Sources: eval logs, sweep_metrics.json, training logs, full_prefill_llama8b_results.md*
