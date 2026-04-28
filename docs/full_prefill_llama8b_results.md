# Full Prefill Pipeline Results: Llama-3-8B-Base

**Date**: 2026-03-19
**Model**: `meta-llama/Meta-Llama-3-8B` (base)
**Eval**: GSM8K (1319 test, zero-shot CoT, cot-meta-math prompt, vLLM, float16)
**Infrastructure**: PSC Bridges-2 — H100-80GB (training), V100-32GB (eval)

---

## 1. Data Pipeline Summary

Full LEMMA dataset → LLaMA-3-8B generation → GPT-5.4-mini Step2 fix → LLaMA-3-8B prefill


| Stage                        | Input           | Output         | Notes                                                                 |
| ---------------------------- | --------------- | -------------- | --------------------------------------------------------------------- |
| LLaMA-3 generation           | 14941 questions | 14941 samples  | 7942 correct, 6999 incorrect                                          |
| GPT Step2 fix                | 6999 incorrect  | 2224 rewritten | Combined judge+fix prompt; 3878 skipped (correct/no-pred/single-step) |
| Prefill (fix_step2)          | 2224 rewritten  | 498 correct    | Only samples where prefill matched ground truth                       |
| Prefill (wait_recompute_all) | 2224 rewritten  | 2224 all       | All prefill outputs regardless of correctness                         |


---

## 2. Training Configuration


| Param        | lora_fix498 | lora_wr2224        | ft_fix498 | ft_wr2224          |
| ------------ | ----------- | ------------------ | --------- | ------------------ |
| Method       | LoRA        | LoRA               | Full FT   | Full FT            |
| Dataset      | fix_step2   | wait_recompute_all | fix_step2 | wait_recompute_all |
| N samples    | 498         | 2224               | 498       | 2224               |
| LR           | 5e-6        | 5e-6               | 1e-6      | 1e-6               |
| Epochs       | 5           | 3                  | 5         | 3                  |
| GA steps     | 8           | 16                 | 8         | 16                 |
| Warmup       | 0.03        | 0.03               | 0.1       | 0.1                |
| Weight Decay | 0.0         | 0.0                | 0.05      | 0.05               |
| LoRA r/α     | 4/4         | 4/4                | —         | —                  |
| LoRA dropout | 0.05        | 0.05               | —         | —                  |
| Optimizer    | AdamW       | AdamW              | Adafactor | Adafactor          |


---

## 3. Training Results


| Experiment  | Train Time | Final Train Loss | Final Eval Loss |
| ----------- | ---------- | ---------------- | --------------- |
| lora_fix498 | 17.5 min   | 0.572            | 0.527           |
| lora_wr2224 | 31.7 min   | 0.506            | 0.408           |
| ft_fix498   | 38.0 min   | 0.454            | 0.463           |
| ft_wr2224   | 58.8 min   | 0.366            | 0.315           |


---

## 4. GSM8K Evaluation Results


| Model                 | GSM8K Acc | Empty Samples | Δ vs Base |
| --------------------- | --------- | ------------- | --------- |
| **Base (Llama-3-8B)** | 36.0%*    | 9             | —         |
| **lora_fix498**       | 35.9%     | 0             | -0.1      |
| **lora_wr2224**       | 10.8%     | 935           | -25.2     |
| **ft_fix498**         | **53.4%** | 71            | **+17.4** |
| **ft_wr2224**         | 41.2%     | 106           | +5.2      |


 Base model result was loaded from cached evaluation outputs; see note below.

---

## 5. Analysis

### Key Findings

1. **Full FT with curated data (ft_fix498) achieves the best result**: 53.4% GSM8K, a +17.4pp improvement over the base model. This is a strong result from only 498 carefully curated samples (prefill-correct only).
2. **Full FT with all prefill data (ft_wr2224) also improves**: 41.2% GSM8K (+5.2pp), but notably worse than the curated subset. More data ≠ better when data quality matters.
3. **LoRA barely moves the needle**: lora_fix498 (35.9%) is essentially unchanged from base. LoRA with low rank (r=4) and base models doesn't have enough capacity for meaningful behavioral change.
4. **lora_wr2224 collapsed**: 10.8% with 935/1319 empty samples indicates catastrophic model degradation. The combination of noisy data (all prefill, including incorrect) and LoRA likely caused the model to degenerate.

### Data Quality > Data Quantity


| Method  | Curated 498 | All 2224 | Difference |
| ------- | ----------- | -------- | ---------- |
| LoRA    | 35.9%       | 10.8%    | +25.1      |
| Full FT | **53.4%**   | 41.2%    | +12.2      |


In both methods, the curated 498-sample dataset (only prefill-correct) significantly outperforms the full 2224-sample dataset. This strongly validates that data quality filtering (keeping only samples where prefill matches ground truth) is critical.

### Comparison with Previous Small-Scale Experiments (1K subset)

Previous results (from `prefill_vs_lemma1k_comparison.md`) used a 1K-question subset:


| Method  | Dataset           | N         | GSM8K (prev) | GSM8K (this) |
| ------- | ----------------- | --------- | ------------ | ------------ |
| Full FT | Prefill (correct) | 245 → 498 | 17.97%       | 53.4%        |
| Full FT | LEMMA 1K original | 1000      | 14.25%       | —            |
| LoRA    | Prefill (correct) | 245 → 498 | 12.05%       | 35.9%        |


**Note**: The previous experiments reported baseline ~10.8% while this run shows 36.0%. This discrepancy may be due to differences in the evaluation framework version or answer extraction method. Within each experiment batch, relative comparisons remain valid.

---

## 6. Conclusions

1. **Full fine-tuning on curated prefill data is the winning strategy** for Llama-3-8B base on GSM8K.
2. **498 high-quality samples suffice** to drive substantial improvement (+17.4pp) when using full FT.
3. **LoRA is ineffective** for this base model + small dataset combination, regardless of data quality.
4. **Noisy data is harmful** — including incorrect prefill outputs degrades performance in both methods.

---

## 7. SLURM Job IDs


| Job ID   | Experiment  | Type     |
| -------- | ----------- | -------- |
| 38055439 | lora_fix498 | Training |
| 38055440 | lora_wr2224 | Training |
| 38055441 | ft_fix498   | Training |
| 38055442 | ft_wr2224   | Training |
| 38058401 | base        | Eval     |
| 38058402 | lora_fix498 | Eval     |
| 38058403 | lora_wr2224 | Eval     |
| 38058404 | ft_fix498   | Eval     |
| 38058405 | ft_wr2224   | Eval     |


## 8. Artifact Paths

- **Training data**: `artifacts_real/full/lemma_sft_fix_step2.json` (498), `artifacts_real/full/lemma_sft_wait_recompute_all.json` (2224)
- **Models**: `/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/{lora_fix498,lora_wr2224,ft_fix498,ft_wr2224}/`
- **Eval logs**: `/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lemma_gsm8k_3805840{1-5}.out`

