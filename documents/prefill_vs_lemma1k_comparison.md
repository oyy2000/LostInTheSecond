# Prefill vs LEMMA 1K Comparison: Llama-3-8B-Base

**Model**: `meta-llama/Meta-Llama-3-8B` (base, non-instruct)
**Eval Tasks**: GSM8K (1319 test, zero-shot CoT) + MATH-500 (500 test)
**Date**: 2026-03-17 20:14

## Baseline: GSM8K EM = **0.10841546626231995** | MATH-500 EM = **0.044**

## LoRA Results

### LoRA Training Parameters

| Param | LEMMA 1K | Prefill 245 | Prefill 50 |
|---|---|---|---|
| N samples | 1000 | 245 | 50 |
| LR | 5e-06 | 5e-06 | 5e-06 |
| Epochs | 3 | 5 | 10 |
| r / alpha | 4 / 4 | 4 / 4 | 4 / 4 |
| Dropout | 0.05 | 0.05 | 0.1 |
| GA steps | 16 | 8 | 4 |
| Warmup | 0.03 | 0.03 | 0.05 |

### LoRA Evaluation Results

| Dataset | N | GSM8K EM | GSM8K Delta | MATH-500 EM | MATH Delta |
|---|---|---|---|---|---|
| LEMMA 1K | 1000 | 10.99% | +0.15% | 6.00% | +1.60% |
| Prefill 245 | 245 | 12.05% | +1.21% | 5.20% | +0.80% |
| Prefill 50 | 50 | 11.14% | +0.30% | 4.60% | +0.20% |

## Full FT Results

### Full FT Training Parameters

| Param | LEMMA 1K | Prefill 245 | Prefill 50 |
|---|---|---|---|
| N samples | 1000 | 245 | 50 |
| LR | 1e-06 | 1e-06 | 1e-06 |
| Epochs | 3 | 5 | 10 |
| Weight Decay | 0.05 | 0.1 | 0.1 |
| Warmup | 0.1 | 0.1 | 0.2 |
| GA steps | 16 | 8 | 4 |
| Optim | adafactor | adafactor | adafactor |

### Full FT Evaluation Results

| Dataset | N | GSM8K EM | GSM8K Delta | MATH-500 EM | MATH Delta |
|---|---|---|---|---|---|
| LEMMA 1K | 1000 | 14.25% | +3.41% | 6.00% | +1.60% |
| Prefill 245 | 245 | 17.97% | +7.13% | 4.60% | +0.20% |
| Prefill 50 | 50 | 16.98% | +6.14% | 6.00% | +1.60% |

## Summary

| Method | Dataset | N | GSM8K EM | MATH-500 EM |
|---|---|---|---|---|
| Baseline | - | - | 10.84% | 4.40% |
| LoRA | LEMMA 1K | 1000 | 10.99% | 6.00% |
| LoRA | Prefill 245 | 245 | 12.05% | 5.20% |
| LoRA | Prefill 50 | 50 | 11.14% | 4.60% |
| Full FT | LEMMA 1K | 1000 | 14.25% | 6.00% |
| Full FT | Prefill 245 | 245 | 17.97% | 4.60% |
| Full FT | Prefill 50 | 50 | 16.98% | 6.00% |

---

*Generated: 2026-03-17 20:14*
