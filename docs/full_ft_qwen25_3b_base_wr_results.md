# Full FT Sweep on Qwen/Qwen2.5-3B (Base) -- Wait+Recompute

**Model**: `Qwen/Qwen2.5-3B` (base, non-instruct)
**Dataset**: Wait+Recompute
**Method**: Full parameter fine-tuning (Adafactor optimizer)
**Eval Tasks**: GSM8K (1319 test) + MATH-500 (500 test)
**Date**: 2026-03-15 02:10

## Baseline: GSM8K EM = **0.731614859742229** | MATH-500 EM = **0.496**

## Full Results (Ranked by GSM8K EM)

| Rank | Name | GSM8K EM | GSM8K Delta | MATH-500 EM | MATH Delta | LR | Epochs | WD | Warmup |
|------|------|----------|-------------|-------------|------------|----|--------|-----|--------|
| 1 | ft2_wd005 | 0.7354 | +0.0038 | 0.5140 | +0.0180 | 1e-06 | 3.0 | 0.05 | 0.1 |
| 2 | ft2_wd0 | 0.7331 | +0.0015 | 0.5060 | +0.0100 | 1e-06 | 3.0 | 0.0 | 0.1 |
| 3 | ft2_wd0005 | 0.7331 | +0.0015 | 0.5060 | +0.0100 | 1e-06 | 3.0 | 0.005 | 0.1 |
| 4 | ft2_wd0_wu02 | 0.7331 | +0.0015 | 0.5160 | +0.0200 | 1e-06 | 3.0 | 0.0 | 0.2 |
| 5 | ft2_wu005 | 0.7331 | +0.0015 | 0.5060 | +0.0100 | 1e-06 | 3.0 | 0.01 | 0.05 |
| 6 | ft2_wu02 | 0.7331 | +0.0015 | 0.5160 | +0.0200 | 1e-06 | 3.0 | 0.01 | 0.2 |
| 7 | ft2_wu03 | 0.7331 | +0.0015 | 0.5060 | +0.0100 | 1e-06 | 3.0 | 0.01 | 0.3 |
| 8 | ft2_wu0 | 0.7316 | +0.0000 | 0.5020 | +0.0060 | 1e-06 | 3.0 | 0.01 | 0.0 |
| 9 | ft2_wd01 | 0.7278 | -0.0038 | 0.5160 | +0.0200 | 1e-06 | 3.0 | 0.1 | 0.1 |
| 10 | ft2_wd01_wu02 | 0.7278 | -0.0038 | 0.5100 | +0.0140 | 1e-06 | 3.0 | 0.1 | 0.2 |

## Key Findings

- **Best full FT**: `ft2_wd005` (GSM8K EM = 0.7354, delta = +0.0038)
- **7/10** experiments improved over base on GSM8K

---

*Generated: 2026-03-15 02:10*
