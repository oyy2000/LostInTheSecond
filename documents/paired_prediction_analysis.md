# Paired Prediction Analysis

**Base Model EM**: 0.8347 (1319 test samples)
**Training data overlap**: 115 question IDs in train set
**Date**: 2026-03-12 15:03

## Summary Table

| Model | Method | Both OK | Both Wrong | Fixed | Broken | Net | Model EM | Fix in Train | Break in Train |
|-------|--------|---------|------------|-------|--------|-----|---------|--------------|----------------|
| attn_only | lora | 1039 | 141 | 77 | 62 | +15 | 0.8461 | 0 | 0 |
| v2_lr1e4_r4_a1x | lora | 1047 | 151 | 67 | 54 | +13 | 0.8446 | 0 | 0 |
| lr_1e-5 | lora | 1069 | 176 | 42 | 32 | +10 | 0.8423 | 0 | 1 |
| ft_lr5e6 | full_ft | 1063 | 165 | 53 | 38 | +15 | 0.8461 | 0 | 2 |
| ft_lr1e6 | full_ft | 1076 | 180 | 38 | 25 | +13 | 0.8446 | 0 | 1 |

## Interpretation

### attn_only (lora)

- Fixed 77 problems the base model got wrong
- Broke 62 problems the base model got right
- Net change: +15 correct answers
- 0/77 fixed problems overlap with training set questions
- 0/62 broken problems overlap with training set questions

### v2_lr1e4_r4_a1x (lora)

- Fixed 67 problems the base model got wrong
- Broke 54 problems the base model got right
- Net change: +13 correct answers
- 0/67 fixed problems overlap with training set questions
- 0/54 broken problems overlap with training set questions

### lr_1e-5 (lora)

- Fixed 42 problems the base model got wrong
- Broke 32 problems the base model got right
- Net change: +10 correct answers
- 0/42 fixed problems overlap with training set questions
- 1/32 broken problems overlap with training set questions

### ft_lr5e6 (full_ft)

- Fixed 53 problems the base model got wrong
- Broke 38 problems the base model got right
- Net change: +15 correct answers
- 0/53 fixed problems overlap with training set questions
- 2/38 broken problems overlap with training set questions

### ft_lr1e6 (full_ft)

- Fixed 38 problems the base model got wrong
- Broke 25 problems the base model got right
- Net change: +13 correct answers
- 0/38 fixed problems overlap with training set questions
- 1/25 broken problems overlap with training set questions

## Detailed Samples

See `artifacts/paired_analysis_details.jsonl` for full response snippets of all fixed and broken samples.

---
*Generated: 2026-03-12 15:03*