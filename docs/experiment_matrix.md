# Experiment Matrix

## Overview

Two-phase experiment design:

1. **Motivation Validation** -- verify the core hypothesis across models and datasets.
2. **Method Comparison** -- apply our method and compare against baselines on a broader benchmark suite.

Baselines:

| Abbreviation | Paper | Method |
|---|---|---|
| Full SC | Wang et al., 2022 | Self-Consistency (fixed sample budget, majority vote) |
| ASC | Aggarwal et al., 2023 (2305.11860) | Adaptive-Consistency (confidence-threshold early stopping) |
| ESC | Li et al., 2024 (2401.10480) | Early-Stopping Self-Consistency (window-based stopping criterion) |
| DSC | Wang et al., 2024 (2408.13457) | Difficulty-Adaptive Self-Consistency (prior + posterior difficulty allocation) |
| RASC | Wan et al., 2024 (2408.17017) | Reasoning-Aware Self-Consistency (rationale quality + answer consistency stopping) |

---

## Phase 1: Motivation Validation

Goal: demonstrate that the motivation (e.g., cascade error patterns, recovery potential) holds across different model scales and reasoning domains.

| Dataset \ Model | Llama 3.2 3B | Qwen 2.5 3B | Qwen 2.5 7B |
|---|:---:|:---:|:---:|
| HotpotQA | | | |
| MATH 500 | | | |
| GSM8K | | | |

---

## Phase 2: Method Comparison

Goal: compare our method against Full SC, ASC, ESC, DSC, RASC on accuracy and sample efficiency.

### Llama 3.2 3B

| Dataset | Full SC | ASC | ESC | DSC | RASC | Ours |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| HotpotQA | | | | | | |
| MATH 500 | | | | | | |
| GSM8K | | | | | | |
| AIME 2024 | | | | | | |
| AMC 2023 | | | | | | |
| OlympiadBench | | | | | | |
| HumanEval | | | | | | |
| CSQA | | | | | | |

### Qwen 2.5 3B

| Dataset | Full SC | ASC | ESC | DSC | RASC | Ours |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| HotpotQA | | | | | | |
| MATH 500 | | | | | | |
| GSM8K | | | | | | |
| AIME 2024 | | | | | | |
| AMC 2023 | | | | | | |
| OlympiadBench | | | | | | |
| HumanEval | | | | | | |
| CSQA | | | | | | |

---

## Metrics

Each cell should report:

- **Accuracy** (or pass@1 for code tasks)
- **Avg Samples** used per question
- **Cost Ratio** relative to Full SC (optional, for efficiency comparison)

---

## Dataset Summary

| Dataset | Domain | Metric | Notes |
|---|---|---|---|
| GSM8K | Arithmetic reasoning | Accuracy | Grade-school math |
| MATH 500 | Competition math | Accuracy | 500-problem subset |
| AIME 2024 | Competition math | Accuracy | AME Invitational Exam |
| AMC 2023 | Competition math | Accuracy | AMC 10/12 |
| OlympiadBench | Competition math | Accuracy | Olympiad-level problems |
| HotpotQA | Multi-hop QA | F1 / EM | Open-domain reasoning |
| HumanEval | Code generation | pass@1 | Function-level synthesis |
| CSQA | Commonsense reasoning | Accuracy | CommonsenseQA |

---

## Running Experiments

Phase 1 and Phase 2 use separate scripts, both backed by the shared engine in `src/sweep_engine.py`.

K = total sample budget per question (drafts + suffixes). K=2 with n_drafts=1 means 1 draft + 1 suffix (1 extra sample).

```bash
# --- Phase 1: Motivation Validation ---

# single run
python scripts/15_0_motivation_sweep.py \
    --dataset gsm8k --model-id Qwen/Qwen2.5-3B-Instruct --gpus 0,1,2,3,4,5,6,7

# all 3 models x 3 datasets
python scripts/15_0_motivation_sweep.py --batch --gpus 0,1,2,3,4,5,6,7

# --- Phase 2: Method Comparison ---

# single run
python scripts/15_1_method_sweep.py \
    --dataset gsm8k --model-id Qwen/Qwen2.5-3B-Instruct --gpus 0,1,2,3,4,5,6,7

# all 2 models x 8 datasets
python scripts/15_1_method_sweep.py --batch --gpus 0,1,2,3,4,5,6,7
```

Config: `configs/sweep_default.yaml`
Reusable modules: `src/sweep_datasets.py`, `src/prompt_templates.py`, `src/sweep_engine.py`
