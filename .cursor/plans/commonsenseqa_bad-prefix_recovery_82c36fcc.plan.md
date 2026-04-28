---
name: CommonsenseQA Bad-Prefix Recovery
overview: Replicate the "bad prefix recovery" motivation experiment (GSM8K/MATH500) on tau/commonsense_qa, adapting the 5-phase pipeline for multiple-choice commonsense reasoning instead of math.
todos:
  - id: csqa-answer
    content: Create src/csqa_answer_equiv.py -- extract_choice_letter() and is_choice_correct()
    status: completed
  - id: csqa-judge
    content: Add build_first_error_prompt_commonsense() to src/step_judge.py
    status: completed
  - id: phase0
    content: Create scripts/10_0_csqa_sample_multi_cot.py -- sample 8 CoT per CSQA question
    status: completed
  - id: phase1
    content: Create scripts/10_1_csqa_find_first_error.py -- GPT first-error locator
    status: completed
  - id: phase2
    content: Create scripts/10_2_csqa_bad_prefix_recovery.py -- bad-prefix continuation
    status: completed
  - id: phase3
    content: Create scripts/10_3_csqa_minimal_repair.py -- minimal repair continuation
    status: completed
  - id: phase45
    content: Create scripts/10_4_csqa_compare.py and 10_5_csqa_fine_relpos.py -- analysis + figures
    status: completed
  - id: runner
    content: Create scripts/10_run_all.sh and run the full pipeline
    status: in_progress
isProject: false
---

# CommonsenseQA Bad-Prefix Recovery Experiment

## Key Differences from Math Pipeline

CommonsenseQA is a 5-way multiple-choice task (A-E), not free-form math. This changes:

- **Answer equivalence**: trivial letter comparison (`pred.strip().upper() == gold.strip().upper()`) instead of LaTeX `is_math_equiv`
- **Answer extraction**: extract the chosen letter from CoT output (regex for "answer is (A|B|C|D|E)" or similar patterns) instead of `extract_boxed_answer`
- **System prompt**: "Think step by step, then give your answer as a single letter (A, B, C, D, or E)." instead of `\boxed{}`
- **Step-error judge prompt**: commonsense reasoning verifier instead of math arithmetic verifier -- check logical consistency, factual accuracy, relevance to the question
- **Dataset loading**: HuggingFace `tau/commonsense_qa` validation split (1221 questions) instead of MATH-500 JSONL
- **Step splitting**: same `split_steps` with `double_newline` mode should work (CoT structure is similar)

## Pipeline (5 scripts, mirroring 9_0 through 9_5)

### Phase 0: `scripts/10_0_csqa_sample_multi_cot.py`
- Load `tau/commonsense_qa` validation split via HuggingFace `datasets`
- Format each question as: `Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nE. ...\n`
- System prompt: step-by-step + answer as single letter
- Sample 8 CoT trajectories per question, multi-GPU parallel (file-based logging)
- Extract predicted letter, compare to `answerKey`, filter `min_steps >= 4`
- Output: `results/csqa_3b_multi_sample/raw_cot_n8.jsonl`

### Phase 1: `scripts/10_1_csqa_find_first_error.py`
- New GPT judge prompt in [src/step_judge.py](scripts/../src/step_judge.py) -- add `build_first_error_prompt_commonsense()`:
  - "You are a rigorous commonsense reasoning verifier"
  - Check each step for: logical validity, factual accuracy, relevance to the question, consistency with prior steps
  - A step is INCORRECT if it makes a factually wrong claim, draws an invalid logical inference, or introduces irrelevant reasoning that leads to the wrong answer
- Bucket into early/late by median tau/N split
- Output: `results/csqa_3b_multi_sample/first_error/`

### Phase 2: `scripts/10_2_csqa_bad_prefix_recovery.py`
- Same structure as [9_2](scripts/9_2_bad_prefix_natural_recovery.py): keep prefix up to error step tau, generate 32 continuations
- Use letter extraction instead of `extract_boxed_answer`
- Output: `results/csqa_3b_multi_sample/bad_prefix_recovery/continuations.jsonl`

### Phase 3: `scripts/10_3_csqa_minimal_repair.py`
- Same structure as [9_3](scripts/9_3_minimal_repair_continuation.py): use GPT correction to replace error step, then continue 32 times
- Output: `results/csqa_3b_multi_sample/minimal_repair/continuations.jsonl`

### Phase 4+5: `scripts/10_4_csqa_compare.py` and `scripts/10_5_csqa_fine_relpos.py`
- Identical analysis logic to [9_4](scripts/9_4_compare_bad_vs_repair.py) and [9_5](scripts/9_5_fine_grained_relpos.py)
- Figures saved to `figures/csqa_bad_prefix_recovery/`

## Shared Code Changes

- `src/step_judge.py`: add `build_first_error_prompt_commonsense(question, choices, gold_letter, steps)` function
- New helper: `src/csqa_answer_equiv.py` with `extract_choice_letter(text)` and `is_choice_correct(pred, gold)`

## Runner Script

- `scripts/10_run_all.sh`: sequential phases, `conda activate fact_yang`, GPUs 0-7

## Key Parameters (same as MATH500)
- Model: `Qwen/Qwen2.5-3B-Instruct`
- n_samples: 8, n_continuations: 32
- max_model_len: 2048, gpu_memory_utilization: 0.95
- GPT model: gpt-5.1 for step-error locator
