# Rebuttal Experiments

## Qwen Same-Draft NLL/PRM Controls

The same-environment signal experiment reuses the exact 8,000 seed-42
Qwen2.5-3B drafts evaluated by the position controls. It runs the
original experiment-6_8 `nll_drop_fb_last` and `prm_drop_fb_last`
definitions with thresholds `0.2` and `0.1`, respectively. The NLL scorer
uses a 4,096-token context, matching the source generation environment;
the old 2,048-token scorer would truncate 93 MATH-500 drafts and three
closed-book HotpotQA drafts.

```bash
sbatch rebuttal/slurm/6_11_qwen_math500_nll_prm_c49.slurm
sbatch rebuttal/slurm/6_14_qwen_hotpotqa_open_all_controls_c49.slurm
```

The `hotpotqa_open` job first creates the context-conditioned shared
random-control checkpoint, then runs NLL/PRM against those exact drafts on
the same node and GPUs. Each dataset produces 8,000 PRM records, 8,000
NLL records, 32,000
suffixes per signal, 16,000 audited rollback assignments, and seven result
rows. Results are saved under:

```text
rebuttal/results/qwen2.5_3b_instruct_nll_prm_controls/seed_42/
  math500/
  hotpotqa_open/
```

The reported `tokens_per_q` is directly comparable with the corrected
random control: each original draft is counted once and contributes one
answer, followed by `ns-1` rollback suffix answers. Thus every row has
exactly `nd + nd*(ns-1) = 32` answers. `signal_tokens_per_q` and
`tokens_per_q_with_signal` additionally expose scoring overhead.

After both runs pass their audits, build the joint reviewer figure:

```bash
python rebuttal/scripts/6_13_build_qwen_control_comparison.py \
  --dataset math500
python rebuttal/scripts/6_13_build_qwen_control_comparison.py \
  --dataset hotpotqa_open
```

## Corrected Qwen MATH-500 Five-Seed Sweep

The corrected sweep assigns a separate deterministic vLLM seed to every
generation request. The seed is derived from the run seed, task type,
document ID, sample index, and suffix strategy fields. This prevents
identical prompts in SC@32 from being reset to the same random state while
preserving reproducibility.

The five configured run seeds are `42`, `123`, `456`, `789`, and `1024`.
Submit the sequential five-element Slurm array:

```bash
sbatch rebuttal/slurm/6_10_qwen_math500_five_seed_c49.slurm
```

When c49 is occupied, the SC pool can be prefetched in parallel on five
single-GPU 16GB nodes. The full c49 run resumes from these checkpoints:

```bash
sbatch rebuttal/slurm/6_10_qwen_math500_sc_prefill_4060ti.slurm
```

After those five array tasks pass, aggregate the corrected SC@32 result with
`scripts/6_10_run_qwen_math500_five_seed.py --aggregate-sc-only`. The
reviewer-facing result is
`math500_five_seed_sc_summary.json` under the result root.

Each array task uses three GPUs for one seed and validates generation counts,
the 32-sample SC pool, and per-request seed uniqueness. Results are written
under:

```text
rebuttal/results/qwen2.5_3b_instruct_math500_five_seed_corrected/
  seed_<seed>/math500/
    checkpoint.jsonl
    eval_summary.json
    seed_validation.json
  math500_five_seed_summary.json
```

The experiment settings are in
`configs/6_10_qwen_math500_five_seed.json`; the runnable entrypoint is
`scripts/6_10_run_qwen_math500_five_seed.py`.

## DeepSeek-R1-7B-Qwen: NLL/PRM versus SC@32

The configuration
`configs/6_8_deepseek_r1_7b_qwen.json` runs 500 MATH-500 questions
and a deterministic 500-example HotpotQA subset. Both datasets use
`nll_drop_fb_last` and `prm_drop_fb_last`, with `(nd, ns)` equal to
`(4, 8)`, `(8, 4)`, and `(16, 2)`. SC@32 is generated and evaluated
from the same sampling pool.

Submit one c31 job per dataset:

```bash
sbatch --job-name=rebuttal_ds7b_math500 \
  --export=ALL,DATASET=math500 \
  rebuttal/slurm/6_8_deepseek_r1_7b_qwen_c31.slurm

sbatch --job-name=rebuttal_ds7b_hotpotqa \
  --export=ALL,DATASET=hotpotqa \
  rebuttal/slurm/6_8_deepseek_r1_7b_qwen_c31.slurm
```

An independent c30 launcher is also available. It uses a separate
configuration and result root, so c30 and c31 can run concurrently without
sharing checkpoints or shard files:

```bash
sbatch --job-name=rebuttal_ds7b_math500_c30 \
  --export=ALL,DATASET=math500 \
  rebuttal/slurm/6_8_deepseek_r1_7b_qwen_c30.slurm

sbatch --job-name=rebuttal_ds7b_hotpotqa_c30 \
  --export=ALL,DATASET=hotpotqa \
  rebuttal/slurm/6_8_deepseek_r1_7b_qwen_c30.slurm
```

Each job runs generation, PRM scoring, NLL scoring, both rollback
strategies, SC@32, tokenizer-based fair cost recounting, comparison
plotting, and a strict completeness validator. Final artifacts are under:

```text
rebuttal/results/deepseek_r1_distill_qwen_7b_budget_multisignal/<node>/seed_42/<dataset>/
  eval_summary.json
  comparison_summary.json
  comparison_table.md
  rebuttal_summary.md
  signal_vs_sc32.pdf
  signal_vs_sc32.png
  validation_report.json
```

## Random Rollback

This directory contains the reviewer-requested random rollback baseline
adapted from `scripts/6_8_budget_controlled_multisignal.py`.

For each draft with `n` reasoning steps, the baseline uniformly samples a
retained-prefix length from `0` through `n - 1`. The sample is deterministic
for `(random_seed, doc_id, draft_idx)`. All other settings match experiment
6_8: temperature `0.7`, top-p `0.95`, budget `nd * ns = 32`, and
majority-vote evaluation.

The experiment reuses drafts and self-consistency results from the matching
6_8 checkpoint. New random suffix generations, token recounts, summaries,
logs, and figures are written under `rebuttal/`.

## Run

```bash
sbatch --export=ALL,DATASET=aime2025,RANDOM_SEED=42 \
  rebuttal/slurm/6_8_random_llama_c49.slurm
```

The final reviewer-facing artifacts are:

```text
rebuttal/results/llama_3.2_3b_instruct_random/seed_42/<dataset>/
  comparison_summary.json
  comparison_table.md
  rebuttal_summary.md
  random_comparison.pdf
  random_comparison.png
```

## Completed Runs

The seed-42 runs completed successfully on three RTX 5000 Ada GPUs:

- MATH-500: 8,000 reused drafts and 32,000 new random suffixes.
- AIME 2025: 480 reused drafts and 1,920 new random suffixes.

For MATH-500, PRM rollback exceeds random rollback by 8.2, 6.0, and
5.4 accuracy percentage points for `nd=4`, `nd=8`, and `nd=16`,
respectively. See
`results/llama_3.2_3b_instruct_random/seed_42/math500/rebuttal_summary.md`
for the compact comparison.

## Qwen Position Controls

`scripts/6_9_qwen_position_controls.py` evaluates Qwen2.5-3B on
MATH-500 and HotpotQA with uniform-random rollback, fixed-position
rollback at 25%, 50%, and 75% of reasoning steps, and final-step-only
resampling. The final-step control retains all complete split reasoning
steps before the last one, then regenerates the entire last step. Each
dataset uses 500 questions, 16 shared drafts per
question, and the `nd * ns = 32` answer budget. Each vote contains the
`nd` original draft answers and `nd * (ns - 1)` suffix answers. Thus the
`nd=4`, `nd=8`, and `nd=16` settings add 7, 3, and 1 suffixes per draft,
respectively.

The fixed-position target is step `ceil(fraction * n_steps)`; all steps
strictly before that target are retained. HotpotQA reports both exact
match and token F1.

```bash
sbatch rebuttal/slurm/6_9_qwen_math500_c49.slurm
sbatch rebuttal/slurm/6_9_qwen_hotpotqa_c32.slurm
```

Outputs are saved under:

```text
rebuttal/results/qwen2.5_3b_instruct_position_controls_final_step/seed_42/
  math500/
  hotpotqa/
```

The corrected MATH-500 recomputation that reuses the completed legacy
checkpoint is saved separately under:

```text
rebuttal/results/qwen2.5_3b_instruct_position_controls_draft_plus_suffix/
  seed_42/math500/
```

Audit either completed dataset with:

```bash
python rebuttal/scripts/6_10_audit_qwen_position_controls.py \
  --result-dir \
  rebuttal/results/qwen2.5_3b_instruct_position_controls/seed_42/hotpotqa
```

Each completed result directory contains:

```text
checkpoint.jsonl
rollback_assignments.jsonl
eval_summary.json
eval_summary_table.md
position_controls.pdf
position_controls.png
phase_times.json
audit_report.json
```

The HotpotQA run completed in `1:07:31` on c32. Its audit verified
8,000 drafts, 160,000 suffixes (32,000 per strategy), 40,000 rollback
assignments, and 16 evaluation rows. The best exact-match and F1 pair
was fixed-position 25% with `nd=8`, `ns=4`: EM `0.1880` and F1
`0.2762`.

The MATH-500 run completed in `2:23:13` on c49 with the same verified
record counts. Its best accuracy was `0.7200`, tied by random rollback
with `nd=8`, fixed-position 25% with `nd=8`, and fixed-position 75%
with `nd=16`. Among those tied rows, fixed-position 75% with `nd=16`
used the fewest generated tokens per question (`12,554.8`), followed
by random rollback with `nd=8` (`12,829.2`).

## Qwen NLL/PRM Threshold Sweep

The threshold sweep stores non-suffix records once and keeps one
suffix-only delta checkpoint per signal threshold. NLL and PRM are
evaluated independently, so the output does not create physical
checkpoints for the NLL-by-PRM Cartesian product:

```text
rebuttal/results/qwen2.5_3b_instruct_nll_prm_threshold_sweep/seed_42/
  <dataset>/
    _shared_seed42/checkpoint.jsonl
    signals/
      nll_0p1/checkpoint.jsonl
      nll_0p2/checkpoint.jsonl
      prm_0p05/checkpoint.jsonl
      prm_0p1/checkpoint.jsonl
    checkpoint_layout.json
    eval_summary.json
    eval_summary_table.md
```

The pipeline reads the shared checkpoint as an immutable
`--checkpoint-part`; newly generated records are written only to the
active signal directory. Run or resume both datasets with:

```bash
python -u rebuttal/scripts/6_17_run_nll_prm_threshold_sweep.py \
  --config rebuttal/configs/6_17_qwen_nll_prm_threshold_sweep.json
```

Use `--reevaluate-only` to recompute tables from the checkpoint union
or `--aggregate-only` to rebuild only the dataset-level summaries.

Each newly executed worker records request-level timing events. The
pipeline publishes:

```text
timing/task_events.jsonl
timing/per_question.jsonl
timing/per_question_summary.json
```

For each question and signal, `per_question.jsonl` reports draft
generation, signal-specific suffix generation, their sum as total
generation, and NLL or PRM signal-computation time. Each value is the
union of active request intervals, so concurrent requests are counted
once rather than summed repeatedly. One-time model loading is excluded
from this per-question comparison and remains available in the worker
timing sidecars. The scope is all checkpoint tasks needed to support
the configured `nd` values, with duplicate task identities counted
only once.

## Qwen Five-Seed Random/NLL/PRM Variance

The variance experiment evaluates MATH-500 and open-book HotpotQA with
run seeds `42`, `123`, `456`, `789`, and `1024`. The dataset seed stays
fixed at `42`; both generation and uniform-random rollback use the run
seed. This keeps the evaluation questions fixed while measuring sampling
variance. Every generation request receives a deterministic content-derived
seed based on the run seed and request identity, so the 16 drafts and
multiple suffix samples remain independent while the run stays reproducible.

Each seed first produces 8,000 shared drafts and the random rollback
control. NLL and PRM then score those exact drafts and use the same
`nd * ns = 32` draft-plus-suffix answer budget. Experiment settings are
stored in `configs/6_15_qwen_seed_variance.json`. The formal variance
run uses the smaller thresholds `NLL=0.1` and `PRM=0.095`, selected from
the audited seed-42 threshold trials.

Submit the ten independent dataset/seed runs, followed by the aggregate
job:

```bash
ARRAY_JOB=$(sbatch --parsable \
  rebuttal/slurm/6_15_qwen_seed_variance_array.slurm)
sbatch --dependency="afterok:${ARRAY_JOB}" \
  rebuttal/slurm/6_16_summarize_qwen_seed_variance.slurm
```

Inside an existing three-GPU allocation, the same array tasks can be
run sequentially while sharing C49 with the resident workload:

```bash
set -e
for task_id in {0..9}; do
  SLURM_ARRAY_TASK_ID="${task_id}" \
    GPU_FREE_THRESHOLD_MIB=12000 \
    bash rebuttal/slurm/6_15_qwen_seed_variance_array.slurm
done
```

Each run is strictly audited before its `COMPLETE` marker is written.
Aggregate artifacts include per-seed values, sample variance, sample
standard deviation, and a bar chart with standard-deviation error bars:

```text
rebuttal/results/qwen2.5_3b_instruct_seed_variance/
  seed_<seed>/<dataset>/
    position_controls/
    nll_prm_controls/
    COMPLETE
  aggregate/
    per_seed_results.json
    per_seed_results.csv
    seed_variance_summary.json
    seed_variance_summary.csv
    seed_variance.pdf
    seed_variance.png
```
