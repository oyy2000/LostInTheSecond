# Preliminary Experiment Plan — Lost in the Second

## Goal

Validate whether **early-step intervention** (via steering vectors) can reduce **error accumulation** in LLM math reasoning.

**Baseline model:** Qwen2.5 3B (pick one and keep consistent throughout)

* `Qwen/Qwen2.5-3B-Instruct` (general instruct)
* or `Qwen/Qwen2.5-Math-3B-Instruct` (math-tuned; usually stronger for MATH-500)

**Eval target:** MATH-500 (500 problems)

**Key idea:** Build contrastive pairs where the *only major difference* is whether the model **recovers from an early mistake** (specifically Step 2).

---

## Phase 0 — Reproducible Baseline (lm-evaluation-harness)

### 0.1 Pin the harness version

MATH-500 is included in newer harness releases (e.g., v0.4.9.2+). The CLI recently changed to `lm-eval run` (v0.4.10+). Keep your repo pinned to a specific commit/tag for stable results.

### 0.2 Confirm task name locally

Run:

```bash
lm-eval ls tasks | grep -i math
```

Use the exact name that shows up (commonly something like `math_500`).

### 0.3 Run baseline on MATH-500 and log samples

Example (adjust task name / CLI syntax to your installed version):

```bash
lm-eval run --model hf \
  --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=float16,device=cuda \
  --tasks math_500 \
  --batch_size 8 \
  --output_path ./runs/baseline_qwen25_3b_math500 \
  --log_samples --apply_chat_template
```

**Artifacts to save (must-have):**

* overall exact-match / accuracy
* per-sample generations (`--log_samples` output)
* generation settings (max tokens, temperature/top_p, etc.)

---

## Phase 1 — PRM Scoring + Plots (diagnose failure & pick “Step-2-wrong” subset)

### 1.1 Choose a PRM

Recommended: `Qwen/Qwen2.5-Math-PRM-7B` (or larger if you can afford it).

### 1.2 Decide what to score

You want two views:

1. **Outcome-level:** score the whole solution (cheap sanity check).
2. **Process-level:** score each step (the real signal).

**Step segmentation (make this stable):**

* In your eval prompt, force numbered steps:

  * “Answer with `Step 1: ...\nStep 2: ...\n...\nFinal Answer:`”
* Or post-hoc split by newlines + regex (`^Step\s*\d+:`).

### 1.3 Score and plot

For each sample:

* correctness (exact match)
* total tokens
* PRM summary stats

  * mean PRM
  * min PRM
  * **PRM at Step 2**
  * PRM drop: PRM(step2) − PRM(step1)

**Plots (baseline):**

* PRM distribution: correct vs wrong (hist)
* PRM(step2) vs correctness (box/violin)
* PRM(step2) vs length (scatter)
* calibration-style: PRM percentile → accuracy

### 1.4 Define “Step 2 wrong” operationally

You need a filter that’s stable and cheap.
Suggested heuristic stack:

* (H1) sample is overall wrong (exact match = 0)
* (H2) Step 2 has *low PRM* (e.g., bottom 20%)
* (H3) Optional: a quick judge confirms Step 2 is wrong

Output: a subset `S_step2_wrong` (e.g., 50–200 examples).

---

## Phase 2 — Build Contrastive Datasets (two variants)

You’ll build pairs for **vector extraction** (and optionally SFT later). All pairs should share:

* same question
* same formatting
* strongly matched length where possible

### Dataset 1: “Hard Correction” (explicitly fix Step 2)

For each example in `S_step2_wrong`:

* **neg_response:** the original model solution (contains the wrong Step 2 and continues)
* **pos_response:** *corrected* version where Step 2 is fixed (and downstream steps are consistent)

**How to generate pos_response:**
Use GPT-5.2 as a corrector:

* input: question + model’s full solution
* output: corrected solution
* constraint: keep Step 1 unchanged if possible; fix Step 2 + re-derive remaining steps

**Quality gate:**

* final answer must be correct (auto-check; if uncertain, use a verifier/judge)

### Dataset 2: “Self-Repair Trigger” (keep the mistake, then recover)

Goal: isolate the effect of adding a **repair transition** rather than rewriting everything.

For each example in `S_step2_wrong`:

* **neg_response (no repair):** original solution (wrong Step 2, continues wrong or inconsistent)
* **pos_response (repair added):** keep the original wrong Step 2 verbatim, then append:

  * `Wait, the previous step is wrong. Let’s recompute.`
  * provide the corrected Step 2
  * continue with correct downstream steps

This creates a clean contrast:

* both share the same early failure token pattern
* only pos exhibits *recognize error → transition → repair* behavior

### Output format for your current vector scripts

Write to `samples_alpha.json` (or `samples_math500_*.json`) with:

```json
{
  "samples": [
    {
      "doc": {"question": "...", "id": 123},
      "pos_response": "...",
      "neg_response": "...",
      "results": {"exact_match": 1.0}
    }
  ]
}
```

Note: Your `01_extract_vectors.py` filters `results.exact_match == 1.0`, so **only keep pairs where pos is verified correct**.

---

## Phase 3 — Extract Steering Vectors (your current pipeline)

### 3.1 Update model + layer range

* Set `MODEL_ID` in `01_extract_vectors.py` to your Qwen2.5-3B model.
* Update `LAYERS` to match Qwen’s `num_hidden_layers` (e.g., `list(range(N))`).

### 3.2 Two vector runs

* Vector A: trained on Dataset 1
* Vector B: trained on Dataset 2

### 3.3 Early-step focus (optional but aligned with your hypothesis)

Your current extraction uses `read_token_index=-1` (last token). For an early-step study, add one of these (in order of ease):

1. **Truncate responses** to end right after Step 2 / repair segment, then keep `read_token_index=-1`.
2. Insert a marker token after Step 2 (e.g., `<<STEP2_END>>`) and set `read_token_index` to that marker’s position (requires small code changes).

---

## Phase 4 — Evaluate Steering on MATH-500 (and analyze)

### 4.1 Update eval script

In `02_apply_vectors.py`:

* set `MODEL_ID` to Qwen2.5-3B
* set `TASKS` to MATH-500 task name (confirmed via `lm-eval ls tasks`)
* create a grid:

  * layers: early/mid/late (e.g., 1/3, 2/3, last few)
  * lambdas: small-to-moderate (+/−), e.g. `[0.25, 0.5, 1.0, 2.0, -0.5, -1.0]`

### 4.2 Primary metrics

* accuracy / exact match on MATH-500
* length (tokens) distribution
* PRM(step2) and PRM(min) shift after steering

### 4.3 “Early-step intervention” specific metrics

You want to show the mechanism, not only final accuracy:

* frequency of explicit repair phrases (“Wait… previous step is wrong…”) in generations
* PRM(step2) improvement on the subset `S_step2_wrong`
* reduction of cascading errors (judge whether later steps become more consistent)

---

## Minimal Deliverables (what you should have after prelim)

1. Baseline MATH-500 results for Qwen2.5-3B (+ logged generations)
2. PRM analysis plots showing a clear separation signal around Step 2
3. Two contrastive datasets (Dataset 1 & 2), each with verified pos correctness
4. Two steering vectors (A & B)
5. A small grid showing whether steering improves:

   * overall MATH-500 acc
   * and/or Step-2 recovery metrics on `S_step2_wrong`

---

## Risk Controls / Common Pitfalls

* **Answer extraction mismatch** (MATH is tricky): validate a few samples manually.
* **Length confound:** try to match pos/neg length or report length changes explicitly.
* **Over-editing in Dataset 1:** keep Step 1 identical if possible; otherwise the vector may learn “rewrite everything.”
* **Leakage via correction model:** keep correction prompts neutral; never include gold answer explicitly.

---

## Suggested Directory Layout

```
runs/
  baseline_qwen25_3b_math500/
  steered_vecA_Lx_lamy/
  steered_vecB_Lx_lamy/
artifacts/
  generations_baseline.jsonl
  prm_scores_baseline.parquet
  prm_plots/
  samples_math500_ds1.json
  samples_math500_ds2.json
vectors_out/
  vecA.pt
  vecB.pt
```

---

## Implemented scripts in this folder

### 1) Build Step-2 subset + DS1/DS2

```bash
python 00_build_contrastive_datasets.py \
  --runs-root ./runs/baseline_qwen25_3b_math500 \
  --score-model Qwen/Qwen2.5-Math-PRM-7B \
  --step-split-mode double_newline \
  --step2-bottom-percentile 0.9 \
  --max-subset 120
```

Outputs:

* `artifacts/prm_scores_baseline.csv`
* `artifacts/prm_plots/*.png`
* `artifacts/samples_math500_ds1.json`
* `artifacts/samples_math500_ds2.json`

### 2) Extract vector (run twice for DS1/DS2)

```bash
python 01_extract_vectors.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --data-file ./artifacts/samples_math500_ds1.json \
  --out-file ./vectors_out/vecA.pt \
  --layers auto \
  --max-samples 200

python 01_extract_vectors.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --data-file ./artifacts/samples_math500_ds2.json \
  --out-file ./vectors_out/vecB.pt \
  --layers auto \
  --max-samples 200
```

### 3) Apply vector on MATH-500 grid

```bash
python 02_apply_vectors.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --task hendrycks_math_500 \
  --vector-path ./vectors_out/vecA.pt \
  --layers auto \
  --lambdas 0.25,0.5,1.0,2.0,-0.5,-1.0 \
  --batch-size 32
```
