# Unified Self-Consistency Interface

This module gives one interface for four related cost-efficient reasoning
methods:

- `ac`: Adaptive Consistency / ASC from `Pranjal2041/AdaptiveConsistency`
- `esc`: Early-stopping Self-Consistency from `Yiwei98/ESC`
- `dsc`: Difficulty-Adaptive Self-Consistency from `WangXinglin/DSC`
- `rasc`: Reasoning-Aware Self-Consistency from `wan19990901/RASC`

It also includes `sc` as the fixed-budget self-consistency baseline.

## Core API

```python
from sc_unified import run_consistency, extract_last_number

result = run_consistency(
    "ac",
    generations=generations,
    answer_extractor=extract_last_number,
    max_samples=40,
)

print(result.answer)
print(result.samples_used)
print(result.counts)
print(result.stop_reason)
```

Every method returns `ConsistencyResult`:

- `answer`: selected final answer
- `generations`: consumed generations
- `counts`: answer vote counts
- `samples_used`: number of samples consumed
- `stopped_early`: whether it stopped before `max_samples`
- `stop_reason`: method-specific stopping reason
- `confidence`: beta confidence or RASC selected score when available
- `selected_rationale`: best rationale for RASC
- `diagnostics`: extra method metadata

## Input Shape

Use raw strings plus an answer extractor:

```python
run_consistency(
    "esc",
    generations=raw_outputs,
    answer_extractor=extract_last_number,
    window_size=5,
    max_samples=40,
)
```

Or use structured generations:

```python
from sc_unified import Generation

generations = [
    Generation(text="...", answer="42", score=0.8, rationale="..."),
]
```

For online sampling, pass `sampler` instead of `generations`:

```python
result = run_consistency(
    "ac",
    sampler=lambda: call_model(prompt),
    answer_extractor=extract_last_number,
    max_samples=40,
)
```

## Method Mapping

`sc` consumes all samples and majority-votes.

`ac` samples one at a time and stops when the beta posterior confidence that
the current winner beats the runner-up reaches `confidence`.

`esc` samples in windows and stops when all answers in the latest window agree.

`dsc` supports the two DSC pieces:

- prior difficulty: set `prior_easy=True` to use one greedy sample
- posterior difficulty: otherwise it runs windowed beta stopping with
  `confidence` and `hard_confidence`

In the DSC paper/repo, the prior difficulty signal is produced by a separate
question-ranking/evaluation step. This interface expects you to pass that
decision as `prior_easy=True/False`; it does not call an LLM to rank question
difficulty for you.

`rasc` keeps high-quality rationales with `score >= rasc_threshold`, stops when
the buffer reaches `rasc_buffer_size`, then weighted-votes by score. Provide
scores through `Generation(score=...)` or pass a `score_fn`.

In the RASC repo, scores come from feature extraction plus a logistic scorer.
This interface consumes those scores directly so your experiment loop can use
any scorer you prefer.

## Example

```bash
PYTHONPATH=. python sc_unified/example_usage.py
```
