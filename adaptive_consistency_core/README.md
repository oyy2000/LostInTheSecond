# Adaptive Consistency Core

This folder contains a small standalone version of the reusable part of
Pranjal2041/AdaptiveConsistency, with `numpy` and `scipy` kept as dependencies.

Core idea:

1. Sample an LLM multiple times for the same input.
2. Extract a final, hashable answer from each sample.
3. Majority vote over extracted answers.
4. Stop early when the current winner is confident enough.

The default `BetaStoppingCriteria` matches the lightweight criterion used by
Adaptive Consistency: compare the top answer count against the runner-up count,
then stop when `P(p_top > 0.5)` under a Beta posterior exceeds the threshold.

Dependencies:

```bash
pip install numpy scipy
```

## Quick Start

```python
from adaptive_consistency import AdaptiveConsistency, BetaStoppingCriteria


def extract_final_answer(text):
    return text.strip().split()[-1]


def sample_once(prompt):
    # Call your model here.
    return "The answer is 42"


ac = AdaptiveConsistency(
    max_samples=40,
    stopping_criteria=BetaStoppingCriteria(confidence=0.95),
    answer_key=extract_final_answer,
)

result = ac.run(sample_once, "Solve the problem.")
print(result.winner, result.num_samples, result.confidence, result.counts)
```

You can also select a criterion by string:

```python
ac = AdaptiveConsistency(max_samples=40, stopping_criteria="dirichlet")
```

Run the example:

```bash
python adaptive_consistency_core/example_usage.py
```

## Practical Notes

- `answer_key` is important: use it to normalize raw generations into final
  answers such as `"A"`, `"42"`, or a canonical math expression.
- `result.answers` keeps raw model outputs.
- `result.winner` is the majority answer after normalization.
- `result.num_samples` tells you how many model calls were used.
- `result.stopped_early` tells you whether the beta criterion stopped before
  hitting `max_samples`.
- Available criteria: `beta`, `dirichlet`, `majority`, `entropy`, `random`,
  and `never`.
