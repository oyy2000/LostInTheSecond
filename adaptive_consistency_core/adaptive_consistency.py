"""Small Adaptive Consistency helper for reuse in experiments.

The main idea is early-stopped self-consistency:
sample answers repeatedly, keep a vote count over normalized final answers,
and stop once the current winner is likely enough to remain the winner.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Type

import numpy as np
from scipy import stats


AnswerKey = Callable[[Any], Hashable]


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    winner: Optional[Hashable]
    confidence: float
    counts: Dict[Hashable, int]
    criterion: str


@dataclass(frozen=True)
class ACResult:
    answers: List[Any]
    winner: Optional[Hashable]
    num_samples: int
    stopped_early: bool
    confidence: float
    counts: Dict[Hashable, int]


class StoppingCriteria:
    name = "base"

    def should_stop(self, answers: List[Hashable]) -> StopDecision:
        raise NotImplementedError


class BetaStoppingCriteria(StoppingCriteria):
    """Paper's lightweight default criterion, implemented with scipy.

    Only compares the current top answer against the runner-up. With a
    uniform Beta(1, 1) prior, counts a and b produce Beta(a + 1, b + 1).
    We stop when P(p_top > 0.5) reaches the chosen confidence threshold.
    """

    name = "beta"

    def __init__(self, confidence: float = 0.95, min_samples: int = 1) -> None:
        self.confidence = confidence
        self.min_samples = min_samples

    def should_stop(self, answers: List[Hashable]) -> StopDecision:
        counts = Counter(answers)
        if not counts:
            return StopDecision(False, None, 0.0, {}, self.name)

        top_two = counts.most_common(2)
        winner = top_two[0][0]
        top_count = top_two[0][1]
        runner_up_count = top_two[1][1] if len(top_two) > 1 else 0
        probability = beta_tail_at_half(top_count, runner_up_count)
        stop = len(answers) >= self.min_samples and probability >= self.confidence
        return StopDecision(stop, winner, probability, dict(counts), self.name)


class DirichletStoppingCriteria(StoppingCriteria):
    """Multi-answer posterior criterion.

    Samples from a Dirichlet posterior over the most common answer classes and
    estimates the probability that the current winner has the largest mass.
    For two classes it falls back to the faster beta criterion.
    """

    name = "dirichlet"

    def __init__(
        self,
        confidence: float = 0.95,
        top_k: int = 5,
        num_samples: int = 50_000,
        min_samples: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        self.confidence = confidence
        self.top_k = top_k
        self.num_samples = num_samples
        self.min_samples = min_samples
        self.rng = np.random.default_rng(seed)

    def should_stop(self, answers: List[Hashable]) -> StopDecision:
        counts = Counter(answers)
        if not counts:
            return StopDecision(False, None, 0.0, {}, self.name)
        if len(counts) < 3:
            beta_decision = BetaStoppingCriteria(
                confidence=self.confidence,
                min_samples=self.min_samples,
            ).should_stop(answers)
            return StopDecision(
                beta_decision.stop,
                beta_decision.winner,
                beta_decision.confidence,
                beta_decision.counts,
                self.name,
            )

        most_common = counts.most_common(self.top_k)
        labels = [label for label, _ in most_common]
        winner = labels[0]
        alphas = np.asarray([count + 1 for _, count in most_common], dtype=float)
        posterior_samples = self.rng.dirichlet(alphas, size=self.num_samples)
        winner_probability = float(np.mean(np.argmax(posterior_samples, axis=1) == 0))
        stop = len(answers) >= self.min_samples and winner_probability >= self.confidence
        return StopDecision(stop, winner, winner_probability, dict(counts), self.name)


class MajorityStoppingCriteria(StoppingCriteria):
    name = "majority"

    def __init__(self, threshold: float = 0.8, min_samples: int = 2) -> None:
        self.threshold = threshold
        self.min_samples = min_samples

    def should_stop(self, answers: List[Hashable]) -> StopDecision:
        counts = Counter(answers)
        if not counts:
            return StopDecision(False, None, 0.0, {}, self.name)

        winner, top_count = counts.most_common(1)[0]
        ratio = top_count / len(answers)
        stop = len(answers) >= self.min_samples and ratio >= self.threshold
        return StopDecision(stop, winner, ratio, dict(counts), self.name)


class EntropyStoppingCriteria(StoppingCriteria):
    name = "entropy"

    def __init__(self, threshold: float = 0.25, min_samples: int = 2) -> None:
        self.threshold = threshold
        self.min_samples = min_samples

    def should_stop(self, answers: List[Hashable]) -> StopDecision:
        counts = Counter(answers)
        if not counts:
            return StopDecision(False, None, 0.0, {}, self.name)

        winner = counts.most_common(1)[0][0]
        normalized_entropy = _normalized_entropy(counts.values())
        confidence = 1.0 - normalized_entropy
        stop = len(answers) >= self.min_samples and normalized_entropy <= self.threshold
        return StopDecision(stop, winner, confidence, dict(counts), self.name)


class RandomStoppingCriteria(StoppingCriteria):
    name = "random"

    def __init__(self, probability: float = 0.1, min_samples: int = 1) -> None:
        self.probability = probability
        self.min_samples = min_samples

    def should_stop(self, answers: List[Hashable]) -> StopDecision:
        counts = Counter(answers)
        if not counts:
            return StopDecision(False, None, 0.0, {}, self.name)

        winner = counts.most_common(1)[0][0]
        stop = (
            len(answers) >= self.min_samples
            and np.random.uniform(0.0, 1.0) < self.probability
        )
        return StopDecision(stop, winner, self.probability, dict(counts), self.name)


class NeverStoppingCriteria(StoppingCriteria):
    name = "never"

    def should_stop(self, answers: List[Hashable]) -> StopDecision:
        counts = Counter(answers)
        winner = counts.most_common(1)[0][0] if counts else None
        return StopDecision(False, winner, 0.0, dict(counts), self.name)


class AdaptiveConsistency:
    def __init__(
        self,
        max_samples: int = 40,
        stopping_criteria: str | Type[StoppingCriteria] | StoppingCriteria = "beta",
        answer_key: Optional[AnswerKey] = None,
    ) -> None:
        self.max_samples = max_samples
        self.stopping_criteria = make_stopping_criteria(stopping_criteria)
        self.answer_key = answer_key or (lambda answer: answer)

    def should_stop(
        self,
        answers: List[Any],
        return_decision: bool = False,
    ) -> bool | StopDecision:
        keyed_answers = self._key_answers(answers)
        decision = self.stopping_criteria.should_stop(keyed_answers)
        return decision if return_decision else decision.stop

    def run(self, sampler: Callable[..., Any], *args: Any, **kwargs: Any) -> ACResult:
        answers: List[Any] = []
        last_decision = StopDecision(False, None, 0.0, {}, self.stopping_criteria.name)

        for _ in range(self.max_samples):
            answers.append(sampler(*args, **kwargs))
            last_decision = self.should_stop(answers, return_decision=True)
            if last_decision.stop:
                break

        final_decision = self.should_stop(answers, return_decision=True)
        return ACResult(
            answers=answers,
            winner=final_decision.winner,
            num_samples=len(answers),
            stopped_early=last_decision.stop and len(answers) < self.max_samples,
            confidence=final_decision.confidence,
            counts=final_decision.counts,
        )

    def vote(self, answers: List[Any]) -> Optional[Hashable]:
        decision = self.should_stop(answers, return_decision=True)
        return decision.winner

    def _key_answers(self, answers: Iterable[Any]) -> List[Hashable]:
        return [self.answer_key(answer) for answer in answers]


AC = AdaptiveConsistency


CRITERIA = {
    "beta": BetaStoppingCriteria,
    "dirichlet": DirichletStoppingCriteria,
    "majority": MajorityStoppingCriteria,
    "entropy": EntropyStoppingCriteria,
    "random": RandomStoppingCriteria,
    "never": NeverStoppingCriteria,
}


def make_stopping_criteria(
    stopping_criteria: str | Type[StoppingCriteria] | StoppingCriteria | None,
) -> StoppingCriteria:
    if stopping_criteria is None:
        return BetaStoppingCriteria()
    if isinstance(stopping_criteria, StoppingCriteria):
        return stopping_criteria
    if isinstance(stopping_criteria, str):
        try:
            return CRITERIA[stopping_criteria]()
        except KeyError as exc:
            choices = ", ".join(sorted(CRITERIA))
            raise ValueError(f"Unknown stopping criterion {stopping_criteria!r}. Choices: {choices}") from exc
    if isinstance(stopping_criteria, type) and issubclass(stopping_criteria, StoppingCriteria):
        return stopping_criteria()
    raise TypeError(f"Unsupported stopping criterion: {stopping_criteria!r}")


def beta_tail_at_half(top_count: int, runner_up_count: int) -> float:
    """Return P(X >= 0.5), where X ~ Beta(top_count + 1, runner_up_count + 1).
    """

    alpha = top_count + 1
    beta = runner_up_count + 1
    return float(stats.beta.sf(0.5, alpha, beta))


def _normalized_entropy(counts: Iterable[int] | Sequence[int]) -> float:
    values = list(counts)
    total = sum(values)
    if total == 0 or len(values) <= 1:
        return 0.0

    entropy = stats.entropy(values, base=2)
    return float(entropy / np.log2(len(values)))
