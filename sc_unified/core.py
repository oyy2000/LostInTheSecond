"""One interface for SC, AC/ASC, ESC, DSC, and RASC-style inference.

The interface works with either pre-generated outputs or an online sampler.
Each method decides how many samples to consume and returns the same result
shape so experiment code can switch methods with one string.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Union


AnswerExtractor = Callable[[Any], Hashable]
Sampler = Callable[[], Any]
ScoreFn = Callable[["Generation", Sequence["Generation"]], float]


@dataclass
class Generation:
    """One sampled chain/rationale plus optional metadata."""

    text: Any
    answer: Optional[Hashable] = None
    score: Optional[float] = None
    rationale: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsistencyResult:
    method: str
    answer: Optional[Hashable]
    generations: List[Generation]
    counts: Dict[Hashable, int]
    samples_used: int
    stopped_early: bool
    stop_reason: str
    confidence: Optional[float] = None
    selected_rationale: Optional[Any] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodConfig:
    method: str = "ac"
    max_samples: int = 40
    window_size: int = 5
    confidence: float = 0.95
    hard_confidence: float = 0.50
    min_samples: int = 1
    prior_easy: bool = False
    initial_window_size: Optional[int] = None
    rasc_threshold: float = 0.5
    rasc_buffer_size: int = 5
    ignore_answers: tuple[Hashable, ...] = ("[invalid]", "", None)


class BasePolicy:
    name = "base"

    def __init__(self, config: Optional[MethodConfig] = None) -> None:
        self.config = config or MethodConfig(method=self.name)

    def run(
        self,
        generations: Optional[Sequence[Any]] = None,
        sampler: Optional[Sampler] = None,
        answer_extractor: Optional[AnswerExtractor] = None,
        score_fn: Optional[ScoreFn] = None,
        greedy_generation: Optional[Any] = None,
    ) -> ConsistencyResult:
        raise NotImplementedError

    def _materialize_generation(
        self,
        raw: Any,
        answer_extractor: Optional[AnswerExtractor],
    ) -> Generation:
        if isinstance(raw, Generation):
            generation = raw
        elif isinstance(raw, Mapping):
            generation = Generation(
                text=raw.get("text", raw.get("completion", raw.get("rationale", raw))),
                answer=raw.get("answer"),
                score=raw.get("score"),
                rationale=raw.get("rationale", raw.get("cot")),
                meta={k: v for k, v in raw.items() if k not in {"text", "completion", "rationale", "cot", "answer", "score"}},
            )
        else:
            generation = Generation(text=raw)

        if generation.answer is None and answer_extractor is not None:
            generation.answer = answer_extractor(generation.text)
        return generation

    def _next_generation(
        self,
        consumed: List[Generation],
        available: Optional[Sequence[Any]],
        sampler: Optional[Sampler],
        answer_extractor: Optional[AnswerExtractor],
    ) -> Optional[Generation]:
        if available is not None:
            if len(consumed) >= len(available):
                return None
            raw = available[len(consumed)]
        elif sampler is not None:
            raw = sampler()
        else:
            raise ValueError("Provide either generations or sampler.")
        return self._materialize_generation(raw, answer_extractor)

    def _consume_one(
        self,
        consumed: List[Generation],
        available: Optional[Sequence[Any]],
        sampler: Optional[Sampler],
        answer_extractor: Optional[AnswerExtractor],
    ) -> bool:
        if len(consumed) >= self.config.max_samples:
            return False
        generation = self._next_generation(consumed, available, sampler, answer_extractor)
        if generation is None:
            return False
        consumed.append(generation)
        return True

    def _result(
        self,
        consumed: List[Generation],
        stop_reason: str,
        confidence: Optional[float] = None,
        selected_rationale: Optional[Any] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> ConsistencyResult:
        answers = [generation.answer for generation in consumed]
        winner = majority_vote(answers, ignore_answers=self.config.ignore_answers)
        counts = dict(Counter(answer for answer in answers if answer not in self.config.ignore_answers))
        stopped_early = len(consumed) < self.config.max_samples and stop_reason != "max_samples"
        return ConsistencyResult(
            method=self.name,
            answer=winner,
            generations=consumed,
            counts=counts,
            samples_used=len(consumed),
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            confidence=confidence,
            selected_rationale=selected_rationale,
            diagnostics=diagnostics or {},
        )


class SCPolicy(BasePolicy):
    """Plain self-consistency: consume max_samples, then majority vote."""

    name = "sc"

    def run(
        self,
        generations: Optional[Sequence[Any]] = None,
        sampler: Optional[Sampler] = None,
        answer_extractor: Optional[AnswerExtractor] = None,
        score_fn: Optional[ScoreFn] = None,
        greedy_generation: Optional[Any] = None,
    ) -> ConsistencyResult:
        del score_fn, greedy_generation
        consumed: List[Generation] = []
        while self._consume_one(consumed, generations, sampler, answer_extractor):
            pass
        return self._result(consumed, stop_reason="max_samples")


class ACPolicy(BasePolicy):
    """Adaptive Consistency / ASC: stop when beta posterior is confident."""

    name = "ac"

    def run(
        self,
        generations: Optional[Sequence[Any]] = None,
        sampler: Optional[Sampler] = None,
        answer_extractor: Optional[AnswerExtractor] = None,
        score_fn: Optional[ScoreFn] = None,
        greedy_generation: Optional[Any] = None,
    ) -> ConsistencyResult:
        del score_fn, greedy_generation
        consumed: List[Generation] = []
        confidence = None
        stop_reason = "max_samples"

        while self._consume_one(consumed, generations, sampler, answer_extractor):
            confidence = beta_winner_confidence([generation.answer for generation in consumed])
            if len(consumed) >= self.config.min_samples and confidence >= self.config.confidence:
                stop_reason = "beta_confidence"
                break

        return self._result(consumed, stop_reason=stop_reason, confidence=confidence)


class ESCPolicy(BasePolicy):
    """Early-stopping Self-Consistency: stop if the latest window agrees."""

    name = "esc"

    def run(
        self,
        generations: Optional[Sequence[Any]] = None,
        sampler: Optional[Sampler] = None,
        answer_extractor: Optional[AnswerExtractor] = None,
        score_fn: Optional[ScoreFn] = None,
        greedy_generation: Optional[Any] = None,
    ) -> ConsistencyResult:
        del score_fn, greedy_generation
        consumed: List[Generation] = []
        stop_reason = "max_samples"

        while len(consumed) < self.config.max_samples:
            before = len(consumed)
            for _ in range(self.config.window_size):
                if not self._consume_one(consumed, generations, sampler, answer_extractor):
                    break
            window = consumed[before:]
            if not window:
                break
            window_answers = [generation.answer for generation in window]
            if len(window_answers) == self.config.window_size and _is_unanimous(window_answers, self.config.ignore_answers):
                stop_reason = "unanimous_window"
                break

        return self._result(consumed, stop_reason=stop_reason)


class DSCPolicy(BasePolicy):
    """Difficulty-adaptive SC.

    If prior_easy is true, use a single greedy sample. Otherwise run a
    windowed beta stopping rule similar to DSC/DSC_hard in WangXinglin/DSC.
    """

    name = "dsc"

    def run(
        self,
        generations: Optional[Sequence[Any]] = None,
        sampler: Optional[Sampler] = None,
        answer_extractor: Optional[AnswerExtractor] = None,
        score_fn: Optional[ScoreFn] = None,
        greedy_generation: Optional[Any] = None,
    ) -> ConsistencyResult:
        del score_fn
        if self.config.prior_easy:
            raw = greedy_generation
            if raw is None:
                raw = generations[0] if generations else None
            if raw is None and sampler is not None:
                raw = sampler()
            if raw is None:
                raise ValueError("prior_easy=True needs greedy_generation, generations, or sampler.")
            consumed = [self._materialize_generation(raw, answer_extractor)]
            return self._result(
                consumed,
                stop_reason="prior_easy_greedy",
                confidence=1.0,
                diagnostics={"prior_easy": True},
            )

        consumed: List[Generation] = []
        confidence = None
        stop_reason = "max_samples"
        halfway_windows = max(1, self.config.max_samples // (2 * self.config.window_size))

        while len(consumed) < self.config.max_samples:
            for _ in range(self.config.window_size):
                if not self._consume_one(consumed, generations, sampler, answer_extractor):
                    break
            if not consumed:
                break

            confidence = beta_winner_confidence([generation.answer for generation in consumed])
            num_windows = math.ceil(len(consumed) / self.config.window_size)
            if confidence >= self.config.confidence:
                stop_reason = "dsc_easy_beta"
                break
            if num_windows >= halfway_windows and confidence <= self.config.hard_confidence:
                stop_reason = "dsc_hard_low_confidence"
                break

        return self._result(
            consumed,
            stop_reason=stop_reason,
            confidence=confidence,
            diagnostics={"prior_easy": False},
        )


class RASCPolicy(BasePolicy):
    """Reasoning-aware SC: keep high-score rationales in a bounded buffer."""

    name = "rasc"

    def run(
        self,
        generations: Optional[Sequence[Any]] = None,
        sampler: Optional[Sampler] = None,
        answer_extractor: Optional[AnswerExtractor] = None,
        score_fn: Optional[ScoreFn] = None,
        greedy_generation: Optional[Any] = None,
    ) -> ConsistencyResult:
        del greedy_generation
        consumed: List[Generation] = []
        high_quality: List[Generation] = []
        stop_reason = "max_samples"

        while self._consume_one(consumed, generations, sampler, answer_extractor):
            current = consumed[-1]
            if current.score is None:
                current.score = score_fn(current, consumed) if score_fn is not None else 1.0
            if current.score >= self.config.rasc_threshold:
                high_quality.append(current)
            if len(high_quality) >= self.config.rasc_buffer_size:
                stop_reason = "rasc_buffer_full"
                break

        candidates = high_quality or consumed
        answer, rationale, best_score = _weighted_vote_with_rationale(candidates, self.config.ignore_answers)
        counts = dict(Counter(gen.answer for gen in candidates if gen.answer not in self.config.ignore_answers))
        stopped_early = len(consumed) < self.config.max_samples and stop_reason != "max_samples"
        return ConsistencyResult(
            method=self.name,
            answer=answer,
            generations=consumed,
            counts=counts,
            samples_used=len(consumed),
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            confidence=best_score,
            selected_rationale=rationale,
            diagnostics={
                "buffer_size": len(high_quality),
                "threshold": self.config.rasc_threshold,
                "used_fallback": not high_quality,
            },
        )


def run_consistency(
    method: str,
    generations: Optional[Sequence[Any]] = None,
    sampler: Optional[Sampler] = None,
    answer_extractor: Optional[AnswerExtractor] = None,
    score_fn: Optional[ScoreFn] = None,
    greedy_generation: Optional[Any] = None,
    **config_kwargs: Any,
) -> ConsistencyResult:
    config = MethodConfig(method=method, **config_kwargs)
    return make_policy(config).run(
        generations=generations,
        sampler=sampler,
        answer_extractor=answer_extractor,
        score_fn=score_fn,
        greedy_generation=greedy_generation,
    )


def make_policy(config: Union[MethodConfig, str]) -> BasePolicy:
    if isinstance(config, str):
        config = MethodConfig(method=config)
    method = config.method.lower()
    aliases = {
        "asc": "ac",
        "adaptive_consistency": "ac",
        "adaptive": "ac",
        "early_stop": "esc",
        "early_stopping": "esc",
    }
    method = aliases.get(method, method)
    policies = {
        "sc": SCPolicy,
        "ac": ACPolicy,
        "esc": ESCPolicy,
        "dsc": DSCPolicy,
        "rasc": RASCPolicy,
    }
    try:
        return policies[method](config)
    except KeyError as exc:
        choices = ", ".join(sorted(policies))
        raise ValueError(f"Unknown method {config.method!r}. Choices: {choices}") from exc


def beta_winner_confidence(answers: Sequence[Hashable]) -> float:
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return 0.0
    top_two = counts.most_common(2)
    top = top_two[0][1]
    runner_up = top_two[1][1] if len(top_two) > 1 else 0
    return beta_tail_at_half(top, runner_up)


def beta_tail_at_half(top_count: int, runner_up_count: int) -> float:
    """P(X >= 0.5) for X ~ Beta(top_count + 1, runner_up_count + 1)."""

    alpha = top_count + 1
    beta = runner_up_count + 1
    n = alpha + beta - 1
    return sum(math.comb(n, j) for j in range(alpha)) / (2**n)


def majority_vote(
    answers: Sequence[Hashable],
    ignore_answers: Iterable[Hashable] = ("[invalid]", "", None),
) -> Optional[Hashable]:
    ignored = set(ignore_answers)
    filtered = [answer for answer in answers if answer not in ignored]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def normalize_answer(answer: Any) -> Hashable:
    try:
        return float(answer)
    except (TypeError, ValueError):
        return str(answer).strip().lower()


def extract_last_number(text: Any) -> str:
    numbers = re.findall(r"[+-]?\d+\.?\d*", str(text).replace(",", ""))
    if not numbers:
        return "[invalid]"
    return normalize_number_string(numbers[-1])


def extract_math_answer(text: Any) -> str:
    text = str(text)
    lowered = text.lower()
    if "boxed" in text:
        answer = extract_boxed_content(text)
    elif "the answer is " in lowered:
        start = lowered.rfind("the answer is ")
        answer = text[start + len("the answer is ") :].strip()
    else:
        answer = extract_last_number(text)
    return strip_math_string(str(answer).rstrip("./ "))


def extract_boxed_content(text: str) -> str:
    marker = text.rfind("boxed")
    if marker < 0:
        return ""
    tail = text[marker + len("boxed") :].strip()
    if not tail:
        return ""
    if tail[0] != "{":
        return tail.split("$")[0].strip()
    depth = 1
    chars: List[str] = []
    for char in tail[1:]:
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
            chars.append(char)
        else:
            chars.append(char)
    return "".join(chars)


def extract_choice_answer(text: Any) -> str:
    answer_text = str(text).lower().split("the answer is")[-1]
    match = re.search(r"\((.*)\)", answer_text, flags=re.S)
    return match.group(1).strip() if match else answer_text.strip()


def extract_yes_no(text: Any) -> str:
    lowered = str(text).lower()
    if "the answer is yes" in lowered:
        return "yes"
    if "the answer is no" in lowered:
        return "no"
    if "yes" in lowered and "no" not in lowered:
        return "yes"
    if "no" in lowered and "yes" not in lowered:
        return "no"
    return "[invalid]"


def extract_last_letters(text: Any) -> str:
    answer_text = str(text).lower().split("the answer is")[-1]
    return "".join(re.split(r"[^a-z]", answer_text))


def normalize_number_string(value: Any) -> str:
    text = str(value).replace(",", "").strip()
    if text.endswith("."):
        text = text[:-1]
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return text
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def strip_math_string(text: str) -> str:
    text = text.replace("\n", "").replace("\\!", "")
    text = text.replace("\\\\", "\\")
    text = text.replace("tfrac", "frac").replace("dfrac", "frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "").replace("\\%", "").replace("%", "")
    if "\\text{ " in text:
        text = text.split("\\text{ ")[0]
    text = text.replace(" .", " 0.").replace("{.", "{0.").strip()
    if not text:
        return text
    if text[0] == ".":
        text = "0" + text
    parts = text.split("=")
    if len(parts) == 2 and len(parts[0]) <= 2:
        text = parts[1]
    text = text.replace(" ", "")
    if text == "0.5":
        return "\\frac{1}{2}"
    return text


def _is_unanimous(answers: Sequence[Hashable], ignored: Iterable[Hashable]) -> bool:
    if not answers:
        return False
    unique = set(answers)
    return len(unique) == 1 and next(iter(unique)) not in set(ignored)


def _weighted_vote_with_rationale(
    candidates: Sequence[Generation],
    ignored: Iterable[Hashable],
) -> tuple[Optional[Hashable], Optional[Any], Optional[float]]:
    ignored_set = set(ignored)
    weighted_votes: Counter = Counter()
    for generation in candidates:
        if generation.answer in ignored_set:
            continue
        weighted_votes[normalize_answer(generation.answer)] += generation.score or 0.0
    if not weighted_votes:
        return None, None, None
    normalized_answer = max(weighted_votes, key=weighted_votes.get)
    supporting = [
        generation
        for generation in candidates
        if normalize_answer(generation.answer) == normalized_answer
    ]
    best = max(supporting, key=lambda generation: generation.score or 0.0)
    return best.answer, best.rationale if best.rationale is not None else best.text, best.score
