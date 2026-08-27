"""Deterministic one-step clarification policy for shared retrieval pools."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from starter.routing.constraints import CATEGORICAL_FIELDS, ShoppingConstraints


SUPPORTED_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
)

ATTRIBUTE_QUESTIONS = {
    "category": "What type of item should I focus on?",
    "material": "Which material should I prioritize?",
    "color": "Do you have a preferred color?",
    "size": "Do you have a preferred size or fit?",
    "style": "Which style or fit should I prioritize?",
    "brand": "Do you have a preferred brand or store?",
    "budget": "What budget range should I stay within?",
    "feature": "Which feature matters most for this item?",
    "use_case": "What will you mainly use this for?",
}

MODE_PRIORS = {
    "BUYING": {
        "material": 1.00,
        "feature": 0.98,
        "color": 0.92,
        "size": 0.90,
        "style": 0.78,
        "use_case": 0.74,
        "brand": 0.68,
        "budget": 0.66,
        "category": 0.60,
    },
    "BROWSING": {
        "style": 1.00,
        "use_case": 0.98,
        "feature": 0.96,
        "color": 0.92,
        "category": 0.90,
        "material": 0.84,
        "size": 0.76,
        "brand": 0.64,
        "budget": 0.62,
    },
}


def _candidate_values(candidate: object, attribute: str) -> tuple[str, ...]:
    attributes = getattr(candidate, "attributes", {})
    if not isinstance(attributes, Mapping):
        return ()
    values = attributes.get(attribute, ())
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _known(constraints: ShoppingConstraints, attribute: str) -> bool:
    if attribute == "budget":
        return (
            getattr(constraints, "price_min", None) is not None
            or getattr(constraints, "price_max", None) is not None
        )
    value = getattr(constraints, attribute, ())
    return bool(value)


def _utility(
    attribute: str,
    candidates: tuple[object, ...],
    mode: str,
) -> float:
    if len(candidates) < 2:
        return float("-inf")

    counts: Counter[str] = Counter()
    covered = 0
    for candidate in candidates:
        values = _candidate_values(candidate, attribute)
        if not values:
            continue
        covered += 1
        share = 1.0 / len(values)
        for value in values:
            counts[value] += share

    coverage = covered / len(candidates)
    if coverage < 0.20 or len(counts) < 2:
        return float("-inf")

    total = sum(counts.values())
    probabilities = [value / total for value in counts.values()]
    gini = 1.0 - sum(probability * probability for probability in probabilities)
    if gini < 0.10:
        return float("-inf")

    prior = MODE_PRIORS.get(mode, MODE_PRIORS["BROWSING"]).get(attribute, 0.70)
    # Coverage rewards known facts; Gini rewards a useful split and suppresses
    # nearly constant attributes.  The small diversity term makes two equally
    # balanced fields prefer the one with more meaningful alternatives.
    diversity = min(1.0, (len(counts) - 1) / 3.0)
    return coverage * gini * (0.75 + 0.25 * diversity) * prior


class ClarificationPolicy:
    """Choose at most one useful, not-yet-asked supported attribute."""

    def choose(
        self,
        candidates: Iterable[object],
        constraints: ShoppingConstraints,
        asked_attributes: Iterable[str] = (),
        *,
        mode: str = "BROWSING",
    ) -> str | None:
        pool = tuple(candidates)
        asked = set(asked_attributes)
        available = [
            attribute
            for attribute in SUPPORTED_ATTRIBUTES
            if attribute not in asked and not _known(constraints, attribute)
        ]
        if not available:
            return None

        scored = [
            (self._score(attribute, pool, mode), -SUPPORTED_ATTRIBUTES.index(attribute), attribute)
            for attribute in available
        ]
        score, _tie, attribute = max(scored)
        if score == float("-inf") or score < 0.035:
            return None
        return attribute

    @staticmethod
    def _score(attribute: str, candidates: tuple[object, ...], mode: str) -> float:
        return _utility(attribute, candidates, mode)

    @staticmethod
    def question(attribute: str | None) -> str:
        if attribute is None:
            return "Here are the closest matches I found."
        return ATTRIBUTE_QUESTIONS.get(attribute, "Which preference matters most for this item?")


def choose_attribute(
    candidates: Iterable[object],
    constraints: ShoppingConstraints,
    asked_attributes: Iterable[str] = (),
    *,
    mode: str = "BROWSING",
) -> str | None:
    """Functional convenience wrapper for the default deterministic policy."""

    return ClarificationPolicy().choose(
        candidates, constraints, asked_attributes, mode=mode
    )


__all__ = [
    "ATTRIBUTE_QUESTIONS",
    "ClarificationPolicy",
    "SUPPORTED_ATTRIBUTES",
    "choose_attribute",
]
