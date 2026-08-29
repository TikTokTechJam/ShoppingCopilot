"""Deterministic one-step clarification policy for shared retrieval pools.

Section 12 scores a question by how well an answer would split the candidate
pool.  Section 12b defines the only objective the Agent may optimize,
``starter.followup.utility``.  This module combines them into a single number,
so one utility decides every question:

    ExpectedGain(a, t) = Split(a) * P(answer | a, mode, profile) * Horizon(t)

``Split`` is the original pool term and is unchanged.  ``P(answer)`` is the
population prior for the shopping mode, updated by the shopper's own profile.
``Horizon`` is the score still reachable if that answer converts the session on
the next turn.  The product is in ``TechnicalScore`` units, which turns the
abstain floor from a magic constant into a bet size and makes the old
``turn < 10`` guard fall out of the arithmetic.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable, Mapping

from starter.followup import MAX_TURNS
from starter.followup import utility as session_utility
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
    "other",
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
    "other": "Is there another preference that matters to you?",
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


# MODE_PRIORS are relative weights on [0.60, 1.00], not calibrated
# probabilities.  Reading the top weight as p = 1.0 would make it infinite odds
# and therefore immune to any finite profile evidence -- the top-priority
# attribute could never be displaced, which is precisely the inertness the
# profile is meant to cure.  The ceiling keeps headroom for evidence.  It
# scales every attribute equally, so it cannot change an argmax on its own; the
# floors below absorb it, leaving profile-free decisions untouched.
PRIOR_CEILING = 0.90

# The historical floor, applied to the pool term alone.
ASK_SPLIT_FLOOR = 0.035

# The same bet after the priors are read as probabilities.  Used when no turn
# is supplied, so those callers keep the exact behaviour they had.
_LEGACY_FLOOR = ASK_SPLIT_FLOOR * PRIOR_CEILING

# And again in score units: the value of a question that clears the historical
# bar and converts on turn 2.  A question worth less expected score than this
# is not worth consuming the attribute slot for.
ASK_UTILITY_FLOOR = _LEGACY_FLOOR * session_utility(2, 1)


def _horizon(turn: int | None) -> float:
    """Score still reachable if the answer converts the session next turn.

    ``None`` keeps the historical unit-free scale for callers that do not track
    the turn.  On the final turn nothing can act on an answer, so the horizon
    -- and with it every question's value -- is exactly zero.  The ``turn < 10``
    guard that used to live in ``Agent.respond`` is this line.
    """
    if turn is None:
        return 1.0
    try:
        current = int(turn)
    except (TypeError, ValueError):
        return 1.0
    if current >= MAX_TURNS:
        return 0.0
    return session_utility(max(current, 1) + 1, 1)


def _answer_probability(
    attribute: str,
    mode: str,
    profile_factor: Callable[[str], float] | None,
) -> float:
    """Probability the shopper answers this question with a usable value.

    ``MODE_PRIORS`` is the population prior for the mode.  The profile factor is
    evidence about *this* shopper, so it enters as a likelihood ratio on the
    odds rather than as a multiplier on the probability: a ratio can never push
    the result outside [0, 1], it never truncates a strong signal on an already
    likely attribute, and it has the most leverage where the prior is least
    certain.  A factor of exactly zero is the single veto, and
    ``ProfileAffinity`` reserves it for direct evidence -- the shopper has
    already declined this attribute.
    """
    weight = MODE_PRIORS.get(mode, MODE_PRIORS["BROWSING"]).get(attribute, 0.70)
    prior = PRIOR_CEILING * weight
    if profile_factor is None:
        return prior
    try:
        ratio = float(profile_factor(attribute))
    except (TypeError, ValueError):
        return prior
    if not math.isfinite(ratio) or ratio < 0.0:
        return prior
    if ratio == 0.0:
        return 0.0
    bounded = min(max(prior, 1e-6), 1.0 - 1e-6)
    odds = bounded / (1.0 - bounded) * ratio
    return odds / (1.0 + odds)

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


def _candidate_price(candidate: object) -> float | None:
    value = getattr(candidate, "price", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0.0 else None


def _price_band(price: float) -> str:
    """Return a coarse, user-meaningful budget band for question utility."""

    if price < 25.0:
        return "under_25"
    if price < 50.0:
        return "25_to_50"
    if price < 100.0:
        return "50_to_100"
    if price < 200.0:
        return "100_to_200"
    return "200_plus"


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
    profile_factor: Callable[[str], float] | None = None,
    turn: int | None = None,
) -> float:
    """Expected score gain from asking about ``attribute`` on ``turn``."""
    if len(candidates) < 2:
        return float("-inf")

    counts: Counter[str] = Counter()
    covered = 0
    for candidate in candidates:
        price = _candidate_price(candidate) if attribute == "budget" else None
        values = (
            (_price_band(price),)
            if price is not None
            else _candidate_values(candidate, attribute)
        )
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

    # Coverage rewards known facts; Gini rewards a useful split and suppresses
    # nearly constant attributes.  The small diversity term makes two equally
    # balanced fields prefer the one with more meaningful alternatives.  This
    # is the Section 12 term: how much an answer would narrow the pool.
    diversity = min(1.0, (len(counts) - 1) / 3.0)
    split = coverage * gini * (0.75 + 0.25 * diversity)

    # The pool cannot say whether the shopper will answer, and it cannot say
    # what an answer is still worth this late in the session.  Those are the
    # other two factors, and the product is the expected score gain.
    return split * _answer_probability(attribute, mode, profile_factor) * _horizon(turn)


class ClarificationPolicy:
    """Choose at most one useful, not-yet-asked supported attribute."""

    def choose(
        self,
        candidates: Iterable[object],
        constraints: ShoppingConstraints,
        asked_attributes: Iterable[str] = (),
        *,
        mode: str = "BROWSING",
        profile_factor: Callable[[str], float] | None = None,
        turn: int | None = None,
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
            (
                self._score(attribute, pool, mode, profile_factor, turn),
                -SUPPORTED_ATTRIBUTES.index(attribute),
                attribute,
            )
            for attribute in available
        ]
        # A turn-aware score is in TechnicalScore units and is measured against
        # the same bet re-expressed in those units; a turn-free score keeps the
        # historical pool-only scale and floor.
        floor = ASK_UTILITY_FLOOR if turn is not None else _LEGACY_FLOOR
        score, _tie, attribute = max(scored)
        if score == float("-inf") or score < floor:
            return None
        return attribute

    @staticmethod
    def _score(
        attribute: str,
        candidates: tuple[object, ...],
        mode: str,
        profile_factor: Callable[[str], float] | None = None,
        turn: int | None = None,
    ) -> float:
        return _utility(attribute, candidates, mode, profile_factor, turn)

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
    profile_factor: Callable[[str], float] | None = None,
    turn: int | None = None,
) -> str | None:
    """Functional convenience wrapper for the default deterministic policy."""

    return ClarificationPolicy().choose(
        candidates,
        constraints,
        asked_attributes,
        mode=mode,
        profile_factor=profile_factor,
        turn=turn,
    )


__all__ = [
    "ASK_SPLIT_FLOOR",
    "ASK_UTILITY_FLOOR",
    "PRIOR_CEILING",
    "ATTRIBUTE_QUESTIONS",
    "ClarificationPolicy",
    "SUPPORTED_ATTRIBUTES",
    "choose_attribute",
]
