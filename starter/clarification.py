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
from typing import Callable, Iterable

from starter.candidate_stats import CandidatePoolAnalyzer, CandidatePoolStats
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

# ``other`` is a cycle boundary rather than a field whose value can be
# clarified.  Keep this derived from the public list so callers do not need a
# second hand-maintained list of askable attributes.
NORMAL_CLARIFICATION_ATTRIBUTES = tuple(
    attribute for attribute in SUPPORTED_ATTRIBUTES if attribute != "other"
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
    candidate_stats: CandidatePoolStats | None = None,
) -> float:
    """Expected score gain from asking about ``attribute`` on ``turn``."""
    stats = candidate_stats or CandidatePoolAnalyzer().analyze(candidates)
    if stats.candidate_count < 2:
        return float("-inf")

    facet = stats.facets.get(attribute)
    if facet is None:
        return float("-inf")
    counts = facet.counts
    coverage = facet.coverage
    if coverage < 0.20 or len(counts) < 2:
        return float("-inf")

    # ``expected_reduction`` is the existing Gini split term, now supplied by
    # the shared candidate-pool analyzer instead of being recomputed per
    # attribute inside this policy.
    gini = facet.expected_reduction
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

    def __init__(self, candidate_analyzer: CandidatePoolAnalyzer | None = None) -> None:
        self.candidate_analyzer = candidate_analyzer or CandidatePoolAnalyzer()

    def analyze(self, candidates: Iterable[object]) -> CandidatePoolStats:
        """Analyze a retrieved candidate pool for selection and question text."""

        return self.candidate_analyzer.analyze(candidates)

    def choose(
        self,
        candidates: Iterable[object],
        constraints: ShoppingConstraints,
        asked_attributes: Iterable[str] = (),
        *,
        mode: str = "BROWSING",
        profile_factor: Callable[[str], float] | None = None,
        turn: int | None = None,
        candidate_stats: CandidatePoolStats | None = None,
    ) -> str | None:
        pool = tuple(candidates)
        stats = candidate_stats or self.candidate_analyzer.analyze(pool)
        asked = set(asked_attributes)
        available = [
            attribute
            for attribute in NORMAL_CLARIFICATION_ATTRIBUTES
            if attribute not in asked and not _known(constraints, attribute)
        ]
        if not available:
            return None

        scored = [
            (
                self._score(
                    attribute,
                    pool,
                    mode,
                    profile_factor,
                    turn,
                    candidate_stats=stats,
                ),
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
        candidate_stats: CandidatePoolStats | None = None,
    ) -> float:
        return _utility(
            attribute,
            candidates,
            mode,
            profile_factor,
            turn,
            candidate_stats=candidate_stats,
        )

    @staticmethod
    def question(
        attribute: str | None,
        candidate_stats: CandidatePoolStats | None = None,
    ) -> str:
        if attribute is None:
            return "Here are the closest matches I found."
        if candidate_stats is not None:
            facet = candidate_stats.facets.get(attribute)
            if facet is not None and facet.coverage >= 0.20:
                values = facet.top_values(3)
                if len(values) >= 2:
                    if attribute == "budget":
                        labels = {
                            "under_25": "under $25",
                            "25_to_50": "$25 to $50",
                            "50_to_100": "$50 to $100",
                            "100_to_200": "$100 to $200",
                            "200_plus": "$200 or more",
                        }
                        options = tuple(labels.get(value, value) for value in values)
                        option_text = (
                            options[0]
                            if len(options) == 1
                            else ", ".join(options[:-1]) + f", or {options[-1]}"
                        )
                        return f"Would you prefer {option_text}?"
                    noun = {
                        "category": "type of item",
                        "use_case": "use case",
                    }.get(attribute, attribute)
                    if attribute == "use_case":
                        prefix = "Is this mainly for"
                    elif attribute == "feature":
                        prefix = "Would you prefer"
                    else:
                        prefix = "Do you prefer"
                    option_text = (
                        values[0]
                        if len(values) == 1
                        else ", ".join(values[:-1]) + f", or another {noun}"
                    )
                    return f"{prefix} {option_text}?"
        return ATTRIBUTE_QUESTIONS.get(attribute, "Which preference matters most for this item?")


def choose_attribute(
    candidates: Iterable[object],
    constraints: ShoppingConstraints,
    asked_attributes: Iterable[str] = (),
    *,
    mode: str = "BROWSING",
    profile_factor: Callable[[str], float] | None = None,
    turn: int | None = None,
    candidate_stats: CandidatePoolStats | None = None,
) -> str | None:
    """Functional convenience wrapper for the default deterministic policy."""

    return ClarificationPolicy().choose(
        candidates,
        constraints,
        asked_attributes,
        mode=mode,
        profile_factor=profile_factor,
        turn=turn,
        candidate_stats=candidate_stats,
    )


__all__ = [
    "ASK_SPLIT_FLOOR",
    "ASK_UTILITY_FLOOR",
    "PRIOR_CEILING",
    "ATTRIBUTE_QUESTIONS",
    "CandidatePoolAnalyzer",
    "CandidatePoolStats",
    "ClarificationPolicy",
    "NORMAL_CLARIFICATION_ATTRIBUTES",
    "SUPPORTED_ATTRIBUTES",
    "choose_attribute",
]
