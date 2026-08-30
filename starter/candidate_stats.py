"""Facet statistics for the current retrieved candidate pool.

The analyzer is intentionally retrieval-side only.  It reads the attributes
already attached to ``Candidate`` objects and does not inspect user messages,
modify constraints, or perform another retrieval pass.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from starter.routing.constraints import CATEGORICAL_FIELDS


# ``budget`` is the runtime clarification name.  Its facet is represented by
# coarse price buckets because the candidate pool has numeric prices rather
# than a categorical budget field.
CANDIDATE_POOL_FACETS: tuple[str, ...] = (*CATEGORICAL_FIELDS, "budget")


def candidate_values(candidate: object, attribute: str) -> tuple[str, ...]:
    """Return distinct non-empty values for one candidate facet."""

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


def candidate_price(candidate: object) -> float | None:
    """Return a finite, non-negative candidate price when available."""

    value = getattr(candidate, "price", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0.0 else None


def price_bucket(price: float) -> str:
    """Return the stable price bucket used by budget facet statistics."""

    if price < 25.0:
        return "under_25"
    if price < 50.0:
        return "25_to_50"
    if price < 100.0:
        return "50_to_100"
    if price < 200.0:
        return "100_to_200"
    return "200_plus"


@dataclass(frozen=True)
class FacetStats:
    """Distribution of one facet over a candidate pool.

    A multi-valued product contributes fractional mass across its values.  For
    example, a product with ``feature=("waterproof", "breathable")`` adds
    0.5 to each value.  This keeps the distribution comparable to the
    existing clarification utility while coverage still counts the product
    once.
    """

    attribute: str
    candidate_count: int
    covered_count: int
    counts: Mapping[str, float]

    @property
    def coverage(self) -> float:
        if self.candidate_count <= 0:
            return 0.0
        return self.covered_count / self.candidate_count

    @property
    def total_mass(self) -> float:
        return sum(float(value) for value in self.counts.values())

    @property
    def probabilities(self) -> tuple[float, ...]:
        total = self.total_mass
        if total <= 0.0:
            return ()
        return tuple(float(value) / total for value in self.counts.values())

    @property
    def expected_reduction(self) -> float:
        """Expected candidate reduction from an answer to this facet."""

        probabilities = self.probabilities
        if not probabilities:
            return 0.0
        return 1.0 - sum(probability * probability for probability in probabilities)

    @property
    def entropy(self) -> float:
        """Shannon entropy of the observed non-empty facet values."""

        return -sum(
            probability * math.log(probability)
            for probability in self.probabilities
            if probability > 0.0
        )

    def top_values(self, limit: int = 3) -> tuple[str, ...]:
        """Return the most common values with deterministic tie-breaking."""

        try:
            count = max(0, int(limit))
        except (TypeError, ValueError):
            count = 0
        return tuple(
            value
            for value, _frequency in sorted(
                self.counts.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )[:count]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "covered_count": self.covered_count,
            "coverage": self.coverage,
            "counts": {
                str(value): float(frequency)
                for value, frequency in sorted(self.counts.items())
            },
            "entropy": self.entropy,
            "expected_reduction": self.expected_reduction,
        }


@dataclass(frozen=True)
class CandidatePoolStats:
    """All facet distributions for one retrieved candidate pool."""

    candidate_count: int
    facets: Mapping[str, FacetStats]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "facets": {
                attribute: stats.as_dict()
                for attribute, stats in self.facets.items()
            },
        }


class CandidatePoolAnalyzer:
    """Compute cheap facet distributions over already-ranked candidates."""

    def __init__(
        self,
        facets: Iterable[str] = CANDIDATE_POOL_FACETS,
    ) -> None:
        self.facets = tuple(dict.fromkeys(str(facet) for facet in facets))

    def analyze(self, candidates: Iterable[object]) -> CandidatePoolStats:
        pool = tuple(candidates)
        counters: dict[str, Counter[str]] = {
            attribute: Counter() for attribute in self.facets
        }
        covered: dict[str, int] = {attribute: 0 for attribute in self.facets}

        for candidate in pool:
            for attribute in self.facets:
                if attribute == "budget":
                    price = candidate_price(candidate)
                    values = (price_bucket(price),) if price is not None else ()
                else:
                    values = candidate_values(candidate, attribute)
                if not values:
                    continue
                covered[attribute] += 1
                share = 1.0 / len(values)
                for value in values:
                    counters[attribute][value] += share

        facet_stats = {
            attribute: FacetStats(
                attribute=attribute,
                candidate_count=len(pool),
                covered_count=covered[attribute],
                counts=dict(counters[attribute]),
            )
            for attribute in self.facets
        }
        return CandidatePoolStats(candidate_count=len(pool), facets=facet_stats)


__all__ = [
    "CANDIDATE_POOL_FACETS",
    "CandidatePoolAnalyzer",
    "CandidatePoolStats",
    "FacetStats",
    "candidate_price",
    "candidate_values",
    "price_bucket",
]
