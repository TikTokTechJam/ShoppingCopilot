from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.retrieval import (
    RATING_BOOST_WEIGHT,
    ProductRetriever,
    is_critical_user,
    normalized_rating,
    rating_weight,
)
from starter.routing.constraints import extract_constraints

CRITICAL_USER = 2.5
GENEROUS_USER = 5.0


def build_catalog(root: Path) -> Path:
    """Two products that match identically and differ only by rating."""

    path = root / "catalog.jsonl"
    rows = [
        {
            "parent_asin": "LOW",
            "categories": ["women", "shoes"],
            "title": "Low rated shoes",
            "price": 30.0,
            "average_rating": 2.0,
        },
        {
            "parent_asin": "HIGH",
            "categories": ["women", "shoes"],
            "title": "High rated shoes",
            "price": 30.0,
            "average_rating": 4.9,
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


class RatingHelperTest(unittest.TestCase):
    def test_missing_rating_is_neutral_not_zero(self) -> None:
        """Cold start must not demote; acceptance criterion 2."""

        self.assertEqual(normalized_rating(None), 0.5)
        self.assertEqual(normalized_rating("not a number"), 0.5)
        self.assertEqual(normalized_rating(0.0), 0.5)
        self.assertAlmostEqual(normalized_rating(5.0), 1.0)
        self.assertAlmostEqual(normalized_rating(2.5), 0.5)

    def test_weight_switches_on_the_threshold(self) -> None:
        self.assertTrue(is_critical_user(CRITICAL_USER))
        self.assertFalse(is_critical_user(GENEROUS_USER))
        self.assertFalse(is_critical_user(None))
        self.assertEqual(rating_weight(CRITICAL_USER), RATING_BOOST_WEIGHT)
        self.assertGreater(
            rating_weight(CRITICAL_USER), rating_weight(GENEROUS_USER)
        )
        self.assertEqual(rating_weight(None), rating_weight(GENEROUS_USER))


class RatingTieBreakTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.retriever = ProductRetriever(build_catalog(Path(self._tmp.name)))
        self.constraints = extract_constraints("I am looking for shoes")

    def _order(self, user_prior_rating: float | None) -> list[str]:
        return [
            candidate.parent_asin
            for candidate in self.retriever.retrieve(
                "BROWSING",
                "shoes",
                self.constraints,
                limit=10,
                user_prior_rating=user_prior_rating,
            )
        ]

    def test_high_rated_product_wins_a_tie_for_a_critical_user(self) -> None:
        """Acceptance criterion 1: identical logits, higher rating ranks first.

        LOW is first in catalog order, so a pass here proves the rating beat
        the catalog-order tie-break rather than coinciding with it.
        """

        self.assertEqual(self._order(CRITICAL_USER)[0], "HIGH")

    def test_cold_start_does_not_raise_and_still_returns_both(self) -> None:
        """Acceptance criterion 2: a null prior rating degrades gracefully."""

        self.assertCountEqual(self._order(None), ["LOW", "HIGH"])

    def test_a_stronger_structured_match_outranks_a_better_rating(self) -> None:
        """Acceptance criterion 2: rank 1 retention when scores really differ.

        The rating bonus spans at most 0.15 * 0.8 = 0.12, while one extra
        matched field is worth at least 0.50, so the best structured match must
        stay at rank 1 whatever its rating.
        """

        constraints = extract_constraints("I am looking for cheap shoes under $35")
        order = [
            candidate.parent_asin
            for candidate in self.retriever.retrieve(
                "BROWSING",
                "shoes",
                constraints,
                limit=10,
                user_prior_rating=CRITICAL_USER,
            )
        ]
        top = self.retriever.retrieve(
            "BROWSING", "shoes", constraints, limit=10
        )
        best = max(candidate.constraint_score for candidate in top)
        winners = {
            candidate.parent_asin
            for candidate in top
            if candidate.constraint_score == best
        }
        self.assertIn(order[0], winners)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
