from __future__ import annotations

import unittest

from dictionary.registry import (
    ATTRIBUTE_FIELDS,
    AttributeDictionary,
    CanonicalValue,
    canonical_id,
    normalize_text,
)
from starter.routing.constraints import (
    AMBIGUITY_MIN_COUNT_RATIO,
    AMBIGUITY_MIN_TOP_SHARE,
    extract_constraints,
)


def _dictionary(*entries: tuple[str, str, int]) -> AttributeDictionary:
    values: list[CanonicalValue] = []
    normalized_index: dict[str, dict[str, list[str]]] = {
        attribute: {} for attribute in ATTRIBUTE_FIELDS
    }
    for attribute, value, count in entries:
        value_id = canonical_id(attribute, value)
        normalized = normalize_text(value)
        values.append(CanonicalValue(value_id, attribute, value, normalized, count))
        normalized_index[attribute].setdefault(normalized, []).append(value_id)
    return AttributeDictionary(values, normalized_index)


class AmbiguityResolutionTests(unittest.TestCase):
    def test_unambiguous_exact_value_is_preserved(self) -> None:
        dictionary = _dictionary(("color", "black", 1))
        constraints = extract_constraints("black", dictionary=dictionary)
        self.assertEqual(constraints.color, ("black",))
        self.assertEqual(constraints.unmapped, ())

    def test_registry_returns_all_ambiguous_candidates_deterministically(self) -> None:
        dictionary = _dictionary(
            ("color", "orange", 10),
            ("brand", "orange", 20),
        )
        candidates = dictionary.get_candidates("ORANGE")
        self.assertEqual(
            tuple(candidate.canonical_id for candidate in candidates),
            ("brand:orange", "color:orange"),
        )
        self.assertEqual(tuple(candidate.count for candidate in candidates), (20, 10))

    def test_longest_phrase_wins_without_overlapping_shorter_match(self) -> None:
        dictionary = _dictionary(
            ("brand", "air max", 10),
            ("brand", "air max 270", 5),
            ("category", "running shoes", 1),
            ("use_case", "running", 100),
        )
        constraints = extract_constraints(
            "I want Air Max 270 running shoes", dictionary=dictionary
        )
        self.assertEqual(constraints.brand, ("air max 270",))
        self.assertEqual(constraints.category, ("running shoes",))
        self.assertEqual(constraints.use_case, ())

    def test_context_resolves_material_color_brand_use_case_and_style(self) -> None:
        dictionary = _dictionary(
            ("material", "leather", 10),
            ("style", "leather", 100),
            ("brand", "orange", 1000),
            ("color", "orange", 10),
            ("style", "running", 100),
            ("use_case", "running", 1),
            ("feature", "casual", 100),
            ("style", "casual", 10),
            ("category", "shoes", 1),
        )
        self.assertEqual(
            extract_constraints("made of leather", dictionary=dictionary).material,
            ("leather",),
        )
        self.assertEqual(
            extract_constraints("color orange", dictionary=dictionary).color,
            ("orange",),
        )
        self.assertEqual(
            extract_constraints("brand orange", dictionary=dictionary).brand,
            ("orange",),
        )
        constraints = extract_constraints("shoes for running", dictionary=dictionary)
        self.assertEqual(constraints.category, ("shoes",))
        self.assertEqual(constraints.use_case, ("running",))
        self.assertEqual(
            extract_constraints("style casual", dictionary=dictionary).style,
            ("casual",),
        )

    def test_explicit_context_does_not_fall_through_to_another_attribute(self) -> None:
        dictionary = _dictionary(
            ("color", "orange", 1000),
            ("material", "leather", 1000),
        )

        brand_orange = extract_constraints("brand orange", dictionary=dictionary)
        self.assertEqual(brand_orange.brand, ())
        self.assertEqual(brand_orange.color, ())
        self.assertIn("orange", brand_orange.unmapped)

        brand_leather = extract_constraints("brand leather", dictionary=dictionary)
        self.assertEqual(brand_leather.brand, ())
        self.assertEqual(brand_leather.material, ())
        self.assertIn("leather", brand_leather.unmapped)

        self.assertEqual(
            extract_constraints("color orange", dictionary=dictionary).color,
            ("orange",),
        )
        self.assertEqual(
            extract_constraints("made of leather", dictionary=dictionary).material,
            ("leather",),
        )

    def test_strong_frequency_dominance_resolves(self) -> None:
        dictionary = _dictionary(
            ("material", "x", 900),
            ("style", "x", 100),
            ("brand", "x", 10),
        )
        constraints = extract_constraints("x", dictionary=dictionary)
        self.assertEqual(constraints.material, ("x",))
        self.assertEqual(constraints.unmapped, ())

    def test_weak_frequency_dominance_stays_unresolved(self) -> None:
        dictionary = _dictionary(
            ("style", "x", 500),
            ("feature", "x", 450),
        )
        constraints = extract_constraints("x", dictionary=dictionary)
        self.assertEqual(constraints.style, ())
        self.assertEqual(constraints.feature, ())
        self.assertEqual(constraints.unmapped, ("x",))

    def test_context_beats_frequency_dominance(self) -> None:
        dictionary = _dictionary(
            ("brand", "x", 1000),
            ("color", "x", 100),
        )
        constraints = extract_constraints("color x", dictionary=dictionary)
        self.assertEqual(constraints.color, ("x",))
        self.assertEqual(constraints.brand, ())

    def test_ambiguity_policy_is_deterministic_and_thresholds_are_conservative(self) -> None:
        self.assertEqual(AMBIGUITY_MIN_TOP_SHARE, 0.75)
        self.assertEqual(AMBIGUITY_MIN_COUNT_RATIO, 3.0)
        dictionary = _dictionary(
            ("material", "x", 900),
            ("style", "x", 100),
            ("brand", "x", 10),
        )
        results = [extract_constraints("x", dictionary=dictionary) for _ in range(5)]
        self.assertTrue(all(result == results[0] for result in results))

    def test_exact_matching_uses_word_boundaries(self) -> None:
        dictionary = _dictionary(("color", "red", 1))
        constraints = extract_constraints("credit", dictionary=dictionary)
        self.assertEqual(constraints.color, ())
        self.assertNotIn("red", constraints.color)

    def test_equal_counts_do_not_choose_an_arbitrary_attribute(self) -> None:
        dictionary = _dictionary(
            ("brand", "orange", 10),
            ("color", "orange", 10),
        )
        constraints = extract_constraints("orange", dictionary=dictionary)
        self.assertEqual(constraints.brand, ())
        self.assertEqual(constraints.color, ())
        self.assertIn("orange", constraints.unmapped)


if __name__ == "__main__":
    unittest.main()