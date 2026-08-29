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


def _neighbor_context_dictionary() -> AttributeDictionary:
    return _dictionary(
        ("brand", "salomon", 50),
        ("brand", "alpha", 1),
        ("category", "shoes", 11689),
        ("style", "shoes", 1),
        ("use_case", "shoes", 5),
        ("color", "black", 7125),
        ("material", "rubber", 6456),
        ("color", "rubber", 1),
        ("style", "rubber", 1),
        ("material", "leather", 10),
        ("style", "leather", 100),
        ("use_case", "running", 2782),
        ("category", "running", 964),
        ("style", "running", 5),
    )


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

    def test_nearby_material_context_does_not_hijack_brand_or_category(self) -> None:
        dictionary = _neighbor_context_dictionary()

        for message in (
            "I want to buy a Salomon shoes, rubber material",
            "I want to buy a Salomon shoes with rubber material",
            "salomon shoes rubber material",
        ):
            constraints = extract_constraints(message, dictionary=dictionary)
            self.assertEqual(constraints.brand, ("salomon",), message)
            self.assertEqual(constraints.category, ("shoes",), message)
            self.assertEqual(constraints.material, ("rubber",), message)

    def test_direct_post_context_resolves_only_the_matched_material(self) -> None:
        dictionary = _neighbor_context_dictionary()
        constraints = extract_constraints("rubber material", dictionary=dictionary)

        self.assertEqual(constraints.material, ("rubber",))
        self.assertEqual(constraints.color, ())
        self.assertEqual(constraints.style, ())

    def test_direct_pre_context_resolves_material_and_keeps_category(self) -> None:
        dictionary = _neighbor_context_dictionary()
        constraints = extract_constraints("made of leather shoes", dictionary=dictionary)

        self.assertEqual(constraints.material, ("leather",))
        self.assertEqual(constraints.category, ("shoes",))
        self.assertEqual(constraints.style, ())

    def test_direct_brand_context_keeps_other_matches_independent(self) -> None:
        dictionary = _neighbor_context_dictionary()
        constraints = extract_constraints(
            "brand salomon black shoes",
            dictionary=dictionary,
        )

        self.assertEqual(constraints.brand, ("salomon",))
        self.assertEqual(constraints.color, ("black",))
        self.assertEqual(constraints.category, ("shoes",))

    def test_directional_brand_and_use_case_context(self) -> None:
        dictionary = _neighbor_context_dictionary()

        from_constraints = extract_constraints(
            "shoes from salomon",
            dictionary=dictionary,
        )
        self.assertEqual(from_constraints.category, ("shoes",))
        self.assertEqual(from_constraints.brand, ("salomon",))

        use_case_constraints = extract_constraints(
            "shoes for running",
            dictionary=dictionary,
        )
        self.assertEqual(use_case_constraints.category, ("shoes",))
        self.assertEqual(use_case_constraints.use_case, ("running",))
        self.assertEqual(use_case_constraints.style, ())

    def test_no_generic_proximity_context_is_applied(self) -> None:
        dictionary = _neighbor_context_dictionary()
        constraints = extract_constraints(
            "alpha shoes rubber material",
            dictionary=dictionary,
        )

        self.assertEqual(constraints.brand, ("alpha",))
        self.assertEqual(constraints.category, ("shoes",))
        self.assertEqual(constraints.material, ("rubber",))

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

    def test_common_word_brand_collision_requires_explicit_brand_context(self) -> None:
        dictionary = _dictionary(
            ("brand", "find", 16),
            ("brand", "nike", 100),
            ("brand", "new balance", 100),
            ("brand", "air max", 100),
            ("brand", "air max 270", 100),
            ("category", "shoes", 1),
        )

        for message in ("find shoes", "Find shoes"):
            constraints = extract_constraints(message, dictionary=dictionary)
            self.assertEqual(constraints.category, ("shoes",))
            self.assertEqual(constraints.brand, ())
            self.assertIn("find", constraints.unmapped)

        brand_find = extract_constraints("brand find shoes", dictionary=dictionary)
        self.assertEqual(brand_find.brand, ("find",))
        self.assertEqual(brand_find.category, ("shoes",))

        for message in ("shoes from find", "made by find", "by find", "find brand"):
            constraints = extract_constraints(message, dictionary=dictionary)
            self.assertEqual(constraints.brand, ("find",), message)

        unrelated_from = extract_constraints("find shoes from nike", dictionary=dictionary)
        self.assertEqual(unrelated_from.brand, ("nike",))

        for message in ("nike shoes", "Nike shoes"):
            self.assertEqual(
                extract_constraints(message, dictionary=dictionary).brand,
                ("nike",),
            )
        self.assertEqual(
            extract_constraints("new balance shoes", dictionary=dictionary).brand,
            ("new balance",),
        )
        self.assertEqual(
            extract_constraints("air max 270", dictionary=dictionary).brand,
            ("air max 270",),
        )

    def test_collision_guard_only_applies_to_brand(self) -> None:
        dictionary = _dictionary(
            ("brand", "find", 16),
            ("style", "find", 100),
        )
        constraints = extract_constraints("find", dictionary=dictionary)
        self.assertEqual(constraints.brand, ())
        self.assertEqual(constraints.style, ("find",))

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


    def test_semantic_path_is_independent_of_structured_claims(self) -> None:
        dictionary = _dictionary(
            ("category", "rain boots", 10),
            ("feature", "waterproof", 10),
        )
        seen_phrases: list[str] = []

        def matcher(phrase: str) -> list[dict[str, object]]:
            seen_phrases.append(phrase)
            matches = {
                "rain boots": {
                    "canonical_id": "category:rain_boots",
                    "similarity": 0.95,
                },
                "waterproof": {
                    "canonical_id": "feature:waterproof",
                    "similarity": 0.93,
                },
            }
            match = matches.get(phrase)
            return [match] if match is not None else []

        constraints = extract_constraints(
            "I'm looking for rain boots. I'd like something waterproof.",
            dictionary=dictionary,
            semantic_matcher=matcher,
        )

        self.assertIn("rain boots", seen_phrases)
        self.assertIn("waterproof", seen_phrases)
        self.assertEqual(constraints.category, ("rain boots",))
        self.assertEqual(constraints.feature, ("waterproof",))
        self.assertEqual(
            constraints.semantic_constraints.category,
            ("rain boots",),
        )
        self.assertEqual(
            constraints.semantic_constraints.feature,
            ("waterproof",),
        )
        self.assertEqual(constraints.structured_only().category, ("rain boots",))
        self.assertEqual(constraints.structured_only().feature, ("waterproof",))


if __name__ == "__main__":
    unittest.main()
