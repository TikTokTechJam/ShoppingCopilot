from __future__ import annotations

import unittest

from scripts.console_canonical_test import (
    ProductTagIndex,
    analyze_utterance,
    canonical_tags,
    count_products_with_all_tags,
)
from starter.routing.constraints import ShoppingConstraints, extract_constraints


def fixture_index() -> ProductTagIndex:
    return ProductTagIndex(
        postings={
            "category": {},
            "brand": {"nike": frozenset({"A", "B"})},
            "color": {
                "black": frozenset({"A"}),
                "white": frozenset({"B"}),
                "red": frozenset({"C"}),
            },
            "material": {"leather": frozenset({"A", "C"})},
            "style": {"casual": frozenset({"A"})},
            "feature": {},
            "use_case": {"running": frozenset({"A", "B"})},
        },
        product_count=3,
    )


class ConsoleCanonicalTest(unittest.TestCase):
    def test_all_tag_intersection(self) -> None:
        constraints = ShoppingConstraints(
            brand=("nike",), color=("black",), use_case=("running",)
        )
        self.assertEqual(count_products_with_all_tags(constraints, fixture_index()), 1)

    def test_multiple_values_are_required_within_one_attribute(self) -> None:
        constraints = ShoppingConstraints(color=("black", "red"))
        self.assertEqual(count_products_with_all_tags(constraints, fixture_index()), 0)

    def test_product_normalization(self) -> None:
        index = ProductTagIndex(
            postings={
                attribute: {} for attribute in (
                    "category", "brand", "color", "material", "style", "feature", "use_case"
                )
            },
            product_count=1,
        )
        index.postings["category"] = {"casual dress socks": frozenset({"A"})}
        constraints = ShoppingConstraints(category=("casual_dress_socks",))
        self.assertEqual(count_products_with_all_tags(constraints, index), 1)

    def test_empty_tags_have_no_intersection_count(self) -> None:
        self.assertIsNone(count_products_with_all_tags(ShoppingConstraints(), fixture_index()))

    def test_canonicalizer_output_feeds_intersection(self) -> None:
        constraints = extract_constraints("nike black running")
        tags = canonical_tags(constraints)
        self.assertEqual(tags.get("brand"), ["nike"])
        self.assertEqual(tags.get("color"), ["black"])
        self.assertNotIn("use_case", tags)

    def test_analyze_utterance_uses_existing_router_result(self) -> None:
        intent, tags, count = analyze_utterance("nike black running", fixture_index())
        self.assertEqual(intent, "BUYING")
        self.assertEqual(tags, {
            "brand": ["nike"],
            "color": ["black"],
        })
        self.assertEqual(count, 1)

    def test_console_helpers_do_not_require_embeddings_or_retrieval(self) -> None:
        intent, tags, count = analyze_utterance(
            "something completely unknown xyzabc", fixture_index()
        )
        self.assertEqual(intent, "BROWSING")
        self.assertEqual(tags, {})
        self.assertIsNone(count)


if __name__ == "__main__":
    unittest.main()
