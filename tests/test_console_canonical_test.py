from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.console_canonical_test import (
    ProductTagIndex,
    analyze_utterance,
    build_product_index,
    canonical_tags,
    count_products_with_all_tags,
    print_analysis,
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
        prices={"A": 40.0, "B": 60.0, "C": None},
    )


def budget_fixture_index() -> ProductTagIndex:
    return ProductTagIndex(
        postings={
            "category": {"earrings": frozenset({"A", "B", "C"})},
            "brand": {},
            "color": {"gold": frozenset({"A", "B", "C"})},
            "material": {},
            "style": {},
            "feature": {},
            "use_case": {},
        },
        product_count=3,
        prices={"A": 40.0, "B": 60.0, "C": None},
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

    def test_build_product_index_reads_top_level_prices_and_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "annotations.jsonl"
            records = [
                {
                    "parent_asin": "A",
                    "price": 40.0,
                    "facts": {"category": ["earrings"]},
                    "annotation": {"status": "success"},
                },
                {
                    "parent_asin": "B",
                    "price": None,
                    "facts": {"category": ["earrings"]},
                    "annotation": {"status": "success"},
                },
                {
                    "parent_asin": "FAILED",
                    "price": 10.0,
                    "facts": {"category": ["earrings"]},
                    "annotation": {"status": "failed"},
                },
            ]
            source.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            index = build_product_index(source)

        self.assertEqual(index.product_count, 2)
        self.assertEqual(index.prices, {"A": 40.0, "B": None})
        self.assertEqual(index.postings["category"]["earrings"], frozenset({"A", "B"}))

    def test_max_price_filters_categorical_intersection(self) -> None:
        constraints = ShoppingConstraints(
            category=("earrings",), color=("gold",), price_max=50
        )
        self.assertEqual(count_products_with_all_tags(constraints, budget_fixture_index()), 1)

    def test_null_price_is_excluded_only_with_active_budget(self) -> None:
        index = budget_fixture_index()
        constrained = ShoppingConstraints(
            category=("earrings",), color=("gold",), price_max=50
        )
        unconstrained = ShoppingConstraints(category=("earrings",), color=("gold",))
        self.assertEqual(count_products_with_all_tags(constrained, index), 1)
        self.assertEqual(count_products_with_all_tags(unconstrained, index), 3)

    def test_minimum_and_range_price_filters(self) -> None:
        index = budget_fixture_index()
        self.assertEqual(count_products_with_all_tags(ShoppingConstraints(price_min=50), index), 1)
        self.assertEqual(
            count_products_with_all_tags(
                ShoppingConstraints(price_min=30, price_max=50), index
            ),
            1,
        )

    def test_price_only_query_is_counted(self) -> None:
        constraints = extract_constraints("under $50")
        self.assertEqual(canonical_tags(constraints), {})
        self.assertEqual(count_products_with_all_tags(constraints, budget_fixture_index()), 1)
        intent, tags, count = analyze_utterance("under $50", budget_fixture_index())
        self.assertEqual(intent, "BUYING")
        self.assertEqual(tags, {"budget": {"max": 50}})
        self.assertEqual(count, 1)

    def test_trailing_currency_symbol_uses_shared_parser(self) -> None:
        self.assertEqual(extract_constraints("less than 50$").price_max, 50)
        self.assertEqual(extract_constraints("less than $50").price_max, 50)

    def test_analysis_prints_budget_and_constraint_label(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            print_analysis("BUYING", {"budget": {"max": 50}}, 1)
        rendered = output.getvalue()
        self.assertIn('"budget": {', rendered)
        self.assertIn('"max": 50', rendered)
        self.assertIn("Products matching all constraints: 1", rendered)

        empty_output = io.StringIO()
        with redirect_stdout(empty_output):
            print_analysis("BROWSING", {}, None)
        self.assertIn("No constraints matched.", empty_output.getvalue())

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
