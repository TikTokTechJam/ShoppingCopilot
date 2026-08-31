from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from starter.agent import Agent
from starter.retrieval import (
    DENSE_SCORE_WEIGHT,
    RATING_DEFAULT_WEIGHT,
    STRUCTURED_FIELD_WEIGHTS,
    ProductRetriever,
    normalized_rating,
)
from starter.routing.constraints import ShoppingConstraints


class ExplodingEncoder:
    def __call__(self, _query: str) -> object:
        raise AssertionError("dense query encoder must not be called")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_fixture(root: Path) -> tuple[Path, Path]:
    catalog_path = root / "catalog.jsonl"
    facts_path = root / "annotations.jsonl"

    write_jsonl(
        catalog_path,
        [
            {
                "parent_asin": "A",
                "categories": ["women", "shoes", "athletic shoes"],
                "store": "Raw Store Value Must Not Override V4",
                "price": 35.0,
            },
            {
                "parent_asin": "B",
                "categories": ["shoes"],
                "store": "Nike",
                "price": 25.0,
            },
            {
                "parent_asin": "C",
                "categories": ["shirts"],
                "store": "Catalog Store",
                "price": None,
            },
            {
                "parent_asin": "X",
                "categories": ["shoes"],
                "store": "Deluxe Gems",
                "price": None,
            },
            {
                "parent_asin": "D",
                "categories": ["shoes"],
                "store": "Other Store",
                "price": 40.0,
            },
        ],
    )
    write_jsonl(
        facts_path,
        [
            {
                "parent_asin": "A",
                "price": 35.0,
                "facts": {
                    "category": ["shoes"],
                    "brand": ["New_Balance", "Gel Kayano Trainer"],
                    "color": ["Black"],
                    "material": ["High_Quality_Mesh"],
                    "style": ["High_Waisted"],
                    "feature": ["Four_Way_Stretch"],
                    "use_case": ["Trail_Running"],
                },
                "annotation": {"status": "success", "prompt_version": "v4"},
            },
            {
                "parent_asin": "B",
                "price": 25.0,
                "facts": {
                    "category": ["shoes"],
                    "brand": ["Nike"],
                    "color": ["red"],
                    "material": ["mesh"],
                    "style": ["casual"],
                    "feature": ["lightweight"],
                    "use_case": ["running"],
                },
                "annotation": {"status": "success", "prompt_version": "v4"},
            },
            {
                "parent_asin": "C",
                "price": None,
                "facts": {
                    "category": ["shirts"],
                    "brand": [],
                    "color": [],
                    "material": [],
                    "style": [],
                    "feature": [],
                    "use_case": [],
                },
                "annotation": {"status": "success", "prompt_version": "v4"},
            },
            {
                "parent_asin": "BAD",
                "facts": {
                    "brand": ["must not load"],
                },
                "annotation": {"status": "failed", "prompt_version": "v4"},
            },
        ],
    )
    return catalog_path, facts_path


class ProductRetrieverTests(unittest.TestCase):
    def make_retriever(self, root: Path, **kwargs: object) -> ProductRetriever:
        catalog_path, facts_path = build_fixture(root)
        return ProductRetriever(
            catalog_path,
            facts_path=facts_path,
            embeddings_path=root / "missing.npy",
            metadata_path=root / "missing.json",
            **kwargs,
        )

    def test_v4_nested_facts_load_as_normalized_lookup_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            facts = retriever.product_by_asin["A"].facts

            self.assertEqual(facts["brand"], ("new balance", "gel kayano trainer"))
            self.assertEqual(facts["color"], ("black",))
            self.assertEqual(facts["material"], ("high quality mesh",))
            self.assertEqual(facts["style"], ("high waisted",))
            self.assertEqual(facts["feature"], ("four way stretch",))
            self.assertEqual(facts["use_case"], ("trail running",))
            self.assertNotIn("BAD", retriever._facts_by_asin)

    def test_missing_annotation_stays_valid_without_invented_semantic_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            product = retriever.product_by_asin["X"]

            self.assertIn("X", retriever.valid_asins)
            self.assertEqual(product.facts["category"], ("shoes",))
            for field_name in ("brand", "color", "material", "style", "feature", "use_case"):
                self.assertEqual(product.facts.get(field_name, ()), ())

    def test_category_exact_match_and_normalized_facts_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            constraints = ShoppingConstraints(
                category=("shoes",),
                brand=("new balance",),
                material=("high quality mesh",),
                style=("high waisted",),
            )
            ranked = retriever.retrieve("BUYING", "", constraints, limit=5)
            best = next(item for item in ranked if item.parent_asin == "A")

            # Buying ranks on BM25 alone, so structured points no longer
            # decide the order. They remain on the Candidate as diagnostics,
            # and the matched-constraint labels still come from the exact
            # posting-list match.
            self.assertEqual(best.matched_constraints[:4], (
                "category:shoes",
                "brand:new balance",
                "material:high quality mesh",
                "style:high waisted",
            ))
            self.assertGreater(
                best.constraint_score,
                next(item for item in ranked if item.parent_asin == "C").constraint_score,
            )

    def test_tier_two_fields_outrank_tier_three_fields(self) -> None:
        cases = (
            ShoppingConstraints(category=("shoes",), brand=("new balance",), style=("casual",)),
            ShoppingConstraints(category=("shoes",), color=("black",), style=("casual",)),
            ShoppingConstraints(
                category=("shoes",),
                material=("high quality mesh",),
                style=("casual",),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            for constraints in cases:
                ranked = retriever.retrieve("BUYING", "", constraints, limit=5)
                best = next(item for item in ranked if item.parent_asin == "A")
                # The tier ordering now lives in the structured *points*, which
                # are diagnostics. Buying rank itself is decided by BM25.
                for other in ranked:
                    if other.parent_asin != "A":
                        self.assertGreaterEqual(
                            best.constraint_score, other.constraint_score
                        )

            self.assertGreater(STRUCTURED_FIELD_WEIGHTS["brand"], STRUCTURED_FIELD_WEIGHTS["style"])
            self.assertGreater(STRUCTURED_FIELD_WEIGHTS["color"], STRUCTURED_FIELD_WEIGHTS["style"])
            self.assertGreater(STRUCTURED_FIELD_WEIGHTS["material"], STRUCTURED_FIELD_WEIGHTS["style"])

    def test_tier_three_mismatch_does_not_eliminate_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            constraints = ShoppingConstraints(category=("shoes",), style=("casual",))
            ranked = retriever.retrieve("BUYING", "", constraints, limit=5)

            self.assertIn("A", [item.parent_asin for item in ranked])
            self.assertIn("style:required", next(
                item for item in ranked if item.parent_asin == "A"
            ).violated_constraints)

    def test_budget_excludes_unknown_and_violating_prices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            constraints = ShoppingConstraints(category=("shoes",), price_max=30.0)
            ranked = retriever.retrieve("BUYING", "", constraints, limit=10)

            self.assertEqual([item.parent_asin for item in ranked], ["B"])

    def test_buying_bm25_changes_order_when_dense_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            facts = root / "facts.jsonl"
            write_jsonl(
                catalog,
                [
                    {
                        "parent_asin": "A",
                        "title": "Plain walking shoes",
                        "categories": ["shoes"],
                    },
                    {
                        "parent_asin": "B",
                        "title": "Waterproof black rain boots",
                        "categories": ["shoes"],
                    },
                ],
            )
            write_jsonl(
                facts,
                [
                    {"parent_asin": "A", "facts": {"category": ["shoes"]}},
                    {"parent_asin": "B", "facts": {"category": ["shoes"]}},
                ],
            )
            retriever = ProductRetriever(
                catalog,
                facts_path=facts,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
            )

            ranked = retriever.retrieve(
                "BUYING",
                "waterproof rain boots",
                ShoppingConstraints(category=("shoes",)),
                limit=2,
            )

            self.assertEqual([item.parent_asin for item in ranked], ["B", "A"])
            self.assertGreater(ranked[0].bm25_score, ranked[1].bm25_score)
            self.assertEqual(ranked[0].score, ranked[0].ranking_score)

    def test_without_budget_unknown_price_remains_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            ranked = retriever.retrieve(
                "BUYING",
                "",
                ShoppingConstraints(category=("shoes",)),
                limit=10,
            )

            self.assertIn("X", [item.parent_asin for item in ranked])

    def test_dense_dependencies_are_optional_and_never_called_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(
                Path(directory),
                query_encoder=ExplodingEncoder(),
            )

            self.assertFalse(retriever.has_dense_index)
            self.assertFalse(retriever.dense_available)
            with patch.object(
                retriever,
                "_dense_scores",
                side_effect=AssertionError("dense search must be skipped"),
            ):
                ranked = retriever.retrieve(
                    "BUYING",
                    "black shoes",
                    ShoppingConstraints(category=("shoes",)),
                    limit=3,
                )

            self.assertEqual(len(ranked), 3)
            # BGE is expansion-only: it reaches the score through the
            # BM25 concept groups, never as a separate canonical term.
            self.assertEqual(DENSE_SCORE_WEIGHT, 0.0)

    def test_structured_browsing_and_empty_browsing_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = self.make_retriever(root)
            structured = retriever.retrieve(
                "BROWSING",
                "",
                ShoppingConstraints(brand=("new balance",)),
                limit=3,
            )
            self.assertEqual(structured[0].parent_asin, "A")

            empty_one = retriever.retrieve("BROWSING", "", ShoppingConstraints(), limit=5)
            empty_two = retriever.retrieve("BROWSING", "", ShoppingConstraints(), limit=5)
            self.assertEqual(empty_one, empty_two)

    def test_top10_completion_is_valid_unique_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            facts_path = root / "empty.jsonl"
            write_jsonl(
                catalog_path,
                [
                    {"parent_asin": f"P{index:02d}", "categories": ["shoes"], "price": 10.0}
                    for index in range(12)
                ],
            )
            facts_path.write_text("", encoding="utf-8")
            retriever = ProductRetriever(
                catalog_path,
                facts_path=facts_path,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
            )

            first = retriever.retrieve("BROWSING", "", ShoppingConstraints(), limit=10)
            second = retriever.retrieve("BROWSING", "", ShoppingConstraints(), limit=10)
            first_ids = [item.parent_asin for item in first]

            self.assertEqual(first_ids, [item.parent_asin for item in second])
            self.assertEqual(len(first_ids), 10)
            self.assertEqual(len(set(first_ids)), 10)
            self.assertTrue(set(first_ids) <= set(retriever.valid_asins))

    def test_agent_runs_without_embeddings_and_returns_valid_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path, facts_path = build_fixture(root)
            agent = Agent(
                catalog_path,
                facts_path=facts_path,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
                query_encoder=ExplodingEncoder(),
            )
            agent.reset("test-session", {})

            with patch(
                "starter.agent.constraint_module.extract_constraints",
                return_value=ShoppingConstraints(category=("shoes",), brand=("nike",)),
            ):
                response = agent.respond("test-session", "nike shoes", 1, 10)

            ids = [item["parent_asin"] for item in response["recommendations"]]
            self.assertTrue(ids)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(set(ids) <= set(agent.retriever.valid_asins))

    def test_mode_weights_make_browsing_dense_dominant_and_buying_structured_dominant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            constraints = ShoppingConstraints(
                category=("shoes",),
                brand=("new balance",),
                color=("black",),
                material=("high quality mesh",),
                style=("high waisted",),
                feature=("four way stretch",),
                use_case=("trail running",),
            )
            dense_scores = {"A": 0.10, "B": 0.90, "C": 0.0, "X": 0.0, "D": 0.0}
            with patch.object(
                retriever,
                "_dense_scores",
                return_value=dense_scores,
            ), patch.object(
                retriever,
                "_bm25_scores",
                return_value={},
            ), patch.object(
                type(retriever),
                "dense_available",
                new_callable=PropertyMock,
                return_value=True,
            ):
                browsing = retriever.retrieve("BROWSING", "semantic query", constraints, limit=2)
                buying = retriever.retrieve("BUYING", "semantic query", constraints, limit=2)

            self.assertEqual(browsing[0].parent_asin, "B")
            self.assertEqual(buying[0].parent_asin, "A")
            for candidate in browsing:
                expected = candidate.fusion_score + RATING_DEFAULT_WEIGHT * normalized_rating(
                    candidate.rating
                )
                self.assertAlmostEqual(candidate.score, expected)
                self.assertEqual(candidate.score, candidate.ranking_score)

    def test_browsing_dense_can_surface_zero_structured_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            dense_scores = {"A": 0.10, "B": 0.90, "C": 0.20, "X": 0.30, "D": 0.40}
            with patch.object(
                retriever,
                "_dense_scores",
                return_value=dense_scores,
            ), patch.object(
                retriever,
                "_bm25_scores",
                return_value={},
            ), patch.object(
                type(retriever),
                "dense_available",
                new_callable=PropertyMock,
                return_value=True,
            ):
                ranked = retriever.retrieve(
                    "BROWSING",
                    "semantic query",
                    ShoppingConstraints(category=("not in this catalog",)),
                    limit=3,
                )

            self.assertEqual(ranked[0].parent_asin, "B")
            self.assertEqual(ranked[0].constraint_score, 0.0)

    def test_browsing_rrf_uses_bm25_as_a_real_rank_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            dense_scores = {"A": 0.90, "B": 0.70, "C": 0.80, "X": 0.60, "D": 0.50}
            bm25_scores = {"A": 0.10, "B": 0.90, "C": 0.20, "X": 0.30, "D": 0.40}
            with patch.object(
                retriever,
                "_dense_scores",
                return_value=dense_scores,
            ), patch.object(
                retriever,
                "_bm25_scores",
                return_value=bm25_scores,
            ), patch.object(
                type(retriever),
                "dense_available",
                new_callable=PropertyMock,
                return_value=True,
            ):
                ranked = retriever.retrieve(
                    "BROWSING",
                    "semantic query",
                    ShoppingConstraints(),
                    limit=5,
                )

            self.assertEqual(ranked[0].parent_asin, "B")
            self.assertGreater(ranked[0].fusion_score, ranked[1].fusion_score)
            self.assertEqual(ranked[0].score, ranked[0].ranking_score)

    def test_browsing_uses_expanded_bm25_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            constraints = ShoppingConstraints(feature=("lightweight",))
            with patch.object(
                retriever,
                "_bm25_scores",
                return_value={"A": 1.0},
            ) as expanded_bm25, patch.object(
                retriever,
                "_raw_bm25_scores",
                side_effect=AssertionError("Browsing must use the expanded BM25 route"),
            ):
                retriever.retrieve(
                    "BROWSING",
                    "lightweight shoes",
                    constraints,
                    limit=1,
                )

            expanded_bm25.assert_called_once_with(
                "lightweight shoes",
                {"A", "B", "C", "X", "D"},
                constraints,
                None,
            )

    def test_dense_ranking_preserves_budget_and_recommendation_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            dense_scores = {"A": 0.99, "B": 0.10, "C": 0.95, "X": 0.90, "D": 0.80}
            with patch.object(
                retriever,
                "_dense_scores",
                return_value=dense_scores,
            ), patch.object(
                type(retriever),
                "dense_available",
                new_callable=PropertyMock,
                return_value=True,
            ):
                budgeted = retriever.retrieve(
                    "BROWSING",
                    "semantic query",
                    ShoppingConstraints(category=("shoes",), price_max=30.0),
                    limit=10,
                )
                excluded = retriever.retrieve(
                    "BROWSING",
                    "semantic query",
                    ShoppingConstraints(category=("shoes",)),
                    limit=10,
                    excluded_asins={"B"},
                )

            self.assertEqual([candidate.parent_asin for candidate in budgeted], ["B"])
            self.assertNotIn("B", [candidate.parent_asin for candidate in excluded])

    def test_debug_ranking_matches_production_ranking_with_dense_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self.make_retriever(Path(directory))
            dense_scores = {"A": 0.1, "B": 0.9, "C": 0.2, "X": 0.3, "D": 0.4}
            with patch.object(
                retriever,
                "_dense_scores",
                return_value=dense_scores,
            ), patch.object(
                type(retriever),
                "dense_available",
                new_callable=PropertyMock,
                return_value=True,
            ):
                production = retriever.retrieve(
                    "BROWSING",
                    "semantic query",
                    ShoppingConstraints(category=("shoes",)),
                    limit=5,
                )
                debug = retriever.debug_rank_all(
                    "BROWSING",
                    "semantic query",
                    ShoppingConstraints(category=("shoes",)),
                )

            self.assertEqual(
                [candidate.parent_asin for candidate in production],
                [candidate.parent_asin for candidate in debug[:5]],
            )


if __name__ == "__main__":
    unittest.main()
