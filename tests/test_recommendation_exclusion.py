from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starter.agent import Agent
from starter.retrieval import ProductRetriever
from starter.routing.constraints import ShoppingConstraints
from starter.session import SessionManager


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_catalog(root: Path) -> Path:
    catalog_path = root / "catalog.jsonl"
    rows: list[dict[str, object]] = []
    for index in range(6):
        rows.append(
            {
                "parent_asin": f"S{index}",
                "categories": ["shoes"],
                "price": float(10 + index),
            }
        )
    for index in range(6):
        rows.append(
            {
                "parent_asin": f"E{index}",
                "categories": ["earrings"],
                "price": float(10 + index),
            }
        )
    write_jsonl(catalog_path, rows)
    return catalog_path



def sequenced_extract(deltas):
    """One delta per distinct user message, whatever the Agent's call count.

    The Agent reads a message unscoped to decide whether it is an override,
    then may re-read it scoped to the attribute it asked about. Keying on the
    message keeps the fixture aligned with turns rather than with calls.
    """

    cache: dict[str, object] = {}

    def _extract(message, **_kwargs):
        if message not in cache:
            cache[message] = next(deltas)
        return cache[message]

    return _extract


class FixedRouter:
    def classify(self, _message: str) -> SimpleNamespace:
        return SimpleNamespace(intent="BUYING")


class RecommendationExclusionTests(unittest.TestCase):
    def make_retriever(self, root: Path) -> ProductRetriever:
        return ProductRetriever(
            build_catalog(root),
            facts_path=root / "missing-facts.jsonl",
            embeddings_path=root / "missing-embeddings.npy",
            metadata_path=root / "missing-metadata.json",
        )

    def make_agent(self, root: Path) -> Agent:
        return Agent(
            build_catalog(root),
            facts_path=root / "missing-facts.jsonl",
            embeddings_path=root / "missing-embeddings.npy",
            metadata_path=root / "missing-metadata.json",
            router=FixedRouter(),
        )

    def test_first_turn_starts_empty_and_ordinary_turns_accumulate_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            shoes = ShoppingConstraints(category=("shoes",))

            with patch(
                "starter.agent.constraint_module.extract_constraints",
                return_value=shoes,
            ):
                first = agent.respond("session", "shoes", 1, 3)
                first_ids = [item["parent_asin"] for item in first["recommendations"]]
                state = agent.sessions.get("session")
                self.assertEqual(state.excluded_recommendations, set())

                second = agent.respond("session", "show me more", 2, 3)
                second_ids = [item["parent_asin"] for item in second["recommendations"]]
                state = agent.sessions.get("session")
                self.assertEqual(
                    state.excluded_recommendations,
                    set(first_ids),
                )

                third = agent.respond("session", "show me different ones", 3, 3)
                third_ids = [item["parent_asin"] for item in third["recommendations"]]

            self.assertEqual(first_ids, ["S0", "S1", "S2"])
            self.assertEqual(second_ids, ["S3", "S4", "S5"])
            self.assertEqual(third_ids, ["E0", "E1", "E2"])
            self.assertTrue(set(first_ids).isdisjoint(second_ids))
            self.assertTrue(set(first_ids + second_ids).isdisjoint(third_ids))
            self.assertEqual(
                state.excluded_recommendations,
                set(first_ids + second_ids),
            )

    def test_exclusions_apply_to_buying_browsing_and_fallback_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = self.make_retriever(root)
            constraints = ShoppingConstraints(brand=("unknown brand",))
            excluded = {"S0", "S1"}

            for mode in ("BUYING", "BROWSING"):
                ranked = retriever.retrieve(
                    mode,
                    "",
                    constraints,
                    limit=5,
                    excluded_asins=excluded,
                )
                ids = [candidate.parent_asin for candidate in ranked]
                self.assertEqual(ids, ["S2", "S3", "S4", "S5", "E0"])
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(excluded.isdisjoint(ids))

    def test_exclusions_respect_active_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = self.make_retriever(root)
            ranked = retriever.retrieve(
                "BUYING",
                "",
                ShoppingConstraints(category=("shoes",), price_max=12.0),
                limit=10,
                excluded_asins={"S0"},
            )

            ids = [candidate.parent_asin for candidate in ranked]
            self.assertNotIn("S0", ids)
            self.assertTrue(
                all(retriever.product_by_asin[asin].price is not None for asin in ids)
            )
            self.assertTrue(
                all(retriever.product_by_asin[asin].price <= 12.0 for asin in ids)
            )

    def test_intent_override_clears_exclusion_history_before_new_goal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            deltas = iter(
                (
                    ShoppingConstraints(category=("shoes",)),
                    ShoppingConstraints(category=("shoes",)),
                    ShoppingConstraints(category=("earrings",)),
                )
            )

            with patch(
                "starter.agent.constraint_module.extract_constraints",
                side_effect=sequenced_extract(deltas),
            ):
                first = agent.respond("session", "shoes", 1, 3)
                agent.respond("session", "show me more", 2, 3)
                override = agent.respond(
                    "session",
                    "forget that, I want earrings instead",
                    3,
                    3,
                )

            override_ids = [item["parent_asin"] for item in override["recommendations"]]
            state = agent.sessions.get("session")
            self.assertEqual(override_ids, ["E0", "E1", "E2"])
            self.assertTrue(
                set(item["parent_asin"] for item in first["recommendations"]).isdisjoint(
                    override_ids
                )
            )
            self.assertEqual(state.excluded_recommendations, set())
            self.assertEqual(state.last_recommendations, tuple(override_ids))

    def test_reset_clears_both_recommendation_histories(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {"preference": "outdoors"})
        manager.set_recommendations("session", ["A", "B"])
        manager.promote_last_recommendations("session")
        self.assertEqual(state.last_recommendations, ("A", "B"))
        self.assertEqual(state.excluded_recommendations, {"A", "B"})

        manager.reset("session", {})
        state = manager.get("session")
        self.assertEqual(state.last_recommendations, ())
        self.assertEqual(state.excluded_recommendations, set())

        manager.set_recommendations("session", ["C"])
        manager.promote_last_recommendations("session")
        manager.reset("session", {"preference": "indoors"})
        state = manager.get("session")
        self.assertEqual(state.last_recommendations, ())
        self.assertEqual(state.excluded_recommendations, set())


if __name__ == "__main__":
    unittest.main()
