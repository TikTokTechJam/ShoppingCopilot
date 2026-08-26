from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent, _category_aliases, _normalize


class AgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "SHOE_BLUE",
                "title": "Blue trail running shoe",
                "features": ["nylon", "Rubber sole", "Breathable trail mesh"],
                "details": {"department": "womens"},
                "description": ["cushioned outdoor runner"],
                "categories": ["Clothing", "Women", "Shoes"],
                "store": "Trail Example",
                "average_rating": 4.7,
                "rating_number": 500,
                "price": 59.0,
            },
            {
                "parent_asin": "SHOE_BLACK",
                "title": "Black office shoe",
                "features": ["leather", "Synthetic sole", "Formal slip on"],
                "details": {"department": "womens"},
                "description": ["work loafer"],
                "categories": ["Clothing", "Women", "Shoes"],
                "store": "Office Example",
                "average_rating": 4.2,
                "rating_number": 50,
                "price": 69.0,
            },
            {
                "parent_asin": "HAT_RED",
                "title": "Red sun hat",
                "features": ["cotton", "Wide brim"],
                "details": {"department": "womens"},
                "description": ["summer hat"],
                "categories": ["Clothing", "Women", "Hats"],
                "store": "Hat Example",
                "average_rating": 4.4,
                "rating_number": 100,
                "price": 19.0,
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(self.catalog_path)
        self.addCleanup(self.agent.connection.close)

    def test_general_category_aliases_cover_labels_and_paths(self) -> None:
        aliases = _category_aliases(["Clothing", "Women", "Shoes"])
        self.assertIn("shoes", aliases)
        self.assertIn("women shoes", aliases)
        self.assertIn("clothing women shoes", aliases)

    def test_multi_turn_constraints_accumulate(self) -> None:
        self.agent.reset("browse", {"preference_tags": ["comfort"]})
        first = self.agent.respond(
            "browse",
            "I'm looking for Women Shoes, but I'm still exploring.",
            1,
            10,
        )
        second = self.agent.respond(
            "browse",
            "For that, what matters is: nylon; color: blue.",
            2,
            10,
        )
        state = self.agent._sessions["browse"]
        self.assertEqual(state.category, "women shoes")
        self.assertEqual(state.constraints, ["nylon", "color blue"])
        self.assertIn(
            first["ask_attribute"],
            {"material", "feature", "style", "use_case", "size", "color", "brand", "budget", "category"},
        )
        self.assertEqual(second["recommendations"][0]["parent_asin"], "SHOE_BLUE")

    def test_intent_override_erases_stale_preference(self) -> None:
        self.agent.reset("override", {})
        self.agent.respond(
            "override",
            "I'm looking for Women Shoes. Formal slip on",
            1,
            10,
        )
        response = self.agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is: nylon.",
            3,
            10,
        )
        state = self.agent._sessions["override"]
        self.assertEqual(state.constraints, [_normalize("nylon")])
        self.assertNotIn("formal slip on", state.constraints)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "SHOE_BLUE")

    def test_free_form_query_and_override_work_for_interactive_clients(self) -> None:
        self.agent.reset("free-form", {"preference_tags": ["comfort"]})
        first = self.agent.respond(
            "free-form",
            "I need blue breathable trail shoes for women, preferably nylon.",
            1,
            10,
        )
        changed = self.agent.respond(
            "free-form",
            "Actually, I changed my mind: find a formal leather office shoe instead.",
            2,
            10,
        )
        state = self.agent._sessions["free-form"]
        self.assertEqual(first["recommendations"][0]["parent_asin"], "SHOE_BLUE")
        self.assertEqual(changed["recommendations"][0]["parent_asin"], "SHOE_BLACK")
        self.assertEqual(len(state.constraints), 1)
        self.assertNotIn("blue breathable", state.constraints[0])

    def test_no_preference_reply_does_not_become_a_constraint(self) -> None:
        self.agent.reset("boundary", {})
        self.agent.respond(
            "boundary",
            "I'm looking for Women Shoes, but I'm still exploring.",
            1,
            10,
        )
        self.agent.respond(
            "boundary",
            "I don't have a preference for material; please use your judgment.",
            2,
            10,
        )
        self.assertEqual(self.agent._sessions["boundary"].constraints, [])

    def test_clarification_never_uses_a_catch_all_attribute(self) -> None:
        self.agent.reset("questions", {})
        message = "I'm looking for Women Shoes, but I'm still exploring."
        allowed = {"material", "feature", "style", "use_case", "size", "color", "brand", "budget", "category", None}
        for turn in range(1, 11):
            response = self.agent.respond("questions", message, turn, 10)
            self.assertIn(response["ask_attribute"], allowed)
            self.assertNotEqual(response["ask_attribute"], "other")
            attribute = response["ask_attribute"] or "feature"
            message = f"I don't have a preference for {attribute}; please use your judgment."

    def test_reset_is_required(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("missing", "hello", 1, 10)


if __name__ == "__main__":
    unittest.main()
