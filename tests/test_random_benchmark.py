from __future__ import annotations

import unittest

from scripts.random_catalog_benchmark import (
    free_form_conversation,
    independent_clues,
    percentile_95,
    scenario_mix,
)


class RandomBenchmarkTest(unittest.TestCase):
    def test_ten_sample_mix_covers_all_official_scenarios(self) -> None:
        scenarios = scenario_mix(10)
        self.assertEqual(scenarios.count("buying"), 4)
        self.assertEqual(scenarios.count("browsing"), 4)
        self.assertEqual(scenarios.count("intent_override"), 1)
        self.assertEqual(scenarios.count("boundary"), 1)

    def test_override_conversation_delays_conversion_until_new_intent(self) -> None:
        product = {
            "title": "Example blue shoe",
            "features": ["nylon", "Rubber sole", "Breathable"],
            "details": {"department": "womens"},
            "categories": ["Clothing", "Women", "Shoes"],
            "price": 50,
        }
        messages, allowed_turn = free_form_conversation(product, "intent_override")
        self.assertEqual(allowed_turn, 3)
        self.assertIn("Actually, ignore", messages[2])
        self.assertEqual(len(messages), 10)

    def test_nearest_rank_p95_is_deterministic(self) -> None:
        self.assertEqual(percentile_95([0.1, 0.2, 0.3, 0.4]), 0.4)
        self.assertEqual(percentile_95([]), 0.0)

    def test_clues_are_deterministic_and_do_not_copy_the_title(self) -> None:
        product = {
            "parent_asin": "TARGET",
            "title": "Do Not Copy This Exact Product Title",
            "features": ["breathable mesh", "rubber sole", "water resistant", "lace closure"],
            "details": {"department": "womens"},
            "description": ["made for trail use"],
            "categories": ["Women", "Shoes"],
        }
        first = independent_clues(product, 42)
        second = independent_clues(product, 42)
        self.assertEqual(first, second)
        self.assertNotIn(product["title"], first)


if __name__ == "__main__":
    unittest.main()
