from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.human_query_benchmark import NEUTRAL_FILLER, normalize_text


class HumanBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = Path(__file__).resolve().parents[1] / "benchmarks" / "human_queries_20.json"
        cls.fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def test_fixture_contains_twenty_unique_human_cases(self) -> None:
        cases = self.fixture["cases"]
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case["case_id"] for case in cases}), 20)
        self.assertEqual(len({case["target_parent_asin"] for case in cases}), 20)
        self.assertIn("read individually", self.fixture["authoring_method"])

    def test_scenario_mix_matches_declared_distribution(self) -> None:
        actual = Counter(case["scenario_type"] for case in self.fixture["cases"])
        self.assertEqual(actual, Counter(self.fixture["scenario_mix"]))
        self.assertEqual(actual, Counter({"buying": 8, "browsing": 8, "intent_override": 3, "boundary": 1}))

    def test_messages_are_nonempty_and_do_not_contain_target_asins(self) -> None:
        for case in self.fixture["cases"]:
            messages = case["messages"]
            self.assertTrue(messages)
            joined = normalize_text(" ".join(messages))
            self.assertNotIn(case["target_parent_asin"].lower(), joined)
            self.assertTrue(all(message.strip() for message in messages))

    def test_neutral_filler_carries_no_new_preference(self) -> None:
        self.assertIn("don't have an additional preference", NEUTRAL_FILLER.lower())


if __name__ == "__main__":
    unittest.main()
