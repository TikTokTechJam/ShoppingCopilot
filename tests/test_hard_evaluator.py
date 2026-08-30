from __future__ import annotations

import random
import threading
import time
import unittest
from pathlib import Path

from evaluator.hard_evaluator import (
    ALLOWED_ATTRIBUTES,
    SCENARIO_COUNTS,
    add_score_fields,
    evaluate,
    fact_id,
    load_catalog_ids,
    load_jsonl,
    metric_summary,
    normalize_recommendations,
    parse_fact_id,
    select_sessions,
    simulate_customer_reply,
    validate_agent_response,
    validate_sessions,
)


ROOT = Path(__file__).parents[1]
SESSIONS_PATH = ROOT / "data" / "derived" / "gptannotation" / "sessions.jsonl"
CATALOG_PATH = ROOT / "data" / "catalog.jsonl"


class SessionTargetAgent:
    def __init__(self, targets: dict[str, str], *, include_similar: bool = False) -> None:
        self.targets = targets
        self.include_similar = include_similar

    def reset(self, session_id: str, user_profile: dict) -> None:
        del user_profile

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        del user_message, top_k
        sample_id = session_id.split(":", 1)[1]
        target = self.targets[sample_id]
        recommendations = [{"parent_asin": target}]
        if self.include_similar:
            recommendations.insert(0, {"parent_asin": f"{target}-similar"})
        return {
            "message": "I can compare these options.",
            "ask_attribute": None,
            "recommendations": recommendations,
        }


class ParallelTargetAgent:
    def __init__(self, targets: dict[str, str], barrier: threading.Barrier) -> None:
        self.targets = targets
        self.barrier = barrier

    def reset(self, session_id: str, user_profile: dict) -> None:
        del session_id, user_profile

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        del user_message, turn, top_k
        self.barrier.wait(timeout=2)
        sample_id = session_id.split(":", 1)[1]
        if sample_id == "slow":
            time.sleep(0.05)
        return {
            "message": "I can compare these options.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": self.targets[sample_id]}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }


class HardEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sessions = load_jsonl(SESSIONS_PATH)
        cls.catalog_ids = load_catalog_ids(CATALOG_PATH)
        cls.validated_sessions = validate_sessions(cls.sessions, cls.catalog_ids)

    def test_fixed_gptannotation_benchmark_shape(self) -> None:
        self.assertEqual(len(self.validated_sessions), 400)
        self.assertEqual(
            {scenario: sum(row["scenario_type"] == scenario for row in self.sessions)
             for scenario in SCENARIO_COUNTS},
            SCENARIO_COUNTS,
        )
        self.assertEqual(len({row["target_asin"] for row in self.sessions}), 400)
        self.assertTrue(all(row["target_asin"] in self.catalog_ids for row in self.sessions))

    def test_override_only_selection_keeps_only_intent_override_sessions(self) -> None:
        selected = select_sessions(self.sessions, override_only=True)
        self.assertEqual(len(selected), SCENARIO_COUNTS["intent_override"])
        self.assertTrue(
            all(row["scenario_type"] == "intent_override" for row in selected)
        )
        self.assertEqual(
            select_sessions(self.sessions, override_only=False),
            self.sessions,
        )

    def test_fact_ids_are_attribute_scoped(self) -> None:
        fact = self.sessions[0]["hidden_facts"][0]
        self.assertEqual(fact_id(fact), (fact["attribute"], fact["canonical"]))
        self.assertEqual(parse_fact_id(["material", "cotton"]), ("material", "cotton"))
        with self.assertRaises(ValueError):
            parse_fact_id(["only-one-value"])

    def test_simulator_answers_requested_hidden_attribute_once(self) -> None:
        session = next(
            row for row in self.sessions
            if any(fact["attribute"] == "material" for fact in row["hidden_facts"])
        )
        state = {
            "disclosed": set(),
            "active_constraints": set(),
            "stale_constraints": set(),
            "no_preference_attributes": set(),
            "boundary_used": False,
        }
        first = simulate_customer_reply(session, "material", state, random.Random(1))
        second = simulate_customer_reply(session, "material", state, random.Random(2))
        material = next(
            fact for fact in session["hidden_facts"] if fact["attribute"] == "material"
        )
        self.assertIn(material["display"].lower(), first.lower())
        self.assertNotIn(material["display"].lower(), second.lower())

    def test_simulator_other_reveals_any_undisclosed_hidden_fact(self) -> None:
        session = next(
            row for row in self.sessions
            if len(row["hidden_facts"]) >= 2
        )
        state = {
            "disclosed": set(),
            "active_constraints": set(),
            "stale_constraints": set(),
            "no_preference_attributes": set(),
            "boundary_used": False,
        }

        first = simulate_customer_reply(session, "other", state, random.Random(1))
        second = simulate_customer_reply(session, "other", state, random.Random(2))
        displays = {str(fact["display"]).lower() for fact in session["hidden_facts"]}

        self.assertTrue(any(display in first.lower() for display in displays))
        self.assertTrue(any(display in second.lower() for display in displays))
        self.assertEqual(len(state["disclosed"]), 2)

    def test_response_contract_rejects_unknown_fields_and_bad_usage(self) -> None:
        valid = {
            "message": "Question",
            "ask_attribute": "feature",
            "recommendations": [{"parent_asin": "B000000001", "score": 0.5}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        self.assertEqual(validate_agent_response(valid), valid)
        with self.assertRaises(ValueError):
            validate_agent_response({**valid, "unexpected": True})
        with self.assertRaises(TypeError):
            validate_agent_response({**valid, "usage": {"prompt_tokens": -1, "completion_tokens": 0}})

    def test_recommendation_normalization_uses_exact_catalog_asins(self) -> None:
        self.assertEqual(
            normalize_recommendations(
                [
                    {"parent_asin": "not-in-catalog"},
                    {"parent_asin": "B000000001"},
                    {"parent_asin": "B000000001"},
                ],
                {"B000000001"},
            ),
            ["B000000001"],
        )

    def test_evaluator_scores_only_after_intent_override(self) -> None:
        targets = {row["sample_id"]: row["target_asin"] for row in self.sessions}
        result = evaluate(SessionTargetAgent(targets), self.sessions, self.catalog_ids)
        self.assertEqual(result["sample_count"], 400)
        for row in result["sessions"]:
            source = next(item for item in self.sessions if item["sample_id"] == row["sample_id"])
            expected_turn = (
                source["override_turn"]
                if source["scenario_type"] == "intent_override"
                else 1
            )
            self.assertTrue(row["hit"])
            self.assertEqual(row["first_hit_turn"], expected_turn)

    def test_similar_asin_does_not_count_as_a_hit(self) -> None:
        targets = {row["sample_id"]: row["target_asin"] for row in self.sessions}
        catalog_ids = self.catalog_ids | {f"{target}-similar" for target in targets.values()}
        result = evaluate(
            SessionTargetAgent(targets, include_similar=True),
            self.sessions,
            catalog_ids,
        )
        self.assertEqual(result["sessions"][0]["best_rank"], 2)

    def test_parallel_evaluation_overlaps_sessions_and_preserves_input_order(self) -> None:
        sessions = [
            {
                "sample_id": "slow",
                "scenario_type": "browsing",
                "target_asin": "A",
                "initial_message": "Show me options.",
            },
            {
                "sample_id": "fast",
                "scenario_type": "browsing",
                "target_asin": "B",
                "initial_message": "Show me options.",
            },
        ]
        result = evaluate(
            ParallelTargetAgent(
                {"slow": "A", "fast": "B"},
                threading.Barrier(2),
            ),
            sessions,
            {"A", "B"},
            validate=False,
            concurrency=2,
        )

        self.assertEqual(
            [row["sample_id"] for row in result["sessions"]],
            ["slow", "fast"],
        )
        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(
            result["reported_token_usage"],
            {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
        )

    def test_evaluation_rejects_nonpositive_concurrency(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(SessionTargetAgent({}), [], set(), validate=False, concurrency=0)

    def test_metric_formula(self) -> None:
        summary = metric_summary([
            {"hit": True, "first_hit_turn": 2, "reciprocal_rank": 1.0},
            {"hit": False, "first_hit_turn": None, "reciprocal_rank": 0.0},
        ])
        scored = add_score_fields(summary)
        self.assertEqual(summary["mttc"], 6.5)
        self.assertEqual(scored["efficiency"], 0.45)
        self.assertAlmostEqual(scored["technical_score"], 0.49)
        self.assertIn("other", ALLOWED_ATTRIBUTES)


if __name__ == "__main__":
    unittest.main()
