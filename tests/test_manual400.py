from __future__ import annotations

import json
import math
import random
import re
import unittest
from pathlib import Path

from evaluator.manual400_evaluator import (
    ALLOWED_ATTRIBUTES,
    ASIN_RE,
    METADATA_RE,
    SCENARIO_COUNTS,
    effective_initial_fact_id,
    evaluate,
    fact_id,
    fact_visible_in_message,
    normalize_recommendations,
    simulate_customer_reply,
    validate_response,
    validate_sessions,
)
from scripts.validate_manual400 import validate_benchmark


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "derived" / "manual400"
SESSIONS_PATH = DATA / "sessions.jsonl"
LABELS_PATH = DATA / "labeled_products.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def synthetic_session(scenario: str = "buying", boundary_first: bool = False) -> dict:
    hidden = [
        {
            "attribute": "material",
            "canonical": "cotton",
            "display": "cotton",
            "evidence_field": "features",
            "evidence_text": "100% cotton fabric",
            "confidence": 0.99,
        },
        {
            "attribute": "feature",
            "canonical": "waterproof_protection",
            "display": "waterproof protection",
            "evidence_field": "features",
            "evidence_text": "waterproof shell",
            "confidence": 0.99,
        },
    ]
    row = {
        "sample_id": "synthetic_0001",
        "scenario_type": scenario,
        "target_asin": "B000000001",
        "category": "jackets",
        "user_profile": {},
        "hidden_facts": hidden,
        "initial_message": "I'm looking for jackets. I'd prefer cotton.",
        "initial_fact_id": ["material", "cotton"],
        "override_turn": None,
        "override_message": None,
        "override_fact_id": None,
        "boundary_first": boundary_first,
    }
    if scenario == "browsing":
        row["initial_message"] = "I'm exploring jackets and would like to compare options."
    if scenario == "intent_override":
        row.update({
            "override_turn": 3,
            "override_message": "Actually, waterproof protection matters more to me now.",
            "override_fact_id": ["feature", "waterproof_protection"],
        })
    return row


class StaticAgent:
    def __init__(self, target: str, ask_attribute: str | None = None) -> None:
        self.target = target
        self.ask_attribute = ask_attribute
        self.messages: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.messages = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.messages.append(user_message)
        return {
            "message": "question",
            "ask_attribute": self.ask_attribute,
            "recommendations": [{"parent_asin": self.target}],
        }


class Manual400StaticArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SESSIONS_PATH.exists():
            raise unittest.SkipTest("committed Manual400 artifacts are unavailable")
        cls.sessions = read_jsonl(SESSIONS_PATH)
        cls.labels = {str(row["parent_asin"]): row for row in read_jsonl(LABELS_PATH)}

    def test_fixed_benchmark_validates_without_building(self) -> None:
        summary = validate_benchmark()
        self.assertEqual(summary["sessions"], 400)
        self.assertEqual(summary["scenario_counts"], SCENARIO_COUNTS)
        self.assertTrue(summary["read_only"])

    def test_exact_counts_unique_targets_and_fixed_rows(self) -> None:
        self.assertEqual(len(self.sessions), 400)
        self.assertEqual(len({row["target_asin"] for row in self.sessions}), 400)
        counts = {
            scenario: sum(row["scenario_type"] == scenario for row in self.sessions)
            for scenario in SCENARIO_COUNTS
        }
        self.assertEqual(counts, SCENARIO_COUNTS)
        self.assertEqual(
            [row["sample_id"] for row in self.sessions],
            [f"manual400_{index:04d}" for index in range(1, 401)],
        )

    def test_hidden_cards_are_validated_evidence_backed_and_bounded(self) -> None:
        for session in self.sessions:
            label = self.labels[session["target_asin"]]
            label_ids = {
                fact_id(fact)
                for fact in label["validated_facts"]
                if isinstance(fact, dict)
            }
            hidden = session["hidden_facts"]
            self.assertTrue(2 <= len(hidden) <= 4)
            self.assertNotIn("category", {fact["attribute"] for fact in hidden})
            self.assertEqual(len({fact_id(fact) for fact in hidden}), len(hidden))
            for fact in hidden:
                self.assertIn(fact_id(fact), label_ids)
                self.assertTrue(fact["evidence_field"])
                self.assertTrue(fact["evidence_text"])

    def test_target_catalog_and_audit_are_static_and_complete(self) -> None:
        result = validate_benchmark()
        audits = read_jsonl(DATA / "label_audit.jsonl")
        self.assertEqual(result["audit_pass_count"], 400)
        self.assertTrue(all(row["status"] == "pass" and not row["issues"] for row in audits))
        self.assertEqual(set(self.labels), {row["target_asin"] for row in self.sessions})

    def test_initial_state_does_not_trust_silent_metadata(self) -> None:
        for session in self.sessions:
            effective_id = effective_initial_fact_id(session)
            if session["scenario_type"] == "browsing":
                self.assertIsNone(effective_id)
            if session["scenario_type"] == "boundary":
                initial_id = tuple(session["initial_fact_id"]) if session.get("initial_fact_id") else None
                facts = {fact_id(fact): fact for fact in session["hidden_facts"]}
                if initial_id is None:
                    self.assertIsNone(effective_id)
                else:
                    explicitly_visible = fact_visible_in_message(
                        facts[initial_id], session["initial_message"]
                    )
                    self.assertEqual(effective_id is not None, explicitly_visible)

    def test_override_rows_keep_target_and_use_turn_three_or_four(self) -> None:
        overrides = [row for row in self.sessions if row["scenario_type"] == "intent_override"]
        self.assertEqual(len(overrides), 60)
        for row in overrides:
            self.assertIn(row["override_turn"], (3, 4))
            self.assertNotEqual(row["initial_fact_id"], row["override_fact_id"])
            self.assertEqual(row["target_asin"], self.labels[row["target_asin"]]["parent_asin"])

    def test_customer_messages_have_no_raw_identifiers(self) -> None:
        for session in self.sessions:
            for message in (session["initial_message"], session.get("override_message") or ""):
                self.assertIsNone(ASIN_RE.search(message))
                self.assertIsNone(METADATA_RE.search(message))

    def test_boundary_first_controls_the_simulated_reply(self) -> None:
        first = synthetic_session("boundary", boundary_first=True)
        state = {"disclosed": set(), "stale_constraints": set(), "boundary_used": False}
        reply = simulate_customer_reply(first, "feature", state, random.Random(1))
        self.assertIn("don't really have a preference", reply)
        self.assertTrue(state["boundary_used"])

        normal = synthetic_session("boundary", boundary_first=False)
        state = {"disclosed": set(), "stale_constraints": set(), "boundary_used": False}
        reply = simulate_customer_reply(normal, "feature", state, random.Random(1))
        self.assertIn("waterproof", reply.lower())
        self.assertFalse(state["boundary_used"])

    def test_other_and_category_do_not_invent_a_hidden_fact(self) -> None:
        session = synthetic_session()
        state = {"disclosed": set(), "stale_constraints": set(), "boundary_used": False}
        reply = simulate_customer_reply(session, "other", state, random.Random(1))
        self.assertNotIn("cotton", reply.lower())
        self.assertNotIn("waterproof", reply.lower())
        reply = simulate_customer_reply(session, "category", state, random.Random(1))
        self.assertNotIn("cotton", reply.lower())
        self.assertNotIn("waterproof", reply.lower())


class Manual400EvaluatorContractTests(unittest.TestCase):
    def test_strict_response_validation_and_top_k_limit(self) -> None:
        valid = {
            "message": "ok",
            "ask_attribute": "feature",
            "recommendations": [{"parent_asin": "B000000001", "score": 0.5}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.assertEqual(validate_response(valid), valid)
        with self.assertRaises((TypeError, ValueError)):
            validate_response({"message": "ok", "ask_attribute": "not_allowed", "recommendations": []})
        with self.assertRaises((TypeError, ValueError)):
            validate_response({"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": ""}]})
        with self.assertRaises((TypeError, ValueError)):
            validate_response({"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": "B", "score": math.nan}]})
        with self.assertRaises((TypeError, ValueError)):
            validate_response({
                "message": "ok",
                "ask_attribute": None,
                "recommendations": [{"parent_asin": f"B{i:09d}"} for i in range(11)],
            })

    def test_exact_parent_asin_scoring_and_normalization(self) -> None:
        self.assertEqual(
            normalize_recommendations(
                [{"parent_asin": "wrong"}, {"parent_asin": "B000000001"}, {"parent_asin": "B000000001"}],
                {"wrong", "B000000001"},
            ),
            ["wrong", "B000000001"],
        )
        result = evaluate(
            StaticAgent("B000000001"),
            [synthetic_session()],
            {"B000000001"},
            expected_count=1,
        )
        self.assertTrue(result["sessions"][0]["hit"])
        self.assertEqual(result["sessions"][0]["best_rank"], 1)

        similar = evaluate(
            StaticAgent("B00000000X"),
            [synthetic_session()],
            {"B00000000X", "B000000001"},
            expected_count=1,
        )
        self.assertFalse(similar["sessions"][0]["hit"])
        self.assertEqual(similar["sessions"][0]["reciprocal_rank"], 0.0)

    def test_target_cannot_score_before_intent_override(self) -> None:
        session = synthetic_session("intent_override")
        result = evaluate(
            StaticAgent("B000000001"),
            [session],
            {"B000000001"},
            expected_count=1,
        )
        self.assertTrue(result["sessions"][0]["hit"])
        self.assertEqual(result["sessions"][0]["first_hit_turn"], 3)
        self.assertEqual(session["target_asin"], "B000000001")

    def test_browsing_does_not_disclose_initial_fact(self) -> None:
        session = synthetic_session("browsing")
        self.assertIsNone(effective_initial_fact_id(session))
        state = {"disclosed": set(), "stale_constraints": set(), "boundary_used": False}
        reply = simulate_customer_reply(session, "material", state, random.Random(1))
        self.assertIn("cotton", reply.lower())

    def test_session_validation_rejects_bad_counts_and_keeps_enum_closed(self) -> None:
        with self.assertRaises(ValueError):
            validate_sessions([synthetic_session()], expected_count=400)
        self.assertIn("other", ALLOWED_ATTRIBUTES)
        self.assertNotIn("wildcard", ALLOWED_ATTRIBUTES)


if __name__ == "__main__":
    unittest.main()
