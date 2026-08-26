from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path

from evaluator.manual400_builder import SEED, SCENARIO_COUNTS, scenario_list
from evaluator.manual400_evaluator import (
    ALLOWED_ATTRIBUTES,
    evaluate,
    normalize_recommendations,
    simulate_customer_reply,
    validate_response,
    validate_sessions,
)


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "derived" / "manual400"
SESSIONS_PATH = DATA / "sessions.jsonl"
LABELS_PATH = DATA / "labeled_products.jsonl"
AUDIT_PATH = DATA / "label_audit.jsonl"
ASIN_RE = re.compile(r"\bB0[0-9A-Z]{8}\b", re.IGNORECASE)
METADATA_RE = re.compile(r"date first available|item model number|product dimensions|package weight|asin", re.I)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def synthetic_session(scenario: str = "buying") -> dict:
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
        "boundary_first": False,
    }
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


class Manual400ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SESSIONS_PATH.exists():
            raise unittest.SkipTest("manual400 artifacts are not generated")
        cls.sessions = read_jsonl(SESSIONS_PATH)
        cls.labels = {str(row["parent_asin"]): row for row in read_jsonl(LABELS_PATH)}

    def test_exact_counts_and_unique_targets(self) -> None:
        self.assertEqual(len(self.sessions), 400)
        self.assertEqual(len({row["target_asin"] for row in self.sessions}), 400)
        counts = {scenario: sum(row["scenario_type"] == scenario for row in self.sessions) for scenario in SCENARIO_COUNTS}
        self.assertEqual(counts, SCENARIO_COUNTS)

    def test_hidden_cards_are_evidence_backed_and_bounded(self) -> None:
        for session in self.sessions:
            label = self.labels[session["target_asin"]]
            label_ids = {(fact["attribute"], fact["canonical"]) for fact in label["validated_facts"]}
            self.assertLessEqual(len(session["hidden_facts"]), 4)
            self.assertGreaterEqual(len(session["hidden_facts"]), 2)
            for fact in session["hidden_facts"]:
                self.assertIn((fact["attribute"], fact["canonical"]), label_ids)
                self.assertTrue(fact["evidence_field"])
                self.assertTrue(fact["evidence_text"])

    def test_audit_has_no_needs_fix_rows(self) -> None:
        audit = read_jsonl(AUDIT_PATH)
        self.assertEqual(len(audit), 400)
        self.assertTrue(all(row["status"] == "pass" and not row["issues"] for row in audit))

    def test_deterministic_seed_and_visible_text_safety(self) -> None:
        self.assertEqual(scenario_list(SEED), scenario_list(SEED))
        self.assertEqual([row["sample_id"] for row in self.sessions], [f"manual400_{i:04d}" for i in range(1, 401)])
        for session in self.sessions:
            for message in (session["initial_message"], session.get("override_message") or ""):
                self.assertIsNone(ASIN_RE.search(message))
                self.assertIsNone(METADATA_RE.search(message))

    def test_override_turns_and_target_invariants(self) -> None:
        overrides = [row for row in self.sessions if row["scenario_type"] == "intent_override"]
        self.assertEqual(len(overrides), 60)
        for row in overrides:
            self.assertIn(row["override_turn"], (3, 4))
            self.assertNotEqual(row["initial_fact_id"], row["override_fact_id"])
            self.assertEqual(row["target_asin"], self.labels[row["target_asin"]]["parent_asin"])

    def test_boundary_sessions_include_no_preference_path(self) -> None:
        self.assertTrue(any(row.get("boundary_first") for row in self.sessions if row["scenario_type"] == "boundary"))


class Manual400EvaluatorContractTests(unittest.TestCase):
    def test_strict_response_validation(self) -> None:
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

    def test_exact_asin_scoring_and_top_k_normalization(self) -> None:
        self.assertEqual(normalize_recommendations([
            {"parent_asin": "wrong"}, {"parent_asin": "B000000001"}, {"parent_asin": "B000000001"},
        ], {"wrong", "B000000001"}), ["wrong", "B000000001"])
        agent = StaticAgent("B000000001")
        result = evaluate(agent, [synthetic_session()], {"B000000001"}, expected_count=1)
        self.assertTrue(result["sessions"][0]["hit"])
        self.assertEqual(result["sessions"][0]["best_rank"], 1)
        self.assertEqual(result["sessions"][0]["first_hit_turn"], 1)

    def test_override_cannot_score_before_override_and_target_is_unchanged(self) -> None:
        agent = StaticAgent("B000000001")
        result = evaluate(agent, [synthetic_session("intent_override")], {"B000000001"}, expected_count=1)
        scored = result["sessions"][0]
        self.assertTrue(scored["hit"])
        self.assertEqual(scored["first_hit_turn"], 3)
        self.assertEqual(scored["sample_id"], "synthetic_0001")

    def test_boundary_and_other_do_not_invent_a_fact(self) -> None:
        session = synthetic_session("boundary")
        state = {"disclosed": set(), "stale_constraints": set(), "boundary_used": False}
        reply = simulate_customer_reply(session, "feature", state, __import__("random").Random(1))
        self.assertIn("don't really have a preference", reply)
        self.assertTrue(state["boundary_used"])
        state = {"disclosed": set(), "stale_constraints": set(), "boundary_used": False}
        reply = simulate_customer_reply(session, "other", state, __import__("random").Random(1))
        self.assertNotIn("cotton", reply.lower())
        self.assertNotIn("waterproof", reply.lower())

    def test_session_validation_rejects_bad_counts_and_allowed_enum_is_closed(self) -> None:
        with self.assertRaises(ValueError):
            validate_sessions([synthetic_session()], expected_count=400)
        self.assertNotIn("other", {"material", "color", "size", "style", "brand", "budget", "feature", "use_case"})
        self.assertIn("other", ALLOWED_ATTRIBUTES)


if __name__ == "__main__":
    unittest.main()
