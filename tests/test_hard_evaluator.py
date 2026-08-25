from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluator.hard_evaluator import (
    Fact,
    StressSession,
    TargetRecord,
    build_stress_sessions,
    evaluate,
    metric_summary,
    normalize_product,
    score_metrics,
    simulate_customer_reply,
)


def make_session(
    facts: dict[str, tuple[Fact, ...]],
    *,
    scenario_type: str = "buying",
    target_asin: str = "A",
    override_turn: int | None = None,
    override_message: str | None = None,
    initial_fact_key: str | None = None,
    override_fact_key: str | None = None,
    boundary_first: bool = False,
) -> StressSession:
    return StressSession(
        sample_id="test_0001",
        scenario_type=scenario_type,
        target_asin=target_asin,
        category="slippers",
        user_profile={"summary": "anonymous"},
        facts=facts,
        initial_message="I'm looking for slippers.",
        initial_fact_key=initial_fact_key,
        override_turn=override_turn,
        override_message=override_message,
        override_fact_key=override_fact_key,
        boundary_first=boundary_first,
    )


class ScriptedAgent:
    def __init__(self, target: str | None = None, target_turn: int | None = None) -> None:
        self.target = target
        self.target_turn = target_turn
        self.calls = 0
        self.messages: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.calls = 0
        self.messages = []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.calls += 1
        self.messages.append(user_message)
        recommendations = []
        if self.target and (self.target_turn is None or turn == self.target_turn):
            recommendations = [{"parent_asin": self.target}]
        return {
            "message": "Which detail matters most?",
            "ask_attribute": "feature",
            "recommendations": recommendations,
        }


class HardEvaluatorTests(unittest.TestCase):
    def test_deterministic_simulation(self) -> None:
        cotton = Fact("material", "cotton", "cotton", priority=4)
        session = make_session({"material": (cotton,)})
        first = evaluate(ScriptedAgent(), [session], {"A"})
        second = evaluate(ScriptedAgent(), [session], {"A"})
        self.assertEqual(first, second)

    def test_generator_is_deterministic(self) -> None:
        facts = {
            "material": (Fact("material", "cotton", "cotton", priority=4),),
            "feature": (Fact("feature", "waterproof", "waterproof protection", priority=5),),
        }
        records = [
            TargetRecord(f"A{index}", "slippers", facts, ("slippers", "soft"))
            for index in range(8)
        ]
        first = build_stress_sessions(records, count=4, seed=7)
        second = build_stress_sessions(records, count=4, seed=7)
        self.assertEqual(first, second)

    def test_no_raw_full_feature_sentence_leakage(self) -> None:
        raw = "MEMORY FOAM INSOLE: These house slippers with memory foam mold to the contours of your feet."
        record = normalize_product({
            "parent_asin": "A",
            "title": "House slippers",
            "features": [raw],
            "description": [],
            "categories": ["Clothing, Shoes & Jewelry", "Slippers"],
            "details": {},
            "store": "Example",
            "price": 19.99,
        })
        session = make_session(record.facts)
        state: dict[str, object] = {"disclosed": set(), "stale_constraints": set()}
        reply = simulate_customer_reply(session, "feature", state, __import__("random").Random(1))
        self.assertNotIn(raw, reply)
        self.assertIn("memory", reply.lower())
        self.assertLess(len(reply), len(raw))

    def test_requested_attribute_controls_answer(self) -> None:
        material = Fact("material", "cotton", "cotton", priority=4)
        feature = Fact("feature", "waterproof", "waterproof protection", priority=5)
        session = make_session({"material": (material,), "feature": (feature,)})
        state: dict[str, object] = {"disclosed": set(), "stale_constraints": set()}
        import random

        material_reply = simulate_customer_reply(session, "material", state, random.Random(1))
        feature_reply = simulate_customer_reply(session, "feature", state, random.Random(1))
        self.assertIn("cotton", material_reply.lower())
        self.assertNotIn("waterproof", material_reply.lower())
        self.assertIn("waterproof", feature_reply.lower())
        self.assertNotIn("cotton", feature_reply.lower())

    def test_other_is_not_a_wildcard(self) -> None:
        material = Fact("material", "cotton", "cotton", priority=4)
        feature = Fact("feature", "waterproof", "waterproof protection", priority=5)
        session = make_session({"material": (material,), "feature": (feature,)})
        state: dict[str, object] = {"disclosed": set(), "stale_constraints": set()}
        import random

        reply = simulate_customer_reply(session, "other", state, random.Random(2))
        self.assertNotIn("cotton", reply.lower())
        self.assertNotIn("waterproof", reply.lower())
        self.assertTrue(reply.endswith("there.") or reply.endswith("there."))

    def test_no_fabricated_fact_and_null_price(self) -> None:
        session = make_session({})
        state: dict[str, object] = {"disclosed": set(), "stale_constraints": set()}
        import random

        reply = simulate_customer_reply(session, "color", state, random.Random(3))
        self.assertIn("preference", reply.lower())

        record = normalize_product({
            "parent_asin": "B",
            "title": "Plain item",
            "features": [],
            "description": [],
            "categories": ["Clothing", "Accessories"],
            "details": {},
            "store": "Example",
            "price": None,
        })
        self.assertNotIn("budget", record.facts)
        null_price_session = make_session(record.facts, target_asin="B")
        null_reply = simulate_customer_reply(null_price_session, "budget", state, random.Random(4))
        self.assertNotIn("$", null_reply)

    def test_duplicate_information_is_not_repeated(self) -> None:
        material = Fact("material", "cotton", "cotton", priority=4)
        session = make_session({"material": (material,)})
        state: dict[str, object] = {"disclosed": set(), "stale_constraints": set()}
        import random

        first = simulate_customer_reply(session, "material", state, random.Random(5))
        second = simulate_customer_reply(session, "material", state, random.Random(6))
        self.assertIn("cotton", first.lower())
        self.assertNotIn("cotton", second.lower())
        self.assertTrue(any(word in second.lower() for word in ("preference", "flexible", "important")))

    def test_override_turn_and_target_are_preserved(self) -> None:
        old = Fact("style", "casual", "casual", priority=3)
        new = Fact("feature", "waterproof", "waterproof protection", priority=5)
        session = make_session(
            {"style": (old,), "feature": (new,)},
            scenario_type="intent_override",
            target_asin="TARGET",
            override_turn=3,
            override_message="I've changed my mind. Waterproof protection is the requirement I need.",
            initial_fact_key=old.key,
            override_fact_key=new.key,
        )
        agent = ScriptedAgent(target="TARGET", target_turn=3)
        result = evaluate(agent, [session], {"TARGET"})
        self.assertEqual(result["sessions"][0]["first_hit_turn"], 3)
        self.assertEqual(session.target_asin, "TARGET")
        self.assertIn("waterproof", agent.messages[2].lower())

    def test_exact_asin_scoring(self) -> None:
        session = make_session({}, target_asin="A")
        result = evaluate(ScriptedAgent(target="A-similar"), [session], {"A", "A-similar"})
        self.assertFalse(result["sessions"][0]["hit"])
        self.assertEqual(result["sessions"][0]["reciprocal_rank"], 0.0)
        self.assertEqual(result["mttc"], 11.0)

    def test_maximum_ten_turns(self) -> None:
        session = make_session({})
        agent = ScriptedAgent()
        result = evaluate(agent, [session], {"A"})
        self.assertEqual(agent.calls, 10)
        self.assertEqual(result["mttc"], 11.0)

    def test_technical_score_formula_matches_contract(self) -> None:
        overall = metric_summary([
            {"hit": True, "first_hit_turn": 2, "reciprocal_rank": 1.0},
            {"hit": False, "first_hit_turn": None, "reciprocal_rank": 0.0},
        ])
        efficiency, technical_score, mttc = score_metrics(overall)
        self.assertEqual(mttc, 6.5)
        self.assertEqual(efficiency, 0.45)
        self.assertEqual(technical_score, 0.49)


if __name__ == "__main__":
    unittest.main()
