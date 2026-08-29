from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluator.debug_web import DebugWebController, SessionPool, STATIC_DIR
from evaluator.hard_evaluator import Manual400SessionRunner
from evaluator.hard_evaluator import _debug_state_snapshot
from starter.routing.constraints import ShoppingConstraints


def make_session(
    sample_id: str,
    *,
    scenario: str = "browsing",
    target: str = "TARGET",
    initial: str = "I am looking for shoes.",
    override_turn: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "target_asin": target,
        "initial_message": initial,
        "user_profile": {},
    }
    if override_turn is not None:
        row.update(
            {
                "override_turn": override_turn,
                "override_message": "Actually, I want a new goal.",
                "initial_fact_id": None,
                "override_fact_id": None,
            }
        )
    return row


class FakeAgent:
    def __init__(self, *, return_target: bool = False) -> None:
        self.calls: list[tuple[str, str, int, int]] = []
        self.reset_calls: list[str] = []
        self.extract_calls = 0
        self.return_target = return_target
        self.sessions = FakeSessionManager()
        self.retriever = SimpleNamespace(
            dense_available=False,
            layer2_index=None,
            query_encoder=None,
            layer2_compatibility_error=None,
            product_by_asin={},
        )

    def reset(self, session_id: str, _profile: dict[str, object]) -> None:
        self.reset_calls.append(session_id)
        self.sessions._states[session_id] = SimpleNamespace(
            constraints=ShoppingConstraints(),
            mode="BROWSING",
            query_text="",
            excluded_recommendations=set(),
            last_recommendations=(),
            asked_attributes=set(),
            last_asked=None,
            last_user_message=None,
            last_override_kind=None,
            last_override_delta=None,
            turn=0,
        )

    def _extract(self, _message: str) -> ShoppingConstraints:
        self.extract_calls += 1
        return ShoppingConstraints()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        self.calls.append((session_id, user_message, turn, top_k))
        state = self.sessions._states[session_id]
        state.turn = turn
        state.last_user_message = user_message
        state.query_text = user_message
        recommendations = [{"parent_asin": "TARGET"}] if self.return_target else []
        return {
            "message": "Here are some options.",
            "ask_attribute": None,
            "recommendations": recommendations,
        }


class FakeSessionManager:
    def __init__(self) -> None:
        self._states: dict[str, SimpleNamespace] = {}

    def get(self, session_id: str) -> SimpleNamespace:
        return self._states[session_id]


class DebugWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = [make_session(f"manual400_{index:04d}") for index in range(1, 7)]

    def test_seeded_pool_is_reproducible(self) -> None:
        first = SessionPool(self.sessions, seed=42)
        second = SessionPool(self.sessions, seed=42)
        first_ids = [first.next()["sample_id"] for _ in range(8)]
        second_ids = [second.next()["sample_id"] for _ in range(8)]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(set(first_ids[:6])), 6)
        self.assertIn(first_ids[6], first_ids[:6])

    def test_debug_snapshot_exposes_clarification_round_state(self) -> None:
        agent = FakeAgent()
        agent.reset("session", {})
        state = agent.sessions._states["session"]
        state.clarification_cycle = 2
        state.attribute_call_count = {"material": 1, "color": 0}
        state.no_preference_attributes = {"color"}
        state.clarification_stopped = False

        snapshot = _debug_state_snapshot(agent, "session")

        self.assertEqual(snapshot["clarification_cycle"], 2)
        self.assertEqual(snapshot["attribute_call_count"]["material"], 1)
        self.assertEqual(snapshot["no_preference_attributes"], ["color"])
        self.assertFalse(snapshot["clarification_stopped"])

    def test_scenario_filter_and_specific_id(self) -> None:
        rows = self.sessions + [make_session("buying_1", scenario="buying")]
        pool = SessionPool(rows, seed=3)
        self.assertEqual(pool.next("BUYING")["sample_id"], "buying_1")
        self.assertEqual(pool.by_id("manual400_0003")["sample_id"], "manual400_0003")
        self.assertIsNone(pool.by_id("missing"))

    def test_runner_advances_exactly_one_turn_and_keeps_target_out_of_agent(self) -> None:
        agent = FakeAgent()
        runner = Manual400SessionRunner(
            agent,
            make_session("manual400_0001", target="TARGET"),
            {"OTHER"},
        )
        first = runner.next_turn()
        self.assertEqual(first["turn"], 1)
        self.assertFalse(runner.done)
        self.assertEqual(len(agent.calls), 1)
        self.assertNotIn("TARGET", repr(agent.calls[0]))
        second = runner.next_turn()
        self.assertEqual(second["turn"], 2)
        self.assertEqual(len(agent.calls), 2)

    def test_runner_uses_configured_override_message_on_exact_turn(self) -> None:
        agent = FakeAgent()
        runner = Manual400SessionRunner(
            agent,
            make_session(
                "manual400_0001",
                scenario="intent_override",
                override_turn=3,
            ),
            {"OTHER"},
        )
        runner.next_turn()
        runner.next_turn()
        runner.next_turn()
        self.assertEqual(agent.calls[2][1], "Actually, I want a new goal.")
        self.assertEqual([event["turn"] for event in runner.events], [1, 2, 3])

    def test_controller_resets_and_runs_to_end_sequentially(self) -> None:
        agent = FakeAgent()
        with patch(
            "evaluator.debug_web._ranking_payload",
            return_value={
                "structured_rank": None,
                "dense_rank": None,
                "hybrid_rank": None,
                "eligible": False,
                "eligible_count": 0,
                "global_count": 0,
                "global_rank": None,
                "global_rank_status": "AVAILABLE",
                "structured_score": None,
                "dense_score": None,
                "final_score": None,
                "top10": [],
                "view_scores": None,
            },
        ):
            controller = DebugWebController(agent, self.sessions, {"OTHER"}, seed=7)
            state = controller.new_random()
            # Which session the seeded pool draws first is an implementation
            # detail of random.shuffle, not a contract; test_seeded_pool_is
            # _reproducible already covers determinism. Assert against the
            # session actually drawn.
            selected = state["session"]["session_id"]
            self.assertEqual(state["turn"], 0)
            controller.next_turn()
            self.assertEqual(controller.turn_records[-1]["turn"], 1)
            self.assertEqual(len(agent.calls), 1)
            self.assertEqual(agent.extract_calls, 0)
            end = controller.run_to_end()
            self.assertTrue(end["done"])
            self.assertEqual(len(controller.turn_records), 10)
            self.assertTrue(end["benchmark"]["complete"])
            self.assertEqual(end["benchmark"]["result"]["sample_id"], selected)
            self.assertEqual(end["benchmark"]["metrics"]["sample_count"], 1)
            controller.load(selected)
            self.assertEqual(len(agent.reset_calls), 2)
            self.assertEqual(controller.state_payload()["turn"], 0)

    def test_static_debug_assets_exist(self) -> None:
        self.assertTrue((STATIC_DIR / "index.html").is_file())
        self.assertTrue((STATIC_DIR / "app.js").is_file())
        self.assertTrue((STATIC_DIR / "style.css").is_file())


if __name__ == "__main__":
    unittest.main()
