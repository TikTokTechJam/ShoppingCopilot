from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluator.hard_evaluator import (
    debug_evaluate,
    debug_session_order,
    evaluate,
)
from starter.agent import Agent
from starter.retrieval import ProductRetriever
from starter.routing.constraints import ShoppingConstraints


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_debug_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    catalog_path = root / "catalog.jsonl"
    facts_path = root / "annotations.jsonl"
    write_jsonl(
        catalog_path,
        [
            {"parent_asin": "S0", "categories": ["shoes"], "price": 20.0},
            {"parent_asin": "S1", "categories": ["shoes"], "price": 30.0},
        ],
    )
    write_jsonl(
        facts_path,
        [
            {
                "parent_asin": "S0",
                "price": 20.0,
                "facts": {
                    "category": ["shoes"],
                    "brand": ["Example"],
                    "color": ["red"],
                    "material": [],
                    "style": [],
                    "feature": ["lightweight"],
                    "use_case": [],
                },
                "annotation": {"status": "success", "prompt_version": "v4"},
            },
            {
                "parent_asin": "S1",
                "price": 30.0,
                "facts": {
                    "category": ["shoes"],
                    "brand": ["Other"],
                    "color": ["blue"],
                    "material": [],
                    "style": [],
                    "feature": [],
                    "use_case": [],
                },
                "annotation": {"status": "success", "prompt_version": "v4"},
            },
        ],
    )
    session = {
        "sample_id": "debug_fixture_1",
        "scenario_type": "browsing",
        "target_asin": "S0",
        "hidden_facts": [
            {
                "attribute": "color",
                "canonical": "red",
                "display": "red",
                "evidence_field": "facts",
                "evidence_text": "red color",
            },
            {
                "attribute": "feature",
                "canonical": "lightweight",
                "display": "lightweight",
                "evidence_field": "facts",
                "evidence_text": "lightweight feature",
            },
        ],
        "initial_message": "I'm looking for shoes, but I'm still exploring.",
        "user_profile": {},
    }
    return catalog_path, facts_path, session


class RecordingRetriever(ProductRetriever):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        super().__init__(*args, **kwargs)

    def retrieve(self, *args: object, **kwargs: object) -> list[object]:
        self.calls.append((args, kwargs))
        return super().retrieve(*args, **kwargs)


class RecordingAgent(Agent):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.respond_calls: list[tuple[str, str, int, int]] = []

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        self.respond_calls.append((session_id, user_message, turn, top_k))
        return super().respond(session_id, user_message, turn, top_k)


class TargetAgent:
    def reset(self, _session_id: str, _user_profile: dict[str, object]) -> None:
        return None

    def respond(
        self,
        _session_id: str,
        _user_message: str,
        _turn: int,
        _top_k: int,
    ) -> dict[str, object]:
        return {
            "message": "Here are some options.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "S0"}],
        }


class HardEvaluatorDebugTests(unittest.TestCase):
    def test_seeded_debug_order_is_reproducible_and_limited(self) -> None:
        rows = [{"sample_id": str(index)} for index in range(8)]
        first = debug_session_order(rows, seed=42)
        second = debug_session_order(rows, seed=42)
        limited = debug_session_order(rows, seed=42, limit=3)

        self.assertEqual(first, second)
        self.assertEqual(len(limited), 3)
        self.assertEqual(
            {row["sample_id"] for row in limited},
            {row["sample_id"] for row in first[:3]},
        )
        self.assertEqual(
            [row["sample_id"] for row in rows],
            [str(index) for index in range(8)],
        )

    def test_normal_evaluate_path_does_not_pause_and_is_deterministic(self) -> None:
        sessions = [
            {
                "sample_id": "s",
                "scenario_type": "browsing",
                "target_asin": "S0",
                "initial_message": "I am looking for shoes.",
            }
        ]
        with patch(
            "evaluator.hard_evaluator.validate_sessions",
            return_value=sessions,
        ):
            with patch("builtins.input", side_effect=AssertionError("must not pause")):
                first = evaluate(TargetAgent(), sessions, {"S0"})
                second = evaluate(TargetAgent(), sessions, {"S0"})

        self.assertEqual(first, second)
        self.assertEqual(first["sessions"][0]["first_hit_turn"], 1)

    def test_debug_prints_rank_facts_state_and_does_not_leak_target_to_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path, facts_path, session = build_debug_fixture(root)
            retriever = RecordingRetriever(
                catalog_path,
                facts_path=facts_path,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
            )
            agent = RecordingAgent(retriever=retriever)
            prompts: list[str] = []
            output = io.StringIO()

            with patch(
                "evaluator.hard_evaluator.validate_sessions",
                return_value=[session],
            ):
                with contextlib.redirect_stdout(output):
                    result = debug_evaluate(
                        agent,
                        [session],
                        {"S0", "S1"},
                        seed=42,
                        debug_sessions=1,
                        input_fn=prompts.append,
                    )

            rendered = output.getvalue()
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(len(prompts), 2)
            self.assertIn("AGENT CONSTRAINTS SO FAR", rendered)
            self.assertIn('"category": [', rendered)
            self.assertIn("TARGET FACTS", rendered)
            self.assertIn('"color": [', rendered)
            self.assertIn("TARGET RANKING", rendered)
            self.assertIn("Global rank: 1 / 2", rendered)
            self.assertIn("Eligible rank: 1 / 2", rendered)
            self.assertIn("Top10 rank: 1", rendered)
            self.assertIn("TARGET MATCH BREAKDOWN", rendered)
            self.assertIn("RUNNING DEBUG BENCHMARK", rendered)
            self.assertIn("Completed sessions: 1", rendered)
            self.assertIn("OVERRIDE DETECTED: NO", rendered)
            self.assertTrue(all("S0" not in repr(call) for call in agent.respond_calls))
            self.assertTrue(all("S0" not in repr(call) for call in retriever.calls))

    def test_debug_ranking_uses_budget_eligible_and_global_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path, facts_path, _ = build_debug_fixture(root)
            retriever = ProductRetriever(
                catalog_path,
                facts_path=facts_path,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
            )
            constraints = ShoppingConstraints(category=("shoes",), price_max=20.0)

            global_ranking = retriever.debug_rank_all(
                "BUYING",
                "",
                constraints,
                apply_budget=False,
            )
            eligible_ranking = retriever.debug_rank_all(
                "BUYING",
                "",
                constraints,
                apply_budget=True,
            )

            # The global view keeps both products because it does not apply
            # the budget. Their relative order is not asserted: price is a
            # numeric eligibility filter and contributes no score, so with the
            # budget off it cannot separate them.
            self.assertEqual(
                sorted(item.parent_asin for item in global_ranking), ["S0", "S1"]
            )
            self.assertEqual([item.parent_asin for item in eligible_ranking], ["S0"])
            self.assertNotIn("S1", [item.parent_asin for item in eligible_ranking])


if __name__ == "__main__":
    unittest.main()