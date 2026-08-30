from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starter.agent import Agent
from starter.clarification import NORMAL_CLARIFICATION_ATTRIBUTES, ClarificationPolicy
from starter.routing.constraints import ShoppingConstraints
from starter.session import SessionManager


def _write_catalog(path: Path) -> None:
    rows = [
        {
            "parent_asin": f"P{index}",
            "categories": ["shoes"],
            "price": 20.0 + index,
        }
        for index in range(8)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class FixedRouter:
    def classify(self, _message: str) -> SimpleNamespace:
        return SimpleNamespace(intent="BUYING")


def _agent(root: Path) -> Agent:
    catalog = root / "catalog.jsonl"
    _write_catalog(catalog)
    return Agent(
        catalog,
        facts_path=root / "missing-facts.jsonl",
        embeddings_path=root / "missing-embeddings.npy",
        metadata_path=root / "missing-metadata.json",
        router=FixedRouter(),
    )


class SessionClarificationStateTests(unittest.TestCase):
    def test_new_state_has_clean_round_state(self) -> None:
        state = SessionManager().reset("session", {})

        self.assertEqual(state.clarification_cycle, 1)
        self.assertEqual(
            state.attribute_call_count,
            {attribute: 0 for attribute in NORMAL_CLARIFICATION_ATTRIBUTES},
        )
        self.assertEqual(state.no_preference_attributes, set())
        self.assertIsNone(state.last_asked)

    def test_mark_asked_counts_only_normal_attributes(self) -> None:
        manager = SessionManager()
        manager.reset("session", {})

        manager.mark_asked("session", "material")
        manager.mark_asked("session", "other")
        state = manager.get("session")

        self.assertEqual(state.attribute_call_count["material"], 1)
        self.assertEqual(state.attribute_call_count["feature"], 0)
        self.assertNotIn("other", state.attribute_call_count)
        self.assertEqual(state.last_asked, "other")

    def test_cycle_reset_preserves_declines_and_goal(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {"profile": "kept"})
        state.mode = "BUYING"
        state.constraints = ShoppingConstraints(category=("shoes",))
        state.no_preference_attributes.add("color")
        manager.mark_asked("session", "material")

        manager.reset_clarification_cycle("session")
        state = manager.get("session")

        self.assertEqual(state.clarification_cycle, 2)
        self.assertEqual(state.attribute_call_count["material"], 0)
        self.assertEqual(state.no_preference_attributes, {"color"})
        self.assertEqual(state.constraints.category, ("shoes",))
        self.assertEqual(state.mode, "BUYING")
        self.assertFalse(state.clarification_stopped)

    def test_full_goal_reset_clears_round_state(self) -> None:
        manager = SessionManager()
        manager.reset("session", {})
        state = manager.get("session")
        state.no_preference_attributes.add("color")
        state.attribute_call_count["material"] = 1
        state.clarification_cycle = 3
        state.clarification_stopped = True

        manager.reset_goal("session")
        state = manager.get("session")

        self.assertEqual(state.clarification_cycle, 1)
        self.assertTrue(all(value == 0 for value in state.attribute_call_count.values()))
        self.assertEqual(state.no_preference_attributes, set())
        self.assertFalse(state.clarification_stopped)


class AgentClarificationRoundTests(unittest.TestCase):
    def _patch_extraction(self, mapping: dict[str, ShoppingConstraints]):
        return patch(
            "starter.agent.constraint_module.extract_constraints",
            side_effect=lambda message, **_kwargs: mapping[message],
        )

    @staticmethod
    def _round_choice(
        _candidates,
        _constraints,
        asked_attributes=(),
        **_kwargs,
    ) -> str | None:
        asked = set(asked_attributes)
        for attribute in ("material", "color", "brand"):
            if attribute not in asked:
                return attribute
        return None

    def test_only_proactive_questions_increment_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = _agent(Path(directory))
            agent.reset("session", {})
            agent.clarification.choose = self._round_choice
            messages = {
                "shoes": ShoppingConstraints(category=("shoes",)),
                "Cotton, black, and machine washable.": ShoppingConstraints(
                    material=("cotton",),
                    color=("black",),
                    feature=("machine washable",),
                ),
            }

            with self._patch_extraction(messages):
                first = agent.respond("session", "shoes", 1, 3)
                second = agent.respond(
                    "session", "Cotton, black, and machine washable.", 2, 3
                )

            state = agent.sessions.get("session")
            self.assertEqual(first["ask_attribute"], "material")
            self.assertEqual(second["ask_attribute"], "color")
            self.assertEqual(state.attribute_call_count["material"], 1)
            self.assertEqual(state.attribute_call_count["color"], 1)
            self.assertEqual(state.attribute_call_count["feature"], 0)
            self.assertEqual(state.constraints.material, ("cotton",))
            self.assertEqual(state.constraints.color, ("black",))
            self.assertEqual(state.constraints.feature, ("machine washable",))

    def test_declined_field_is_blocked_and_other_answer_starts_new_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = _agent(Path(directory))
            agent.reset("session", {})
            agent.clarification.choose = self._round_choice
            messages = {
                "shoes": ShoppingConstraints(category=("shoes",)),
                "cotton": ShoppingConstraints(material=("cotton",)),
                "machine washable": ShoppingConstraints(
                    feature=("machine washable",)
                ),
            }

            with self._patch_extraction(messages):
                agent.respond("session", "shoes", 1, 3)
                agent.respond("session", "cotton", 2, 3)
                # color is asked next and is declined.
                agent.respond(
                    "session", "I don't have a preference.", 3, 3
                )
                # brand is asked next and is declined, leaving only OTHER.
                agent.respond(
                    "session", "I don't have a preference.", 4, 3
                )
                before_other = agent.sessions.get("session")
                self.assertEqual(before_other.last_asked, "other")
                response = agent.respond(
                    "session", "machine washable", 5, 3
                )

            state = agent.sessions.get("session")
            self.assertEqual(state.clarification_cycle, 2)
            self.assertEqual(response["ask_attribute"], "material")
            self.assertEqual(state.attribute_call_count["material"], 1)
            self.assertEqual(state.attribute_call_count["color"], 0)
            self.assertEqual(state.attribute_call_count["brand"], 0)
            self.assertEqual(state.no_preference_attributes, {"color", "brand"})
            self.assertEqual(state.constraints.feature, ("machine washable",))

    def test_other_without_information_stops_without_resetting_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = _agent(Path(directory))
            agent.reset("session", {})
            state = agent.sessions.get("session")
            state.mode = "BUYING"
            state.last_asked = "other"
            state.attribute_call_count["material"] = 1
            state.clarification_cycle = 2

            response = agent.respond(
                "session", "No, that's all.", 3, 3
            )

            self.assertIsNone(response["ask_attribute"])
            self.assertTrue(state.clarification_stopped)
            self.assertEqual(state.clarification_cycle, 2)
            self.assertEqual(state.attribute_call_count["material"], 1)

    def test_blocked_field_can_still_be_volunteered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = _agent(Path(directory))
            agent.reset("session", {})
            state = agent.sessions.get("session")
            state.mode = "BUYING"
            state.no_preference_attributes.add("color")
            agent.clarification.choose = lambda *_args, **_kwargs: None

            with self._patch_extraction(
                {"Black would be nice.": ShoppingConstraints(color=("black",))}
            ):
                agent.respond("session", "Black would be nice.", 1, 3)

            self.assertEqual(state.constraints.color, ("black",))
            self.assertIn("color", state.no_preference_attributes)
            self.assertNotEqual(state.last_asked, "color")

    def test_unrelated_answer_does_not_reclassify_or_reask_pending_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = _agent(Path(directory))
            agent.reset("session", {})
            agent.clarification.choose = self._round_choice
            messages = {
                "shoes": ShoppingConstraints(category=("shoes",)),
                "Keep it under $50.": ShoppingConstraints(price_max=50.0),
            }

            with self._patch_extraction(messages):
                agent.respond("session", "shoes", 1, 3)
                response = agent.respond(
                    "session", "Keep it under $50.", 2, 3
                )

            state = agent.sessions.get("session")
            self.assertEqual(state.constraints.price_max, 50.0)
            self.assertEqual(state.attribute_call_count["material"], 1)
            self.assertNotIn("material", state.no_preference_attributes)
            self.assertEqual(response["ask_attribute"], "color")


class ClarificationPolicyRoundTests(unittest.TestCase):
    def test_asked_count_excludes_field_but_utility_still_selects_another(self) -> None:
        candidates = tuple(
            SimpleNamespace(
                attributes={
                    "material": ("cotton", "wool"),
                    "color": ("red", "blue", "green", "black"),
                    "brand": (),
                },
                price=30.0,
            )
            for _ in range(4)
        )
        policy = ClarificationPolicy()

        selected = policy.choose(
            candidates,
            ShoppingConstraints(),
            {"material"},
            mode="BUYING",
            turn=1,
        )

        self.assertNotEqual(selected, "material")


if __name__ == "__main__":
    unittest.main()
