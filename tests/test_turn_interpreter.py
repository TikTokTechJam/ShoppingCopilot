from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from starter.agent import Agent
from starter.routing.constraints import ShoppingConstraints
from starter.turn_interpreter import (
    TurnInterpretation,
    build_turn_prompt,
    parse_turn_interpretation,
)


class TurnInterpreterParsingTests(unittest.TestCase):
    def test_parses_schema_delta_and_preference_override(self) -> None:
        result = parse_turn_interpretation(
            {
                "intent": "browsing",
                "updates": {
                    "category": ["sweatshirts"],
                    "use_case": ["exploring"],
                    "unknown_slot": ["must be ignored"],
                },
                "override": {
                    "type": "preference_override",
                    "fields": ["use_case"],
                },
            }
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.intent, "BROWSING")
        self.assertEqual(result.updates, {
            "category": ("sweatshirts",),
            "use_case": ("exploring",),
        })
        self.assertEqual(result.override_kind, "PREFERENCE")
        self.assertEqual(result.override_fields, ("use_case",))

    def test_accepts_constraints_alias_and_json_fence(self) -> None:
        result = parse_turn_interpretation(
            "```json\n"
            '{"intent":"buying","constraints":{"category":"boots"},'
            '"replaces":["category"]}\n```'
        )

        self.assertEqual(result, TurnInterpretation(
            intent="BUYING",
            updates={"category": ("boots",)},
            override_kind="NONE",
            override_fields=("category",),
        ))

    def test_invalid_payload_does_not_become_session_state(self) -> None:
        self.assertIsNone(parse_turn_interpretation("not JSON"))
        result = parse_turn_interpretation({
            "intent": "shopping",
            "updates": {"category": [123, True], "admin": ["ignore"]},
        })
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsNone(result.intent)
        self.assertEqual(result.updates, {})

    def test_prompt_contains_schema_and_only_agent_visible_state(self) -> None:
        state = SimpleNamespace(
            mode="BROWSING",
            constraints=ShoppingConstraints(category=("shoes",)),
            semantic_constraints=SimpleNamespace(as_dict=lambda: {"feature": []}),
            last_asked="feature",
            no_preference_attributes={"color"},
        )

        prompt = build_turn_prompt("I'm exploring sweatshirts", state)

        self.assertIn("exploring sweatshirts", prompt)
        self.assertIn("use_case", prompt)
        self.assertIn("conversation framing", prompt)
        self.assertNotIn("target", prompt.casefold())


class InterpretedExtractionTests(unittest.TestCase):
    def test_interpreter_delta_is_canonicalized_without_semantic_query_noise(self) -> None:
        agent = Agent.__new__(Agent)
        interpretation = TurnInterpretation(
            intent="BROWSING",
            updates={"category": ("sweatshirt",)},
        )

        def fake_extract(message: str, **_kwargs: object) -> ShoppingConstraints:
            if message == "I'm exploring sweatshirts and would like to compare some options.":
                return ShoppingConstraints()
            return ShoppingConstraints(category=(message,))

        with patch(
            "starter.agent.constraint_module.extract_constraints",
            side_effect=fake_extract,
        ):
            delta = agent._extract_interpreted_turn(
                "I'm exploring sweatshirts and would like to compare some options.",
                interpretation,
            )

        self.assertEqual(delta.category, ("sweatshirt",))

    def test_interpreter_categorical_delta_replaces_old_exact_categorical_pass(self) -> None:
        agent = Agent.__new__(Agent)
        interpretation = TurnInterpretation(
            intent="BROWSING",
            updates={"category": ("sweatshirt",)},
        )
        message = "I'm exploring sweatshirts and would like to compare some options."

        def fake_extract(text: str, **kwargs: object) -> ShoppingConstraints:
            if text == message and "semantic_matcher" in kwargs:
                return ShoppingConstraints(use_case=("exploring",))
            return ShoppingConstraints(category=(text,))

        with patch(
            "starter.agent.constraint_module.extract_constraints",
            side_effect=fake_extract,
        ):
            delta = agent._extract_interpreted_turn(message, interpretation)

        self.assertEqual(delta.category, ("sweatshirt",))
        self.assertEqual(delta.use_case, ())


if __name__ == "__main__":
    unittest.main()
