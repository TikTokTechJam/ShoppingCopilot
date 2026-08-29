from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starter.agent import Agent
from starter.retrieval import (
    MODE_SCORE_WEIGHTS,
    OVERRIDE_BM25_RETENTION,
    OVERRIDE_SCORE_WEIGHTS,
    score_weights,
)
from starter.routing.constraints import ShoppingConstraints, extract_constraints
from starter.session import (
    OverrideKind,
    SessionManager,
    detect_override_kind,
    is_generic_clarification_reply,
    is_no_preference_reply,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )



def sequenced_extract(deltas):
    """One delta per distinct user message, whatever the Agent's call count.

    The Agent reads a message unscoped to decide whether it is an override,
    then may re-read it scoped to the attribute it asked about. Keying on the
    message keeps the fixture aligned with turns rather than with calls.
    """

    cache: dict[str, object] = {}

    def _extract(message, **_kwargs):
        if message not in cache:
            cache[message] = next(deltas)
        return cache[message]

    return _extract


class FixedRouter:
    def classify(self, _message: str) -> SimpleNamespace:
        return SimpleNamespace(intent="BUYING")


def build_catalog(root: Path) -> Path:
    catalog_path = root / "catalog.jsonl"
    rows = [
        {"parent_asin": f"S{index}", "categories": ["shoes"], "price": 10.0 + index}
        for index in range(6)
    ]
    rows.extend(
        {"parent_asin": f"E{index}", "categories": ["earrings"], "price": 20.0 + index}
        for index in range(6)
    )
    write_jsonl(catalog_path, rows)
    return catalog_path


class OverrideHandlingTests(unittest.TestCase):
    def make_agent(self, root: Path) -> Agent:
        return Agent(
            build_catalog(root),
            facts_path=root / "missing-facts.jsonl",
            embeddings_path=root / "missing-embeddings.npy",
            metadata_path=root / "missing-metadata.json",
            router=FixedRouter(),
        )

    def test_override_markers_are_shared_and_scope_is_separate(self) -> None:
        current = ShoppingConstraints(category=("shoes",), color=("red",))
        preference_delta = ShoppingConstraints(feature=("pockets",))
        full_delta = ShoppingConstraints(category=("earrings",))

        for message in (
            "actually pockets",
            "my priority changed: pockets matter",
            "changed my mind about pockets",
            "instead I want pockets",
        ):
            self.assertEqual(
                detect_override_kind(message, current, preference_delta),
                OverrideKind.PREFERENCE,
                message,
            )

        self.assertEqual(
            detect_override_kind(
                "forget that, I want earrings instead",
                current,
                full_delta,
            ),
            OverrideKind.FULL_GOAL,
        )

    def test_preference_override_preserves_goal_and_mode_but_replaces_initial_preference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {"profile": "kept"})
            deltas = iter(
                (
                    ShoppingConstraints(
                        category=("shoes",),
                        color=("red",),
                        style=("casual",),
                    ),
                    ShoppingConstraints(material=("nylon",)),
                    ShoppingConstraints(feature=("pockets",)),
                )
            )

            with patch(
                "starter.agent.constraint_module.extract_constraints",
                side_effect=sequenced_extract(deltas),
            ):
                agent.respond("session", "shoes in red, casual", 1, 3)
                agent.respond("session", "nylon", 2, 3)
                agent.respond(
                    "session",
                    "Actually, my priority changed. Pockets matter.",
                    3,
                    3,
                )

            state = agent.sessions.get("session")
            self.assertEqual(state.mode, "BUYING")
            self.assertEqual(state.profile, {"profile": "kept"})
            self.assertEqual(state.constraints.category, ("shoes",))
            self.assertEqual(state.constraints.material, ("nylon",))
            self.assertEqual(state.constraints.feature, ("pockets",))
            self.assertEqual(state.constraints.color, ())
            self.assertEqual(state.constraints.style, ())
            self.assertEqual(
                state.query_text,
                "Actually, my priority changed. Pockets matter.",
            )
            self.assertEqual(state.excluded_recommendations, set())
            self.assertEqual(state.last_override_kind, "PREFERENCE")

    def test_preference_transition_clears_clarification_and_recommendation_history(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {})
        state.mode = "BROWSING"
        state.messages[:] = ["old goal"]
        state.asked_attributes.update({"color", "material"})
        state.no_preference_attributes.add("color")
        state.attribute_call_count["material"] = 1
        state.clarification_cycle = 2
        state.last_asked = "material"
        state.last_recommendations = ("A", "B")
        state.excluded_recommendations.update({"A", "B"})
        state.constraints = ShoppingConstraints(
            category=("shoes",),
            color=("red",),
            material=("nylon",),
        )
        state.constraint_provenance = {
            "category": {"shoes": "initial"},
            "color": {"red": "initial"},
            "material": {"nylon": "clarification"},
        }

        manager.reset_preference("session")
        state = manager.get("session")
        self.assertEqual(state.mode, "BROWSING")
        self.assertEqual(state.constraints.category, ("shoes",))
        self.assertEqual(state.constraints.color, ())
        self.assertEqual(state.constraints.material, ("nylon",))
        self.assertEqual(state.messages, [])
        self.assertEqual(state.asked_attributes, set())
        self.assertEqual(state.no_preference_attributes, {"color"})
        self.assertEqual(state.attribute_call_count["material"], 0)
        self.assertEqual(state.clarification_cycle, 1)
        self.assertIsNone(state.last_asked)
        self.assertEqual(state.last_recommendations, ())
        self.assertEqual(state.excluded_recommendations, set())

    def test_full_goal_override_still_resets_category_and_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            deltas = iter(
                (
                    ShoppingConstraints(category=("shoes",)),
                    ShoppingConstraints(category=("shoes",)),
                    ShoppingConstraints(category=("earrings",)),
                )
            )
            with patch(
                "starter.agent.constraint_module.extract_constraints",
                side_effect=sequenced_extract(deltas),
            ):
                agent.respond("session", "shoes", 1, 2)
                agent.respond("session", "show more", 2, 2)
                agent.respond(
                    "session",
                    "forget that, I want earrings instead",
                    3,
                    2,
                )

            state = agent.sessions.get("session")
            self.assertEqual(state.constraints.category, ("earrings",))
            self.assertEqual(state.query_text, "forget that, I want earrings instead")
            self.assertEqual(state.last_override_kind, "FULL_GOAL")

    def test_actually_field_correction_is_not_full_goal_reset(self) -> None:
        current = ShoppingConstraints(category=("shoes",), color=("red",))
        delta = ShoppingConstraints(color=("black",))
        self.assertEqual(
            detect_override_kind("actually black", current, delta),
            OverrideKind.PREFERENCE,
        )

    def test_no_preference_reply_skips_extraction_and_excludes_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            state = agent.sessions.get("session")
            state.mode = "BUYING"
            state.constraints = ShoppingConstraints(category=("shoes",))
            state.messages.append("I need shoes")
            state.last_asked = "material"
            state.asked_attributes.add("material")

            with patch(
                "starter.agent.Agent._extract",
                side_effect=AssertionError(
                    "no-preference replies must not be extracted"
                ),
            ):
                response = agent.respond(
                    "session",
                    "I don't know.",
                    2,
                    2,
                )

            self.assertNotEqual(response["ask_attribute"], "material")
            self.assertIn("material", state.asked_attributes)
            self.assertEqual(state.constraints.category, ("shoes",))
            self.assertEqual(state.constraints.material, ())
            self.assertEqual(state.last_user_message, "I don't know.")
            self.assertEqual(state.query_text, "I need shoes")

    def test_no_preference_detection_requires_pending_clarification(self) -> None:
        self.assertTrue(is_no_preference_reply("I don't know.", "material"))
        self.assertTrue(is_no_preference_reply("I don't have.", "material"))
        self.assertFalse(is_no_preference_reply("I don't know.", None))

    def test_generic_clarification_reply_matches_evaluator_variants_only(self) -> None:
        self.assertTrue(
            is_generic_clarification_reply(
                "Those options are not quite right yet. You can ask me about one specific attribute."
            )
        )
        self.assertTrue(
            is_generic_clarification_reply(
                "Those options are not quite right yet. Ask me about one specific attribute."
            )
        )
        self.assertFalse(
            is_generic_clarification_reply(
                "Those options are not quite right yet. I want one specific attribute."
            )
        )

    def test_generic_clarification_reply_skips_extraction_and_semantic_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            state = agent.sessions.get("session")
            state.mode = "BUYING"
            state.constraints = ShoppingConstraints(category=("shoes",))
            state.messages.append("I need shoes")
            state.last_asked = "other"
            state.asked_attributes.add("other")

            with patch(
                "starter.agent.Agent._extract",
                side_effect=AssertionError(
                    "generic clarification filler must not be extracted"
                ),
            ):
                agent.respond(
                    "session",
                    "Those options are not quite right yet. You can ask me about one specific attribute.",
                    2,
                    2,
                )

            self.assertEqual(state.constraints.category, ("shoes",))
            self.assertEqual(state.constraints.brand, ())
            self.assertEqual(state.constraints.feature, ())
            self.assertEqual(state.query_text, "I need shoes")
            self.assertEqual(
                state.last_user_message,
                "Those options are not quite right yet. You can ask me about one specific attribute.",
            )

    def test_other_without_new_information_stops_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            write_jsonl(
                catalog_path,
                [
                    {
                        "parent_asin": f"S{index}",
                        "categories": ["shoes"],
                        "price": 20.0,
                    }
                    for index in range(4)
                ],
            )
            agent = Agent(
                catalog_path,
                facts_path=root / "missing-facts.jsonl",
                embeddings_path=root / "missing-embeddings.npy",
                metadata_path=root / "missing-metadata.json",
                router=FixedRouter(),
            )
            agent.reset("session", {})

            first = agent.respond("session", "shoes", 1, 3)
            second = agent.respond(
                "session",
                "I don't have a specific preference.",
                2,
                3,
            )

            self.assertEqual(first["ask_attribute"], "other")
            self.assertIsNone(second["ask_attribute"])
            self.assertTrue(agent.sessions.get("session").clarification_stopped)

    def test_normal_clarification_keeps_transcript_and_promotes_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            deltas = iter(
                (
                    ShoppingConstraints(category=("shoes",)),
                    ShoppingConstraints(feature=("pockets",)),
                )
            )
            with patch(
                "starter.agent.constraint_module.extract_constraints",
                side_effect=sequenced_extract(deltas),
            ):
                first = agent.respond("session", "shoes", 1, 2)
                agent.respond("session", "show me more", 2, 2)

            state = agent.sessions.get("session")
            first_ids = {item["parent_asin"] for item in first["recommendations"]}
            self.assertEqual(state.last_override_kind, None)
            self.assertIn("shoes", state.query_text)
            self.assertIn("show me more", state.query_text)
            self.assertTrue(first_ids <= state.excluded_recommendations)

    def test_override_phrase_aliases_and_false_positive_guards(self) -> None:
        cases = {
            "Some practical storage pockets would be useful.": ("feature", "pockets"),
            "Sun protection matters to me.": ("feature", "uv protection"),
            "It is easy to machine wash.": ("feature", "machine washable"),
            "A zip closure would work best for me.": ("feature", "zipper closure"),
            "I want a fit I can adjust.": ("feature", "adjustable"),
            "It should dry quickly.": ("feature", "quick drying"),
        }
        for message, (field_name, expected) in cases.items():
            constraints = extract_constraints(message)
            values = {str(value).replace("_", " ") for value in getattr(constraints, field_name)}
            self.assertIn(expected, values, message)

        false_work = extract_constraints("A zip closure would work best for me.")
        self.assertNotIn("work", false_work.use_case)
        false_it = extract_constraints("It should dry quickly.")
        self.assertNotIn("it", false_it.brand)
        false_machine = extract_constraints("machine wash")
        self.assertNotIn("machine", false_machine.brand)
        self.assertNotIn("wash", false_machine.brand)


class OverrideScoreWeightTests(unittest.TestCase):
    """The lexical channel is demoted once an override is detected."""

    make_agent = OverrideHandlingTests.make_agent

    def test_redistribution_is_mass_preserving_and_only_demotes_bm25(self) -> None:
        for mode, normal in MODE_SCORE_WEIGHTS.items():
            override = OVERRIDE_SCORE_WEIGHTS[mode]
            with self.subTest(mode=mode):
                self.assertAlmostEqual(
                    sum(override.values()), sum(normal.values())
                )
                self.assertAlmostEqual(
                    override["bm25"], normal["bm25"] * OVERRIDE_BM25_RETENTION
                )
                self.assertGreater(override["structured"], normal["structured"])
                self.assertGreater(override["dense"], normal["dense"])
                # The freed weight is split in proportion to what each
                # surviving channel already carried, so their ratio is fixed.
                self.assertAlmostEqual(
                    override["structured"] / override["dense"],
                    normal["structured"] / normal["dense"],
                )

    def test_score_weights_selects_the_table_from_the_flag(self) -> None:
        self.assertEqual(score_weights("BUYING"), MODE_SCORE_WEIGHTS["BUYING"])
        self.assertEqual(
            score_weights("BUYING", override_active=True),
            OVERRIDE_SCORE_WEIGHTS["BUYING"],
        )

    def test_override_latches_the_flag_for_the_rest_of_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            deltas = iter(
                (
                    ShoppingConstraints(category=("shoes",), color=("red",)),
                    ShoppingConstraints(feature=("pockets",)),
                    ShoppingConstraints(material=("nylon",)),
                )
            )

            with patch(
                "starter.agent.constraint_module.extract_constraints",
                side_effect=sequenced_extract(deltas),
            ):
                agent.respond("session", "shoes in red", 1, 3)
                state = agent.sessions.get("session")
                self.assertFalse(state.override_active)

                agent.respond(
                    "session",
                    "Actually, my priority changed. Pockets matter.",
                    2,
                    3,
                )
                self.assertTrue(state.override_active)

                # A plain clarification answer afterwards must not clear it:
                # the discarded transcript never comes back.
                agent.respond("session", "nylon", 3, 3)
                self.assertTrue(state.override_active)

    def test_retrieval_receives_the_flag_once_an_override_is_seen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = self.make_agent(root)
            agent.reset("session", {})
            seen: list[bool] = []
            inner = agent.retriever.retrieve

            def spy(*args, **kwargs):
                seen.append(bool(kwargs.get("override_active")))
                return inner(*args, **kwargs)

            agent.retriever.retrieve = spy
            deltas = iter(
                (
                    ShoppingConstraints(category=("shoes",)),
                    ShoppingConstraints(feature=("pockets",)),
                )
            )
            with patch(
                "starter.agent.constraint_module.extract_constraints",
                side_effect=sequenced_extract(deltas),
            ):
                agent.respond("session", "shoes", 1, 3)
                agent.respond(
                    "session",
                    "Actually, my priority changed. Pockets matter.",
                    2,
                    3,
                )

            self.assertEqual(seen[0], False)
            self.assertEqual(seen[-1], True)


if __name__ == "__main__":
    unittest.main()
