from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starter.agent import Agent
from starter.routing.constraints import (
    ConstraintEvidence,
    SemanticShoppingConstraints,
    ShoppingConstraints,
    extract_constraints,
)
from starter.session import (
    ConstraintProvenance,
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

    def test_lexical_markers_do_not_classify_override_scope(self) -> None:
        current = ShoppingConstraints(category=("shoes",), color=("red",))
        preference_delta = ShoppingConstraints(feature=("pockets",))
        replacement_delta = ShoppingConstraints(category=("earrings",))

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
                replacement_delta,
            ),
            OverrideKind.PREFERENCE,
        )

    def test_preference_override_preserves_independent_constraints_and_goal(self) -> None:
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
                excluded_before_override = set(
                    agent.sessions.get("session").excluded_recommendations
                )
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
            self.assertEqual(state.constraints.color, ("red",))
            self.assertEqual(state.constraints.style, ("casual",))
            self.assertEqual(
                state.query_text,
                "shoes in red, casual nylon Actually, my priority changed. Pockets matter.",
            )
            self.assertEqual(state.excluded_recommendations, excluded_before_override)
            self.assertEqual(state.last_override_kind, "PREFERENCE")

    def test_preference_transition_restarts_clarification_without_clearing_goal_state(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {})
        state.mode = "BROWSING"
        state.messages[:] = ["old goal"]
        state.llm_summary_messages[:] = ["old summarized preference"]
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

        manager.reset_preference("session", overridden_fields=("color",))
        state = manager.get("session")
        self.assertEqual(state.mode, "BROWSING")
        self.assertEqual(state.constraints.category, ("shoes",))
        self.assertEqual(state.constraints.color, ())
        self.assertEqual(state.constraints.material, ("nylon",))
        self.assertEqual(state.messages, ["old goal"])
        self.assertEqual(state.llm_summary_messages, ["old summarized preference"])
        self.assertEqual(state.asked_attributes, set())
        self.assertEqual(state.no_preference_attributes, {"color"})
        self.assertEqual(state.attribute_call_count["material"], 0)
        self.assertEqual(state.clarification_cycle, 1)
        self.assertIsNone(state.last_asked)
        self.assertEqual(state.last_recommendations, ("A", "B"))
        self.assertEqual(state.excluded_recommendations, {"A", "B"})

    def test_actually_field_correction_is_not_a_full_reset(self) -> None:
        current = ShoppingConstraints(category=("shoes",), color=("red",))
        delta = ShoppingConstraints(color=("black",))
        self.assertEqual(
            detect_override_kind("actually black", current, delta),
            OverrideKind.PREFERENCE,
        )
        self.assertEqual(
            detect_override_kind(
                "actually boots instead",
                current,
                ShoppingConstraints(category=("boots",)),
            ),
            OverrideKind.PREFERENCE,
        )

    def test_category_override_removes_inferred_descendants_but_keeps_explicit_values(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {})
        state.semantic_constraints = state.semantic_constraints.__class__(
            category=("shirt",),
            color=("black",),
            use_case=("sunny",),
            feature=("uv protection",),
        )
        state.semantic_constraint_provenance = {
            "category": {
                "shirt": ConstraintProvenance("category", "shirt", "inferred")
            },
            "color": {
                "black": ConstraintProvenance("color", "black", "explicit")
            },
            "use_case": {
                "sunny": ConstraintProvenance(
                    "use_case", "sunny", "inferred", ("category", "shirt")
                )
            },
            "feature": {
                "uv protection": ConstraintProvenance(
                    "feature", "uv protection", "inferred", ("use_case", "sunny")
                )
            },
        }

        manager.reset_preference("session", overridden_fields=("category",))

        self.assertEqual(state.semantic_constraints.category, ())
        self.assertEqual(state.semantic_constraints.use_case, ())
        self.assertEqual(state.semantic_constraints.feature, ())
        self.assertEqual(state.semantic_constraints.color, ("black",))

    def test_use_case_override_removes_only_inferred_feature_branch(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {})
        state.semantic_constraints = state.semantic_constraints.__class__(
            category=("shoes",),
            color=("black",),
            use_case=("running",),
            feature=("moisture wicking",),
            material=("mesh",),
        )
        state.semantic_constraint_provenance = {
            "category": {
                "shoes": ConstraintProvenance("category", "shoes", "explicit")
            },
            "color": {
                "black": ConstraintProvenance("color", "black", "explicit")
            },
            "use_case": {
                "running": ConstraintProvenance("use_case", "running", "inferred")
            },
            "feature": {
                "moisture wicking": ConstraintProvenance(
                    "feature", "moisture wicking", "inferred", ("use_case", "running")
                )
            },
            "material": {
                "mesh": ConstraintProvenance(
                    "material", "mesh", "inferred", ("use_case", "running")
                )
            },
        }

        manager.reset_preference("session", overridden_fields=("use_case",))

        self.assertEqual(state.semantic_constraints.category, ("shoes",))
        self.assertEqual(state.semantic_constraints.use_case, ())
        self.assertEqual(state.semantic_constraints.feature, ())
        self.assertEqual(state.semantic_constraints.material, ())
        self.assertEqual(state.semantic_constraints.color, ("black",))

    def test_color_override_does_not_remove_other_semantic_constraints(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {})
        state.semantic_constraints = state.semantic_constraints.__class__(
            color=("red",),
            use_case=("running",),
            feature=("breathable",),
        )
        state.semantic_constraint_provenance = {
            "color": {
                "red": ConstraintProvenance("color", "red", "explicit")
            },
            "use_case": {
                "running": ConstraintProvenance("use_case", "running", "inferred")
            },
            "feature": {
                "breathable": ConstraintProvenance(
                    "feature", "breathable", "inferred", ("use_case", "running")
                )
            },
        }

        manager.reset_preference("session", overridden_fields=("color",))

        self.assertEqual(state.semantic_constraints.color, ())
        self.assertEqual(state.semantic_constraints.use_case, ("running",))
        self.assertEqual(state.semantic_constraints.feature, ("breathable",))

    def test_brand_override_preserves_color_and_removes_old_brand(self) -> None:
        manager = SessionManager()
        state = manager.reset("session", {})
        state.constraints = ShoppingConstraints(brand=("nike",), color=("black",))
        state.constraint_provenance = {
            "brand": {"nike": ConstraintProvenance("brand", "nike", "explicit")},
            "color": {
                "black": ConstraintProvenance("color", "black", "explicit")
            },
        }

        manager.reset_preference("session", overridden_fields=("brand",))

        self.assertEqual(state.constraints.brand, ())
        self.assertEqual(state.constraints.color, ("black",))

    def test_updates_record_explicit_and_inferred_dependency_provenance(self) -> None:
        manager = SessionManager()
        manager.reset("session", {})
        semantic_delta = SemanticShoppingConstraints(
            category=("shirts",),
            use_case=("rain",),
            feature=("waterproof",),
            evidence=(
                ConstraintEvidence("category:shirts", "category", "shirts", "semantic_1gram", 1.0, "layer2"),
                ConstraintEvidence("use_case:rain", "use_case", "rain", "semantic_1gram", 1.0, "layer2"),
                ConstraintEvidence("feature:waterproof", "feature", "waterproof", "semantic_1gram", 1.0, "layer2"),
            ),
        )

        manager.update_constraints(
            "session",
            ShoppingConstraints(brand=("nike",)),
            semantic_delta=semantic_delta,
        )
        state = manager.get("session")

        self.assertEqual(
            state.constraint_provenance["brand"]["nike"].source,
            "explicit",
        )
        self.assertEqual(
            state.semantic_constraint_provenance["use_case"]["rain"].parent,
            ("category", "shirts"),
        )
        self.assertEqual(
            state.semantic_constraint_provenance["feature"]["waterproof"].parent,
            ("use_case", "rain"),
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


if __name__ == "__main__":
    unittest.main()
