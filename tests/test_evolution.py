from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starter.agent import Agent
from starter.evolution import (
    EvolutionConfig,
    EvolutionLoop,
    TurnObservation,
    distill,
    field_factors,
)
from starter.retrieval import STRUCTURED_FIELD_WEIGHTS
from starter.routing.constraints import ShoppingConstraints
from starter.session import SessionManager

CONFIG = EvolutionConfig()


def _obs(
    *,
    turn: int = 2,
    called_again: bool = True,
    new_pairs: tuple = (),
    reinforced_pairs: tuple = (),
    no_preference: bool = False,
    override_kind: str | None = None,
) -> TurnObservation:
    return TurnObservation(
        turn=turn,
        called_again=called_again,
        new_pairs=new_pairs,
        reinforced_pairs=reinforced_pairs,
        no_preference=no_preference,
        override_kind=override_kind,
    )


# --------------------------------------------------------------------------- #
# DISTILL — pure unit tests
# --------------------------------------------------------------------------- #


class DistillTests(unittest.TestCase):
    def test_reinforcement_raises_field_factor_and_clamps(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(reinforced_pairs=(("feature", "waterproof"),))

        belief = distill({}, obs, constraints=constraints, trace=[], config=CONFIG)
        self.assertAlmostEqual(belief["feature"]["waterproof"], 1.12)

        belief = distill(belief, obs, constraints=constraints, trace=[], config=CONFIG)
        self.assertAlmostEqual(belief["feature"]["waterproof"], 1.24)

        belief = distill(belief, obs, constraints=constraints, trace=[], config=CONFIG)
        # 1.24 + 0.12 = 1.36, clamped to w_max.
        self.assertAlmostEqual(belief["feature"]["waterproof"], CONFIG.w_max)

    def test_unreinforced_one_off_value_stays_absent(self) -> None:
        constraints = ShoppingConstraints(
            feature=("waterproof",), color=("blue",)
        )
        obs = _obs(
            new_pairs=(("color", "blue"),),
            reinforced_pairs=(("feature", "waterproof"),),
        )
        belief = distill({}, obs, constraints=constraints, trace=[], config=CONFIG)
        self.assertNotIn("color", belief)
        self.assertIn("feature", belief)

    def test_no_preference_reply_does_not_reinforce(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(no_preference=True)  # empty delta -> no reinforced pairs
        belief = distill(
            {"feature": {"waterproof": 1.12}},
            obs,
            constraints=constraints,
            trace=[],
            config=CONFIG,
        )
        self.assertAlmostEqual(belief["feature"]["waterproof"], 1.12)

    def test_replace_fields_resets_field_weights(self) -> None:
        obs = _obs(new_pairs=(("color", "brown"),))
        belief = distill(
            {"color": {"black": 1.24}},
            obs,
            constraints=ShoppingConstraints(color=("brown",)),
            replacements=("color",),
            trace=[],
            config=CONFIG,
        )
        self.assertNotIn("color", belief)  # black dropped, brown at unity

    def test_preference_override_seeds_new_value_at_neutral(self) -> None:
        obs = _obs(
            new_pairs=(("feature", "waterproof"),), override_kind="PREFERENCE"
        )
        belief = distill(
            {"feature": {"waterproof": 1.24}},
            obs,
            constraints=ShoppingConstraints(feature=("waterproof",)),
            trace=[],
            config=CONFIG,
        )
        self.assertNotIn("feature", belief)

    def test_new_value_reinforced_only_after_two_distinct_turns(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(new_pairs=(("feature", "waterproof"),))

        first = distill({}, obs, constraints=constraints, trace=[], config=CONFIG)
        self.assertNotIn("feature", first)

        trace = [{"new_pairs": ["feature:waterproof"], "reinforced": []}]
        second = distill({}, obs, constraints=constraints, trace=trace, config=CONFIG)
        self.assertAlmostEqual(second["feature"]["waterproof"], 1.12)

    def test_distiller_is_pure(self) -> None:
        belief = {"feature": {"waterproof": 1.12}}
        snapshot = deepcopy(belief)
        obs = _obs(reinforced_pairs=(("feature", "waterproof"),))
        constraints = ShoppingConstraints(feature=("waterproof",))

        out_a = distill(belief, obs, constraints=constraints, trace=[], config=CONFIG)
        out_b = distill(belief, obs, constraints=constraints, trace=[], config=CONFIG)

        self.assertEqual(belief, snapshot)  # input untouched
        self.assertEqual(out_a, out_b)  # deterministic

    def test_stale_pair_is_pruned_when_no_longer_a_live_constraint(self) -> None:
        belief = {"feature": {"waterproof": 1.24}}
        obs = _obs()
        # constraints no longer carry waterproof.
        pruned = distill(
            belief,
            obs,
            constraints=ShoppingConstraints(category=("jacket",)),
            trace=[],
            config=CONFIG,
        )
        self.assertEqual(pruned, {})


# --------------------------------------------------------------------------- #
# field_factors / act_field_weights
# --------------------------------------------------------------------------- #


class FieldFactorTests(unittest.TestCase):
    def test_mean_collapse_and_clamp(self) -> None:
        belief = {"feature": {"a": 1.30, "b": 1.00}}
        factors = field_factors(
            belief, ShoppingConstraints(feature=("a", "b")), CONFIG
        )
        self.assertAlmostEqual(factors["feature"], 1.15)

        clamped = field_factors(
            {"feature": {"a": 9.0}},
            ShoppingConstraints(feature=("a",)),
            CONFIG,
        )
        self.assertAlmostEqual(clamped["feature"], CONFIG.w_max)

    def test_act_field_weights_returns_none_when_all_factors_unit(self) -> None:
        loop = EvolutionLoop(CONFIG)
        self.assertIsNone(
            loop.act_field_weights({}, ShoppingConstraints(feature=("waterproof",)))
        )

    def test_act_field_weights_scales_only_the_moved_field(self) -> None:
        loop = EvolutionLoop(CONFIG)
        weights = loop.act_field_weights(
            {"feature": {"waterproof": 1.12}},
            ShoppingConstraints(feature=("waterproof",)),
        )
        assert weights is not None
        self.assertEqual(set(weights), set(STRUCTURED_FIELD_WEIGHTS))
        self.assertAlmostEqual(
            weights["feature"], STRUCTURED_FIELD_WEIGHTS["feature"] * 1.12
        )
        self.assertAlmostEqual(weights["brand"], STRUCTURED_FIELD_WEIGHTS["brand"])


# --------------------------------------------------------------------------- #
# SessionManager sidecar clearing
# --------------------------------------------------------------------------- #


class SessionSidecarTests(unittest.TestCase):
    def test_reset_goal_clears_belief_weights_and_trace(self) -> None:
        manager = SessionManager()
        state = manager.reset("s", {})
        state.belief_weights = {"feature": {"waterproof": 1.24}}
        state.evolution_trace = [{"turn": 1, "pool_size": 5, "shown": ("A",)}]
        manager.reset_goal("s")
        self.assertEqual(state.belief_weights, {})
        self.assertEqual(state.evolution_trace, [])

    def test_reset_preference_drops_belief_for_the_overridden_field_only(self) -> None:
        manager = SessionManager()
        state = manager.reset("s", {})
        state.constraints = ShoppingConstraints(
            category=("boots",), color=("black", "navy")
        )
        state.belief_weights = {
            "color": {"black": 1.24, "navy": 1.12},
            "category": {"boots": 1.30},
        }
        state.evolution_trace = [{"turn": 1, "pool_size": 3, "shown": ()}]

        # color is an independent root, so reset_preference removes every color
        # value; category is untouched.
        manager.reset_preference("s", overridden_fields=["color"])

        self.assertEqual(state.belief_weights, {"category": {"boots": 1.30}})
        self.assertEqual(state.evolution_trace, [])


# --------------------------------------------------------------------------- #
# Agent integration
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class _FixedRouter:
    def classify(self, _message: str) -> SimpleNamespace:
        return SimpleNamespace(intent="BUYING")


def _patch_extract(mapping: dict[str, ShoppingConstraints]):
    """Patch the extractor by message text, tolerant of repeat calls per turn.

    `Agent.respond` may call the extractor twice on a turn (once broad, once
    scoped to the asked attribute), so a message-keyed lookup is more robust
    than a call-count iterator.
    """

    def side_effect(message: str, **_kwargs) -> ShoppingConstraints:
        return mapping[message]

    return patch(
        "starter.agent.constraint_module.extract_constraints",
        side_effect=side_effect,
    )


TARGET = "P25"


def _build_fixture(root: Path) -> tuple[Path, Path]:
    """30 jackets. P0..P19 have style=quilted; the target P25 has
    feature=waterproof; the rest match category only.

    On turn 1 (category + style + feature) every quilted item and the target
    tie at 1.2, so the target sits at rank 21 and turn 1 shows P0..P9. On
    turn 2 those ten are excluded, leaving ten quilted items (P10..P19) still
    tied with the target -- so the target is rank 11 unless the reinforced
    feature weight lifts it.
    """

    catalog = root / "catalog.jsonl"
    facts = root / "facts.jsonl"
    catalog_rows: list[dict] = []
    fact_rows: list[dict] = []
    for index in range(30):
        asin = f"P{index}"
        catalog_rows.append(
            {"parent_asin": asin, "categories": ["jacket"], "price": 20.0}
        )
        if index < 20:
            fact_rows.append({"parent_asin": asin, "facts": {"style": ["quilted"]}})
        elif asin == TARGET:
            fact_rows.append(
                {"parent_asin": asin, "facts": {"feature": ["waterproof"]}}
            )
    _write_jsonl(catalog, catalog_rows)
    _write_jsonl(facts, fact_rows)
    return catalog, facts


def _make_agent(root: Path, *, enable_evolution: bool) -> Agent:
    catalog, facts = _build_fixture(root)
    return Agent(
        catalog,
        facts_path=facts,
        embeddings_path=root / "missing-embeddings.npy",
        metadata_path=root / "missing-metadata.json",
        router=_FixedRouter(),
        enable_evolution=enable_evolution,
    )


def _target_rank(response: dict, target: str) -> int | None:
    ids = [item["parent_asin"] for item in response["recommendations"]]
    return ids.index(target) + 1 if target in ids else None


class AgentEvolutionTests(unittest.TestCase):
    def test_enabled_reweights_after_repeated_constraint(self) -> None:
        target = TARGET
        mapping = {
            "a quilted waterproof jacket": ShoppingConstraints(
                category=("jacket",), style=("quilted",), feature=("waterproof",)
            ),
            # Turn 2 restates only the feature -- so only that field is
            # reinforced and the target gains a relative edge over the
            # equally-scored quilted items.
            "waterproof is what matters": ShoppingConstraints(
                feature=("waterproof",)
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            off = _make_agent(root, enable_evolution=False)
            off.reset("s", {})
            with _patch_extract(mapping):
                off.respond("s", "a quilted waterproof jacket", 1, 10)
                off_turn2 = off.respond("s", "waterproof is what matters", 2, 10)

            on = _make_agent(root, enable_evolution=True)
            on.reset("s", {})
            with _patch_extract(mapping):
                on.respond("s", "a quilted waterproof jacket", 1, 10)
                on_turn2 = on.respond("s", "waterproof is what matters", 2, 10)

        self.assertIsNone(_target_rank(off_turn2, target))
        self.assertEqual(_target_rank(on_turn2, target), 1)

    def test_no_reinforcement_script_is_identical_on_and_off(self) -> None:
        # Each turn introduces a brand-new value, so nothing is ever reinforced
        # and field_weights stays None -- the loop must be a no-op.
        mapping = {
            "a jacket": ShoppingConstraints(category=("jacket",)),
            "black one": ShoppingConstraints(category=("jacket",), color=("black",)),
            "nylon please": ShoppingConstraints(
                category=("jacket",), color=("black",), material=("nylon",)
            ),
        }
        messages = list(mapping)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses: dict[bool, list[dict]] = {}
            for flag in (False, True):
                agent = _make_agent(root, enable_evolution=flag)
                agent.reset("s", {})
                with _patch_extract(mapping):
                    responses[flag] = [
                        agent.respond("s", message, turn, 10)
                        for turn, message in enumerate(messages, 1)
                    ]

        self.assertEqual(responses[False], responses[True])

    def test_turn_one_is_identical_on_and_off(self) -> None:
        delta = ShoppingConstraints(category=("jacket",), feature=("waterproof",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for flag in (False, True):
                agent = _make_agent(root, enable_evolution=flag)
                agent.reset("s", {})
                with patch(
                    "starter.agent.constraint_module.extract_constraints",
                    return_value=delta,
                ):
                    results.append(agent.respond("s", "waterproof jacket", 1, 10))
        self.assertEqual(results[0], results[1])

    def test_response_contract_still_valid_with_evolution(self) -> None:
        allowed = {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other", None,
        }
        delta = ShoppingConstraints(category=("jacket",), feature=("waterproof",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = _make_agent(root, enable_evolution=True)
            agent.reset("s", {})
            with patch(
                "starter.agent.constraint_module.extract_constraints",
                return_value=delta,
            ):
                for turn in (1, 2, 3):
                    response = agent.respond("s", "waterproof jacket", turn, 10)
                    self.assertEqual(
                        set(response),
                        {"message", "ask_attribute", "recommendations", "usage"},
                    )
                    self.assertEqual(
                        response["usage"],
                        {"prompt_tokens": 0, "completion_tokens": 0},
                    )
                    self.assertIn(response["ask_attribute"], allowed)
                    self.assertLessEqual(len(response["recommendations"]), 10)
                    for item in response["recommendations"]:
                        self.assertIn(
                            item["parent_asin"], agent.retriever.valid_asins
                        )

    def test_telemetry_is_deterministic_and_counters_are_monotonic(self) -> None:
        delta = ShoppingConstraints(category=("jacket",), feature=("waterproof",))

        def run(root: Path) -> tuple[dict, list[int]]:
            agent = _make_agent(root, enable_evolution=True)
            agent.reset("s", {})
            seen: list[int] = []
            with patch(
                "starter.agent.constraint_module.extract_constraints",
                return_value=delta,
            ):
                for turn in (1, 2, 3):
                    agent.respond("s", "waterproof jacket", turn, 10)
                    seen.append(
                        int(agent.telemetry()["evolution.turns_observed"])
                    )
            return agent.telemetry(), seen

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, counters = run(root)
            second, _ = run(root)

        self.assertEqual(first, second)
        self.assertEqual(counters, sorted(counters))
        self.assertEqual(counters[-1], 3)
        self.assertGreaterEqual(first["evolution.reinforce_events"], 2)


# --------------------------------------------------------------------------- #
# retrieve() field_weights kwarg
# --------------------------------------------------------------------------- #


class RetrieveFieldWeightsTests(unittest.TestCase):
    def _retriever(self, root: Path):
        from starter.retrieval import ProductRetriever

        catalog, facts = _build_fixture(root)
        return ProductRetriever(
            catalog,
            facts_path=facts,
            embeddings_path=root / "missing.npy",
            metadata_path=root / "missing.json",
        )

    def test_none_and_explicit_default_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self._retriever(Path(directory))
            constraints = ShoppingConstraints(
                category=("jacket",), style=("quilted",), feature=("waterproof",)
            )
            base = retriever.retrieve("BUYING", "q", constraints, limit=20)
            explicit = retriever.retrieve(
                "BUYING",
                "q",
                constraints,
                limit=20,
                field_weights=dict(STRUCTURED_FIELD_WEIGHTS),
            )
            none = retriever.retrieve(
                "BUYING", "q", constraints, limit=20, field_weights=None
            )
            self.assertEqual(base, none)
            self.assertEqual(base, explicit)

    def test_debug_rank_all_prefix_matches_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = self._retriever(Path(directory))
            constraints = ShoppingConstraints(category=("jacket",))
            full = retriever.debug_rank_all("BUYING", "q", constraints)
            top5 = retriever.retrieve("BUYING", "q", constraints, limit=5)
            self.assertEqual([c.parent_asin for c in full[:5]],
                             [c.parent_asin for c in top5])


if __name__ == "__main__":
    unittest.main()
