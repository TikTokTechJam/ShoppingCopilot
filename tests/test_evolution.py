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
    FULL_CONFIG,
    CrossSessionStore,
    EvolutionConfig,
    EvolutionLoop,
    Strategy,
    StrategyController,
    TurnObservation,
    distill,
    field_factors,
)
from starter.retrieval import STRUCTURED_FIELD_WEIGHTS, _final_score
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

    def test_score_weights_none_matches_mode_table(self) -> None:
        for mode in ("BUYING", "BROWSING"):
            default = _final_score(mode, 2.0, 0.5, 1.0)
            explicit = _final_score(mode, 2.0, 0.5, 1.0, None)
            self.assertEqual(default, explicit)

    def test_score_weights_override_changes_the_blend(self) -> None:
        base = _final_score("BUYING", 1.0, 1.0, 1.0)
        heavy_dense = _final_score("BUYING", 1.0, 1.0, 1.0, (1.0, 3.0, 0.0))
        self.assertAlmostEqual(heavy_dense, 1.0 * 1.0 + 3.0 * 1.0 + 0.0 * 1.0)
        self.assertNotEqual(base, heavy_dense)


# --------------------------------------------------------------------------- #
# Stage 1 — implicit-negative decay
# --------------------------------------------------------------------------- #

_DECAY_CONFIG = EvolutionConfig(enable_implicit_negative=True)


class ImplicitNegativeDecayTests(unittest.TestCase):
    def _trace(self, *, made_progress: bool, shown: tuple[str, ...]) -> list[dict]:
        return [{"turn": 2, "pool_size": 40, "shown": shown,
                 "reinforced": [], "new_pairs": [], "made_progress": made_progress,
                 "churn": 1.0, "pool_delta": 0}]

    def test_decays_value_common_to_every_shown_and_missed_candidate(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(turn=3, called_again=True)
        facts = {("A", "feature"): ("waterproof",), ("B", "feature"): ("waterproof",)}
        out = distill(
            {}, obs, constraints=constraints,
            trace=self._trace(made_progress=False, shown=("A", "B")),
            fact_lookup=lambda a, f: facts.get((a, f), ()),
            config=_DECAY_CONFIG,
        )
        # 1.0 - neg_decay - oneoff_extra_penalty = 0.82
        self.assertAlmostEqual(out["feature"]["waterproof"], 0.82)

    def test_no_decay_when_value_discriminates(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(turn=3, called_again=True)
        facts = {("A", "feature"): ("waterproof",), ("B", "feature"): ()}
        out = distill(
            {}, obs, constraints=constraints,
            trace=self._trace(made_progress=False, shown=("A", "B")),
            fact_lookup=lambda a, f: facts.get((a, f), ()),
            config=_DECAY_CONFIG,
        )
        self.assertEqual(out, {})

    def test_no_decay_before_min_turn(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(turn=2, called_again=True)
        facts = {("A", "feature"): ("waterproof",)}
        out = distill(
            {}, obs, constraints=constraints,
            trace=self._trace(made_progress=False, shown=("A",)),
            fact_lookup=lambda a, f: facts.get((a, f), ()),
            config=_DECAY_CONFIG,
        )
        self.assertEqual(out, {})

    def test_gated_off_by_default(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(turn=3, called_again=True)
        facts = {("A", "feature"): ("waterproof",)}
        out = distill(
            {}, obs, constraints=constraints,
            trace=self._trace(made_progress=False, shown=("A",)),
            fact_lookup=lambda a, f: facts.get((a, f), ()),
            config=CONFIG,  # enable_implicit_negative = False
        )
        self.assertEqual(out, {})

    def test_no_decay_when_last_turn_made_progress(self) -> None:
        constraints = ShoppingConstraints(feature=("waterproof",))
        obs = _obs(turn=3, called_again=True)
        facts = {("A", "feature"): ("waterproof",)}
        out = distill(
            {}, obs, constraints=constraints,
            trace=self._trace(made_progress=True, shown=("A",)),
            fact_lookup=lambda a, f: facts.get((a, f), ()),
            config=_DECAY_CONFIG,
        )
        self.assertEqual(out, {})


# --------------------------------------------------------------------------- #
# Stage 2 — StrategyController
# --------------------------------------------------------------------------- #

_REPLAN_CONFIG = EvolutionConfig(enable_replan=True)


class StrategyControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = StrategyController(_REPLAN_CONFIG)
        self.off = StrategyController(CONFIG)

    def test_disabled_always_neutral(self) -> None:
        obs = _obs(turn=5, new_pairs=(("color", "black"),))
        self.assertIs(self.off.choose(obs, [], ("color",)), Strategy.NEUTRAL)

    def test_turn_one_or_no_constraints_is_recall_broad(self) -> None:
        self.assertIs(
            self.ctrl.choose(_obs(turn=1), [], ()), Strategy.RECALL_BROAD
        )
        self.assertIs(
            self.ctrl.choose(_obs(turn=4), [], ()), Strategy.RECALL_BROAD
        )

    def test_stuck_is_diversify(self) -> None:
        trace = [{"turn": 4, "pool_size": 50, "made_progress": False,
                  "churn": 0.0, "pool_delta": 0}]
        obs = _obs(turn=5, called_again=True)
        self.assertIs(
            self.ctrl.choose(obs, trace, ("color",)), Strategy.DIVERSIFY
        )

    def test_small_pool_is_relax_weakest(self) -> None:
        trace = [{"turn": 4, "pool_size": 5, "made_progress": True,
                  "churn": 1.0, "pool_delta": -3}]
        obs = _obs(turn=5, called_again=True)
        self.assertIs(
            self.ctrl.choose(obs, trace, ("color",)), Strategy.RELAX_WEAKEST
        )

    def test_new_constraint_plus_contraction_is_exploit_narrow(self) -> None:
        trace = [{"turn": 4, "pool_size": 60, "made_progress": True,
                  "churn": 1.0, "pool_delta": -20}]
        obs = _obs(turn=5, called_again=True, new_pairs=(("color", "black"),))
        self.assertIs(
            self.ctrl.choose(obs, trace, ("color",)), Strategy.EXPLOIT_NARROW
        )

    def test_apply_relax_weakest_softens_the_lowest_weight_field(self) -> None:
        adjusted, score_weights = self.ctrl.apply(
            Strategy.RELAX_WEAKEST, None, ("brand", "feature")
        )
        self.assertIsNone(score_weights)
        # feature (0.5) is weaker than brand (>=3.0)
        self.assertAlmostEqual(
            adjusted["feature"],
            STRUCTURED_FIELD_WEIGHTS["feature"] * _REPLAN_CONFIG.relax_scale,
        )
        self.assertEqual(adjusted["brand"], STRUCTURED_FIELD_WEIGHTS["brand"])

    def test_apply_exploit_narrow_returns_structured_heavy_score_weights(self) -> None:
        _adjusted, score_weights = self.ctrl.apply(
            Strategy.EXPLOIT_NARROW, None, ("color",)
        )
        self.assertEqual(
            score_weights,
            (1.0, _REPLAN_CONFIG.exploit_dense_weight,
             _REPLAN_CONFIG.exploit_bm25_weight),
        )


# --------------------------------------------------------------------------- #
# Stage 3 — CrossSessionStore
# --------------------------------------------------------------------------- #

_LEARN_CONFIG = EvolutionConfig(enable_learn=True)


class CrossSessionStoreTests(unittest.TestCase):
    def test_disabled_prior_is_one(self) -> None:
        store = CrossSessionStore(CONFIG)
        store.observe_session_end(
            belief_weights={"feature": {"waterproof": 1.24}},
            trace=[{"turn": 1, "reinforced": ["feature:waterproof"],
                    "new_pairs": [], "made_progress": True, "pool_delta": -5}],
        )
        self.assertEqual(store.prior_factor("feature"), 1.0)

    def test_progress_after_reinforcement_nudges_prior_up(self) -> None:
        store = CrossSessionStore(_LEARN_CONFIG)
        trace = [
            {"turn": 1, "reinforced": [], "new_pairs": ["feature:waterproof"],
             "made_progress": True, "pool_delta": 0},
            {"turn": 2, "reinforced": ["feature:waterproof"], "new_pairs": [],
             "made_progress": True, "pool_delta": -8},
            {"turn": 3, "reinforced": [], "new_pairs": [],
             "made_progress": True, "pool_delta": -4},
        ]
        touched = store.observe_session_end(
            belief_weights={"feature": {"waterproof": 1.24}}, trace=trace
        )
        self.assertEqual(touched, 1)
        self.assertGreater(store.prior_factor("feature"), 1.0)
        self.assertLessEqual(
            store.prior_factor("feature"), _LEARN_CONFIG.learn_prior_ceiling
        )

    def test_never_reinforced_field_decays_toward_one(self) -> None:
        from starter.evolution.store import _FieldPrior

        store = CrossSessionStore(_LEARN_CONFIG)
        store._priors["color"] = _FieldPrior(mean=1.14)  # a prior above neutral
        store.observe_session_end(
            belief_weights={},
            trace=[{"turn": 1, "reinforced": [], "new_pairs": ["color:black"],
                    "made_progress": False, "pool_delta": 0}],
        )
        self.assertLess(store.prior_factor("color"), 1.14)


# --------------------------------------------------------------------------- #
# FULL_CONFIG integration
# --------------------------------------------------------------------------- #


class FullLoopIntegrationTests(unittest.TestCase):
    def test_full_config_keeps_the_response_contract(self) -> None:
        allowed = {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other", None,
        }
        delta = ShoppingConstraints(category=("jacket",), feature=("waterproof",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, facts = _build_fixture(root)
            agent = Agent(
                catalog, facts_path=facts,
                embeddings_path=root / "m.npy", metadata_path=root / "m.json",
                router=_FixedRouter(), enable_evolution=True,
                evolution_config=FULL_CONFIG,
            )
            for session in ("s1", "s2"):
                agent.reset(session, {})
                with patch(
                    "starter.agent.constraint_module.extract_constraints",
                    return_value=delta,
                ):
                    for turn in range(1, 11):
                        r = agent.respond(session, "waterproof jacket", turn, 10)
                        self.assertEqual(
                            set(r),
                            {"message", "ask_attribute", "recommendations", "usage"},
                        )
                        self.assertEqual(
                            r["usage"],
                            {"prompt_tokens": 0, "completion_tokens": 0},
                        )
                        self.assertIn(r["ask_attribute"], allowed)
                        self.assertLessEqual(len(r["recommendations"]), 10)

    def test_full_config_is_deterministic(self) -> None:
        delta = ShoppingConstraints(category=("jacket",), feature=("waterproof",))

        def run(root: Path) -> list[list[str]]:
            catalog, facts = _build_fixture(root)
            agent = Agent(
                catalog, facts_path=facts,
                embeddings_path=root / "m.npy", metadata_path=root / "m.json",
                router=_FixedRouter(), enable_evolution=True,
                evolution_config=FULL_CONFIG,
            )
            out = []
            agent.reset("s", {})
            with patch(
                "starter.agent.constraint_module.extract_constraints",
                return_value=delta,
            ):
                for turn in range(1, 6):
                    r = agent.respond("s", "waterproof jacket", turn, 10)
                    out.append([x["parent_asin"] for x in r["recommendations"]])
            return out

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            self.assertEqual(run(Path(d1)), run(Path(d2)))

    def test_learn_priors_survive_reset_and_appear_in_telemetry(self) -> None:
        delta = ShoppingConstraints(category=("jacket",), feature=("waterproof",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, facts = _build_fixture(root)
            agent = Agent(
                catalog, facts_path=facts,
                embeddings_path=root / "m.npy", metadata_path=root / "m.json",
                router=_FixedRouter(), enable_evolution=True,
                evolution_config=FULL_CONFIG,
            )
            for session in ("s1", "s2", "s3"):
                agent.reset(session, {})
                with patch(
                    "starter.agent.constraint_module.extract_constraints",
                    return_value=delta,
                ):
                    for turn in range(1, 11):
                        agent.respond(session, "waterproof jacket", turn, 10)
            # priors accumulated across sessions and are exposed
            self.assertIsInstance(agent.evolution_priors(), dict)
            self.assertGreaterEqual(
                int(agent.telemetry().get("evolution.learn_sessions", 0)), 1
            )


if __name__ == "__main__":
    unittest.main()
