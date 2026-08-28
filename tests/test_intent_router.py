from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from evaluator.hard_evaluator import customer_sentence
from starter.routing import (
    BROWSING,
    BUYING,
    CascadingIntentRouter,
    IntentResult,
    LexicalIntentRouter,
    SessionIntentTracker,
    TwoPhaseIntentRouter,
    extract_constraints,
    lexicon,
)
from starter.routing.constraints import (
    CATEGORICAL_FIELDS,
    _load_default_dictionary,
)
from starter.routing.local_model import LABEL_QUERIES, QwenRerankerBackend


ROOT = Path(__file__).parents[1]
GOLDEN_PATH = ROOT / "tests" / "data" / "intent_golden.jsonl"
DEV_PATH = ROOT / "data" / "derived" / "intent" / "dev_set.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class StubBackend:
    """Stands in for the reranker so the cascade is testable without 1.2 GB."""

    def __init__(self, result=None, *, raises: BaseException | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[str] = []

    def score(self, message: str):
        self.calls.append(message)
        if self.raises is not None:
            raise self.raises
        return self.result


# ---------------------------------------------------------------------------
# Issue #6 acceptance criteria, stated as assertions
# ---------------------------------------------------------------------------


class ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = LexicalIntentRouter()

    def test_returns_only_the_two_allowed_intents(self) -> None:
        for row in load_jsonl(GOLDEN_PATH) + load_jsonl(DEV_PATH):
            self.assertIn(self.router.classify(row["message"]).intent, {BUYING, BROWSING})

    def test_confidence_is_in_the_unit_interval(self) -> None:
        for row in load_jsonl(DEV_PATH):
            result = self.router.classify(row["message"])
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            # Confidence describes the label returned, so it is never a
            # minority probability.
            self.assertGreaterEqual(result.confidence, 0.5)

    def test_serialises_to_the_shape_the_issue_specifies(self) -> None:
        payload = self.router.classify("I need black boots under $100.").as_dict()
        self.assertEqual(payload["intent"], BUYING)
        self.assertIsInstance(payload["confidence"], float)
        self.assertTrue(0.0 <= payload["confidence"] <= 1.0)

    def test_classification_is_deterministic(self) -> None:
        message = "I'm browsing running shoes and not sure what I want yet."
        first = self.router.classify(message)
        for _ in range(5):
            self.assertEqual(self.router.classify(message), first)

    def test_never_raises_on_degenerate_input(self) -> None:
        for message in ["", "   ", "?!?", "🙂🙂🙂", "a" * 20_000, "\n\t\n"]:
            result = self.router.classify(message)
            self.assertIsInstance(result, IntentResult)
            self.assertIn(result.intent, {BUYING, BROWSING})

    def test_needs_no_catalog_and_no_network(self) -> None:
        self.assertFalse(hasattr(self.router, "catalog_path"))
        self.assertFalse(hasattr(self.router, "connection"))


class GoldenSetTest(unittest.TestCase):
    """The 13 examples in issue #6. The spec, never a training set."""

    def test_every_issue_example(self) -> None:
        router = LexicalIntentRouter()
        failures = [
            f"want {row['intent']} got {result.intent} @{result.confidence:.2f} :: {row['message']}"
            for row in load_jsonl(GOLDEN_PATH)
            if (result := router.classify(row["message"])).intent != row["intent"]
        ]
        self.assertEqual(failures, [], "\n".join(failures))


class AmbiguityTest(unittest.TestCase):
    """The two cases the issue calls out by name, plus the traps found building it."""

    def setUp(self) -> None:
        self.router = LexicalIntentRouter()

    def test_category_keyword_does_not_override_exploratory_language(self) -> None:
        for message in [
            "I'm browsing running shoes and not sure what I want yet.",
            "Exploring leather jackets, not sure what I want yet.",
            "Just looking at winter coats for now.",
        ]:
            self.assertEqual(self.router.classify(message).intent, BROWSING, message)

    def test_strong_constraints_push_a_broad_request_to_buying(self) -> None:
        for message in [
            "I need something for hiking, but it must be waterproof and under $80.",
            "Something for travel, but it has to be lightweight and machine washable.",
            "I want anything comfortable as long as it's cotton and under $25.",
        ]:
            self.assertEqual(self.router.classify(message).intent, BUYING, message)

    def test_category_alone_contributes_no_weight(self) -> None:
        without = self.router.classify("I'm just browsing, not sure yet.")
        with_category = self.router.classify("I'm just browsing hiking boots, not sure yet.")
        self.assertEqual(without.intent, with_category.intent)

    def test_request_verb_needs_an_actual_request(self) -> None:
        result = self.router.classify("Can you show me some gift ideas?")
        self.assertEqual(result.intent, BROWSING)
        self.assertNotIn("request_verb", {s.name for s in result.signals})

    def test_single_letter_size_tokens_do_not_match_contractions(self) -> None:
        r"""Regression: \bm\b matched the m in "I'm" and invented a size."""
        result = self.router.classify("I'm looking for something nice for a trip.")
        self.assertNotIn("size", {s.name for s in result.signals})

    def test_down_jacket_is_a_category_not_a_material(self) -> None:
        result = self.router.classify("I'm looking for down jackets, undecided so far.")
        self.assertNotIn("material", {s.name for s in result.signals})
        self.assertEqual(result.intent, BROWSING)

    def test_contentless_message_is_weak_browsing(self) -> None:
        for message in ["hi", "hello there", "hmm", "ok thanks"]:
            result = self.router.classify(message)
            self.assertEqual(result.intent, BROWSING, message)
            self.assertTrue(result.weak, message)


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------


class CascadeTest(unittest.TestCase):
    CONFIDENT = "I need black waterproof hiking boots under $100."
    UNCERTAIN = "Show me belts, I'm just getting a feel for the range."

    def test_confident_messages_do_not_reach_the_model(self) -> None:
        backend = StubBackend((BROWSING, 0.99))
        router = CascadingIntentRouter(backend=backend)
        result = router.classify(self.CONFIDENT)
        self.assertEqual(result.tier, "rules")
        self.assertEqual(backend.calls, [])
        self.assertEqual(router.escalations, 0)

    def test_uncertain_messages_escalate(self) -> None:
        backend = StubBackend((BROWSING, 0.95))
        router = CascadingIntentRouter(backend=backend)
        result = router.classify(self.UNCERTAIN)
        self.assertEqual(result.tier, "reranker")
        self.assertEqual(result.intent, BROWSING)
        self.assertEqual(len(backend.calls), 1)

    def test_contentless_messages_are_not_escalated(self) -> None:
        """A greeting holds nothing for a model to read."""
        backend = StubBackend((BUYING, 0.99))
        router = CascadingIntentRouter(backend=backend)
        result = router.classify("hi")
        self.assertEqual(backend.calls, [])
        self.assertEqual(result.intent, BROWSING)

    def test_unsure_disagreement_does_not_override_the_rules(self) -> None:
        rules_says = LexicalIntentRouter().classify(self.UNCERTAIN)
        backend = StubBackend(
            (BROWSING if rules_says.intent == BUYING else BUYING, 0.55)
        )
        router = CascadingIntentRouter(backend=backend)
        result = router.classify(self.UNCERTAIN)
        self.assertEqual(result.intent, rules_says.intent)
        self.assertEqual(result.tier, "rules_fallback")
        self.assertTrue(result.weak)

    def test_backend_exception_falls_back_to_rules(self) -> None:
        backend = StubBackend(raises=RuntimeError("model exploded"))
        router = CascadingIntentRouter(backend=backend)
        expected = LexicalIntentRouter().classify(self.UNCERTAIN)
        result = router.classify(self.UNCERTAIN)
        self.assertEqual(result.intent, expected.intent)
        self.assertEqual(result.tier, "rules_fallback")
        self.assertEqual(router.backend_failures, 1)

    def test_backend_returning_none_falls_back_to_rules(self) -> None:
        router = CascadingIntentRouter(backend=StubBackend(None))
        self.assertEqual(router.classify(self.UNCERTAIN).tier, "rules_fallback")

    def test_absent_backend_is_pure_rules(self) -> None:
        router = CascadingIntentRouter(backend=None)
        for row in load_jsonl(GOLDEN_PATH):
            result = router.classify(row["message"])
            self.assertEqual(result.tier, "rules")
            self.assertEqual(result.intent, row["intent"])


# ---------------------------------------------------------------------------
# Phase 1: constraint extraction and the tag rule
# ---------------------------------------------------------------------------


class ConstraintExtractionTest(unittest.TestCase):
    def test_produces_the_runtime_record_shape(self) -> None:
        payload = extract_constraints(
            "I need dark blue waterproof hiking boots under $100."
        ).as_dict()
        self.assertEqual(
            set(payload),
            {"category", "brand", "price_min", "price_max", "color", "material",
             "size", "style", "feature", "use_case"},
        )
        self.assertEqual(payload["price_max"], 100)
        self.assertIn("dark blue", payload["color"])
        self.assertIn("waterproof", payload["feature"])

    def test_preserves_exact_dictionary_values(self) -> None:
        constraints = extract_constraints("I want a dark blue coat.")
        self.assertIn("dark blue", constraints.color)

    def test_normalises_price_bounds(self) -> None:
        self.assertEqual(extract_constraints("less than 80 bucks").price_max, 80)
        self.assertEqual(extract_constraints("under $100").price_max, 100)
        self.assertEqual(extract_constraints("at least $30").price_min, 30)
        both = extract_constraints("between $50 and $100")
        self.assertEqual((both.price_min, both.price_max), (50, 100))

    def test_a_size_number_is_not_a_price(self) -> None:
        constraints = extract_constraints("I need a running shoe in size 10.")
        self.assertIsNone(constraints.price_max)
        self.assertIsNone(constraints.price_min)
        self.assertIn("10", constraints.size)

    def test_does_not_invent_constraints(self) -> None:
        constraints = extract_constraints("I'm just having a look around today.")
        for name in CATEGORICAL_FIELDS:
            self.assertEqual(getattr(constraints, name), (), name)
        self.assertFalse(constraints.has_price())

    def test_naming_an_attribute_is_not_committing_to_a_value(self) -> None:
        """"What style?" is a topic. It must not count as a style constraint."""
        constraints = extract_constraints("I'm not sure what style I want yet.")
        self.assertEqual(constraints.style, ())
        self.assertEqual(constraints.tag_count(), 0)
        self.assertIn("style", constraints.unmapped)

    def test_overlapping_aliases_are_counted_once(self) -> None:
        """"dark blue" is one colour, not navy plus blue."""
        self.assertEqual(extract_constraints("a dark blue coat").color, ("dark blue",))

class TwoPhaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = TwoPhaseIntentRouter()

    def test_two_tags_decide_buying_without_consulting_the_ledger(self) -> None:
        result = self.router.classify("I'm looking for a Nike running shoe in size 10.")
        self.assertEqual(result.intent, BUYING)
        self.assertEqual(result.tier, "tags")
        self.assertEqual(result.signals, ())
        self.assertGreaterEqual(len(result.tags), 2)

    def test_one_tag_falls_through_to_the_ledger(self) -> None:
        result = self.router.classify("I'm browsing running shoes and not sure what I want yet.")
        self.assertEqual(result.intent, BROWSING)
        self.assertNotEqual(result.tier, "tags")

    def test_the_threshold_is_configurable(self) -> None:
        message = "I want a black shirt."
        self.assertEqual(TwoPhaseIntentRouter(tag_threshold=1).classify(message).tier, "tags")
        self.assertNotEqual(TwoPhaseIntentRouter(tag_threshold=4).classify(message).tier, "tags")

    def test_undecided_messages_default_to_browsing(self) -> None:
        """The terminal rule: browsing is the recoverable error."""
        result = self.router.classify("Sandals, please.")
        self.assertEqual(result.intent, BROWSING)
        self.assertEqual(result.tier, "default")
        self.assertTrue(result.weak)

    def test_the_default_never_downgrades_a_confident_verdict(self) -> None:
        for message, expected in [
            ("I need black waterproof hiking boots under $100.", BUYING),
            ("I'm just exploring some options for my vacation.", BROWSING),
        ]:
            result = self.router.classify(message)
            self.assertEqual(result.intent, expected, message)
            self.assertNotEqual(result.tier, "default", message)

    def test_no_result_is_ever_below_the_decision_confidence(self) -> None:
        for row in load_jsonl(GOLDEN_PATH) + load_jsonl(DEV_PATH):
            result = self.router.classify(row["message"])
            self.assertGreaterEqual(result.confidence, 0.5)
            if result.intent == BUYING:
                self.assertGreaterEqual(
                    result.confidence, self.router.decision_confidence, row["message"]
                )

    def test_constraints_are_carried_on_the_result(self) -> None:
        result = self.router.classify("I need a leather handbag under $150.")
        self.assertIsNotNone(result.constraints)
        self.assertEqual(result.constraints.price_max, 150)
        self.assertIn("leather", result.constraints.material)


# ---------------------------------------------------------------------------
# Session behaviour
# ---------------------------------------------------------------------------


class SessionFlowTest(unittest.TestCase):
    """Seed on turn one, re-run only when the customer volunteers something."""

    def setUp(self) -> None:
        self.tracker = SessionIntentTracker()

    @staticmethod
    def _simulator_reply(attribute: str, display: str) -> str:
        """Build a reply exactly as the evaluator's simulator would render it."""
        return customer_sentence({"attribute": attribute, "display": display})

    def test_answer_turns_do_not_move_a_browsing_decision(self) -> None:
        session = "s1"
        seed = self.tracker.observe(session, "I'm considering slippers.", turn=1)
        self.assertEqual(seed.intent, BROWSING)

        answers = [
            ("material", "cotton"),
            ("budget", "under $25"),
            ("feature", "machine washable"),
            ("color", "black"),
        ]
        for turn, (attribute, display) in enumerate(answers, start=2):
            message = self._simulator_reply(attribute, display)
            result = self.tracker.observe(session, message, turn, asked_attribute=attribute)
            self.assertEqual(result.intent, BROWSING, message)

    def test_naive_per_round_routing_would_have_converted_that_session(self) -> None:
        """Why the gate exists rather than trusting each message on its own."""
        router = LexicalIntentRouter()
        for attribute, display in [
            ("material", "cotton"),
            ("budget", "under $25"),
            ("feature", "machine washable"),
        ]:
            message = self._simulator_reply(attribute, display)
            self.assertEqual(router.classify(message).intent, BUYING, message)

    def test_no_preference_reply_is_an_answer_not_a_new_intent(self) -> None:
        session = "s2"
        self.tracker.observe(session, "I need waterproof boots under $90.", turn=1)
        result = self.tracker.observe(
            session,
            "I don't really have a preference for color.",
            turn=2,
            asked_attribute="color",
        )
        self.assertEqual(result.intent, BUYING)

    def test_buying_does_not_drift_to_browsing_without_an_explicit_marker(self) -> None:
        session = "s3"
        self.tracker.observe(session, "I need a leather handbag under $150.", turn=1)
        result = self.tracker.observe(
            session, "It should also be nice.", turn=2, asked_attribute=None
        )
        self.assertEqual(result.intent, BUYING)

    def test_explicit_exploratory_turn_can_flip_a_buyer(self) -> None:
        session = "s4"
        self.tracker.observe(session, "I need a leather handbag under $150.", turn=1)
        result = self.tracker.observe(
            session,
            "Actually, forget that. I'm just browsing and not sure what I want.",
            turn=3,
            asked_attribute="material",
        )
        self.assertEqual(result.intent, BROWSING)

    def test_override_message_does_not_change_buying_intent(self) -> None:
        """An override replaces a constraint outside the routing layer."""
        session = "s5"
        self.tracker.observe(session, "I'm looking for socks. I'd prefer nylon.", turn=1)
        result = self.tracker.observe(
            session,
            "Actually, my priority changed. I need solid arch support.",
            turn=4,
            asked_attribute="material",
        )
        self.assertEqual(result.intent, BUYING)

    def test_reset_clears_session_state(self) -> None:
        self.tracker.observe("s6", "I need boots under $50.", turn=1)
        self.assertIsNotNone(self.tracker.current("s6"))
        self.tracker.reset("s6")
        self.assertIsNone(self.tracker.current("s6"))

    def test_sessions_are_isolated(self) -> None:
        self.tracker.observe("a", "I need boots under $50.", turn=1)
        self.tracker.observe("b", "Just browsing, not sure yet.", turn=1)
        self.assertEqual(self.tracker.current("a").intent, BUYING)
        self.assertEqual(self.tracker.current("b").intent, BROWSING)


# ---------------------------------------------------------------------------
# Quality floors
# ---------------------------------------------------------------------------


class DevSetTest(unittest.TestCase):
    def test_pipeline_floor_without_the_model(self) -> None:
        """The pipeline must stay usable if the reranker is never installed."""
        router = TwoPhaseIntentRouter()
        rows = load_jsonl(DEV_PATH)
        correct = sum(1 for r in rows if router.classify(r["message"]).intent == r["intent"])
        self.assertGreaterEqual(correct / len(rows), 0.90, f"{correct}/{len(rows)}")

    def test_golden_set_holds_without_the_model(self) -> None:
        router = TwoPhaseIntentRouter()
        rows = load_jsonl(GOLDEN_PATH)
        correct = sum(1 for r in rows if router.classify(r["message"]).intent == r["intent"])
        self.assertEqual(correct, len(rows))

    def test_every_remaining_error_is_a_browsing_error(self) -> None:
        """The asymmetry the terminal rule buys.

        A message wrongly sent to Browsing still converges through broad
        retrieval and a clarifying question. A message wrongly sent to Buying
        narrows onto constraints nobody gave. The pipeline should make only
        the recoverable mistake.
        """
        router = TwoPhaseIntentRouter()
        wrongly_buying = [
            row["message"]
            for row in load_jsonl(DEV_PATH) + load_jsonl(GOLDEN_PATH)
            if router.classify(row["message"]).intent == BUYING
            and row["intent"] == BROWSING
        ]
        self.assertEqual(wrongly_buying, [], f"unrecoverable errors: {wrongly_buying}")

    def test_misses_are_flagged_weak(self) -> None:
        """A wrong answer must at least not be a confident one.

        This is the property the cascade depends on: everything the rules tier
        gets wrong has to land in the band that escalates.
        """
        router = LexicalIntentRouter()
        confident_misses = [
            row["message"]
            for row in load_jsonl(DEV_PATH) + load_jsonl(GOLDEN_PATH)
            if (r := router.classify(row["message"])).intent != row["intent"] and not r.weak
        ]
        self.assertEqual(confident_misses, [], f"confident misses: {confident_misses}")


class VocabularyOwnershipTest(unittest.TestCase):
    """One source of attribute vocabulary, shared by both components.

    The ledger used to restate the brand / budget / size / colour / material /
    feature / use-case / style value lists that the extractor already owned.
    Two copies could disagree about what "navy" or "water resistant" means;
    these tests fail if a second copy ever reappears.
    """

    LEDGER_SIGNAL = {
        "brand": "brand",
        "color": "color",
        "material": "material",
        "feature": "feature",
        "size": "size",
        "style": "style",
    }

    @staticmethod
    def _signal(name: str):
        return next(spec for spec in lexicon.SIGNALS if spec.name == name)

    def test_ledger_patterns_are_built_from_the_canonical_dictionary(self) -> None:
        """Every generated attribute has a corresponding ledger pattern."""
        dictionary = _load_default_dictionary()
        self.assertIsNotNone(dictionary)
        for field_name, signal_name in self.LEDGER_SIGNAL.items():
            if field_name == "size":
                continue
            source = self._signal(signal_name).pattern.pattern
            values = [
                value.normalized
                for value in dictionary.values
                if value.attribute == field_name
            ]
            self.assertTrue(values, field_name)
            self.assertTrue(any(re.search(source, f"for {value}") for value in values))

    def test_use_case_signal_is_built_from_the_dictionary_too(self) -> None:
        """Anchored on "for", but the values still come from one place."""
        source = self._signal("use_case").pattern.pattern
        dictionary = _load_default_dictionary()
        self.assertIsNotNone(dictionary)
        values = [
            value.normalized
            for value in dictionary.values
            if value.attribute == "use_case"
        ]
        self.assertTrue(any(re.search(source, f"for {value}") for value in values))

    def test_extractor_and_ledger_agree_on_an_alias(self) -> None:
        """A canonical mapping and a ledger signal, from the same string."""
        message = "I want a dark blue waterproof coat."
        constraints = extract_constraints(message)
        self.assertIn("dark blue", constraints.color)
        self.assertIn("waterproof", constraints.feature)
        fired = {s.name for s in LexicalIntentRouter().classify(message).signals}
        self.assertIn("color", fired)
        self.assertIn("feature", fired)

    def test_no_attribute_values_are_hardcoded_in_the_lexicon(self) -> None:
        """A tripwire against pasting a value list back into the ledger."""
        source = (Path(lexicon.__file__)).read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        for value in ("cotton", "polyester", "waterproof", "nike", "casual", "beige"):
            self.assertNotIn(
                value, code.lower(), f"{value!r} is hardcoded in lexicon.py again"
            )

    def test_budget_signal_uses_the_extractor_price_expression(self) -> None:
        pattern = self._signal("budget").pattern
        for phrase in ("under $80", "less than 80 bucks", "between $50 and $100"):
            self.assertTrue(pattern.search(phrase), phrase)


class LexiconHygieneTest(unittest.TestCase):
    def test_every_signal_has_a_polarity_and_a_positive_weight(self) -> None:
        for spec in lexicon.SIGNALS:
            self.assertIn(spec.polarity, (1, -1), spec.name)
            self.assertGreater(spec.weight, 0.0, spec.name)

    def test_signal_names_are_unique(self) -> None:
        names = [spec.name for spec in lexicon.SIGNALS]
        self.assertEqual(len(names), len(set(names)))

    def test_attribute_signal_map_points_at_real_signals(self) -> None:
        names = {spec.name for spec in lexicon.SIGNALS}
        for attribute, signal in lexicon.ATTRIBUTE_SIGNALS.items():
            self.assertIn(signal, names, attribute)

    def test_label_queries_cover_exactly_the_two_intents(self) -> None:
        self.assertEqual(set(LABEL_QUERIES), {BUYING, BROWSING})


# ---------------------------------------------------------------------------
# The real model, when it is installed
# ---------------------------------------------------------------------------


class RerankerTest(unittest.TestCase):
    """Skipped unless `requirements-reranker.txt` and the weights are present."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = QwenRerankerBackend()
        missing = cls.backend.missing_requirements()
        if missing:
            raise unittest.SkipTest(f"reranker unavailable: {missing}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_scores_the_ambiguity_cases_the_rules_tier_cannot(self) -> None:
        cases = [
            ("Show me belts, I'm just getting a feel for the range.", BROWSING),
            ("I might want a raincoat, still weighing it up.", BROWSING),
            ("I'm looking for cardigans.", BUYING),
        ]
        for message, expected in cases:
            scored = self.backend.score(message)
            self.assertIsNotNone(scored, message)
            self.assertEqual(scored[0], expected, message)

    def test_confidence_is_a_probability(self) -> None:
        scored = self.backend.score("I need running shoes.")
        self.assertIsNotNone(scored)
        self.assertGreaterEqual(scored[1], 0.5)
        self.assertLessEqual(scored[1], 0.99)

    def test_empty_message_scores_nothing(self) -> None:
        self.assertIsNone(self.backend.score("   "))

    def test_scoring_is_deterministic(self) -> None:
        message = "I'd like to see a range before I narrow anything down."
        first = self.backend.score(message)
        self.backend._score_cached.cache_clear()
        self.assertEqual(self.backend.score(message), first)

    def test_cascade_beats_rules_alone_on_the_dev_set(self) -> None:
        """The reason to accept the dependency at all."""
        rows = load_jsonl(DEV_PATH) + load_jsonl(GOLDEN_PATH)
        rules = LexicalIntentRouter()
        cascade = CascadingIntentRouter(backend=self.backend)

        rules_correct = sum(
            1 for r in rows if rules.classify(r["message"]).intent == r["intent"]
        )
        cascade_correct = sum(
            1 for r in rows if cascade.classify(r["message"]).intent == r["intent"]
        )
        self.assertGreater(
            cascade_correct,
            rules_correct,
            f"cascade {cascade_correct} vs rules {rules_correct} of {len(rows)}",
        )


if __name__ == "__main__":
    unittest.main()
