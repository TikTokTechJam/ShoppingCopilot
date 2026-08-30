"""Buying vs Browsing intent routing (issue #6).

The router answers exactly one question: should the current user message be
treated as a targeted purchase or as open-ended exploration? It knows nothing
about retrieval, ranking, conversation state, or clarification policy.

Three pieces live here:

`LexicalIntentRouter`
    A pure function over one message. Scores the signal ledger in
    `lexicon.py`, turns the signed margin into a confidence, and returns an
    auditable `IntentResult`. Standard library only, microseconds per call.

`CascadingIntentRouter`
    Runs the lexical tier first and escalates only low-confidence messages to
    a local reranker (`local_model.QwenRerankerBackend`). Confidence therefore
    does real work: it decides whether the cheap path is trusted. If the model
    is unavailable for any reason, the lexical answer stands.

`SessionIntentTracker`
    Seeds a decision on the first turn and re-runs the router only on
    *unsolicited* turns. From turn two the customer is normally answering the
    attribute the agent just asked about, so the message is a constraint by
    construction and reads as BUYING whatever the session really is --
    `customer_sentence()` in `evaluator/hard_evaluator.py` renders every reply
    that way. Classifying every round would convert Browsing sessions by turn
    two; classifying only the first utterance would go deaf at the turn-3/4
    intent override. The tracker does neither.

The tracker stores its own decision and nothing else. Shopping-constraint
Constraint state belongs to the constraint/session layer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from starter.routing import lexicon
from starter.routing.constraints import ShoppingConstraints, extract_constraints
from starter.routing.lexicon import BROWSING, BUYING


Intent = Literal["BUYING", "BROWSING"]

# Which implementation produced a result. Recorded so a regression can be
# attributed to a tier rather than to "the router".
Tier = Literal["tags", "rules", "reranker", "rules_fallback", "default"]


@dataclass(frozen=True)
class Signal:
    """One piece of evidence that fired, kept for auditing."""

    name: str
    polarity: int
    weight: float
    evidence: str
    span: tuple[int, int]


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    margin: float = 0.0
    signals: tuple[Signal, ...] = ()
    weak: bool = False
    tier: Tier = "rules"
    tags: tuple[str, ...] = ()
    constraints: ShoppingConstraints | None = None

    def as_dict(self) -> dict[str, object]:
        """The shape issue #6 asks for, plus the audit trail."""
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "weak": self.weak,
            "tier": self.tier,
            "tags": list(self.tags),
            "signals": [signal.name for signal in self.signals],
        }


class IntentRouter(Protocol):
    def classify(self, message: str) -> IntentResult: ...


def _confidence_from_margin(margin: float) -> tuple[Intent, float]:
    probability = 1.0 / (1.0 + math.exp(-lexicon.LOGISTIC_K * margin))
    intent: Intent = BUYING if probability >= 0.5 else BROWSING
    # Report the probability of the label actually returned, so confidence is
    # always >= 0.5 and never claims certainty it has not earned.
    confidence = min(max(max(probability, 1.0 - probability), 0.5), 0.99)
    return intent, confidence


class LexicalIntentRouter:
    """Deterministic, dependency-free, offline. The floor every tier falls back to."""

    tier: Tier = "rules"

    def classify(self, message: str) -> IntentResult:
        text = message or ""

        if self.is_contentless(text):
            # A greeting is not a decided buyer. Falling through to the BUYING
            # bias here would point precision retrieval at nothing at all.
            return IntentResult(
                intent=BROWSING,
                confidence=lexicon.NO_CONTENT_CONFIDENCE,
                margin=0.0,
                signals=(),
                weak=True,
                tier=self.tier,
            )

        matches = self._match_signals(text)
        has_hard = any(spec.hard for spec, _ in matches)
        # A conditional signal must not satisfy its own condition: "show me"
        # is not evidence that the customer said what they want.
        has_buying = any(
            spec.polarity > 0 and not spec.requires_buying_evidence
            for spec, _ in matches
        )

        fired: list[Signal] = []
        buying_weight = 0.0
        browsing_weight = 0.0

        for spec, match in matches:
            if spec.suppressed_by_hard_evidence and has_hard:
                # "something", "nice", "yet" stop meaning hesitation once the
                # customer has named a price, a colour or a material.
                continue
            if spec.requires_buying_evidence and not has_buying:
                continue

            fired.append(
                Signal(
                    name=spec.name,
                    polarity=spec.polarity,
                    weight=spec.weight,
                    evidence=match.group(0),
                    span=match.span(),
                )
            )
            if spec.polarity > 0:
                buying_weight += spec.weight
            else:
                browsing_weight += spec.weight

        margin = (
            lexicon.BIAS
            + lexicon.BUYING_SCALE * buying_weight
            - lexicon.BROWSING_SCALE * browsing_weight
        )
        intent, confidence = _confidence_from_margin(margin)

        return IntentResult(
            intent=intent,
            confidence=confidence,
            margin=margin,
            signals=tuple(fired),
            weak=confidence < lexicon.WEAK_CONFIDENCE,
            tier=self.tier,
        )

    @staticmethod
    def is_contentless(text: str) -> bool:
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        if not tokens:
            return True
        return all(
            token.replace("'", "") in lexicon.CONVERSATIONAL_FILLER for token in tokens
        )

    @staticmethod
    def _match_signals(text: str):
        found = []
        for spec in lexicon.SIGNALS:
            match = lexicon.match_signal(spec, text)
            if match is not None:
                # A signal counts once however often it appears; three colours
                # are not three times the evidence of one.
                found.append((spec, match))
        return found


class CascadingIntentRouter:
    """Rules first; the reranker only when the rules tier does not trust itself.

    The escalation band is narrow and the tracker below fires the router once
    or twice per session, so a model costing tens of milliseconds is
    affordable where one costing that on every turn would not be.
    """

    def __init__(
        self,
        backend: "object | None" = None,
        *,
        threshold: float | None = None,
        rules: LexicalIntentRouter | None = None,
    ) -> None:
        self.rules = rules or LexicalIntentRouter()
        self.backend = backend
        self.threshold = (
            lexicon.WEAK_CONFIDENCE if threshold is None else float(threshold)
        )
        self.escalations = 0
        self.backend_failures = 0

    def classify(self, message: str) -> IntentResult:
        base = self.rules.classify(message)
        if self.backend is None or base.confidence >= self.threshold:
            return base
        if self.rules.is_contentless(message):
            # There is nothing in a greeting for a model to read. Escalating
            # it only invites a confident answer about no evidence.
            return base

        self.escalations += 1
        try:
            scored = self.backend.score(message)
        except Exception:
            # An optional model must never take the router down with it.
            self.backend_failures += 1
            return IntentResult(
                intent=base.intent,
                confidence=base.confidence,
                margin=base.margin,
                signals=base.signals,
                weak=True,
                tier="rules_fallback",
            )

        if scored is None:
            return IntentResult(
                intent=base.intent,
                confidence=base.confidence,
                margin=base.margin,
                signals=base.signals,
                weak=True,
                tier="rules_fallback",
            )

        intent, confidence = scored
        if (
            intent != base.intent
            and confidence < lexicon.RERANKER_ACCEPT_CONFIDENCE
        ):
            # An unsure disagreement is not enough to discard a deterministic
            # answer. Keep the rules verdict, flagged weak either way.
            return IntentResult(
                intent=base.intent,
                confidence=base.confidence,
                margin=base.margin,
                signals=base.signals,
                weak=True,
                tier="rules_fallback",
            )

        return IntentResult(
            intent=intent,
            confidence=confidence,
            margin=base.margin,
            signals=base.signals,
            weak=confidence < lexicon.WEAK_CONFIDENCE,
            tier="reranker",
        )


@dataclass
class _SessionState:
    result: IntentResult
    decided_turn: int
    evaluations: int = 1


@dataclass
class SessionIntentTracker:
    """Incremental session intent with conservative, explicit hysteresis."""

    router: IntentRouter = field(default_factory=lambda: TwoPhaseIntentRouter())
    _sessions: dict[str, _SessionState] = field(default_factory=dict, init=False)

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def seed(self, session_id: str, intent: str, turn: int = 1) -> IntentResult:
        """Install an already-decided first-turn intent without reclassifying.

        The Agent may obtain the initial intent from the schema-guided turn
        interpreter.  Seeding lets later unsolicited turns use this same
        tracker while avoiding a second first-turn classifier call.
        """

        normalized: Intent = BUYING if str(intent).upper() == BUYING else BROWSING
        result = IntentResult(
            intent=normalized,
            confidence=0.5,
            margin=0.0,
            weak=True,
            tier="default",
        )
        self._sessions[session_id] = _SessionState(
            result=result,
            decided_turn=int(turn),
        )
        return result

    def current(self, session_id: str) -> IntentResult | None:
        state = self._sessions.get(session_id)
        return None if state is None else state.result

    def observe(
        self,
        session_id: str,
        message: str,
        turn: int,
        asked_attribute: str | None = None,
        extracted_constraints: ShoppingConstraints | None = None,
    ) -> IntentResult:
        """Update and return the session intent.

        `asked_attribute` is what the agent asked on the *previous* turn.
        `Agent.respond()` already has it, so threading it through costs
        nothing. None means the agent asked nothing, which makes any
        substantive reply unprompted.
        """
        state = self._sessions.get(session_id)

        if state is None:
            result = self._classify(message, extracted_constraints)
            self._sessions[session_id] = _SessionState(result=result, decided_turn=turn)
            return result

        if not self._is_unsolicited(message, asked_attribute):
            return state.result

        state.evaluations += 1
        candidate = self._classify(message, extracted_constraints)
        held = state.result

        if candidate.intent == held.intent:
            state.result = candidate
            state.decided_turn = turn
            return state.result

        if not self._may_flip(held, candidate):
            return held

        state.result = candidate
        state.decided_turn = turn
        return state.result

    def _classify(
        self,
        message: str,
        extracted_constraints: ShoppingConstraints | None,
    ) -> IntentResult:
        """Classify without repeating the Agent's current-turn extraction."""

        if (
            extracted_constraints is not None
            and isinstance(self.router, TwoPhaseIntentRouter)
        ):
            return self.router.classify(
                message,
                extracted_constraints=extracted_constraints,
            )
        return self.router.classify(message)

    @staticmethod
    def _is_unsolicited(message: str, asked_attribute: str | None) -> bool:
        text = message or ""

        if lexicon.OVERRIDE_MARKER.search(text):
            return True
        if lexicon.NO_PREFERENCE_MARKER.search(text) or lexicon.FILLER_MARKER.search(text):
            # Declining to answer is still an answer, not a change of direction.
            return False

        signal_name = lexicon.ATTRIBUTE_SIGNALS.get(asked_attribute or "")
        if signal_name is None:
            # Nothing was asked, so anything substantive is volunteered.
            return True

        for spec in lexicon.SIGNALS:
            if spec.name == signal_name and spec.pattern.search(text):
                return False
        return True

    @staticmethod
    def _may_flip(held: IntentResult, candidate: IntentResult) -> bool:
        if abs(candidate.margin) < lexicon.FLIP_MARGIN:
            return False
        if held.intent == BUYING and candidate.intent == BROWSING:
            # A buyer answering questions does not drift back into browsing by
            # accident. Demand that they actually say so.
            return any(
                signal.name in ("undecided", "explore_verb", "option_seeking")
                for signal in candidate.signals
            )
        return True


class TwoPhaseIntentRouter:
    """Constraint tags first, signal ledger second, BROWSING if still unsure.

    Phase 1 asks a factual question: how many distinct canonical constraint
    fields did the customer actually fill in? A message carrying two or more
    -- a colour and a price, a brand and a size -- has committed to enough
    that no further reading is needed. The constraint extractor does the
    work, and it is the cheapest evidence available -- but not an unarguable
    one. The count is only as good as the extractor's precision, so Phase 1 is
    vetoed when the ledger reads the same message as confidently exploratory.

    Phase 2 handles everything Phase 1 cannot settle, which is every message
    whose intent lives in *how* it is phrased rather than in what it names.
    That is the signal ledger, optionally refined by the reranker on the
    messages the ledger itself is unsure about.

    The terminal rule is deliberate asymmetry: anything still undecided is
    routed BROWSING. The two errors are not symmetric. A Browsing session that
    was really a buyer still converges -- broad retrieval plus a clarifying
    question recovers it within a turn or two. A Buying session that was
    really a browser narrows hard onto constraints the customer never gave and
    has no natural way back.
    """

    def __init__(
        self,
        *,
        backend: "object | None" = None,
        tag_threshold: int | None = None,
        decision_confidence: float | None = None,
        rules: LexicalIntentRouter | None = None,
        escalation_threshold: float | None = None,
    ) -> None:
        self.rules = rules or LexicalIntentRouter()
        self.tag_threshold = (
            lexicon.BUYING_TAG_THRESHOLD if tag_threshold is None else int(tag_threshold)
        )
        self.decision_confidence = (
            lexicon.DECISION_CONFIDENCE
            if decision_confidence is None
            else float(decision_confidence)
        )
        self.cascade = CascadingIntentRouter(
            backend=backend, threshold=escalation_threshold, rules=self.rules
        )
        self.phase1_decisions = 0
        self.phase1_vetoed = 0
        self.defaulted = 0

    @property
    def escalations(self) -> int:
        return self.cascade.escalations

    @property
    def backend_failures(self) -> int:
        return self.cascade.backend_failures

    def _browsing_veto(self, message: str) -> bool:
        """Whether the ledger reads this message as confidently exploratory.

        Deliberately the pure lexical tier, not the cascade: a veto must stay
        a cheap pure-function check and must never escalate to the reranker.
        """
        result = self.rules.classify(message)
        return (
            result.intent == BROWSING
            and result.confidence >= lexicon.BROWSING_VETO_CONFIDENCE
        )

    def classify(
        self,
        message: str,
        *,
        extracted_constraints: ShoppingConstraints | None = None,
    ) -> IntentResult:
        # Agent.respond() may already have extracted this turn.  Reuse that
        # delta so routing does not duplicate dictionary/semantic extraction.
        constraints = (
            extracted_constraints
            if extracted_constraints is not None
            else extract_constraints(message)
        )
        tags = constraints.populated_fields(exclude=lexicon.TAG_COUNT_EXCLUDE)

        # -- Phase 1: enough named constraints is decisive on its own --------
        if len(tags) >= self.tag_threshold:
            if self._browsing_veto(message):
                self.phase1_vetoed += 1
            else:
                self.phase1_decisions += 1
                # Confidence grows with the evidence, saturating quickly: three
                # constraints is not meaningfully surer than two.
                confidence = min(0.99, 0.80 + 0.06 * (len(tags) - self.tag_threshold))
                return IntentResult(
                    intent=BUYING,
                    confidence=confidence,
                    margin=float(len(tags)),
                    signals=(),
                    weak=False,
                    tier="tags",
                    tags=tags,
                    constraints=constraints,
                )

        # -- Phase 2: fall back to the signal ledger (and the reranker) ------
        result = self.cascade.classify(message)

        # -- Terminal rule: undecided means BROWSING -------------------------
        if result.confidence < self.decision_confidence and result.intent != BROWSING:
            self.defaulted += 1
            return IntentResult(
                intent=BROWSING,
                confidence=self.decision_confidence,
                margin=result.margin,
                signals=result.signals,
                weak=True,
                tier="default",
                tags=tags,
                constraints=constraints,
            )

        return IntentResult(
            intent=result.intent,
            confidence=result.confidence,
            margin=result.margin,
            signals=result.signals,
            weak=result.weak,
            tier=result.tier,
            tags=tags,
            constraints=constraints,
        )


_DEFAULT_ROUTER = LexicalIntentRouter()


def classify(message: str) -> IntentResult:
    """Convenience entry point for one-off rules-tier classification."""
    return _DEFAULT_ROUTER.classify(message)
