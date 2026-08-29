"""Evaluator-facing Agent integrating routing, state, retrieval, and asking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from starter.clarification import ClarificationPolicy
from starter.retrieval import ProductRetriever
from starter.routing import constraints as constraint_module
from starter.routing.constraints import ShoppingConstraints
from starter.routing.intent_router import LexicalIntentRouter, TwoPhaseIntentRouter
from starter.session import (
    OverrideKind,
    SessionManager,
    correction_fields,
    detect_override_kind,
)


class Agent:
    """In-memory conversational shopping agent for the evaluator contract.

    Product artifacts are loaded once at construction. Session intent is sticky
    after the first turn; only an explicit new shopping goal resets it. Every
    response contains the current recommendations, with an optional single
    clarification question selected from the same shared candidate pool.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        facts_path: str | Path | None = None,
        embeddings_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        query_encoder: object | None = None,
        layer2_artifact_dir: str | Path | None = None,
        layer2_weights: Mapping[str, float] | None = None,
        retriever: ProductRetriever | None = None,
        router: object | None = None,
    ) -> None:
        self.retriever = retriever or ProductRetriever(
            catalog_path,
            facts_path=facts_path,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            query_encoder=query_encoder,
            layer2_artifact_dir=layer2_artifact_dir,
            layer2_weights=layer2_weights,
        )
        self.sessions = SessionManager()
        # Keep the old private attribute available to lightweight integrations
        # that inspected the starter, while the manager owns all mutations.
        self._sessions = self.sessions._sessions
        self.router = router or TwoPhaseIntentRouter()
        self._fallback_router = LexicalIntentRouter()
        self.clarification = ClarificationPolicy()

    @staticmethod
    def _extract(message: str) -> ShoppingConstraints:
        """Extract constraints from the required generated dictionary."""

        return constraint_module.extract_constraints(message)

    def _route(self, message: str) -> str:
        try:
            result = self.router.classify(message)
        except Exception:
            result = self._fallback_router.classify(message)
        intent = str(getattr(result, "intent", "BROWSING")).upper()
        return "BUYING" if intent == "BUYING" else "BROWSING"

    def reset(self, session_id: str, user_profile: Mapping[str, Any]) -> None:
        self.sessions.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        state = self.sessions.get(session_id)
        message = user_message or ""
        delta = self._extract(message)
        semantic_delta = getattr(delta, "semantic_constraints", None)
        structured_delta = (
            delta.structured_only()
            if hasattr(delta, "structured_only")
            else delta
        )
        had_messages = bool(state.messages)
        override_kind = OverrideKind.NONE

        if state.mode is not None:
            override_kind = detect_override_kind(message, state.constraints, delta)

        if override_kind is OverrideKind.FULL_GOAL:
            state = self.sessions.reset_goal(session_id)
        elif override_kind is OverrideKind.PREFERENCE:
            state = self.sessions.reset_preference(session_id)
        else:
            self.sessions.promote_last_recommendations(session_id)

        # Keep evaluator/debug tooling informed using only Agent-visible
        # information. This is state metadata, not benchmark knowledge.
        state.last_override_kind = override_kind.value if override_kind is not OverrideKind.NONE else None
        state.last_override_delta = delta if override_kind is not OverrideKind.NONE else None

        if state.mode is None:
            state.mode = self._route(message)

        replacements = correction_fields(
            message,
            state.constraints,
            delta,
            current_semantic=getattr(state, "semantic_constraints", None),
            delta_semantic=semantic_delta,
        )
        if override_kind is OverrideKind.FULL_GOAL:
            source = "initial"
        elif override_kind is OverrideKind.PREFERENCE:
            source = "override"
        else:
            source = "initial" if not had_messages else "clarification"
        self.sessions.update_constraints(
            session_id,
            structured_delta,
            semantic_delta=semantic_delta,
            replace_fields=replacements,
            source=source,
        )
        self.sessions.record_message(session_id, message)
        state.turn = int(turn)

        candidates = self.retriever.retrieve(
            state.mode or "BROWSING",
            state.query_text,
            state.constraints,
            semantic_constraints=getattr(state, "semantic_constraints", None),
            limit=100,
            minimum_candidates=50,
            excluded_asins=state.excluded_recommendations,
        )

        try:
            requested_k = max(0, int(top_k))
        except (TypeError, ValueError):
            requested_k = 0
        valid_asins = self.retriever.valid_asins
        recommendations: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            asin = str(candidate.parent_asin).strip()
            if asin not in valid_asins or asin in seen:
                continue
            seen.add(asin)
            recommendations.append({"parent_asin": asin})
            if len(recommendations) >= requested_k:
                break

        ask_attribute = None
        if int(turn) < 10:
            ask_attribute = self.clarification.choose(
                candidates,
                state.constraints,
                state.asked_attributes,
                mode=state.mode or "BROWSING",
            )
        self.sessions.mark_asked(session_id, ask_attribute)
        self.sessions.set_recommendations(
            session_id,
            (item["parent_asin"] for item in recommendations),
        )

        return {
            "message": self.clarification.question(ask_attribute),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


__all__ = ["Agent"]
