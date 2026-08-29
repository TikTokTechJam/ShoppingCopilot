"""Evaluator-facing Agent integrating routing, state, retrieval, and asking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from starter.clarification import ClarificationPolicy
from starter.followup import fill_to_top_k
from starter.profile_affinity import ProfileAffinity
from starter.retrieval import ProductRetriever, is_critical_user, normalized_rating
from starter.routing import constraints as constraint_module
from starter.routing.constraints import CATEGORICAL_FIELDS, ShoppingConstraints
from starter.routing.intent_router import LexicalIntentRouter, TwoPhaseIntentRouter
from starter.session import (
    OverrideKind,
    SessionManager,
    correction_fields,
    detect_override_kind,
    is_no_preference_reply,
)



def _user_prior_rating(profile: Mapping[str, Any] | None) -> float | None:
    """Read ``average_prior_rating`` off the session profile, if usable.

    A cold-start shopper has no rating history; ``None`` takes the default
    weight rather than raising.
    """

    if not isinstance(profile, Mapping):
        return None
    try:
        rating = float(profile.get("average_prior_rating"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return rating if rating == rating else None


def _scoping_could_change(
    delta: ShoppingConstraints,
    asked_attribute: str,
) -> bool:
    """Whether re-reading a message scoped to ``asked_attribute`` can differ.

    The unscoped read already resolved every surface it could. Narrowing only
    changes the outcome when it has something to remove or to re-resolve: a
    value that landed on another attribute, or a surface left unmapped because
    the ambiguity between attributes could not be broken. When the reading is
    already entirely within the asked attribute and nothing was left over, the
    scoped pass would return the same constraints, so it is skipped.
    """

    if getattr(delta, "unmapped", ()):
        return True
    return any(
        getattr(delta, field_name, ())
        for field_name in CATEGORICAL_FIELDS
        if field_name != asked_attribute
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
        use_user_profile: bool = True,
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
        self.use_user_profile = bool(use_user_profile)
        self._profile_affinity: dict[str, ProfileAffinity] = {}

    @staticmethod
    def _extract(
        message: str,
        asked_attribute: str | None = None,
    ) -> ShoppingConstraints:
        """Extract constraints from the required generated dictionary."""

        return constraint_module.extract_constraints(
            message,
            asked_attribute=asked_attribute,
        )

    def _route(self, message: str) -> str:
        try:
            result = self.router.classify(message)
        except Exception:
            result = self._fallback_router.classify(message)
        intent = str(getattr(result, "intent", "BROWSING")).upper()
        return "BUYING" if intent == "BUYING" else "BROWSING"

    def reset(self, session_id: str, user_profile: Mapping[str, Any]) -> None:
        self.sessions.reset(session_id, user_profile)
        # preference_tags are fixed for a session, so the prior is built once
        # here and reuses the already-loaded Layer 2 encoder.
        self._profile_affinity.pop(session_id, None)
        if self.use_user_profile:
            self._profile_affinity[session_id] = ProfileAffinity(
                user_profile,
                encoder=getattr(self.retriever, "query_encoder", None),
            )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict[str, object]:
        state = self.sessions.get(session_id)
        message = user_message or ""
        pending_attribute = state.last_asked
        no_preference_reply = is_no_preference_reply(message, pending_attribute)
        # A no-preference answer is clarification metadata, not a product
        # constraint. Skip the extractor so its words cannot become facts.
        delta = (
            ShoppingConstraints()
            if no_preference_reply
            else self._extract(message)
        )
        had_messages = bool(state.messages)
        override_kind = OverrideKind.NONE

        if state.mode is not None:
            override_kind = detect_override_kind(message, state.constraints, delta)

        if (
            not no_preference_reply
            and override_kind is OverrideKind.NONE
            and pending_attribute
            and _scoping_could_change(delta, pending_attribute)
        ):
            # This message is an answer to our own question, so it is read as
            # being about that attribute alone. An override is a change of goal
            # rather than an answer and keeps the full reading above. A
            # no-preference reply produced no delta at all, so there is nothing
            # to narrow.
            delta = self._extract(message, asked_attribute=pending_attribute)

        semantic_delta = getattr(delta, "semantic_constraints", None)
        structured_delta = (
            delta.structured_only()
            if hasattr(delta, "structured_only")
            else delta
        )

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
        self.sessions.record_message(
            session_id,
            message,
            include_in_query=not no_preference_reply,
        )
        state.turn = int(turn)

        # The rating tie-breaker is profile-derived, so the ablation arm must
        # not see it: with no prior rating the weight falls back to the default
        # and no critical-shopper boost applies.
        user_prior_rating = (
            _user_prior_rating(state.profile) if self.use_user_profile else None
        )

        candidates = self.retriever.retrieve(
            state.mode or "BROWSING",
            state.query_text,
            state.constraints,
            semantic_constraints=getattr(state, "semantic_constraints", None),
            limit=100,
            minimum_candidates=50,
            excluded_asins=state.excluded_recommendations,
            user_prior_rating=user_prior_rating,
        )

        try:
            requested_k = max(0, int(top_k))
        except (TypeError, ValueError):
            requested_k = 0
        valid_asins = self.retriever.valid_asins
        ranked = fill_to_top_k(
            (candidate.parent_asin for candidate in candidates),
            (),
            requested_k,
            valid_asins=valid_asins,
        )
        if len(ranked) < requested_k:
            # Invariant 2 of Section 12b.4: returning fewer than top_k while
            # more valid catalog IDs exist is a pure expected-value loss, since
            # padding below the ranked items cannot change their ranks. The
            # backfill is lazy, so a full pool costs nothing.
            ranked = fill_to_top_k(
                ranked,
                self._relaxed_backfill(state, requested_k, user_prior_rating),
                requested_k,
                valid_asins=valid_asins,
            )
        recommendations = [{"parent_asin": asin} for asin in ranked]

        affinity = self._profile_affinity.get(session_id)
        # No turn cutoff is needed here any more: a question asked on the final
        # turn has zero horizon in the Section 12b utility, so the policy
        # abstains on its own arithmetic rather than on a literal.
        ask_attribute = self.clarification.choose(
            candidates,
            state.constraints,
            state.asked_attributes,
            mode=state.mode or "BROWSING",
            profile_factor=affinity.factor if affinity is not None else None,
            turn=state.turn,
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

    def _relaxed_backfill(
        self,
        state: Any,
        limit: int,
        user_prior_rating: float | None = None,
    ) -> list[str]:
        """Candidates for padding a short list, weakest evidence relaxed first.

        Section 10.2 ordering: drop the budget filter and the previously-shown
        exclusions before returning fewer than the requested top_k.
        """
        try:
            relaxed = self.retriever.retrieve(
                state.mode or "BROWSING",
                state.query_text,
                state.constraints,
                limit=max(int(limit) * 4, 50),
                apply_budget=False,
                user_prior_rating=user_prior_rating,
            )
        except Exception:
            return []
        if is_critical_user(user_prior_rating):
            # Task 2: a critical shopper's padding is ordered by rating alone.
            # Python's sort is stable, so equal ratings keep retrieval order.
            relaxed = sorted(
                relaxed,
                key=lambda candidate: -normalized_rating(candidate.rating),
            )
        return [candidate.parent_asin for candidate in relaxed]


__all__ = ["Agent"]
