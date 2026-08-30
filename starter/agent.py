"""Evaluator-facing Agent integrating routing, state, retrieval, and asking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from starter.clarification import (
    ClarificationPolicy,
    NORMAL_CLARIFICATION_ATTRIBUTES,
)
from starter.evolution import (
    PHASE_A_CONFIG,
    CrossSessionStore,
    EvolutionConfig,
    EvolutionLoop,
    constraint_pairs,
)
from starter.followup import MAX_TURNS, fill_to_top_k
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
    is_generic_clarification_reply,
    is_no_preference_reply,
)


CLARIFICATION_CANDIDATE_LIMIT = 500


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


def _belief_lost_weight(
    before: Mapping[str, Mapping[str, float]],
    after: Mapping[str, Mapping[str, float]],
) -> bool:
    """Whether a still-present per-value factor dropped between two snapshots.

    Used only to flag an implicit-negative decay event for telemetry. A value
    that simply left the belief (correction / no longer a live constraint) is
    not a decay.
    """

    for field_name, values in after.items():
        prior = before.get(field_name, {})
        for value, weight in values.items():
            if weight < prior.get(value, 1.0) - 1e-9:
                return True
    return False


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
        enable_evolution: bool = True,
        evolution_config: EvolutionConfig | None = None,
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
        # Runtime feedback loop. None == the pre-loop code path (byte-identical).
        self._evolution_config = evolution_config or PHASE_A_CONFIG
        self.evolution = (
            EvolutionLoop(self._evolution_config) if enable_evolution else None
        )
        # Cross-session learned priors (Stage 3). Survives reset(); a no-op
        # unless the config enables it.
        self._evo_store = (
            CrossSessionStore(self._evolution_config) if enable_evolution else None
        )
        self._evo_last_state: object | None = None
        self._evo_last_session: str | None = None
        # Cumulative across the whole run; never reset between sessions.
        self.last_diagnostics: dict[str, object] = {}

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
        # LEARN (Stage 3): a new session id means the previous one is over, so
        # fold its surrogate signal into the cross-session priors before its
        # state is discarded. A no-op unless the config enables it.
        self._finalize_evolution_session(session_id)
        self.sessions.reset(session_id, user_profile)
        # preference_tags are fixed for a session, so the prior is built once
        # here and reuses the already-loaded Layer 2 encoder.
        self._profile_affinity.pop(session_id, None)
        if self.use_user_profile:
            self._profile_affinity[session_id] = ProfileAffinity(
                user_profile,
                encoder=getattr(self.retriever, "query_encoder", None),
            )

    def telemetry(self) -> dict[str, object]:
        """Snapshot of the run-level feedback-loop diagnostics."""

        diagnostics = dict(self.last_diagnostics)
        if self._evo_store is not None:
            priors = self._evo_store.snapshot()
            if priors:
                diagnostics["evolution.learned_priors"] = priors
        return diagnostics

    def evolution_priors(self) -> dict[str, float]:
        """Current cross-session learned per-field starting factors."""

        return self._evo_store.snapshot() if self._evo_store is not None else {}

    def _finalize_evolution_session(self, next_session_id: str | None) -> int:
        """Fold the just-finished session into the LEARN priors. Returns updates."""

        store = getattr(self, "_evo_store", None)
        last_state = getattr(self, "_evo_last_state", None)
        if (
            store is None
            or last_state is None
            or getattr(self, "_evo_last_session", None) == next_session_id
        ):
            return 0
        state = last_state
        updates = store.observe_session_end(
            belief_weights=getattr(state, "belief_weights", {}) or {},
            trace=getattr(state, "evolution_trace", ()) or (),
        )
        self._evo_last_state = None
        self._evo_last_session = None
        return updates

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
        pending_other = pending_attribute == "other"
        no_preference_reply = is_no_preference_reply(message, pending_attribute)
        generic_clarification_reply = is_generic_clarification_reply(message)
        skip_constraint_extraction = (
            no_preference_reply or generic_clarification_reply
        )
        # No-preference answers and evaluator clarification filler are
        # conversation metadata, not product constraints. Skip the extractor
        # so words such as "you" and "one" cannot become accidental facts.
        delta = (
            ShoppingConstraints()
            if skip_constraint_extraction
            else self._extract(message)
        )
        had_messages = bool(state.messages)
        override_kind = OverrideKind.NONE

        if state.mode is not None:
            override_kind = detect_override_kind(message, state.constraints, delta)

        if (
            no_preference_reply
            and pending_attribute in NORMAL_CLARIFICATION_ATTRIBUTES
        ):
            state.no_preference_attributes.add(pending_attribute)

        if (
            not skip_constraint_extraction
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
        has_new_information = bool(delta.populated_fields())
        other_cycle_has_information = pending_other and has_new_information

        # Compute the replacement scope against the pre-override state.  The
        # old flow reset preference state first, which made it impossible to
        # identify the dependency branch that needed pruning.
        replacements = correction_fields(
            message,
            state.constraints,
            delta,
            current_semantic=getattr(state, "semantic_constraints", None),
            delta_semantic=semantic_delta,
        )

        if override_kind is OverrideKind.FULL_GOAL:
            state = self.sessions.reset_goal(session_id)
        elif override_kind is OverrideKind.PREFERENCE:
            state = self.sessions.reset_preference(
                session_id,
                overridden_fields=replacements,
            )
        else:
            self.sessions.promote_last_recommendations(session_id)

        # Keep evaluator/debug tooling informed using only Agent-visible
        # information. This is state metadata, not benchmark knowledge.
        state.last_override_kind = override_kind.value if override_kind is not OverrideKind.NONE else None
        state.last_override_delta = delta if override_kind is not OverrideKind.NONE else None

        if state.mode is None:
            state.mode = self._route(message)

        if override_kind is OverrideKind.FULL_GOAL:
            source = "initial"
        elif override_kind is OverrideKind.PREFERENCE:
            source = "override"
        else:
            source = "initial" if not had_messages else "clarification"

        # OBSERVE (part 1): the constraint set held going into this turn's merge,
        # captured after any override reset so an override turn reads as fresh.
        evolution_pre_pairs: set[tuple[str, str]] = set()
        if self.evolution is not None:
            evolution_pre_pairs = constraint_pairs(state.constraints)

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
            include_in_query=not skip_constraint_extraction,
        )
        state.turn = int(turn)

        # ``other`` is a boundary between clarification cycles.  It is not a
        # field answer and therefore does not consume an ordinary attribute
        # slot.  A useful answer starts a new cycle after the current delta has
        # been applied; a non-answer stops clarification instead of reopening
        # the same questions indefinitely.
        if other_cycle_has_information and override_kind is not OverrideKind.FULL_GOAL:
            state = self.sessions.reset_clarification_cycle(session_id)
        other_cycle_stopped = (
            pending_other
            and not other_cycle_has_information
            and override_kind is OverrideKind.NONE
        )
        if other_cycle_stopped:
            state.clarification_stopped = True
        # The rating tie-breaker is profile-derived, so the ablation arm must
        # not see it: with no prior rating the weight falls back to the default
        # and no critical-shopper boost applies.
        user_prior_rating = (
            _user_prior_rating(state.profile) if self.use_user_profile else None
        )

        # OBSERVE (part 2) + DISTILL + RE-PLAN + ACT. `field_weights` and
        # `score_weights` stay None -- the byte-identical retrieval path --
        # until the belief has moved or a non-neutral strategy is chosen.
        field_weights = None
        score_weights = None
        evolution_obs = None
        evolution_strategy = "neutral"
        evolution_decayed = False
        if self.evolution is not None:
            cfg = self._evolution_config
            evolution_obs = self.evolution.observe(
                turn=state.turn,
                trace=state.evolution_trace,
                structured_delta=structured_delta,
                prev_pairs=evolution_pre_pairs,
                new_pairs=constraint_pairs(state.constraints) - evolution_pre_pairs,
                no_preference=skip_constraint_extraction,
                override_kind=(
                    override_kind.value
                    if override_kind is not OverrideKind.NONE
                    else None
                ),
            )
            fact_lookup = (
                self._evolution_fact_lookup
                if cfg.enable_implicit_negative
                else None
            )
            prior_factor = (
                self._evo_store.prior_factor
                if (self._evo_store is not None and cfg.enable_learn)
                else None
            )
            before_belief = state.belief_weights
            new_belief = self.evolution.distill(
                before_belief,
                evolution_obs,
                constraints=state.constraints,
                provenance=state.constraint_provenance,
                replacements=replacements,
                trace=state.evolution_trace,
                fact_lookup=fact_lookup,
                prior_factor=prior_factor,
            )
            evolution_decayed = fact_lookup is not None and _belief_lost_weight(
                before_belief, new_belief
            )
            self.sessions.set_belief_weights(session_id, new_belief)
            field_weights = self.evolution.act_field_weights(
                new_belief, state.constraints
            )
            constraint_fields = tuple(
                f for f in CATEGORICAL_FIELDS if getattr(state.constraints, f, ())
            )
            field_weights, score_weights, evolution_strategy = self.evolution.plan(
                evolution_obs,
                state.evolution_trace,
                constraint_fields,
                field_weights,
            )

        candidates = self.retriever.retrieve(
            state.mode or "BROWSING",
            state.query_text,
            state.constraints,
            semantic_constraints=getattr(state, "semantic_constraints", None),
            limit=CLARIFICATION_CANDIDATE_LIMIT,
            minimum_candidates=50,
            excluded_asins=state.excluded_recommendations,
            field_weights=field_weights,
            user_prior_rating=user_prior_rating,
            score_weights=score_weights,
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

        # OBSERVE (finalize): record this turn's pool/churn signal and roll the
        # run-level telemetry forward.
        if self.evolution is not None and evolution_obs is not None:
            self.sessions.record_evolution_turn(
                session_id,
                self.evolution.finalize_turn(
                    evolution_obs,
                    pool_size=len(candidates),
                    ranked=ranked,
                    trace=state.evolution_trace,
                ),
            )
            # LEARN (Stage 3): remember this session so reset() -- or the final
            # turn below -- can fold its surrogate signal into the priors.
            self._evo_last_state = state
            self._evo_last_session = session_id
            learn_updates = 0
            if state.turn >= MAX_TURNS:
                learn_updates = self._finalize_evolution_session(None)
            self.last_diagnostics = self.evolution.telemetry_snapshot(
                self.last_diagnostics,
                evolution_obs,
                state,
                reweighted=(field_weights is not None or score_weights is not None),
                strategy=evolution_strategy,
                decayed=evolution_decayed,
                learn_updates=learn_updates,
            )

        affinity = self._profile_affinity.get(session_id)
        # No turn cutoff is needed here any more: a question asked on the final
        # turn has zero horizon in the Section 12b utility, so the policy
        # abstains on its own arithmetic rather than on a literal.
        if state.clarification_stopped:
            ask_attribute = None
        else:
            # Counts are the source of truth for the current cycle. The
            # lifetime asked_attributes set remains available for legacy
            # consumers/debugging, but it must not block a field after a
            # legitimate cycle reset.
            clarification_asked = {
                field_name
                for field_name in NORMAL_CLARIFICATION_ATTRIBUTES
                if state.attribute_call_count.get(field_name, 0) > 0
            }
            clarification_asked.update(state.no_preference_attributes)
            ask_attribute = self.clarification.choose(
                candidates,
                state.constraints,
                clarification_asked,
                mode=state.mode or "BROWSING",
                profile_factor=affinity.factor if affinity is not None else None,
                turn=state.turn,
            )
            if ask_attribute is None:
                # No useful normal field remains in this cycle. ``other`` is
                # the boundary marker and may be used again after a useful
                # answer starts a new cycle. If its next answer is empty, the
                # stopped flag above prevents another question.
                ask_attribute = "other"
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

    def _evolution_fact_lookup(self, asin: str, field_name: str) -> tuple[str, ...]:
        """Facts of a shown candidate, for the implicit-negative decay check."""

        record = self.retriever.product_by_asin.get(asin)
        if record is None:
            return ()
        return tuple(record.facts.get(field_name, ()))

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
