"""Evaluator-facing Agent integrating routing, state, retrieval, and asking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from starter.clarification import (
    ClarificationPolicy,
    NORMAL_CLARIFICATION_ATTRIBUTES,
)
from starter.followup import fill_to_top_k
from starter.profile_affinity import ProfileAffinity
from starter.retrieval import ProductRetriever, is_critical_user, normalized_rating
from starter.routing import constraints as constraint_module
from starter.routing.constraints import (
    CATEGORICAL_FIELDS,
    CanonicalShoppingConstraints,
    SemanticShoppingConstraints,
    ShoppingConstraints,
)
from starter.routing.intent_router import (
    LexicalIntentRouter,
    SessionIntentTracker,
    TwoPhaseIntentRouter,
)
from starter.session import (
    OverrideKind,
    SessionManager,
    correction_fields,
    detect_override_kind,
    is_generic_clarification_reply,
    is_no_preference_reply,
    merge_constraints,
    merge_semantic_constraints,
)
from starter.turn_interpreter import (
    TurnInterpretation,
    build_turn_interpreter,
    parse_turn_interpretation,
)


# Keep a broader pool for facet distributions than the final recommendation
# list.  This changes only clarification evidence; retrieval and ranking still
# return the same production candidate type and downstream top_k.
CLARIFICATION_CANDIDATE_LIMIT = 1000


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

    Product artifacts are loaded once at construction. Session intent is
    incremental with conservative hysteresis: unsolicited turns may move a
    session when the existing tracker has sufficient evidence, while answers
    to clarification questions do not cause random flips. Every response
    contains current recommendations and an optional clarification question.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        facts_path: str | Path | None = None,
        embeddings_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        query_encoder: object | None = None,
        product_query_encoder: object | None = None,
        use_user_profile: bool = True,
        layer2_artifact_dir: str | Path | None = None,
        layer2_weights: Mapping[str, float] | None = None,
        product_embedding_artifact_dir: str | Path | None = None,
        retriever: ProductRetriever | None = None,
        router: object | None = None,
        turn_interpreter: object | None = None,
    ) -> None:
        self.retriever = retriever or ProductRetriever(
            catalog_path,
            facts_path=facts_path,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            query_encoder=query_encoder,
            product_query_encoder=product_query_encoder,
            layer2_artifact_dir=layer2_artifact_dir,
            layer2_weights=layer2_weights,
            product_embedding_artifact_dir=product_embedding_artifact_dir,
        )
        self.sessions = SessionManager()
        # Keep the old private attribute available to lightweight integrations
        # that inspected the starter, while the manager owns all mutations.
        self._sessions = self.sessions._sessions
        self.router = router or TwoPhaseIntentRouter()
        self.intent_tracker = SessionIntentTracker(router=self.router)
        self._fallback_router = LexicalIntentRouter()
        self.clarification = ClarificationPolicy()
        self.use_user_profile = bool(use_user_profile)
        self._profile_affinity: dict[str, ProfileAffinity] = {}
        # Optional local model, loaded once per Agent.  If it is absent or
        # fails to load, the existing deterministic extraction remains active.
        self.turn_interpreter = (
            turn_interpreter
            if turn_interpreter is not None
            else build_turn_interpreter()
        )

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

    @staticmethod
    def _extract_without_semantics(
        message: str,
        asked_attribute: str | None = None,
    ) -> ShoppingConstraints:
        """Run deterministic parsing without a second BGE semantic pass."""

        return constraint_module.extract_constraints(
            message,
            asked_attribute=asked_attribute,
            semantic_matcher=lambda _phrase: (),
        )

    @staticmethod
    def _merge_extraction_deltas(
        first: ShoppingConstraints,
        second: ShoppingConstraints,
    ) -> ShoppingConstraints:
        """Merge deterministic and schema-guided facts without losing Layer 2."""

        merged = merge_constraints(first, second)
        semantic = merge_semantic_constraints(
            getattr(first, "semantic_constraints", SemanticShoppingConstraints()),
            getattr(second, "semantic_constraints", SemanticShoppingConstraints()),
        )
        if not isinstance(merged, CanonicalShoppingConstraints):
            return merged
        return CanonicalShoppingConstraints(
            category=merged.category,
            brand=merged.brand,
            price_min=merged.price_min,
            price_max=merged.price_max,
            color=merged.color,
            material=merged.material,
            size=merged.size,
            style=merged.style,
            feature=merged.feature,
            use_case=merged.use_case,
            unmapped=merged.unmapped,
            evidence=merged.evidence,
            semantic_constraints=semantic,
        )

    def _interpret(self, message: str, state: object) -> TurnInterpretation | None:
        interpreter = self.turn_interpreter
        if interpreter is None:
            return None
        try:
            result = interpreter.interpret(message, state)
        except Exception as exc:
            print(
                "[turn_interpreter] turn failed: "
                f"{type(exc).__name__}: {exc}; using deterministic fallback",
                flush=True,
            )
            return None
        if isinstance(result, TurnInterpretation):
            return result
        if isinstance(result, (Mapping, str)):
            return parse_turn_interpretation(result)
        return None

    def _constraints_from_interpretation(
        self,
        interpretation: TurnInterpretation,
    ) -> ShoppingConstraints:
        """Validate LLM categorical slots through the existing dictionary."""

        delta: ShoppingConstraints = ShoppingConstraints()
        for field_name, values in (interpretation.updates or {}).items():
            # Price and size remain deterministic fields.  The interpreter can
            # describe them, but it cannot bypass the existing typed parser.
            if field_name in {"price_min", "price_max", "size"}:
                continue
            if field_name not in CATEGORICAL_FIELDS:
                continue
            for value in values:
                try:
                    value_delta = self._extract(
                        value,
                        asked_attribute=field_name,
                    )
                except Exception:
                    continue
                delta = self._merge_extraction_deltas(delta, value_delta)
        return delta

    def _extract_interpreted_turn(
        self,
        message: str,
        interpretation: TurnInterpretation,
    ) -> ShoppingConstraints:
        parsed = self._extract_without_semantics(message)
        # Once the schema-guided interpreter is active, categorical extraction
        # comes from its current-turn delta.  The old exact dictionary would
        # otherwise still turn dialogue framing such as "exploring" into the
        # canonical use_case value ``exploring``.  Numeric price and size stay
        # with the deterministic parser as required by the slot schema.
        deterministic = ShoppingConstraints(
            price_min=parsed.price_min,
            price_max=parsed.price_max,
            size=parsed.size,
        )
        interpreted = self._constraints_from_interpretation(interpretation)
        return self._merge_extraction_deltas(deterministic, interpreted)

    def _route(
        self,
        message: str,
        *,
        interpreted_intent: str | None = None,
        extracted_constraints: ShoppingConstraints | None = None,
    ) -> str:
        if interpreted_intent:
            intent = str(interpreted_intent).upper()
            if intent in {"BUYING", "BROWSING"}:
                return intent
        try:
            if (
                extracted_constraints is not None
                and isinstance(self.router, TwoPhaseIntentRouter)
            ):
                result = self.router.classify(
                    message,
                    extracted_constraints=extracted_constraints,
                )
            else:
                result = self.router.classify(message)
        except Exception:
            result = self._fallback_router.classify(message)
        intent = str(getattr(result, "intent", "BROWSING")).upper()
        return "BUYING" if intent == "BUYING" else "BROWSING"

    def reset(self, session_id: str, user_profile: Mapping[str, Any]) -> None:
        self.sessions.reset(session_id, user_profile)
        self.intent_tracker.reset(session_id)
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
        pending_other = pending_attribute == "other"
        no_preference_reply = is_no_preference_reply(message, pending_attribute)
        generic_clarification_reply = is_generic_clarification_reply(message)
        skip_constraint_extraction = (
            no_preference_reply or generic_clarification_reply
        )
        interpretation = (
            None
            if skip_constraint_extraction
            else self._interpret(message, state)
        )
        # No-preference answers and evaluator clarification filler are
        # conversation metadata, not product constraints. Skip the extractor
        # so words such as "you" and "one" cannot become accidental facts.
        delta = (
            ShoppingConstraints()
            if skip_constraint_extraction
            else (
                self._extract_interpreted_turn(message, interpretation)
                if interpretation is not None
                else self._extract(message)
            )
        )
        had_messages = bool(state.messages)
        override_kind = OverrideKind.NONE

        if state.mode is not None:
            override_kind = detect_override_kind(message, state.constraints, delta)
            if (
                override_kind is OverrideKind.NONE
                and interpretation is not None
                and interpretation.override_kind == OverrideKind.PREFERENCE.value
                and delta.populated_fields()
            ):
                # Natural preference-override wording may not yet be covered
                # by the lexical marker set.  Full-goal resets stay guarded by
                # the existing explicit reset markers.
                override_kind = OverrideKind.PREFERENCE

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
            if interpretation is None:
                delta = self._extract(message, asked_attribute=pending_attribute)
            else:
                # The schema-guided interpreter is intentionally allowed to
                # preserve several volunteered fields in one answer.
                delta = self._extract_interpreted_turn(message, interpretation)

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
        if override_kind is OverrideKind.PREFERENCE and interpretation is not None:
            replacement_values = list(replacements)
            for field_name in interpretation.override_fields:
                if field_name not in replacement_values:
                    replacement_values.append(field_name)
            if not replacement_values:
                for field_name in delta.populated_fields():
                    if field_name not in replacement_values:
                        replacement_values.append(field_name)
            replacements = tuple(replacement_values)

        if override_kind is OverrideKind.FULL_GOAL:
            state = self.sessions.reset_goal(session_id)
            self.intent_tracker.reset(session_id)
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
            state.mode = self._route(
                message,
                interpreted_intent=(
                    interpretation.intent if interpretation is not None else None
                ),
                extracted_constraints=delta,
            )
            self.intent_tracker.seed(session_id, state.mode, turn=int(turn))
        else:
            # Re-evaluate only through the existing tracker. It ignores normal
            # replies to our clarification questions and uses its explicit
            # margin/signal hysteresis for genuine unsolicited direction shifts.
            try:
                routed = self.intent_tracker.observe(
                    session_id,
                    message,
                    int(turn),
                    asked_attribute=pending_attribute,
                    extracted_constraints=delta,
                )
            except Exception:
                routed = None
            intent = str(getattr(routed, "intent", "")).upper()
            if intent in {"BUYING", "BROWSING"}:
                state.mode = intent

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

        candidates = self.retriever.retrieve(
            state.mode or "BROWSING",
            state.retrieval_query_text,
            state.constraints,
            semantic_constraints=getattr(state, "semantic_constraints", None),
            limit=CLARIFICATION_CANDIDATE_LIMIT,
            minimum_candidates=50,
            excluded_asins=state.excluded_recommendations,
            user_prior_rating=user_prior_rating,
        )
        # Analyze the already-ranked pool once.  The policy uses these facet
        # distributions for utility, and the selected question can reuse them
        # to show the most common values from the same pool.
        candidate_stats = self.clarification.analyze(candidates)

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
                candidate_stats=candidate_stats,
            )
            if (
                ask_attribute is None
                and self.clarification.is_broad_pool(candidate_stats)
            ):
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
            "message": self.clarification.question(ask_attribute, candidate_stats),
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

        Backfill remains subject to the active budget and recommendation
        exclusions so padding cannot violate explicit user constraints or
        re-show products already rejected by the session.
        """
        try:
            relaxed = self.retriever.retrieve(
                state.mode or "BROWSING",
                state.retrieval_query_text,
                state.constraints,
                limit=max(int(limit) * 4, 50),
                apply_budget=True,
                excluded_asins=state.excluded_recommendations,
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
