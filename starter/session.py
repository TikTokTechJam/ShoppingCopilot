"""Small in-memory conversation state for the shopping agent.

The evaluator creates a fresh session through :meth:`SessionManager.reset` and
then calls the agent repeatedly.  This module deliberately owns no storage
outside the process and keeps intent mode separate from canonical constraint
merging.  Retrieval and clarification can therefore be replaced without
changing the session protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping

from starter.clarification import NORMAL_CLARIFICATION_ATTRIBUTES
from starter.routing.constraints import (
    CATEGORICAL_FIELDS,
    SemanticShoppingConstraints,
    ShoppingConstraints,
)
from starter.routing import lexicon


class OverrideKind(str, Enum):
    """Scope of an explicit change to the active shopping request."""

    NONE = "NONE"
    PREFERENCE = "PREFERENCE"


ConstraintRef = tuple[str, str]


@dataclass(frozen=True)
class ConstraintProvenance:
    """Origin and dependency metadata for one stored constraint value."""

    attribute: str
    value: str
    source: str
    parent: ConstraintRef | None = None


# Only inferred descendants are removed when a preference is replaced.  The
# graph is intentionally small and deterministic; independent fields such as
# brand, color, and price never inherit a dependency from another field.
DEPENDENCY_CHILDREN: dict[str, tuple[str, ...]] = {
    "category": ("use_case", "size", "style"),
    "use_case": ("feature", "material", "style"),
}
DEPENDENCY_PARENTS: dict[str, tuple[str, ...]] = {
    "use_case": ("category",),
    "size": ("category",),
    "feature": ("use_case", "category"),
    "material": ("use_case", "category"),
    "style": ("use_case", "category"),
}


def _dependency_descendants(roots: Iterable[str]) -> set[str]:
    descendants: set[str] = set()
    pending = list(roots)
    while pending:
        parent = pending.pop(0)
        for child in DEPENDENCY_CHILDREN.get(parent, ()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _dependency_roots(fields: Iterable[str]) -> tuple[str, ...]:
    selected = {str(field) for field in fields}
    if not selected:
        return ()
    return tuple(
        field_name
        for field_name in (*CATEGORICAL_FIELDS, "price")
        if field_name in selected
        and not any(
            field_name in _dependency_descendants((other,))
            for other in selected
            if other != field_name
        )
    )


def _provenance_record(
    raw: object,
    *,
    attribute: str,
    value: str,
) -> ConstraintProvenance:
    """Read current and legacy provenance representations safely."""

    if isinstance(raw, ConstraintProvenance):
        return raw
    if isinstance(raw, Mapping):
        source = str(raw.get("source", "explicit"))
        parent_value = raw.get("parent")
        parent: ConstraintRef | None = None
        if isinstance(parent_value, (tuple, list)) and len(parent_value) == 2:
            parent = (str(parent_value[0]), str(parent_value[1]))
        return ConstraintProvenance(attribute, value, source, parent)
    # Older sessions stored "initial", "clarification", or "override".
    # Those values were all user-originated, so retain them as explicit facts.
    source = "inferred" if str(raw) == "inferred" else "explicit"
    return ConstraintProvenance(attribute, value, source)


def _evidence_value(item: object) -> str:
    canonical_id = str(getattr(item, "canonical_id", ""))
    if ":" in canonical_id:
        return canonical_id.split(":", 1)[1].replace("_", " ")
    return canonical_id


@dataclass
class SessionState:
    """Mutable state for one evaluator session."""

    session_id: str
    profile: dict[str, Any]
    mode: str | None = None
    constraints: ShoppingConstraints = field(default_factory=ShoppingConstraints)
    semantic_constraints: SemanticShoppingConstraints = field(
        default_factory=SemanticShoppingConstraints
    )
    asked_attributes: set[str] = field(default_factory=set)
    no_preference_attributes: set[str] = field(default_factory=set)
    # Counts only proactive Agent questions in the current clarification
    # cycle.  User-volunteered values never increment these counters.
    attribute_call_count: dict[str, int] = field(
        default_factory=lambda: {
            attribute: 0 for attribute in NORMAL_CLARIFICATION_ATTRIBUTES
        }
    )
    clarification_cycle: int = 1
    clarification_stopped: bool = False
    last_recommendations: tuple[str, ...] = ()
    excluded_recommendations: set[str] = field(default_factory=set)
    last_user_message: str | None = None
    turn: int = 0
    messages: list[str] = field(default_factory=list)
    # Retrieval text is intentionally separate from the transcript.  A
    # preference override keeps the conversation visible for debugging but
    # starts a fresh lexical query context so obsolete goal wording cannot
    # pollute BM25.
    retrieval_messages: list[str] = field(default_factory=list)
    # Extractive summaries returned by the turn interpreter form a separate
    # current-goal BM25 stream.  They are intentionally not exposed as a new
    # debug/UI state field.
    llm_summary_messages: list[str] = field(default_factory=list)
    last_asked: str | None = None
    # Value-level provenance distinguishes explicit user facts from inferred
    # semantic facts and records the optional dependency that produced an
    # inferred value.
    constraint_provenance: dict[str, dict[str, ConstraintProvenance | str]] = field(
        default_factory=dict
    )
    semantic_constraint_provenance: dict[
        str, dict[str, ConstraintProvenance | str]
    ] = field(
        default_factory=dict
    )
    last_override_kind: str | None = None
    last_override_delta: ShoppingConstraints | None = None
    # Per-turn LLM diagnostic payload for the local debug UI. This is not part
    # of retrieval state or the Agent response contract.
    last_llm_return: dict[str, Any] | None = None

    @property
    def query_text(self) -> str:
        """Return the current shopping context in chronological order."""

        return " ".join(message for message in self.messages if message).strip()

    @property
    def retrieval_query_text(self) -> str:
        """Return only lexical text belonging to the active goal segment."""

        return " ".join(
            message for message in self.retrieval_messages if message
        ).strip()

    @property
    def llm_summary_text(self) -> str:
        """Return accumulated interpreter summaries for the active goal."""

        return " ".join(
            summary for summary in self.llm_summary_messages if summary
        ).strip()


def _fresh_attribute_call_count() -> dict[str, int]:
    return {attribute: 0 for attribute in NORMAL_CLARIFICATION_ATTRIBUTES}


def _unique(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _field_values(constraints: ShoppingConstraints, field_name: str) -> tuple[str, ...]:
    value = getattr(constraints, field_name, ())
    if isinstance(value, (list, tuple, set, frozenset)):
        return _unique(value)
    return ()


def _evidence(constraints: object) -> tuple[object, ...]:
    value = getattr(constraints, "evidence", ())
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def merge_constraints(
    current: ShoppingConstraints,
    delta: ShoppingConstraints,
    *,
    replace_fields: Iterable[str] = (),
) -> ShoppingConstraints:
    """Merge one canonical extraction into session constraints.

    Values accumulate by default.  A field in ``replace_fields`` is treated as
    an explicit correction, which lets a later ``"actually brown"`` replace a
    previous ``color=black`` without deleting unrelated constraints.  Numeric
    bounds refine one another unless ``price`` is explicitly replaced.
    """

    replacements = set(replace_fields)
    values: dict[str, object] = {}
    for field_name in CATEGORICAL_FIELDS:
        incoming = _field_values(delta, field_name)
        previous = _field_values(current, field_name)
        if field_name in replacements:
            values[field_name] = incoming
        else:
            values[field_name] = _unique((*previous, *incoming))

    old_min = getattr(current, "price_min", None)
    old_max = getattr(current, "price_max", None)
    new_min = getattr(delta, "price_min", None)
    new_max = getattr(delta, "price_max", None)
    if "price" in replacements:
        values["price_min"] = new_min
        values["price_max"] = new_max
    else:
        values["price_min"] = (
            max(float(old_min), float(new_min))
            if old_min is not None and new_min is not None
            else new_min if new_min is not None else old_min
        )
        values["price_max"] = (
            min(float(old_max), float(new_max))
            if old_max is not None and new_max is not None
            else new_max if new_max is not None else old_max
        )

    values["unmapped"] = _unique(
        (*getattr(current, "unmapped", ()), *getattr(delta, "unmapped", ()))
    )

    evidence = list(_evidence(current))
    if replacements:
        evidence = [
            item
            for item in evidence
            if str(getattr(item, "attribute", "")) not in replacements
        ]
    for item in _evidence(delta):
        if item not in evidence:
            evidence.append(item)

    if evidence:
        try:
            from starter.routing.constraints import CanonicalShoppingConstraints

            return CanonicalShoppingConstraints(evidence=tuple(evidence), **values)
        except (ImportError, TypeError):
            # Keep the starter compatible with a checkout that predates the
            # optional provenance extension.
            pass
    return ShoppingConstraints(**values)


def merge_semantic_constraints(
    current: SemanticShoppingConstraints,
    delta: SemanticShoppingConstraints,
    *,
    replace_fields: Iterable[str] = (),
) -> SemanticShoppingConstraints:
    """Accumulate Layer 2 values without merging them into Layer 1 state."""

    replacements = set(replace_fields)
    values: dict[str, tuple[str, ...]] = {}
    for field_name in (
        "category",
        "color",
        "material",
        "style",
        "feature",
        "use_case",
    ):
        incoming = _field_values(delta, field_name)
        previous = _field_values(current, field_name)
        values[field_name] = (
            _unique(incoming)
            if field_name in replacements
            else _unique((*previous, *incoming))
        )

    evidence = list(_evidence(current))
    if replacements:
        evidence = [
            item
            for item in evidence
            if str(getattr(item, "attribute", "")) not in replacements
        ]
    for item in _evidence(delta):
        if item not in evidence:
            evidence.append(item)
    return SemanticShoppingConstraints(evidence=tuple(evidence), **values)


_CORRECTION_MARKER = re.compile(
    r"\b(?:actually|instead|rather|change|changed|switch(?:ing)?|not)\b",
    re.IGNORECASE,
)
def correction_fields(
    message: str,
    current: ShoppingConstraints,
    delta: ShoppingConstraints,
    *,
    current_semantic: SemanticShoppingConstraints | None = None,
    delta_semantic: SemanticShoppingConstraints | None = None,
) -> tuple[str, ...]:
    """Return populated fields that the user explicitly corrected."""

    if not _CORRECTION_MARKER.search(message or ""):
        return ()
    current_semantic = current_semantic or SemanticShoppingConstraints()
    delta_semantic = delta_semantic or SemanticShoppingConstraints()
    fields: list[str] = []
    for field_name in CATEGORICAL_FIELDS:
        has_current = bool(
            _field_values(current, field_name)
            or _field_values(current_semantic, field_name)
        )
        has_delta = bool(
            _field_values(delta, field_name)
            or _field_values(delta_semantic, field_name)
        )
        if not has_current or not has_delta:
            continue
        semantic_values = _field_values(delta_semantic, field_name)
        if semantic_values:
            semantic_evidence = tuple(
                item
                for item in _evidence(delta_semantic)
                if str(getattr(item, "attribute", "")) == field_name
            )
            # Dense matches participate in override detection only at the
            # same high-confidence level used for replacement decisions.
            if semantic_evidence and not any(
                float(getattr(item, "confidence", 0.0)) >= 0.9
                for item in semantic_evidence
            ):
                continue
        fields.append(field_name)
    if (
        getattr(current, "price_min", None) is not None
        or getattr(current, "price_max", None) is not None
    ) and (
        getattr(delta, "price_min", None) is not None
        or getattr(delta, "price_max", None) is not None
    ):
        fields.append("price")
    return tuple(fields)


def is_no_preference_reply(message: str, asked_attribute: str | None) -> bool:
    """Return whether a pending clarification received no usable preference."""

    return bool(
        asked_attribute
        and lexicon.NO_PREFERENCE_MARKER.search(message or "")
    )


def is_generic_clarification_reply(message: str) -> bool:
    """Return whether a reply is evaluator-generated clarification filler.

    This is deliberately an exact sentence-level check. Generic words such as
    ``you`` and ``one`` remain valid dictionary/product tokens elsewhere.
    """

    return bool(lexicon.GENERIC_CLARIFICATION_REPLY.fullmatch(message or ""))


def detect_override_kind(
    message: str,
    current: ShoppingConstraints,
    delta: ShoppingConstraints,
) -> OverrideKind:
    """Classify an explicit change without conflating it with reset scope."""

    text = message or ""
    if not lexicon.OVERRIDE_MARKER.search(text):
        return OverrideKind.NONE

    # Lexical markers only identify a possible preference change. Override
    # scope is not inferred by this helper from a second hardcoded vocabulary.
    # Marker-only messages are not enough to mutate state. A preference
    # override must carry at least one extracted current-turn fact.
    if delta.populated_fields():
        return OverrideKind.PREFERENCE
    return OverrideKind.NONE


def is_intent_override(
    message: str,
    current: ShoppingConstraints,
    delta: ShoppingConstraints,
) -> bool:
    """Backward-compatible boolean view of :func:`detect_override_kind`."""

    return detect_override_kind(message, current, delta) is not OverrideKind.NONE


class SessionManager:
    """Own all mutable session state for one in-memory Agent instance."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: Mapping[str, Any]) -> SessionState:
        state = SessionState(session_id=session_id, profile=dict(user_profile))
        self._sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise RuntimeError("reset must be called before respond") from exc

    def reset_preference(
        self,
        session_id: str,
        overridden_fields: Iterable[str] = (),
    ) -> SessionState:
        """Remove only the preference branch being replaced.
        """

        state = self.get(session_id)
        roots = _dependency_roots(overridden_fields)
        if not roots:
            return state

        # Keep the human-readable transcript, but discard raw lexical text
        # from the replaced preference segment. The accumulated LLM summary is
        # intentionally preserved across intent/preference overrides, so it
        # remains part of the Buying BM25 context.
        state.retrieval_messages.clear()

        structured_values = {
            field_name: _field_values(state.constraints, field_name)
            for field_name in CATEGORICAL_FIELDS
        }
        semantic_values = {
            field_name: _field_values(state.semantic_constraints, field_name)
            for field_name in (
                "category",
                "color",
                "material",
                "style",
                "feature",
                "use_case",
            )
        }
        if getattr(state.constraints, "price_min", None) is not None:
            structured_values["price"] = ("price_min",)
        if getattr(state.constraints, "price_max", None) is not None:
            structured_values.setdefault("price", ())
            structured_values["price"] = (*structured_values["price"], "price_max")

        root_attributes = set(roots)
        dependent_attributes = _dependency_descendants(roots)
        removed_refs: set[ConstraintRef] = {
            (attribute, value)
            for attribute in root_attributes
            for value in (
                *structured_values.get(attribute, ()),
                *semantic_values.get(attribute, ()),
            )
        }

        def provenance_for(attribute: str, value: str) -> ConstraintProvenance:
            raw = state.constraint_provenance.get(attribute, {}).get(value)
            if raw is None:
                raw = state.semantic_constraint_provenance.get(attribute, {}).get(value)
            return _provenance_record(
                raw,
                attribute=attribute,
                value=value,
            )

        # A missing parent is treated as dependent only for an inferred value
        # in the affected subtree.  Explicit values are always preserved.
        changed = True
        while changed:
            changed = False
            for attribute in CATEGORICAL_FIELDS:
                if attribute not in dependent_attributes:
                    continue
                for value in (
                    *structured_values.get(attribute, ()),
                    *semantic_values.get(attribute, ()),
                ):
                    reference = (attribute, value)
                    if reference in removed_refs:
                        continue
                    provenance = provenance_for(attribute, value)
                    if provenance.source != "inferred":
                        continue
                    parent = provenance.parent
                    if parent is None or parent in removed_refs:
                        removed_refs.add(reference)
                        changed = True

        removed_attributes = {attribute for attribute, _ in removed_refs}

        def keep_evidence(item: object) -> bool:
            attribute = str(getattr(item, "attribute", ""))
            if attribute in root_attributes:
                return False
            if attribute not in dependent_attributes:
                return True
            if str(getattr(item, "layer", "layer1")) != "layer2":
                return True
            return (attribute, _evidence_value(item)) not in removed_refs

        constraint_payload: dict[str, object] = {
            field_name: tuple(
                value
                for value in _field_values(state.constraints, field_name)
                if (field_name, value) not in removed_refs
            )
            for field_name in CATEGORICAL_FIELDS
        }
        constraint_payload["price_min"] = (
            None
            if "price" in removed_attributes
            else getattr(state.constraints, "price_min", None)
        )
        constraint_payload["price_max"] = (
            None
            if "price" in removed_attributes
            else getattr(state.constraints, "price_max", None)
        )
        try:
            state.constraints = replace(
                state.constraints,
                **constraint_payload,
                evidence=tuple(
                    item for item in _evidence(state.constraints) if keep_evidence(item)
                ),
            )
        except TypeError:
            state.constraints = replace(state.constraints, **constraint_payload)

        semantic_payload = {
            field_name: tuple(
                value
                for value in _field_values(state.semantic_constraints, field_name)
                if (field_name, value) not in removed_refs
            )
            for field_name in (
                "category",
                "color",
                "material",
                "style",
                "feature",
                "use_case",
            )
        }
        state.semantic_constraints = replace(
            state.semantic_constraints,
            **semantic_payload,
            evidence=tuple(
                item
                for item in _evidence(state.semantic_constraints)
                if keep_evidence(item)
            ),
        )

        def prune_provenance(
            source_map: dict[str, dict[str, ConstraintProvenance | str]],
        ) -> dict[str, dict[str, ConstraintProvenance | str]]:
            result: dict[str, dict[str, ConstraintProvenance | str]] = {}
            for attribute, values in source_map.items():
                kept = {
                    value: raw
                    for value, raw in values.items()
                    if attribute not in root_attributes
                    and (attribute, value) not in removed_refs
                }
                if kept:
                    result[attribute] = kept
            return result

        state.constraint_provenance = prune_provenance(state.constraint_provenance)
        state.semantic_constraint_provenance = prune_provenance(
            state.semantic_constraint_provenance
        )

        # The goal and its recommendation exclusions remain active.  Only the
        # clarification cursor is restarted so the new preference can be
        # collected without erasing independent state.
        state.asked_attributes.clear()
        state.attribute_call_count = _fresh_attribute_call_count()
        state.clarification_cycle = 1
        state.clarification_stopped = False
        state.last_asked = None
        # The replaced preference branch must not keep old conversational
        # wording in query_text. Active constraints remain above, while the
        # next retrieval query is rebuilt from the new goal context.
        state.messages.clear()
        state.last_user_message = None
        return state

    def reset_clarification_cycle(self, session_id: str) -> SessionState:
        """Start a fresh clarification pass without changing the goal."""

        state = self.get(session_id)
        state.clarification_cycle += 1
        state.attribute_call_count = _fresh_attribute_call_count()
        state.last_asked = None
        state.clarification_stopped = False
        return state

    def record_message(
        self,
        session_id: str,
        message: str,
        *,
        include_in_query: bool = True,
    ) -> None:
        """Record the latest user message, optionally in semantic context."""

        state = self.get(session_id)
        text = (message or "").strip()
        state.last_user_message = text
        if text and include_in_query:
            state.messages.append(text)
            state.retrieval_messages.append(text)

    def record_llm_summary(self, session_id: str, summary: str | None) -> None:
        """Record one non-empty current-turn interpreter summary."""

        state = self.get(session_id)
        text = str(summary or "").strip()
        if text:
            state.llm_summary_messages.append(text)

    def update_constraints(
        self,
        session_id: str,
        delta: ShoppingConstraints,
        *,
        semantic_delta: SemanticShoppingConstraints | None = None,
        replace_fields: Iterable[str] = (),
        source: str | None = None,
    ) -> ShoppingConstraints:
        state = self.get(session_id)
        replacements = set(replace_fields)
        update_source = source or ("initial" if not state.messages else "clarification")
        semantic_delta = semantic_delta or getattr(
            delta, "semantic_constraints", None
        )
        if not isinstance(semantic_delta, SemanticShoppingConstraints):
            semantic_delta = SemanticShoppingConstraints()

        structured_delta = (
            delta.structured_only()
            if hasattr(delta, "structured_only")
            else delta
        )
        for field_name in replacements:
            state.constraint_provenance.pop(field_name, None)
            state.semantic_constraint_provenance.pop(field_name, None)
        state.constraints = merge_constraints(
            state.constraints, structured_delta, replace_fields=replacements
        )
        state.semantic_constraints = merge_semantic_constraints(
            state.semantic_constraints,
            semantic_delta,
            replace_fields=replacements,
        )

        def parent_for(attribute: str) -> ConstraintRef | None:
            for parent_attribute in DEPENDENCY_PARENTS.get(attribute, ()):
                parent_values = _unique(
                    (
                        *_field_values(semantic_delta, parent_attribute),
                        *_field_values(structured_delta, parent_attribute),
                        *_field_values(state.semantic_constraints, parent_attribute),
                        *_field_values(state.constraints, parent_attribute),
                    )
                )
                if parent_values:
                    return (parent_attribute, parent_values[0])
            return None

        def semantic_source(attribute: str, value: str) -> str:
            matches = tuple(
                item
                for item in _evidence(semantic_delta)
                if str(getattr(item, "attribute", "")) == attribute
                and _evidence_value(item).casefold() == value.casefold()
            )
            if not matches:
                return "inferred"
            return (
                "inferred"
                if any(str(getattr(item, "layer", "layer1")) == "layer2" for item in matches)
                else "explicit"
            )

        def store_provenance(
            target: dict[str, dict[str, ConstraintProvenance | str]],
            attribute: str,
            value: str,
            source_kind: str,
            parent: ConstraintRef | None = None,
        ) -> None:
            target.setdefault(attribute, {})[value] = ConstraintProvenance(
                attribute=attribute,
                value=value,
                source=source_kind,
                parent=parent if source_kind == "inferred" else None,
            )

        for field_name in CATEGORICAL_FIELDS:
            incoming = _field_values(structured_delta, field_name)
            if not incoming:
                continue
            for value in incoming:
                store_provenance(
                    state.constraint_provenance,
                    field_name,
                    value,
                    "explicit",
                )
        if getattr(structured_delta, "price_min", None) is not None:
            store_provenance(
                state.constraint_provenance,
                "price",
                "price_min",
                "explicit",
            )
        if getattr(structured_delta, "price_max", None) is not None:
            store_provenance(
                state.constraint_provenance,
                "price",
                "price_max",
                "explicit",
            )
        for field_name in (
            "category",
            "color",
            "material",
            "style",
            "feature",
            "use_case",
        ):
            incoming = _field_values(semantic_delta, field_name)
            if not incoming:
                continue
            for value in incoming:
                source_kind = semantic_source(field_name, value)
                store_provenance(
                    state.semantic_constraint_provenance,
                    field_name,
                    value,
                    source_kind,
                    parent_for(field_name),
                )
        return state.constraints

    def mark_asked(self, session_id: str, attribute: str | None) -> None:
        state = self.get(session_id)
        state.last_asked = attribute
        if attribute:
            state.asked_attributes.add(attribute)
            if attribute in NORMAL_CLARIFICATION_ATTRIBUTES:
                state.attribute_call_count[attribute] = (
                    state.attribute_call_count.get(attribute, 0) + 1
                )

    def set_recommendations(self, session_id: str, asins: Iterable[str]) -> None:
        state = self.get(session_id)
        state.last_recommendations = _unique(asins)

    def promote_last_recommendations(self, session_id: str) -> None:
        """Treat the prior turn's recommendations as misses for this goal."""

        state = self.get(session_id)
        state.excluded_recommendations.update(state.last_recommendations)


__all__ = [
    "SessionManager",
    "SessionState",
    "ConstraintProvenance",
    "OverrideKind",
    "correction_fields",
    "detect_override_kind",
    "is_generic_clarification_reply",
    "is_no_preference_reply",
    "is_intent_override",
    "merge_constraints",
    "merge_semantic_constraints",
]
