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

from starter.routing.constraints import CATEGORICAL_FIELDS, ShoppingConstraints
from starter.routing import lexicon


class OverrideKind(str, Enum):
    """Scope of an explicit change to the active shopping request."""

    NONE = "NONE"
    FULL_GOAL = "FULL_GOAL"
    PREFERENCE = "PREFERENCE"


@dataclass
class SessionState:
    """Mutable state for one evaluator session."""

    session_id: str
    profile: dict[str, Any]
    mode: str | None = None
    constraints: ShoppingConstraints = field(default_factory=ShoppingConstraints)
    asked_attributes: set[str] = field(default_factory=set)
    last_recommendations: tuple[str, ...] = ()
    excluded_recommendations: set[str] = field(default_factory=set)
    last_user_message: str | None = None
    turn: int = 0
    messages: list[str] = field(default_factory=list)
    last_asked: str | None = None
    # Value-level provenance is deliberately small: it only distinguishes
    # facts from the initial request, later clarifications, and an override.
    # This is enough to remove an obsolete initial preference without creating
    # a second session-state system.
    constraint_provenance: dict[str, dict[str, str]] = field(default_factory=dict)
    last_override_kind: str | None = None
    last_override_delta: ShoppingConstraints | None = None

    @property
    def query_text(self) -> str:
        """Return the current shopping context in chronological order."""

        return " ".join(message for message in self.messages if message).strip()


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


def _evidence(constraints: ShoppingConstraints) -> tuple[object, ...]:
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
        values["price_min"] = new_min if new_min is not None else old_min
        values["price_max"] = new_max if new_max is not None else old_max
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


_CORRECTION_MARKER = re.compile(
    r"\b(?:actually|instead|rather|change|changed|switch(?:ing)?|not)\b",
    re.IGNORECASE,
)
_GOAL_LANGUAGE = re.compile(
    r"\b(?:need|want|looking for|search(?:ing)? for|show me|find me|shopping for)\b",
    re.IGNORECASE,
)


def correction_fields(
    message: str,
    current: ShoppingConstraints,
    delta: ShoppingConstraints,
) -> tuple[str, ...]:
    """Return populated fields that the user explicitly corrected."""

    if not _CORRECTION_MARKER.search(message or ""):
        return ()
    fields = [
        field_name
        for field_name in CATEGORICAL_FIELDS
        if _field_values(current, field_name) and _field_values(delta, field_name)
    ]
    if (
        getattr(current, "price_min", None) is not None
        or getattr(current, "price_max", None) is not None
    ) and (
        getattr(delta, "price_min", None) is not None
        or getattr(delta, "price_max", None) is not None
    ):
        fields.append("price")
    return tuple(fields)


def detect_override_kind(
    message: str,
    current: ShoppingConstraints,
    delta: ShoppingConstraints,
) -> OverrideKind:
    """Classify an explicit change without conflating it with reset scope."""

    text = message or ""
    lowered = text.casefold()
    if not lexicon.OVERRIDE_MARKER.search(text):
        return OverrideKind.NONE

    old_categories = set(_field_values(current, "category"))
    new_categories = set(_field_values(delta, "category"))
    category_replacement = bool(
        new_categories
        and _GOAL_LANGUAGE.search(text)
        and (not old_categories or new_categories.isdisjoint(old_categories))
        and _CORRECTION_MARKER.search(text)
    )

    if lexicon.FULL_GOAL_OVERRIDE_MARKER.search(text) or category_replacement:
        return OverrideKind.FULL_GOAL

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

    def reset_goal(self, session_id: str) -> SessionState:
        """Clear stale shopping state while retaining the session profile."""

        state = self.get(session_id)
        state.mode = None
        state.constraints = ShoppingConstraints()
        state.asked_attributes.clear()
        state.last_recommendations = ()
        state.excluded_recommendations.clear()
        state.last_user_message = None
        state.turn = 0
        state.messages.clear()
        state.last_asked = None
        state.constraint_provenance.clear()
        state.last_override_kind = None
        state.last_override_delta = None
        return state

    def reset_preference(self, session_id: str) -> SessionState:
        """Replace stale preference state while retaining the active goal.

        Category, mode, profile, and an explicit budget remain active. Values
        recorded on the initial turn are treated as the obsolete priority;
        facts learned during later clarification turns survive unless the
        current override explicitly replaces their field.
        """

        state = self.get(session_id)
        kept_values: dict[str, tuple[str, ...]] = {}
        kept_provenance: dict[str, dict[str, str]] = {}
        for field_name in CATEGORICAL_FIELDS:
            values = _field_values(state.constraints, field_name)
            origins = state.constraint_provenance.get(field_name, {})
            if field_name == "category":
                kept = values
            else:
                kept = tuple(
                    value
                    for value in values
                    # Unknown provenance is preserved conservatively. The
                    # Agent records provenance for all normal updates, while
                    # this protects callers that construct state directly.
                    if origins.get(value) != "initial"
                )
            kept_values[field_name] = kept
            if kept:
                kept_provenance[field_name] = {
                    value: origins[value]
                    for value in kept
                    if value in origins
                }

        payload: dict[str, object] = {
            **kept_values,
            "price_min": getattr(state.constraints, "price_min", None),
            "price_max": getattr(state.constraints, "price_max", None),
            # Unresolved text belongs to the old semantic goal and must not be
            # carried into the new query context.
            "unmapped": (),
        }
        try:
            state.constraints = replace(state.constraints, **payload, evidence=())
        except TypeError:
            state.constraints = replace(state.constraints, **payload)

        state.constraint_provenance = kept_provenance
        state.asked_attributes.clear()
        state.last_recommendations = ()
        state.excluded_recommendations.clear()
        state.last_user_message = None
        state.messages.clear()
        state.last_asked = None
        return state

    def record_message(self, session_id: str, message: str) -> None:
        state = self.get(session_id)
        text = (message or "").strip()
        state.last_user_message = text
        if text:
            state.messages.append(text)

    def update_constraints(
        self,
        session_id: str,
        delta: ShoppingConstraints,
        *,
        replace_fields: Iterable[str] = (),
        source: str | None = None,
    ) -> ShoppingConstraints:
        state = self.get(session_id)
        replacements = set(replace_fields)
        update_source = source or ("initial" if not state.messages else "clarification")
        for field_name in replacements:
            state.constraint_provenance.pop(field_name, None)
        state.constraints = merge_constraints(
            state.constraints, delta, replace_fields=replacements
        )
        for field_name in CATEGORICAL_FIELDS:
            incoming = _field_values(delta, field_name)
            if not incoming:
                continue
            provenance = state.constraint_provenance.setdefault(field_name, {})
            for value in incoming:
                provenance[value] = update_source
        return state.constraints

    def mark_asked(self, session_id: str, attribute: str | None) -> None:
        state = self.get(session_id)
        state.last_asked = attribute
        if attribute:
            state.asked_attributes.add(attribute)

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
    "OverrideKind",
    "correction_fields",
    "detect_override_kind",
    "is_intent_override",
    "merge_constraints",
]
