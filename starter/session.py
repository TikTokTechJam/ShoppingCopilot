"""Small in-memory conversation state for the shopping agent.

The evaluator creates a fresh session through :meth:`SessionManager.reset` and
then calls the agent repeatedly.  This module deliberately owns no storage
outside the process and keeps intent mode separate from canonical constraint
merging.  Retrieval and clarification can therefore be replaced without
changing the session protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from starter.routing.constraints import CATEGORICAL_FIELDS, ShoppingConstraints


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
_OVERRIDE_MARKER = re.compile(
    r"\b(?:forget|disregard|ignore|start over|new search|different search)\b",
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


def is_intent_override(
    message: str,
    current: ShoppingConstraints,
    delta: ShoppingConstraints,
) -> bool:
    """Recognize an explicit new shopping goal, not an ordinary answer."""

    text = message or ""
    lowered = text.casefold()
    if _OVERRIDE_MARKER.search(text) and (
        re.search(r"\b(?:earlier|previous|old|that|those)\b", lowered)
        or _GOAL_LANGUAGE.search(text)
    ):
        return True
    if re.search(r"\bstart over\b|\bnew search\b", lowered):
        return True

    old_categories = set(_field_values(current, "category"))
    new_categories = set(_field_values(delta, "category"))
    if not new_categories or not _GOAL_LANGUAGE.search(text):
        return False
    if old_categories and new_categories - old_categories and (
        _CORRECTION_MARKER.search(text) or "forget" in lowered
    ):
        return True
    return bool(old_categories and new_categories.isdisjoint(old_categories) and "instead" in lowered)


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
    ) -> ShoppingConstraints:
        state = self.get(session_id)
        state.constraints = merge_constraints(
            state.constraints, delta, replace_fields=replace_fields
        )
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
    "correction_fields",
    "is_intent_override",
    "merge_constraints",
]
