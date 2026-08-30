"""Active-slot query serialization for the Browsing Qwen dense route.

Only the current structured session state is serialized. Conversation history,
stale messages, prices, and sizes are intentionally excluded from the dense
semantic query; numeric/structured eligibility remains a separate concern.
"""

from __future__ import annotations

from collections.abc import Mapping


BROWSING_QUERY_FIELDS: tuple[str, ...] = (
    "category",
    "brand",
    "color",
    "material",
    "feature",
    "use_case",
    "style",
)


def _field_values(source: object, field_name: str) -> tuple[str, ...]:
    value = getattr(source, field_name, None)
    if value is None and isinstance(source, Mapping):
        value = source.get(field_name)
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _constraints_from_state(state_or_constraints: object) -> object:
    constraints = getattr(state_or_constraints, "constraints", None)
    if constraints is not None:
        return constraints
    if isinstance(state_or_constraints, Mapping) and "constraints" in state_or_constraints:
        return state_or_constraints.get("constraints") or {}
    return state_or_constraints


def build_browsing_query(state_or_constraints: object) -> str:
    """Serialize active slot values with their field labels.

    The order is deterministic and follows the shared shopping schema. Values
    are not semantically expanded here; Qwen receives the active slot state,
    while BGE expansion remains owned by the sparse BM25 route.
    """

    constraints = _constraints_from_state(state_or_constraints)
    lines: list[str] = []
    for field_name in BROWSING_QUERY_FIELDS:
        for value in _field_values(constraints, field_name):
            lines.append(f"{field_name}: {value}")
    return "\n".join(lines)


__all__ = ["BROWSING_QUERY_FIELDS", "build_browsing_query"]
