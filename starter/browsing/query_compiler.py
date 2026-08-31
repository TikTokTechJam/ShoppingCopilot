"""Active-slot query serialization for the Browsing Qwen dense route.

The dense query is compiled from the current active state, not from the
conversation transcript. Brand is the exact structured identity field; the
remaining product attributes come from the semantic slot state. Prices and
sizes remain outside the embedding query.
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
BROWSING_STRUCTURED_QUERY_FIELDS: tuple[str, ...] = ("brand",)
BROWSING_SEMANTIC_QUERY_FIELDS: tuple[str, ...] = (
    "category",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
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


def _semantic_constraints_from_state(
    state_or_constraints: object,
    semantic_constraints: object | None,
) -> object:
    if semantic_constraints is not None:
        return semantic_constraints
    value = getattr(state_or_constraints, "semantic_constraints", None)
    if value is not None:
        return value
    if isinstance(state_or_constraints, Mapping):
        return state_or_constraints.get("semantic_constraints") or {}
    return {}


def build_browsing_query(
    state_or_constraints: object,
    semantic_constraints: object | None = None,
) -> str:
    """Serialize the active structured and semantic slots with field labels.

    The order is deterministic and follows the shared product-card schema.
    Only ``brand`` is read from Layer 1 structured constraints. Category,
    color, material, style, feature, and use-case values are read from the
    separate semantic state. BGE expansions are intentionally not copied into
    this query; Qwen receives the active semantic slots as they were accepted
    by the session state.
    """

    constraints = _constraints_from_state(state_or_constraints)
    semantic = _semantic_constraints_from_state(
        state_or_constraints,
        semantic_constraints,
    )
    lines: list[str] = []
    for field_name in BROWSING_QUERY_FIELDS:
        source = (
            constraints
            if field_name in BROWSING_STRUCTURED_QUERY_FIELDS
            else semantic
        )
        for value in _field_values(source, field_name):
            lines.append(f"{field_name}: {value}")
    return "\n".join(lines)


__all__ = [
    "BROWSING_QUERY_FIELDS",
    "BROWSING_SEMANTIC_QUERY_FIELDS",
    "BROWSING_STRUCTURED_QUERY_FIELDS",
    "build_browsing_query",
]
