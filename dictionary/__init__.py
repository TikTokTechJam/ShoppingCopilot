"""Canonical shopping-attribute registry and lookup utilities."""

from .registry import (
    ATTRIBUTE_FIELDS,
    DEFAULT_MIN_SIMILARITY,
    SEMANTIC_ATTRIBUTES,
    CanonicalValue,
    LookupMatch,
    AttributeDictionary,
    normalize_text,
)

__all__ = [
    "ATTRIBUTE_FIELDS",
    "DEFAULT_MIN_SIMILARITY",
    "SEMANTIC_ATTRIBUTES",
    "AttributeDictionary",
    "CanonicalValue",
    "LookupMatch",
    "normalize_text",
]
