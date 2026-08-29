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
from .semantic import (
    ATTRIBUTE_EMBEDDING_DIMENSION,
    ATTRIBUTE_EMBEDDING_MODEL,
    ATTRIBUTE_MODEL_ENV,
    load_bge_attribute_encoder,
)

__all__ = [
    "ATTRIBUTE_FIELDS",
    "DEFAULT_MIN_SIMILARITY",
    "SEMANTIC_ATTRIBUTES",
    "AttributeDictionary",
    "CanonicalValue",
    "LookupMatch",
    "normalize_text",
    "ATTRIBUTE_EMBEDDING_DIMENSION",
    "ATTRIBUTE_EMBEDDING_MODEL",
    "ATTRIBUTE_MODEL_ENV",
    "load_bge_attribute_encoder",
]
