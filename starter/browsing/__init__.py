"""Browsing-specific dense retrieval building blocks."""

from .dense_retriever import (
    BROWSING_QWEN_INSTRUCTION,
    BrowsingDenseIndex,
    BrowsingDenseMatch,
    format_qwen_query,
    load_qwen_browsing_encoder,
)
from .product_cards import (
    BROWSING_CARD_FIELDS,
    build_product_cards,
    serialize_product_card,
)
from .query_compiler import (
    BROWSING_QUERY_FIELDS,
    BROWSING_SEMANTIC_QUERY_FIELDS,
    BROWSING_STRUCTURED_QUERY_FIELDS,
    build_browsing_query,
)

__all__ = [
    "BROWSING_CARD_FIELDS",
    "BROWSING_QUERY_FIELDS",
    "BROWSING_SEMANTIC_QUERY_FIELDS",
    "BROWSING_STRUCTURED_QUERY_FIELDS",
    "BROWSING_QWEN_INSTRUCTION",
    "BrowsingDenseIndex",
    "BrowsingDenseMatch",
    "build_browsing_query",
    "build_product_cards",
    "format_qwen_query",
    "load_qwen_browsing_encoder",
    "serialize_product_card",
]
