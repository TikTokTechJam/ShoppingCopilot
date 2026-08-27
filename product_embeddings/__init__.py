"""Offline product-embedding artifact generation and exact retrieval."""

from .loader import ProductEmbeddingIndex, ProductEmbeddingMatch, load_product_embedding_index
from .pipeline import (
    BUILDER_VERSION,
    HashEmbeddingModel,
    build_product_embeddings,
)
from .text import PRODUCT_TEXT_VERSION, build_product_text
from .tier4 import (
    TIER4_ARTIFACT_VERSION,
    TIER4_FIELDS,
    build_tier4_raw_text,
    build_tier4_record,
    normalize_tier4_source,
)

__all__ = [
    "BUILDER_VERSION",
    "HashEmbeddingModel",
    "PRODUCT_TEXT_VERSION",
    "ProductEmbeddingIndex",
    "ProductEmbeddingMatch",
    "build_product_embeddings",
    "build_product_text",
    "TIER4_ARTIFACT_VERSION",
    "TIER4_FIELDS",
    "build_tier4_raw_text",
    "build_tier4_record",
    "normalize_tier4_source",
    "load_product_embedding_index",
]
