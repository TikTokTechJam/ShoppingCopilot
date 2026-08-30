"""Offline product-embedding artifact generation and exact retrieval."""

from .loader import ProductEmbeddingIndex, ProductEmbeddingMatch, load_product_embedding_index
from .layer2 import (
    DEFAULT_LAYER2_WEIGHTS,
    LAYER2_ARTIFACT_VERSION,
    LAYER2_FILES,
    LAYER2_TEXT_VERSION,
    LAYER2_VIEWS,
    Layer2EmbeddingIndex,
    Layer2EmbeddingMatch,
    build_layer2_embeddings,
    build_layer2_view_documents,
    build_layer2_view_text,
    load_layer2_embedding_index,
)
from .pipeline import (
    HashEmbeddingModel,
)
from .v5 import (
    V5_PRODUCT_ARTIFACT_VERSION,
    V5_PRODUCT_CARDS_FILE,
    V5_PRODUCT_EMBEDDING_FILE,
    V5_PRODUCT_FACT_FIELDS,
    V5_PRODUCT_METADATA_FILE,
    V5_PRODUCT_MODEL,
    V5_PRODUCT_TEXT_VERSION,
    V5ProductEmbeddingIndex,
    V5ProductEmbeddingMatch,
    build_v5_product_card,
    build_v5_product_embeddings,
    load_v5_product_embedding_index,
)

__all__ = [
    "DEFAULT_LAYER2_WEIGHTS",
    "HashEmbeddingModel",
    "LAYER2_ARTIFACT_VERSION",
    "LAYER2_FILES",
    "LAYER2_TEXT_VERSION",
    "LAYER2_VIEWS",
    "Layer2EmbeddingIndex",
    "Layer2EmbeddingMatch",
    "ProductEmbeddingIndex",
    "ProductEmbeddingMatch",
    "build_layer2_embeddings",
    "build_layer2_view_documents",
    "build_layer2_view_text",
    "V5_PRODUCT_ARTIFACT_VERSION",
    "V5_PRODUCT_CARDS_FILE",
    "V5_PRODUCT_EMBEDDING_FILE",
    "V5_PRODUCT_FACT_FIELDS",
    "V5_PRODUCT_METADATA_FILE",
    "V5_PRODUCT_MODEL",
    "V5_PRODUCT_TEXT_VERSION",
    "V5ProductEmbeddingIndex",
    "V5ProductEmbeddingMatch",
    "build_v5_product_card",
    "build_v5_product_embeddings",
    "load_product_embedding_index",
    "load_layer2_embedding_index",
    "load_v5_product_embedding_index",
]
