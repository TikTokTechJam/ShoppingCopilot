"""Local-only BGE query encoding for canonical attribute search."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from product_embeddings.pipeline import load_local_sentence_transformer


ATTRIBUTE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
ATTRIBUTE_EMBEDDING_DIMENSION = 384
ATTRIBUTE_EMBEDDING_NORMALIZATION = "l2"
ATTRIBUTE_MODEL_ENV = "SHOPPING_ATTRIBUTE_EMBEDDING_MODEL"
DEFAULT_ATTRIBUTE_MODEL_DIRS = (
    Path("models/bge-small-en-v1.5"),
    Path("model/bge-small-en-v1.5"),
)


def is_bge_small_en_v1_5(model_identifier: object) -> bool:
    normalized = str(model_identifier).strip().casefold()
    normalized = normalized.replace("\\", "/").replace("_", "-")
    return "bge-small-en-v1.5" in normalized


def resolve_attribute_model_path(model_path: str | Path | None = None) -> str:
    """Resolve a local BGE model without permitting a download fallback."""

    configured = str(model_path or os.environ.get(ATTRIBUTE_MODEL_ENV, "")).strip()
    if configured:
        if not is_bge_small_en_v1_5(configured):
            raise RuntimeError(
                f"{ATTRIBUTE_MODEL_ENV} must point to {ATTRIBUTE_EMBEDDING_MODEL}"
            )
        return configured
    for candidate in DEFAULT_ATTRIBUTE_MODEL_DIRS:
        if candidate.is_dir():
            return candidate.as_posix()
    raise RuntimeError(
        "local BGE attribute model is unavailable; set "
        f"{ATTRIBUTE_MODEL_ENV} or place it in models/bge-small-en-v1.5"
    )


def load_bge_attribute_encoder(model_path: str | Path | None = None) -> Any:
    """Load the exact BGE family used by the canonical-value matrices."""

    resolved = resolve_attribute_model_path(model_path)
    encoder = load_local_sentence_transformer(
        resolved,
        task=None,
        document_prompt_name=None,
        query_prompt_name=None,
        trust_remote_code=False,
    )
    actual_model = getattr(encoder, "model_id", resolved)
    if not is_bge_small_en_v1_5(actual_model):
        raise RuntimeError(
            "attribute query encoder does not match the BGE artifact: "
            f"{actual_model} != {ATTRIBUTE_EMBEDDING_MODEL}"
        )
    dimension = getattr(encoder, "embedding_dimension", None)
    if dimension is None or int(dimension) != ATTRIBUTE_EMBEDDING_DIMENSION:
        raise RuntimeError(
            "BGE attribute query encoder dimension mismatch: "
            f"{dimension} != {ATTRIBUTE_EMBEDDING_DIMENSION}"
        )
    if not callable(getattr(encoder, "embed_query", None)):
        raise RuntimeError("BGE attribute query encoder does not expose embed_query()")
    return encoder


__all__ = [
    "ATTRIBUTE_EMBEDDING_DIMENSION",
    "ATTRIBUTE_EMBEDDING_MODEL",
    "ATTRIBUTE_EMBEDDING_NORMALIZATION",
    "ATTRIBUTE_MODEL_ENV",
    "DEFAULT_ATTRIBUTE_MODEL_DIRS",
    "is_bge_small_en_v1_5",
    "load_bge_attribute_encoder",
    "resolve_attribute_model_path",
]
