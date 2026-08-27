from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingModel(Protocol):
    """Small injection boundary for local embedding implementations."""

    def embed_documents(self, texts: Sequence[str]) -> Any:
        ...


class HashEmbeddingModel:
    """Dependency-light deterministic fallback for pipeline smoke generation.

    This is a reproducible lexical sketch, not a quality semantic model. Pass a
    local model or injected embedder for the benchmark artifact.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_id = "hashing-fallback-v1"

    def embed_documents(self, texts: Sequence[str]) -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "NumPy is required to write .npy embedding artifacts"
            ) from exc

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.blake2b(
                    token.encode("utf-8"), digest_size=16
                ).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vectors[row, index] += sign
            # Distinct token hashes can cancel in a small smoke-test space.
            # Keep every non-empty document representable without changing the
            # deterministic lexical nature of this fallback.
            if text.strip() and not np.any(vectors[row]):
                digest = hashlib.blake2b(
                    text.casefold().encode("utf-8"), digest_size=16
                ).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                vectors[row, index] = 1.0
        return vectors


def load_local_sentence_transformer(model_path: str) -> Any:
    """Load a cached/local SentenceTransformer without permitting downloads."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "--model requires the optional sentence-transformers dependency; "
            "inject a local embedder or use the hashing fallback instead"
        ) from exc
    try:
        return SentenceTransformer(model_path, local_files_only=True)
    except TypeError as exc:  # pragma: no cover - old optional dependency
        raise RuntimeError(
            "the installed sentence-transformers version does not support "
            "local_files_only; refusing to load a potentially network-backed model"
        ) from exc


def load_injected_embedder(spec: str) -> Any:
    """Load ``module:attribute`` as an embedder or zero-argument factory."""
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder must use the module:attribute form")
    target = getattr(importlib.import_module(module_name), attribute_name)
    if any(hasattr(target, name) for name in ("embed_documents", "encode", "embed")):
        return target
    if callable(target):
        instance = target()
        if any(hasattr(instance, name) for name in ("embed_documents", "encode", "embed")):
            return instance
    raise TypeError(
        "injected embedder must be an embedder object or zero-argument factory"
    )
