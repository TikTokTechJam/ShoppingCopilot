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


class SentenceTransformerRetrievalEncoder:
    """Adapt a local SentenceTransformer to the document/query boundary.

    Jina's retrieval models use different prompt names for catalog documents
    and user queries.  Keeping that distinction here makes the offline
    artifact builder and the runtime Agent share exactly the same encoding
    contract.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        task: str | None = None,
        document_prompt_name: str | None = None,
        query_prompt_name: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model = model
        self.model_id = model_id
        dimension_getter = getattr(model, "get_sentence_embedding_dimension", None)
        try:
            dimension = dimension_getter() if callable(dimension_getter) else None
        except (TypeError, ValueError):
            dimension = None
        self.embedding_dimension = (
            int(dimension)
            if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
            else None
        )
        self.task = task
        self.document_prompt_name = document_prompt_name
        self.query_prompt_name = query_prompt_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.supports_full_document_batch = True

    def _encode(self, texts: Sequence[str], *, prompt_name: str | None) -> Any:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": self.show_progress_bar,
            "convert_to_numpy": True,
            "normalize_embeddings": False,
        }
        if self.task is not None:
            kwargs["task"] = self.task
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        return self._model.encode(list(texts), **kwargs)

    def embed_documents(self, texts: Sequence[str]) -> Any:
        return self._encode(texts, prompt_name=self.document_prompt_name)

    def embed_query(self, text: str) -> Any:
        return self._encode([text], prompt_name=self.query_prompt_name)


class HashEmbeddingModel:
    """Dependency-light deterministic fallback for pipeline smoke generation.

    This is a reproducible lexical sketch, not a quality semantic model. Pass a
    local model or injected embedder for the benchmark artifact.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.embedding_dimension = dimension
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


def is_jina_v5_text_nano(model_path: str) -> bool:
    """Return whether a model path identifies the requested Jina model."""
    normalized = re.sub(r"[^a-z0-9]+", "-", model_path.casefold())
    return (
        "jina-embeddings-v5-text-nano" in normalized
        or "jina-v5-text-nano" in normalized
    )


def embedding_models_compatible(expected: object, actual: object) -> bool:
    """Return whether two model identifiers refer to the same model family."""

    expected_id = str(expected).strip().casefold()
    actual_id = str(actual).strip().casefold()
    if not expected_id or not actual_id:
        return False
    if expected_id == actual_id:
        return True
    # A cached local directory and the Hugging Face identifier can be different
    # strings while still referring to the same Jina model.
    return is_jina_v5_text_nano(expected_id) and is_jina_v5_text_nano(actual_id)


def load_local_sentence_transformer(
    model_path: str,
    *,
    task: str | None = None,
    document_prompt_name: str | None = None,
    query_prompt_name: str | None = None,
    trust_remote_code: bool = False,
    batch_size: int = 32,
    device: str | None = None,
    half_precision: bool = False,
    show_progress_bar: bool = False,
) -> Any:
    """Load a cached/local SentenceTransformer without permitting downloads."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "--model requires the optional sentence-transformers dependency; "
            "inject a local embedder or use the hashing fallback instead"
        ) from exc
    try:
        model = SentenceTransformer(
            model_path,
            device=device,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except TypeError as exc:  # pragma: no cover - old optional dependency
        raise RuntimeError(
            "the installed sentence-transformers version does not support "
            "local_files_only; refusing to load a potentially network-backed model"
        ) from exc
    if half_precision:
        model.half()
    return SentenceTransformerRetrievalEncoder(
        model,
        model_id=model_path,
        task=task,
        document_prompt_name=document_prompt_name,
        query_prompt_name=query_prompt_name,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )


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
