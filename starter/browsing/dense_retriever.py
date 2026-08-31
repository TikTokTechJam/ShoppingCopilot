"""Qwen3 dense product-card index for the Browsing retrieval path."""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from product_embeddings.pipeline import (
    embedding_models_compatible,
    load_local_sentence_transformer,
)


BROWSING_QWEN_INSTRUCTION = (
    "Given an open-ended shopping request, retrieve products that best match "
    "the shopper's use case, preferences, and desired attributes."
)


def format_qwen_query(query_text: str, *, instruction: str = BROWSING_QWEN_INSTRUCTION) -> str:
    """Format a query with an instruction while leaving documents unprefixed."""

    cleaned = "\n".join(line.strip() for line in str(query_text).splitlines() if line.strip())
    if not cleaned:
        return ""
    return f"Instruct: {instruction}\nQuery: {cleaned}"


def load_qwen_browsing_encoder(
    model_path: str,
    *,
    batch_size: int = 32,
    device: str | None = None,
    half_precision: bool = False,
    show_progress_bar: bool = False,
) -> Any:
    """Load Qwen locally with plain document encoding and query-only instruction.

    Documents receive no instruction or query prefix. The shared adapter adds
    the instruction only in ``embed_query``.
    """

    return load_local_sentence_transformer(
        model_path,
        batch_size=batch_size,
        device=device,
        half_precision=half_precision,
        show_progress_bar=show_progress_bar,
        trust_remote_code=True,
        document_prompt_name=None,
        query_prompt_name=None,
        query_instruction=BROWSING_QWEN_INSTRUCTION,
    )


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embedding matrix must be 2D, got shape {matrix.shape}")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not bool(np.isfinite(norms).all()):
        raise ValueError("embedding matrix norms are non-finite")
    nonzero = norms > 0.0
    matrix = matrix.copy()
    matrix[nonzero] /= norms[nonzero, None].astype(np.float32)
    return matrix


def _encode_query(encoder: Any, text: str) -> np.ndarray:
    if hasattr(encoder, "embed_query"):
        value = encoder.embed_query(text)
    elif hasattr(encoder, "encode"):
        value = encoder.encode([text])
    else:
        raise TypeError("Qwen browsing encoder must provide embed_query or encode")
    query = np.asarray(value, dtype=np.float32)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1:
        raise ValueError(f"query embedding must be 1D, got shape {query.shape}")
    if not bool(np.isfinite(query).all()):
        raise ValueError("query embedding contains non-finite values")
    norm = float(np.linalg.norm(query.astype(np.float64)))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("query embedding has zero or invalid norm")
    return (query / np.float32(norm)).astype(np.float32)


@dataclass(frozen=True)
class BrowsingDenseMatch:
    parent_asin: str
    score: float
    row: int


@dataclass
class BrowsingDenseIndex:
    embeddings: np.ndarray
    parent_asins: tuple[str, ...]
    manifest: dict[str, Any]

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        expected_model: str | None = None,
    ) -> "BrowsingDenseIndex":
        directory = Path(artifact_dir)
        embeddings = np.load(directory / "product_embeddings.npy", allow_pickle=False)
        with (directory / "parent_asins.json").open(encoding="utf-8") as handle:
            parent_asins = json.load(handle)
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(parent_asins, list) or not all(
            isinstance(asin, str) and asin for asin in parent_asins
        ):
            raise ValueError("browsing dense parent_asins.json must be a list of strings")
        if len(set(parent_asins)) != len(parent_asins):
            raise ValueError("browsing dense parent_asins contain duplicates")
        embeddings = np.asarray(embeddings)
        if embeddings.dtype != np.float32:
            raise ValueError(f"browsing dense embeddings must be float32, got {embeddings.dtype}")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(parent_asins):
            raise ValueError("browsing dense embedding/ASIN alignment is invalid")
        norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
        nonzero = norms > 0.0
        if not bool(np.isfinite(norms).all()):
            raise ValueError("browsing dense embedding norms are non-finite")
        if bool((np.abs(norms[nonzero] - 1.0) > 1e-3).any()):
            raise ValueError("browsing dense non-zero vectors are not L2 normalized")
        dimension = manifest.get("dimension")
        if dimension != int(embeddings.shape[1]):
            raise ValueError("browsing dense manifest dimension does not match matrix")
        if expected_model is not None and not embedding_models_compatible(
            manifest.get("model", ""),
            expected_model,
        ):
            raise ValueError(
                f"browsing dense model mismatch: artifact={manifest.get('model')} "
                f"expected={expected_model}"
            )
        return cls(embeddings, tuple(parent_asins), manifest)

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    @cached_property
    def _positions(self) -> dict[str, int]:
        return {asin: row for row, asin in enumerate(self.parent_asins)}

    def search(
        self,
        query_text: str,
        encoder: Any,
        *,
        top_k: int = 100,
        allowed_asins: Collection[str] | None = None,
    ) -> list[BrowsingDenseMatch]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        cleaned = "\n".join(
            line.strip() for line in str(query_text).splitlines() if line.strip()
        )
        if not cleaned:
            return []
        # The shared local adapter applies the instruction in its query
        # method. Raw/fake encoders do not, so retain the compatible manual
        # formatting fallback for those injected encoders.
        query_input = (
            cleaned
            if getattr(encoder, "query_instruction", None)
            else format_qwen_query(cleaned)
        )
        query = _encode_query(encoder, query_input)
        if query.size != self.dimension:
            raise ValueError(
                f"query dimension {query.size} does not match index dimension {self.dimension}"
            )
        scores = self.embeddings @ query
        if allowed_asins is None:
            candidate_rows = np.arange(len(self.parent_asins), dtype=np.int64)
        else:
            allowed = {str(asin) for asin in allowed_asins}
            candidate_rows = np.asarray(
                [
                    row
                    for row, asin in enumerate(self.parent_asins)
                    if asin in allowed
                ],
                dtype=np.int64,
            )
        if candidate_rows.size == 0:
            return []
        candidate_scores = scores[candidate_rows]
        count = min(top_k, int(candidate_rows.size))
        if count == 0:
            return []
        if count < len(candidate_scores):
            local_candidates = np.argpartition(-candidate_scores, count - 1)[:count]
            local_order = local_candidates[
                np.argsort(-candidate_scores[local_candidates], kind="stable")
            ]
        else:
            local_order = np.argsort(-candidate_scores, kind="stable")
        return [
            BrowsingDenseMatch(
                parent_asin=self.parent_asins[int(candidate_rows[row])],
                score=float(candidate_scores[row]),
                row=int(candidate_rows[row]),
            )
            for row in local_order
        ]

    def similarity(self, first_asin: str, second_asin: str) -> float:
        """Return cosine similarity between two indexed product cards."""

        first = self._positions.get(str(first_asin))
        second = self._positions.get(str(second_asin))
        if first is None or second is None:
            return 0.0
        return float(self.embeddings[first] @ self.embeddings[second])


__all__ = [
    "BROWSING_QWEN_INSTRUCTION",
    "BrowsingDenseIndex",
    "BrowsingDenseMatch",
    "format_qwen_query",
    "load_qwen_browsing_encoder",
]
