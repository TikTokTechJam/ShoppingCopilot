from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "NumPy is required to load product embedding artifacts"
        ) from exc
    return np


@dataclass(frozen=True)
class ProductEmbeddingMatch:
    row: int
    parent_asin: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "parent_asin": self.parent_asin,
            "score": self.score,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


class ProductEmbeddingIndex:
    """Validated in-memory exact cosine search over normalized product rows."""

    def __init__(
        self,
        vectors: Any,
        metadata: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
    ) -> None:
        np = _require_numpy()
        self._vectors = vectors
        self._metadata = tuple(dict(item) for item in metadata)
        self._manifest = dict(manifest)
        self._asins = tuple(str(item["parent_asin"]) for item in metadata)
        self._asin_to_row = {asin: row for row, asin in enumerate(self._asins)}
        self._np = np

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        expected_asins: Iterable[str] | None = None,
    ) -> "ProductEmbeddingIndex":
        directory = Path(artifact_dir)
        matrix_path = directory / "product_embeddings.npy"
        metadata_path = directory / "product_embedding_metadata.json"
        manifest_path = directory / "manifest.json"
        np = _require_numpy()

        try:
            vectors = np.load(matrix_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"unable to load product embedding matrix: {matrix_path}") from exc
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid product embedding metadata in {directory}") from exc

        if not isinstance(manifest, Mapping):
            raise ValueError("manifest must be an object")
        if not isinstance(metadata, list):
            raise ValueError("product embedding metadata must be an array")
        if vectors.ndim != 2:
            raise ValueError("product embedding matrix must be two-dimensional")
        if vectors.dtype != np.dtype(np.float32):
            raise ValueError("product embedding matrix must have dtype float32")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("embedding rows and metadata rows differ")

        rows: list[dict[str, Any]] = []
        seen_asins: set[str] = set()
        for expected_row, item in enumerate(metadata):
            if not isinstance(item, Mapping):
                raise ValueError(f"metadata row {expected_row} must be an object")
            row = item.get("row")
            asin = item.get("parent_asin")
            if isinstance(row, bool) or not isinstance(row, int) or row != expected_row:
                raise ValueError(
                    f"metadata row {expected_row} has invalid row index {row!r}"
                )
            if not isinstance(asin, str) or not asin.strip():
                raise ValueError(f"metadata row {expected_row} has invalid parent_asin")
            asin = asin.strip()
            if asin in seen_asins:
                raise ValueError(f"duplicate parent_asin in metadata: {asin}")
            seen_asins.add(asin)
            normalized_item = dict(item)
            normalized_item["parent_asin"] = asin
            rows.append(normalized_item)

        manifest_dimension = manifest.get("embedding_dimension", manifest.get("dimension"))
        if manifest_dimension is not None and manifest_dimension != vectors.shape[1]:
            raise ValueError("manifest dimension does not match embedding matrix")
        manifest_count = manifest.get("product_count", manifest.get("row_count"))
        if manifest_count is not None and manifest_count != vectors.shape[0]:
            raise ValueError("manifest product count does not match embedding matrix")
        if manifest.get("dtype") not in (None, "float32"):
            raise ValueError("manifest dtype does not describe float32 vectors")
        if manifest.get("normalization") not in (None, "l2", "L2"):
            raise ValueError("product embedding artifact is not marked as L2-normalized")

        if not np.isfinite(vectors).all():
            raise ValueError("embedding matrix contains non-finite values")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if (norms <= 0.0).any() or not np.allclose(
            norms, 1.0, rtol=1e-4, atol=1e-5
        ):
            raise ValueError("embedding matrix contains vectors that are not L2-normalized")

        result = cls(vectors, rows, manifest)
        if expected_asins is not None:
            result.validate_asins(expected_asins)
        return result

    @property
    def vectors(self) -> Any:
        return self._vectors

    @property
    def metadata(self) -> tuple[Mapping[str, Any], ...]:
        return self._metadata

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def asins(self) -> tuple[str, ...]:
        return self._asins

    def validate_asins(self, expected_asins: Iterable[str]) -> None:
        expected = tuple(expected_asins)
        if any(not isinstance(asin, str) or not asin.strip() for asin in expected):
            raise ValueError("expected_asins must contain non-empty strings")
        if len(set(expected)) != len(expected):
            raise ValueError("expected_asins contains duplicates")
        normalized = tuple(asin.strip() for asin in expected)
        if normalized != self._asins:
            missing = sorted(set(normalized) - set(self._asins))
            extra = sorted(set(self._asins) - set(normalized))
            raise ValueError(
                "embedding metadata ASIN rows do not match expected catalog order; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    def row_for_asin(self, parent_asin: str) -> int:
        try:
            return self._asin_to_row[parent_asin]
        except KeyError as exc:
            raise KeyError(f"unknown parent_asin: {parent_asin}") from exc

    def search(self, query_embedding: Any, top_k: int = 10) -> list[ProductEmbeddingMatch]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query = self._np.asarray(query_embedding, dtype=self._np.float32)
        if query.ndim != 1 or query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                "query embedding must be a one-dimensional vector matching artifact dimension"
            )
        if not self._np.isfinite(query).all():
            raise ValueError("query embedding contains non-finite values")
        norm = float(self._np.linalg.norm(query.astype(self._np.float64)))
        if not norm or not math_is_finite(norm):
            raise ValueError("query embedding must have a non-zero finite norm")

        normalized_query = (query.astype(self._np.float64) / norm).astype(self._np.float32)
        scores = self._vectors @ normalized_query
        order = self._np.argsort(-scores, kind="stable")[: min(top_k, len(self._asins))]
        return [
            ProductEmbeddingMatch(
                row=int(row),
                parent_asin=self._asins[int(row)],
                score=float(scores[int(row)]),
            )
            for row in order
        ]


def math_is_finite(value: float) -> bool:
    # Kept as a tiny helper so the loader does not expose NumPy scalar details.
    return value == value and value not in (float("inf"), float("-inf"))


def load_product_embedding_index(
    artifact_dir: str | Path,
    *,
    expected_asins: Iterable[str] | None = None,
) -> ProductEmbeddingIndex:
    return ProductEmbeddingIndex.load(artifact_dir, expected_asins=expected_asins)
