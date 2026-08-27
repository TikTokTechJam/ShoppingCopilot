"""Direct catalog field embeddings for the Layer 2 retrieval path.

Layer 2 is deliberately independent from canonical facts.  It reads only the
immutable catalog and creates one matrix per semantic catalog view.  Empty
views are represented by zero rows plus a presence mask in metadata; they do
not become negative evidence during scoring.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAYER2_ARTIFACT_VERSION = "layer2-product-field-embeddings-v1"
LAYER2_TEXT_VERSION = "layer2-field-text-v1"
LAYER2_VIEWS = ("categories", "title", "features", "description")
LAYER2_FILES = {
    "categories": "category_embeddings.npy",
    "title": "title_embeddings.npy",
    "features": "features_embeddings.npy",
    "description": "description_embeddings.npy",
}

# Starting weights only.  The architecture leaves final weights for
# benchmark tuning; callers can provide a mapping to search().
DEFAULT_LAYER2_WEIGHTS = {
    "categories": 1.5,
    "title": 2.0,
    "features": 1.25,
    "description": 0.75,
}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "NumPy is required for Layer 2 embedding artifacts; install "
            "requirements-embeddings.txt"
        ) from exc
    return np


def _clean_text(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"catalog field {field} must be a string or null")
    return " ".join(value.split())


def _clean_text_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise ValueError(f"catalog field {field} must be an array, string, or null")
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"catalog field {field} must contain only strings")
        cleaned = " ".join(item.split())
        if cleaned:
            result.append(cleaned)
    return result


def build_layer2_view_text(product: Mapping[str, Any], view: str) -> str:
    """Build one deterministic raw-catalog text view for a product."""

    if view not in LAYER2_VIEWS:
        raise ValueError(f"unknown Layer 2 view: {view}")
    if not isinstance(product, Mapping):
        raise ValueError("catalog product must be an object")
    if view == "title":
        return _clean_text(product.get("title"), field="title")
    field = view
    return " ".join(_clean_text_list(product.get(field, []), field=field))


def build_layer2_view_documents(
    products: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Return one source-only text list per Layer 2 view."""

    return {
        view: [build_layer2_view_text(product, view) for product in products]
        for view in LAYER2_VIEWS
    }


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read catalog: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number}: catalog row must be an object")
            asin = record.get("parent_asin")
            if not isinstance(asin, str) or not asin.strip():
                raise ValueError(f"{path}:{line_number}: missing parent_asin")
            asin = asin.strip()
            if asin in seen:
                raise ValueError(f"{path}:{line_number}: duplicate parent_asin {asin}")
            seen.add(asin)
            product = dict(record)
            product["parent_asin"] = asin
            products.append(product)
    if not products:
        raise ValueError(f"{path}: catalog contains no products")
    return products


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_id(model: Any) -> str:
    for name in ("model_id", "model_name", "name"):
        value = getattr(model, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(model).__name__


def _call_encode(model: Any, texts: Sequence[str], batch_size: int) -> Any:
    method = model.encode
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    for name, value in (
        ("batch_size", batch_size),
        ("show_progress_bar", False),
        ("convert_to_numpy", True),
        ("normalize_embeddings", False),
    ):
        if accepts_kwargs or name in parameters:
            kwargs[name] = value
    return method(list(texts), **kwargs)


def _embed(model: Any, texts: Sequence[str], batch_size: int) -> Any:
    if hasattr(model, "embed_documents"):
        return model.embed_documents(list(texts))
    if hasattr(model, "encode"):
        return _call_encode(model, texts, batch_size)
    if hasattr(model, "embed"):
        return model.embed(list(texts))
    if callable(model):
        return model(list(texts))
    raise TypeError(
        "embedder must expose embed_documents, encode, embed, or be callable"
    )


def _as_matrix(value: Any, expected_rows: int, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 1:
        if expected_rows != 1:
            raise ValueError("embedder returned one vector for multiple texts")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != expected_rows:
        raise ValueError(
            "embedder output must be a 2-D matrix with one row per catalog text"
        )
    if array.shape[1] < 1:
        raise ValueError("embedder output must have a positive dimension")
    return array


def _embed_present(
    model: Any,
    texts: Sequence[str],
    present_rows: Sequence[int],
    *,
    product_count: int,
    batch_size: int,
    dimension: int | None,
) -> tuple[Any, int]:
    np = _require_numpy()
    if not present_rows:
        if dimension is None:
            raise ValueError("catalog has no non-empty Layer 2 text views")
        return np.zeros((product_count, dimension), dtype=np.float32), dimension

    batches: list[Any] = []
    present_texts = [texts[row] for row in present_rows]
    for start in range(0, len(present_texts), batch_size):
        batch = present_texts[start : start + batch_size]
        batches.append(_as_matrix(_embed(model, batch, batch_size), len(batch), np))
    raw = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if dimension is not None and int(raw.shape[1]) != dimension:
        raise ValueError("all Layer 2 views must use the same embedding dimension")
    if not np.isfinite(raw).all():
        raise ValueError("embedder returned non-finite values")
    norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0.0).any():
        raise ValueError("cannot L2-normalize a zero or non-finite Layer 2 vector")
    normalized = (raw.astype(np.float64) / norms).astype(np.float32)
    dimension = int(normalized.shape[1])
    output = np.zeros((product_count, dimension), dtype=np.float32)
    output[list(present_rows)] = normalized
    return output, dimension


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_npy(path: Path, array: Any, np: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def build_layer2_embeddings(
    catalog_path: str | Path,
    output_dir: str | Path,
    model: Any,
    *,
    batch_size: int = 32,
    catalog_version: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the four core Layer 2 matrices directly from ``catalog.jsonl``."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    catalog = Path(catalog_path)
    output = Path(output_dir)
    products = _read_catalog(catalog)
    documents = build_layer2_view_documents(products)
    product_count = len(products)
    dimension: int | None = None
    matrices: dict[str, Any] = {}
    presence: dict[str, list[bool]] = {}

    for view in LAYER2_VIEWS:
        texts = documents[view]
        present_rows = [row for row, text in enumerate(texts) if text]
        presence[view] = [bool(text) for text in texts]
        matrix, dimension = _embed_present(
            model,
            texts,
            present_rows,
            product_count=product_count,
            batch_size=batch_size,
            dimension=dimension,
        )
        matrices[view] = matrix

    if dimension is None:  # guarded by _embed_present, kept for type clarity
        raise ValueError("catalog has no non-empty Layer 2 text views")

    if generated_at_utc is None:
        generated_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.strip():
        raise ValueError("generated_at_utc must be a non-empty string when provided")

    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for row, product in enumerate(products):
        metadata.append(
            {
                "row": row,
                "parent_asin": product["parent_asin"],
                **{f"has_{view}": presence[view][row] for view in LAYER2_VIEWS},
            }
        )

    source_hash = _source_sha256(catalog)
    resolved_catalog_version = catalog_version or f"sha256:{source_hash}"
    model_name = _model_id(model)
    manifest = {
        "artifact_format_version": LAYER2_ARTIFACT_VERSION,
        "generation_version": LAYER2_ARTIFACT_VERSION,
        "generated_at_utc": generated_at_utc,
        "catalog_path": catalog.as_posix(),
        "source_catalog_version": resolved_catalog_version,
        "catalog_version": resolved_catalog_version,
        "catalog_sha256": source_hash,
        "embedding_model": model_name,
        "model": model_name,
        "embedding_dimension": dimension,
        "dimension": dimension,
        "dtype": "float32",
        "normalization": "l2",
        "product_count": product_count,
        "views": list(LAYER2_VIEWS),
        "implemented_views": list(LAYER2_VIEWS),
        "optional_views": ["details"],
        "row_order": "catalog order",
        "text_version": LAYER2_TEXT_VERSION,
        "generation_config": {
            "batch_size": batch_size,
            "views": list(LAYER2_VIEWS),
            "presence_masks": "metadata.has_<view>",
        },
    }

    np = _require_numpy()
    for view, matrix in matrices.items():
        _atomic_npy(output / LAYER2_FILES[view], matrix, np)
    _atomic_json(output / "product_embedding_metadata.json", metadata)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class Layer2EmbeddingMatch:
    row: int
    parent_asin: str
    score: float
    view_scores: Mapping[str, float | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "parent_asin": self.parent_asin,
            "score": self.score,
            "view_scores": dict(self.view_scores),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


class Layer2EmbeddingIndex:
    """Validated in-memory exact multi-view Layer 2 search."""

    def __init__(
        self,
        matrices: Mapping[str, Any],
        metadata: Sequence[Mapping[str, Any]],
        presence: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        self._np = _require_numpy()
        self._matrices = dict(matrices)
        self._metadata = tuple(dict(row) for row in metadata)
        self._presence = dict(presence)
        self._manifest = dict(manifest)
        self._asins = tuple(str(row["parent_asin"]) for row in metadata)
        self._asin_to_row = {asin: row for row, asin in enumerate(self._asins)}

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        expected_asins: Iterable[str] | None = None,
    ) -> "Layer2EmbeddingIndex":
        directory = Path(artifact_dir)
        np = _require_numpy()
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            metadata = json.loads(
                (directory / "product_embedding_metadata.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Layer 2 metadata in {directory}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("Layer 2 manifest must be an object")
        if not isinstance(metadata, list):
            raise ValueError("Layer 2 product metadata must be an array")

        views = tuple(manifest.get("views", LAYER2_VIEWS))
        if views != LAYER2_VIEWS:
            raise ValueError("Layer 2 artifact must contain exactly the four core views")
        matrices: dict[str, Any] = {}
        expected_shape: tuple[int, int] | None = None
        for view in LAYER2_VIEWS:
            try:
                matrix = np.load(directory / LAYER2_FILES[view], allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f"unable to load Layer 2 matrix for {view}") from exc
            if matrix.ndim != 2 or matrix.dtype != np.dtype(np.float32):
                raise ValueError(f"Layer 2 {view} matrix must be 2-D float32")
            if expected_shape is None:
                expected_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
            elif tuple(matrix.shape) != expected_shape:
                raise ValueError("Layer 2 matrices must have identical shapes")
            matrices[view] = matrix
        assert expected_shape is not None
        product_count, dimension = expected_shape
        if len(metadata) != product_count:
            raise ValueError("Layer 2 metadata rows and matrix rows differ")

        rows: list[dict[str, Any]] = []
        seen_asins: set[str] = set()
        presence: dict[str, Any] = {}
        for expected_row, item in enumerate(metadata):
            if not isinstance(item, Mapping):
                raise ValueError(f"Layer 2 metadata row {expected_row} must be an object")
            row = item.get("row")
            asin = item.get("parent_asin")
            if isinstance(row, bool) or not isinstance(row, int) or row != expected_row:
                raise ValueError(f"Layer 2 metadata row {expected_row} has invalid row {row!r}")
            if not isinstance(asin, str) or not asin.strip():
                raise ValueError(f"Layer 2 metadata row {expected_row} has invalid parent_asin")
            asin = asin.strip()
            if asin in seen_asins:
                raise ValueError(f"duplicate parent_asin in Layer 2 metadata: {asin}")
            seen_asins.add(asin)
            normalized = dict(item)
            normalized["parent_asin"] = asin
            rows.append(normalized)
        for view in LAYER2_VIEWS:
            key = f"has_{view}"
            values = []
            for row_number, row in enumerate(rows):
                value = row.get(key)
                if not isinstance(value, bool):
                    raise ValueError(f"Layer 2 metadata row {row_number} has invalid {key}")
                values.append(value)
            presence[view] = np.asarray(values, dtype=bool)

        manifest_dimension = manifest.get("embedding_dimension", manifest.get("dimension"))
        if manifest_dimension is not None and manifest_dimension != dimension:
            raise ValueError("Layer 2 manifest dimension does not match matrices")
        manifest_count = manifest.get("product_count")
        if manifest_count is not None and manifest_count != product_count:
            raise ValueError("Layer 2 manifest product count does not match matrices")
        if manifest.get("dtype") not in (None, "float32"):
            raise ValueError("Layer 2 manifest must describe float32 vectors")
        if manifest.get("normalization") not in (None, "l2", "L2"):
            raise ValueError("Layer 2 artifact is not marked as L2-normalized")

        for view, matrix in matrices.items():
            if not np.isfinite(matrix).all():
                raise ValueError(f"Layer 2 {view} matrix contains non-finite values")
            norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
            present = presence[view]
            if (norms[present] <= 0.0).any() or not np.allclose(
                norms[present], 1.0, rtol=1e-4, atol=1e-5
            ):
                raise ValueError(f"Layer 2 {view} present rows are not L2-normalized")
            if (norms[~present] > 1e-6).any():
                raise ValueError(f"Layer 2 {view} missing rows must be zero vectors")

        result = cls(matrices, rows, presence, manifest)
        if expected_asins is not None:
            result.validate_asins(expected_asins)
        return result

    @property
    def matrices(self) -> Mapping[str, Any]:
        return self._matrices

    @property
    def metadata(self) -> tuple[Mapping[str, Any], ...]:
        return self._metadata

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def asins(self) -> tuple[str, ...]:
        return self._asins

    @property
    def dimension(self) -> int:
        return int(self._matrices["title"].shape[1])

    @property
    def presence(self) -> Mapping[str, Any]:
        return self._presence

    def validate_asins(self, expected_asins: Iterable[str]) -> None:
        expected = tuple(expected_asins)
        if any(not isinstance(asin, str) or not asin.strip() for asin in expected):
            raise ValueError("expected_asins must contain non-empty strings")
        normalized = tuple(asin.strip() for asin in expected)
        if len(set(normalized)) != len(normalized):
            raise ValueError("expected_asins contains duplicates")
        if normalized != self._asins:
            missing = sorted(set(normalized) - set(self._asins))
            extra = sorted(set(self._asins) - set(normalized))
            raise ValueError(
                "Layer 2 metadata ASIN rows do not match catalog order; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    def row_for_asin(self, parent_asin: str) -> int:
        try:
            return self._asin_to_row[parent_asin]
        except KeyError as exc:
            raise KeyError(f"unknown parent_asin: {parent_asin}") from exc

    def _normalized_query(self, query_embedding: Any) -> Any:
        query = self._np.asarray(query_embedding, dtype=self._np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError(
                "Layer 2 query embedding must be a one-dimensional vector "
                "matching the artifact dimension"
            )
        if not self._np.isfinite(query).all():
            raise ValueError("Layer 2 query embedding contains non-finite values")
        norm = float(self._np.linalg.norm(query.astype(self._np.float64)))
        if not norm or not (norm == norm and norm not in (float("inf"), float("-inf"))):
            raise ValueError("Layer 2 query embedding must have a non-zero norm")
        return (query.astype(self._np.float64) / norm).astype(self._np.float32)

    @staticmethod
    def _weights(weights: Mapping[str, float] | None) -> dict[str, float]:
        selected = dict(DEFAULT_LAYER2_WEIGHTS if weights is None else weights)
        unknown = sorted(set(selected) - set(LAYER2_VIEWS))
        if unknown:
            raise ValueError(f"unknown Layer 2 view weights: {unknown}")
        result: dict[str, float] = {}
        for view in LAYER2_VIEWS:
            value = selected.get(view, DEFAULT_LAYER2_WEIGHTS[view])
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Layer 2 weight for {view} must be numeric")
            value = float(value)
            if not value == value or value in (float("inf"), float("-inf")) or value < 0:
                raise ValueError(f"Layer 2 weight for {view} must be finite and non-negative")
            result[view] = value
        if not any(result.values()):
            raise ValueError("at least one Layer 2 view weight must be positive")
        return result

    def score_all(
        self,
        query_embedding: Any,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Return the presence-aware score and raw per-view score arrays."""

        query = self._normalized_query(query_embedding)
        selected = self._weights(weights)
        view_scores = {
            view: self._matrices[view] @ query for view in LAYER2_VIEWS
        }
        denominator = self._np.zeros(len(self._asins), dtype=self._np.float32)
        numerator = self._np.zeros(len(self._asins), dtype=self._np.float32)
        for view in LAYER2_VIEWS:
            weight = selected[view]
            mask = self._presence[view]
            numerator += weight * view_scores[view] * mask
            denominator += weight * mask
        score = self._np.divide(
            numerator,
            denominator,
            out=self._np.zeros_like(numerator),
            where=denominator > 0,
        )
        return score, view_scores

    def search(
        self,
        query_embedding: Any,
        top_k: int = 10,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> list[Layer2EmbeddingMatch]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores, view_scores = self.score_all(query_embedding, weights=weights)
        order = self._np.argsort(-scores, kind="stable")[: min(top_k, len(self._asins))]
        matches: list[Layer2EmbeddingMatch] = []
        for row_value in order:
            row = int(row_value)
            matches.append(
                Layer2EmbeddingMatch(
                    row=row,
                    parent_asin=self._asins[row],
                    score=float(scores[row]),
                    view_scores={
                        view: float(view_scores[view][row])
                        if bool(self._presence[view][row])
                        else None
                        for view in LAYER2_VIEWS
                    },
                )
            )
        return matches


def load_layer2_embedding_index(
    artifact_dir: str | Path,
    *,
    expected_asins: Iterable[str] | None = None,
) -> Layer2EmbeddingIndex:
    return Layer2EmbeddingIndex.load(artifact_dir, expected_asins=expected_asins)


__all__ = [
    "DEFAULT_LAYER2_WEIGHTS",
    "LAYER2_ARTIFACT_VERSION",
    "LAYER2_FILES",
    "LAYER2_TEXT_VERSION",
    "LAYER2_VIEWS",
    "Layer2EmbeddingIndex",
    "Layer2EmbeddingMatch",
    "build_layer2_embeddings",
    "build_layer2_view_documents",
    "build_layer2_view_text",
    "load_layer2_embedding_index",
]
