"""V5 semantic product-card embeddings and exact cosine retrieval.

The V5 product path is intentionally separate from the BGE canonical-attribute
matcher.  It embeds one deterministic product card per catalog row and is used
as the product-level semantic signal for Browsing.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


V5_PRODUCT_ARTIFACT_VERSION = "v5-semantic-product-card-v1"
V5_PRODUCT_TEXT_VERSION = "v5-facts-product-card-v2"
V5_PRODUCT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
V5_PRODUCT_EMBEDDING_FILE = "product_embeddings.npy"
V5_PRODUCT_METADATA_FILE = "product_embedding_metadata.json"
V5_PRODUCT_CARDS_FILE = "product_cards.jsonl"
V5_PRODUCT_FACT_FIELDS = (
    "category",
    "brand",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
)


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "NumPy is required for V5 product embedding artifacts; install "
            "requirements-embeddings.txt"
        ) from exc
    return np


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    source = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    result: list[str] = []
    for item in source:
        text = _clean_text(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def build_v5_product_card(
    product: Mapping[str, Any],
    facts: Mapping[str, object] | None = None,
) -> str:
    """Serialize one catalog product and its V5 facts deterministically."""

    if not isinstance(product, Mapping):
        raise ValueError("product must be an object")
    source = facts if isinstance(facts, Mapping) else product.get("facts", {})
    if not isinstance(source, Mapping):
        source = {}

    parts: list[str] = []
    title = _clean_text(product.get("title"))
    if title:
        parts.append(f"title: {title}")
    for field_name in V5_PRODUCT_FACT_FIELDS:
        values = _values(source.get(field_name))
        if values:
            parts.append(f"{field_name}: {', '.join(values)}")
    return " | ".join(parts)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read JSONL: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"{path}: contains no rows")
    return rows


def _catalog_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows):
        asin = _clean_text(row.get("parent_asin"))
        if not asin:
            raise ValueError(f"{path}:{row_number + 1}: missing parent_asin")
        if asin in seen:
            raise ValueError(f"{path}:{row_number + 1}: duplicate parent_asin {asin}")
        seen.add(asin)
        row["parent_asin"] = asin
        normalized.append(row)
    return normalized


def _annotation_facts(path: Path) -> dict[str, Mapping[str, object]]:
    facts_by_asin: dict[str, Mapping[str, object]] = {}
    for row_number, row in enumerate(_read_jsonl(path), 1):
        asin = _clean_text(row.get("parent_asin"))
        if not asin:
            raise ValueError(f"{path}:{row_number}: missing parent_asin")
        if asin in facts_by_asin:
            raise ValueError(f"{path}:{row_number}: duplicate parent_asin {asin}")
        facts = row.get("facts", {})
        facts_by_asin[asin] = facts if isinstance(facts, Mapping) else {}
    return facts_by_asin


def _sha256(path: Path) -> str:
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


def _model_dimension(model: Any) -> int | None:
    for name in ("embedding_dimension", "dimension"):
        value = getattr(model, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        value = getter()
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _embed(model: Any, texts: Sequence[str]) -> Any:
    if callable(getattr(model, "embed_documents", None)):
        return model.embed_documents(list(texts))
    if callable(getattr(model, "encode", None)):
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
    if callable(getattr(model, "embed", None)):
        return model.embed(list(texts))
    if callable(model):
        return model(list(texts))
    raise TypeError("model must expose embed_documents, encode, embed, or be callable")


def _normalise(value: Any, *, expected_rows: int, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim == 1:
        if expected_rows != 1:
            raise ValueError("encoder returned one vector for multiple product cards")
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError("encoder output must have one row per product card")
    if matrix.shape[1] < 1 or not bool(np.isfinite(matrix).all()):
        raise ValueError("encoder output must be finite with a positive dimension")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    if bool((norms == 0).any()):
        raise ValueError("encoder output contains a zero vector")
    return (matrix / norms).astype(np.float32)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npy(path: Path, matrix: Any, np: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary.replace(path)


def build_v5_product_embeddings(
    catalog_path: str | Path = "data/catalog.jsonl",
    annotations_path: str | Path = "data/derived/annotations/v5/annotations.jsonl",
    output_dir: str | Path = "data/derived/product_embeddings_v5",
    model: Any | None = None,
    *,
    batch_size: int = 32,
    progress: bool = False,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Build one normalized Qwen product-card vector per catalog row."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    catalog = Path(catalog_path)
    annotations = Path(annotations_path)
    output = Path(output_dir)
    products = _catalog_rows(catalog)
    facts_by_asin = _annotation_facts(annotations)
    cards = [
        build_v5_product_card(product, facts_by_asin.get(product["parent_asin"], {}))
        for product in products
    ]
    if model is None:
        raise ValueError("a local product-card embedding model is required")

    np = _require_numpy()
    dimension = _model_dimension(model)
    matrices: list[Any] = []
    started = time.perf_counter()
    non_empty = [index for index, card in enumerate(cards) if card]
    if non_empty:
        for start in range(0, len(non_empty), batch_size):
            batch_rows = non_empty[start : start + batch_size]
            encoded = _normalise(
                _embed(model, [cards[row] for row in batch_rows]),
                expected_rows=len(batch_rows),
                np=np,
            )
            if dimension is None:
                dimension = int(encoded.shape[1])
            if int(encoded.shape[1]) != dimension:
                raise ValueError(
                    f"model dimension changed during encoding: "
                    f"expected={dimension}, actual={encoded.shape[1]}"
                )
            matrices.append((batch_rows, encoded))
            if progress:
                print(
                    f"[v5-product-embeddings] encoded "
                    f"{min(start + len(batch_rows), len(non_empty)):,}/"
                    f"{len(non_empty):,} cards "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    if dimension is None:
        raise ValueError("cannot determine embedding dimension from an empty catalog")

    matrix = np.zeros((len(products), dimension), dtype=np.float32)
    for rows, encoded in matrices:
        matrix[rows, :] = encoded
    metadata = [
        {
            "row": row,
            "parent_asin": product["parent_asin"],
            "has_card": bool(cards[row]),
        }
        for row, product in enumerate(products)
    ]

    output.mkdir(parents=True, exist_ok=True)
    with (output / V5_PRODUCT_CARDS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for product, card in zip(products, cards):
            handle.write(
                json.dumps(
                    {"parent_asin": product["parent_asin"], "text": card},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    source_catalog_hash = _sha256(catalog)
    source_annotations_hash = _sha256(annotations)
    model_name = (model_id or _model_id(model)).strip()
    if not model_name:
        raise ValueError("embedding model identifier must be non-empty")
    manifest = {
        "artifact_format_version": V5_PRODUCT_ARTIFACT_VERSION,
        "generation_version": V5_PRODUCT_ARTIFACT_VERSION,
        "text_version": V5_PRODUCT_TEXT_VERSION,
        "catalog_path": catalog.as_posix(),
        "annotations_path": annotations.as_posix(),
        "catalog_sha256": source_catalog_hash,
        "annotations_sha256": source_annotations_hash,
        "embedding_model": model_name,
        "model": model_name,
        "embedding_dimension": int(dimension),
        "dimension": int(dimension),
        "dtype": "float32",
        "normalization": "l2",
        "product_count": len(products),
        "product_card_fields": list(V5_PRODUCT_FACT_FIELDS),
        "row_order": "catalog order",
        "embedding_file": V5_PRODUCT_EMBEDDING_FILE,
        "metadata_file": V5_PRODUCT_METADATA_FILE,
        "cards_file": V5_PRODUCT_CARDS_FILE,
        "generation_config": {"batch_size": batch_size},
    }
    _write_npy(output / V5_PRODUCT_EMBEDDING_FILE, matrix, np)
    _write_json(output / V5_PRODUCT_METADATA_FILE, metadata)
    _write_json(output / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class V5ProductEmbeddingMatch:
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


class V5ProductEmbeddingIndex:
    """Validated in-memory cosine search over V5 product-card vectors."""

    def __init__(
        self,
        vectors: Any,
        metadata: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
    ) -> None:
        self._np = _require_numpy()
        self._vectors = vectors
        self._metadata = tuple(dict(row) for row in metadata)
        self._manifest = dict(manifest)
        self._asins = tuple(str(row["parent_asin"]) for row in self._metadata)
        self._asin_to_row = {asin: row for row, asin in enumerate(self._asins)}

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        expected_asins: Iterable[str] | None = None,
    ) -> "V5ProductEmbeddingIndex":
        directory = Path(artifact_dir)
        np = _require_numpy()
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            metadata = json.loads(
                (directory / V5_PRODUCT_METADATA_FILE).read_text(encoding="utf-8")
            )
            vectors = np.load(
                directory / V5_PRODUCT_EMBEDDING_FILE,
                allow_pickle=False,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid V5 product embedding artifact in {directory}") from exc
        if not isinstance(manifest, Mapping) or not isinstance(metadata, list):
            raise ValueError("V5 product manifest and metadata must be objects/arrays")
        if vectors.ndim != 2 or vectors.dtype != np.dtype(np.float32):
            raise ValueError("V5 product matrix must be a 2-D float32 array")
        if len(metadata) != int(vectors.shape[0]):
            raise ValueError("V5 product metadata and matrix rows differ")
        if manifest.get("normalization") not in (None, "l2", "L2"):
            raise ValueError("V5 product artifact must be L2-normalized")
        manifest_dimension = manifest.get("embedding_dimension", manifest.get("dimension"))
        if manifest_dimension is not None and int(manifest_dimension) != int(vectors.shape[1]):
            raise ValueError("V5 product manifest dimension does not match matrix")
        manifest_count = manifest.get("product_count")
        if manifest_count is not None and int(manifest_count) != len(metadata):
            raise ValueError("V5 product manifest count does not match matrix")

        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for expected_row, item in enumerate(metadata):
            if not isinstance(item, Mapping):
                raise ValueError(f"V5 product metadata row {expected_row} is not an object")
            row = item.get("row")
            asin = item.get("parent_asin")
            if isinstance(row, bool) or not isinstance(row, int) or row != expected_row:
                raise ValueError(f"V5 product metadata row {expected_row} has invalid row")
            if not isinstance(asin, str) or not asin.strip() or asin in seen:
                raise ValueError(f"V5 product metadata row {expected_row} has invalid ASIN")
            seen.add(asin)
            normalized.append({**dict(item), "parent_asin": asin.strip()})

        if not bool(np.isfinite(vectors).all()):
            raise ValueError("V5 product matrix contains non-finite values")
        present = np.asarray(
            [bool(row.get("has_card", True)) for row in normalized],
            dtype=bool,
        )
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if (norms[present] <= 0.0).any() or not np.allclose(
            norms[present], 1.0, rtol=1e-4, atol=1e-5
        ):
            raise ValueError("V5 product present rows are not L2-normalized")
        if (norms[~present] > 1e-6).any():
            raise ValueError("V5 product missing rows must be zero vectors")
        result = cls(vectors, normalized, manifest)
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

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    def validate_asins(self, expected_asins: Iterable[str]) -> None:
        expected = tuple(str(asin).strip() for asin in expected_asins)
        if any(not asin for asin in expected) or len(set(expected)) != len(expected):
            raise ValueError("expected_asins must contain unique non-empty strings")
        if expected != self._asins:
            raise ValueError("V5 product metadata ASIN rows do not match catalog order")

    def row_for_asin(self, parent_asin: str) -> int:
        try:
            return self._asin_to_row[parent_asin]
        except KeyError as exc:
            raise KeyError(f"unknown parent_asin: {parent_asin}") from exc

    def _normalized_query(self, query_embedding: Any) -> Any:
        query = self._np.asarray(query_embedding, dtype=self._np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError("V5 product query dimension does not match the artifact")
        if not bool(self._np.isfinite(query).all()):
            raise ValueError("V5 product query contains non-finite values")
        norm = float(self._np.linalg.norm(query.astype(self._np.float64)))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("V5 product query must have a non-zero norm")
        return (query.astype(self._np.float64) / norm).astype(self._np.float32)

    def score_all(self, query_embedding: Any) -> Any:
        return self._vectors @ self._normalized_query(query_embedding)

    def search(self, query_embedding: Any, top_k: int = 100) -> list[V5ProductEmbeddingMatch]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores = self.score_all(query_embedding)
        order = self._np.argsort(-scores, kind="stable")[: min(top_k, len(self._asins))]
        return [
            V5ProductEmbeddingMatch(
                row=int(row),
                parent_asin=self._asins[int(row)],
                score=float(scores[int(row)]),
            )
            for row in order
        ]

    def similarity(self, first_asin: str, second_asin: str) -> float:
        first = self._vectors[self.row_for_asin(first_asin)]
        second = self._vectors[self.row_for_asin(second_asin)]
        return float(first @ second)


def load_v5_product_embedding_index(
    artifact_dir: str | Path,
    *,
    expected_asins: Iterable[str] | None = None,
) -> V5ProductEmbeddingIndex:
    return V5ProductEmbeddingIndex.load(artifact_dir, expected_asins=expected_asins)


__all__ = [
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
    "load_v5_product_embedding_index",
]
