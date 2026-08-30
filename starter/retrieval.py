"""Dependency-light in-memory product retrieval for the Layer 1/2/3 MVP.

The retriever consumes canonical product facts, optional direct Layer 2 field
vectors, and a BM25 product-text index. It does not parse user language.
Structured, semantic, and lexical scores are kept separate until the shared
final scorer combines them.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping

from dictionary.registry import normalize_text

from product_embeddings.layer2 import (
    Layer2EmbeddingIndex,
    load_layer2_embedding_index,
)
from product_embeddings.pipeline import embedding_models_compatible
from starter.bm25 import (
    BM25Index,
    DEFAULT_BM25F_ARTIFACT_DIR,
    DEFAULT_BM25F_DB_NAME,
    PythonBM25FIndex,
    SQLiteBM25FIndex,
    _catalog_sha256,
    default_extension_path,
)
from starter.routing.constraints import CATEGORICAL_FIELDS


FACT_FIELDS = tuple(CATEGORICAL_FIELDS)
DEFAULT_FACT_PATHS = (
    Path("data/derived/annotations/v5/annotations.jsonl"),
    Path("data/derived/annotations/v4/annotations.jsonl"),
    Path("data/derived/catalog_facts/catalog_facts.jsonl"),
    Path("data/derived/annotations/v2/annotations.jsonl"),
    Path("data/derived/annotations/v1/annotations.jsonl"),
    Path("data/derived/facts/facts.jsonl"),
    Path("data/facts.jsonl"),
)
DEFAULT_EMBEDDING_PATHS = (
    Path("data/derived/product_embeddings/product_embeddings.npy"),
    Path("data/derived/product_embeddings.npy"),
    Path("data/product_embeddings.npy"),
    Path("product_embeddings.npy"),
)
DEFAULT_METADATA_PATHS = (
    Path("data/derived/product_embeddings/product_embedding_metadata.json"),
    Path("data/derived/product_embedding_metadata.json"),
    Path("data/product_embedding_metadata.json"),
    Path("product_embedding_metadata.json"),
)
DEFAULT_LAYER2_ARTIFACT_PATHS = (
    Path("data/derived/product_embeddings"),
    Path("data/derived/layer2_embeddings"),
    Path("data/layer2_embeddings"),
    Path("layer2_embeddings"),
)


# One shared structured score is used by Buying and Browsing. Values are
# configured weighted points, and the final score is the accumulated weighted sum.
STRUCTURED_FIELD_WEIGHTS: dict[str, float] = {
    "category": 0.70,
    "price": 1.50,
    "brand": 7.00,
    "size": 0.80,
    "color": 1.00,
    "material": 1.20,
    "style": 0.50,
    "feature": 0.50,
    "use_case": 0.50,
}
MODE_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "BUYING": {"structured": 1.00, "dense": 1.00, "bm25": 0.20},
    "BROWSING": {"structured": 1.00, "dense": 1.00, "bm25": 0.20},
}
# Backward-compatible name for callers that inspect the Buying contribution.
DENSE_SCORE_WEIGHT = MODE_SCORE_WEIGHTS["BUYING"]["dense"]
BM25_SCORE_WEIGHT = MODE_SCORE_WEIGHTS["BUYING"]["bm25"]

# Rating tie-breaker (INSTRUCTION.md section 1).  The structured matcher scores
# most candidates identically -- 9.6 of every 10 returned candidates share one
# score -- so without a continuous term the top-10 order is catalog position.
# The catalog's own average_rating supplies that term.
RATING_SCALE = 5.0
NEUTRAL_NORMALIZED_RATING = 0.5
CRITICAL_USER_RATING_THRESHOLD = 3.5
RATING_BOOST_WEIGHT = 0.15
RATING_DEFAULT_WEIGHT = 0.02


def normalized_rating(rating: float | None) -> float:
    """Catalog rating on [0, 1].  An unusable rating is neutral, never zero.

    Scoring an unrated product as 0.0 would demote it; the shopper said nothing
    about it, so it is left where the retrieval score put it.
    """

    try:
        value = float(rating)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return NEUTRAL_NORMALIZED_RATING
    if value != value or value <= 0.0:
        return NEUTRAL_NORMALIZED_RATING
    return min(1.0, value / RATING_SCALE)


def is_critical_user(user_prior_rating: float | None) -> bool:
    """Whether the shopper rates strictly enough to weight catalog rating up."""

    try:
        value = float(user_prior_rating)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return value == value and value < CRITICAL_USER_RATING_THRESHOLD


def rating_weight(user_prior_rating: float | None) -> float:
    """``w_r``: boosted for a critical shopper, default for everyone else.

    A missing prior rating is a cold-start shopper and takes the default, which
    is the spec's null fallback.
    """

    return (
        RATING_BOOST_WEIGHT
        if is_critical_user(user_prior_rating)
        else RATING_DEFAULT_WEIGHT
    )


def _rating(value: object) -> float | None:
    """Parse ``average_rating`` off a raw catalog row."""

    try:
        rating = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if rating != rating or rating <= 0.0:
        return None
    return rating


def _encoder_model_id(encoder: object | None) -> str | None:
    if encoder is None:
        return None
    for name in ("model_id", "model_name", "embedding_model"):
        value = getattr(encoder, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _encoder_dimension(encoder: object | None) -> int | None:
    if encoder is None:
        return None
    for name in ("embedding_dimension", "dimension"):
        value = getattr(encoder, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    getter = getattr(encoder, "get_sentence_embedding_dimension", None)
    if callable(getter):
        try:
            value = getter()
        except (TypeError, ValueError):
            return None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _final_score(
    mode: str,
    structured_score: float,
    dense_score: float,
    bm25_score: float = 0.0,
) -> float:
    weights = MODE_SCORE_WEIGHTS[mode]
    return float(
        weights["structured"] * structured_score
        + weights["dense"] * dense_score
        + weights.get("bm25", 0.0) * bm25_score
    )


def _normalise_value(value: object) -> str:
    return normalize_text(str(value).strip())


def _values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        source = value
    else:
        source = (value,)
    output: list[str] = []
    for item in source:
        normalised = _normalise_value(item)
        if normalised and normalised not in output:
            output.append(normalised)
    return tuple(output)


def _price(value: object) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if match is None:
            return None
        result = float(match.group(0))
    return result if math.isfinite(result) and result >= 0.0 else None


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except (OSError, UnicodeError):
        return ()

    def rows() -> Iterable[Mapping[str, Any]]:
        with handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(item, Mapping):
                    yield item

    return rows()


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def _record_facts(record: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    nested = record.get("facts")
    source = nested if isinstance(nested, Mapping) else record
    facts: dict[str, tuple[str, ...]] = {}
    for field_name in FACT_FIELDS:
        if field_name in source:
            found = _values(source.get(field_name))
        elif field_name == "category":
            found = _values(record.get("categories"))
        elif field_name == "brand":
            found = _values(record.get("store"))
        else:
            found = ()
        if found:
            facts[field_name] = found
    return facts


def _catalog_structured_facts(record: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    facts = _record_facts(record)
    category = facts.get("category", ())
    return {"category": category} if category else {}

def _merge_facts(
    base: Mapping[str, tuple[str, ...]],
    supplement: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}
    for field_name in FACT_FIELDS:
        values = list(supplement.get(field_name, ()))
        if not values:
            values = list(base.get(field_name, ()))
        merged[field_name] = tuple(dict.fromkeys(values))
    return {field_name: values for field_name, values in merged.items() if values}


@dataclass(frozen=True)
class ProductRecord:
    """One validated catalog record plus canonical facts."""

    parent_asin: str
    facts: Mapping[str, tuple[str, ...]]
    price: float | None
    catalog_order: int
    rating: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class Candidate:
    """Shared candidate contract returned by both retrieval modes."""

    parent_asin: str
    score: float
    dense_score: float
    constraint_score: float
    matched_constraints: tuple[str, ...] = ()
    violated_constraints: tuple[str, ...] = ()
    relaxed_constraints: tuple[str, ...] = ()
    retrieval_mode: str = "BROWSING"
    attributes: Mapping[str, tuple[str, ...]] = field(default_factory=dict, repr=False, compare=False)
    semantic_score: float = 0.0
    matched_semantic_constraints: tuple[str, ...] = ()
    price: float | None = None
    bm25_score: float = 0.0
    rating: float | None = None
    ranking_score: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_asin": self.parent_asin,
            "score": self.score,
            "rating": self.rating,
            "ranking_score": self.ranking_score,
            "dense_score": self.dense_score,
            "semantic_score": self.semantic_score,
            "bm25_score": self.bm25_score,
            "constraint_score": self.constraint_score,
            "matched_constraints": list(self.matched_constraints),
            "matched_semantic_constraints": list(self.matched_semantic_constraints),
            "violated_constraints": list(self.violated_constraints),
            "relaxed_constraints": list(self.relaxed_constraints),
            "retrieval_mode": self.retrieval_mode,
        }


SharedCandidate = Candidate
QueryEncoder = Any


class ProductRetriever:
    """Load catalog artifacts once and retrieve from deterministic memory indexes."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        facts_path: str | Path | None = None,
        embeddings_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        query_encoder: QueryEncoder | None = None,
        layer2_artifact_dir: str | Path | None = None,
        layer2_weights: Mapping[str, float] | None = None,
        bm25f_artifact_dir: str | Path | None = None,
        bm25f_extension_path: str | Path | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.query_encoder = query_encoder
        self.layer2_weights = dict(layer2_weights) if layer2_weights is not None else None
        self.product_by_asin: dict[str, ProductRecord] = {}
        self._catalog_order: list[str] = []
        self.inverted_index: dict[str, dict[str, set[str]]] = {
            field_name: {} for field_name in FACT_FIELDS
        }
        self.price_lookup: dict[str, float | None] = {}
        self.rating_lookup: dict[str, float | None] = {}
        self._facts_by_asin, self._annotated_prices = self._load_fact_artifact(facts_path)
        self._load_catalog()
        self.bm25_index: BM25Index | PythonBM25FIndex | SQLiteBM25FIndex | None = None
        self.bm25_state = "loading"
        self.bm25_error: str | None = None
        self.bm25_build_seconds: float | None = None
        self.bm25_backend: str | None = None
        try:
            requested_backend = os.environ.get(
                "SHOPPING_BM25F_BACKEND", "auto"
            ).casefold()
            if requested_backend not in {"auto", "sqlite", "python"}:
                raise ValueError(
                    "SHOPPING_BM25F_BACKEND must be auto, sqlite, or python"
                )
            artifact_dir = (
                Path(bm25f_artifact_dir)
                if bm25f_artifact_dir is not None
                else DEFAULT_BM25F_ARTIFACT_DIR
            )
            db_path = artifact_dir / DEFAULT_BM25F_DB_NAME
            if requested_backend == "python":
                self.bm25_index = PythonBM25FIndex(
                    self.product_by_asin,
                    self._catalog_order,
                )
                self.bm25_state = "reference_python"
                self.bm25_error = "Python BM25F reference selected by environment"
            elif db_path.is_file():
                self.bm25_index = SQLiteBM25FIndex(
                    db_path,
                    extension_path=bm25f_extension_path or default_extension_path(),
                    expected_catalog_sha256=_catalog_sha256(self.catalog_path),
                    expected_catalog_rows=len(self._catalog_order),
                )
            else:
                self.bm25_index = PythonBM25FIndex(
                    self.product_by_asin,
                    self._catalog_order,
                )
                self.bm25_state = "fallback_python"
                self.bm25_error = (
                    f"Native BM25F artifact not found at {db_path}; "
                    "using Python BM25F reference"
                )
            if self.bm25_state == "loading":
                self.bm25_state = "ready"
            self.bm25_build_seconds = self.bm25_index.build_seconds
            self.bm25_backend = getattr(self.bm25_index, "backend", None)
            if self.bm25_state in {"fallback_python", "reference_python"}:
                print(f"[retrieval] {self.bm25_error}", flush=True)
        except Exception as exc:
            # A missing/incompatible native artifact must not prevent the
            # Agent from starting. The explicit reference implementation is
            # the safe fallback; it is slower but preserves lexical service.
            try:
                self.bm25_index = PythonBM25FIndex(
                    self.product_by_asin,
                    self._catalog_order,
                )
                self.bm25_state = "fallback_python"
                self.bm25_backend = self.bm25_index.backend
                self.bm25_build_seconds = self.bm25_index.build_seconds
                self.bm25_error = f"Native BM25F unavailable: {exc}"
            except Exception as fallback_exc:
                self.bm25_state = "unavailable"
                self.bm25_error = f"BM25F unavailable: {fallback_exc}"
            print(f"[retrieval] {self.bm25_error}", flush=True)
        self.embedding_matrix: Any = None
        self.embedding_asins: tuple[str, ...] = ()
        self._embedding_norms: Any = None
        self._load_embeddings(embeddings_path, metadata_path)
        self.layer2_index: Layer2EmbeddingIndex | None = None
        self._layer2_encoder_compatible = False
        self.layer2_compatibility_error: str | None = None
        self._load_layer2(layer2_artifact_dir)

    @property
    def valid_asins(self) -> frozenset[str]:
        return frozenset(self.product_by_asin)

    @property
    def has_dense_index(self) -> bool:
        return bool(self.layer2_index is not None and self.layer2_index.asins) or (
            self.embedding_matrix is not None and bool(self.embedding_asins)
        )

    def _load_layer2(self, artifact_dir: str | Path | None) -> None:
        if artifact_dir is None:
            if self.query_encoder is None:
                return
            try:
                is_default_catalog = self.catalog_path.resolve() == Path(
                    "data/catalog.jsonl"
                ).resolve()
            except OSError:
                is_default_catalog = False
            if not is_default_catalog:
                return
        candidates = (
            (Path(artifact_dir),) if artifact_dir is not None else DEFAULT_LAYER2_ARTIFACT_PATHS
        )
        expected_asins = tuple(self._catalog_order)
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                index = load_layer2_embedding_index(
                    candidate,
                    expected_asins=expected_asins,
                )
            except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self.layer2_index = index
            self._layer2_encoder_compatible = self._validate_layer2_encoder(index)
            return

    def _validate_layer2_encoder(self, index: Layer2EmbeddingIndex) -> bool:
        encoder = self.query_encoder
        if encoder is None:
            self.layer2_compatibility_error = "Layer 2 query encoder is not configured"
            return False

        manifest = index.manifest
        expected_model = manifest.get("embedding_model", manifest.get("model"))
        actual_model = _encoder_model_id(encoder)
        if expected_model and actual_model and not embedding_models_compatible(
            expected_model, actual_model
        ):
            self.layer2_compatibility_error = (
                "Layer 2 embedding model does not match the query encoder: "
                f"artifact={expected_model!r}, encoder={actual_model!r}"
            )
            return False

        actual_dimension = _encoder_dimension(encoder)
        if actual_dimension is not None and actual_dimension != index.dimension:
            self.layer2_compatibility_error = (
                "Layer 2 embedding dimension does not match the query encoder: "
                f"artifact={index.dimension}, encoder={actual_dimension}"
            )
            return False
        return True

    @property
    def dense_available(self) -> bool:
        encoder = self.query_encoder
        if self.layer2_index is not None and not self._layer2_encoder_compatible:
            return False
        return self.has_dense_index and (
            callable(encoder)
            or any(
                callable(getattr(encoder, name, None))
                for name in ("encode", "embed_documents", "embed")
            )
        )

    @property
    def bm25_available(self) -> bool:
        return self.bm25_index is not None

    def _load_fact_artifact(
        self, facts_path: str | Path | None
    ) -> tuple[dict[str, Mapping[str, tuple[str, ...]]], dict[str, float | None]]:
        selected = Path(facts_path) if facts_path is not None else _first_existing(DEFAULT_FACT_PATHS)
        if selected is None:
            return {}, {}
        facts: dict[str, Mapping[str, tuple[str, ...]]] = {}
        prices: dict[str, float | None] = {}
        for row in _read_jsonl(selected):
            annotation = row.get("annotation")
            if isinstance(annotation, Mapping) and str(annotation.get("status", "")).casefold() != "success":
                continue
            asin = str(row.get("parent_asin", "")).strip()
            if not asin:
                continue
            facts[asin] = _record_facts(row)
            if "price" in row:
                prices[asin] = _price(row.get("price"))
        return facts, prices

    def _load_catalog(self) -> None:
        for row in _read_jsonl(self.catalog_path):
            asin = str(row.get("parent_asin", "")).strip()
            if not asin or asin in self.product_by_asin:
                continue
            annotation_facts = self._facts_by_asin.get(asin)
            catalog_facts = _catalog_structured_facts(row)
            if annotation_facts is None:
                # The raw catalog remains the product-universe authority, but
                # an unannotated product contributes only safe category data.
                facts = catalog_facts
            else:
                facts = _merge_facts(catalog_facts, annotation_facts)
            raw_price = _price(row.get("price"))
            price = self._annotated_prices.get(asin, raw_price)
            if asin in self._annotated_prices and price is None and raw_price is not None:
                price = raw_price
            product = ProductRecord(
                parent_asin=asin,
                facts=facts,
                price=price,
                catalog_order=len(self._catalog_order),
                rating=_rating(row.get("average_rating")),
                raw=dict(row),
            )
            self.product_by_asin[asin] = product
            self._catalog_order.append(asin)
            self.price_lookup[asin] = price
            self.rating_lookup[asin] = product.rating
            for field_name, values in facts.items():
                for value in values:
                    self.inverted_index[field_name].setdefault(value, set()).add(asin)

    @staticmethod
    def _metadata_asins(payload: object) -> tuple[str, ...]:
        rows: object = payload
        if isinstance(payload, Mapping):
            rows = payload.get("rows", payload.get("metadata", payload.get("items", ())))
            if isinstance(rows, Mapping):
                rows = rows.values()
        if not isinstance(rows, (list, tuple)):
            return ()
        result: list[str] = []
        for row in rows:
            if isinstance(row, Mapping):
                asin = str(row.get("parent_asin", row.get("asin", ""))).strip()
            else:
                asin = str(row).strip()
            if not asin:
                return ()
            result.append(asin)
        return tuple(result)

    def _load_embeddings(
        self,
        embeddings_path: str | Path | None,
        metadata_path: str | Path | None,
    ) -> None:
        selected_embeddings = (
            Path(embeddings_path)
            if embeddings_path is not None
            else _first_existing(DEFAULT_EMBEDDING_PATHS)
        )
        selected_metadata = (
            Path(metadata_path)
            if metadata_path is not None
            else _first_existing(DEFAULT_METADATA_PATHS)
        )
        if selected_embeddings is None or selected_metadata is None:
            return
        try:
            import numpy as np

            matrix = np.load(selected_embeddings, allow_pickle=False)
            with selected_metadata.open(encoding="utf-8") as handle:
                asins = self._metadata_asins(json.load(handle))
            if getattr(matrix, "ndim", None) != 2 or len(asins) != int(matrix.shape[0]):
                return
            if len(set(asins)) != len(asins) or any(asin not in self.product_by_asin for asin in asins):
                return
            if not bool(np.isfinite(matrix).all()):
                return
            norms = np.linalg.norm(matrix, axis=1)
            if not bool(np.isfinite(norms).all()) or not bool((norms > 0).all()):
                return
        except (ImportError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        self.embedding_matrix = matrix
        self.embedding_asins = asins
        self._embedding_norms = norms

    @staticmethod
    def _constraint_values(constraints: object, field_name: str) -> tuple[str, ...]:
        value = getattr(constraints, field_name, None)
        if value is None and isinstance(constraints, Mapping):
            value = constraints.get(field_name)
        return _values(value)

    @classmethod
    def _price_bounds(cls, constraints: object) -> tuple[float | None, float | None]:
        def get(name: str) -> float | None:
            value = getattr(constraints, name, None)
            if value is None and isinstance(constraints, Mapping):
                value = constraints.get(name)
            return _price(value)

        return get("price_min"), get("price_max")

    @classmethod
    def _constraint_fields(cls, constraints: object) -> tuple[str, ...]:
        fields = [
            field_name
            for field_name in FACT_FIELDS
            if cls._constraint_values(constraints, field_name)
        ]
        price_min, price_max = cls._price_bounds(constraints)
        if price_min is not None or price_max is not None:
            fields.append("price")
        return tuple(fields)

    @classmethod
    def _constraint_similarities(cls, constraints: object) -> dict[str, float]:
        """Read Layer 2 canonical-match similarity from persisted evidence."""

        evidence = getattr(constraints, "evidence", None)
        if evidence is None and isinstance(constraints, Mapping):
            evidence = constraints.get("evidence")
        if not isinstance(evidence, (list, tuple)):
            return {}
        similarities: dict[str, float] = {}
        for item in evidence:
            canonical_id = getattr(item, "canonical_id", None)
            confidence = getattr(item, "confidence", None)
            if isinstance(item, Mapping):
                canonical_id = item.get("canonical_id", canonical_id)
                confidence = item.get("confidence", confidence)
            if not isinstance(canonical_id, str):
                continue
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            similarities[canonical_id] = max(
                similarities.get(canonical_id, 0.0),
                min(max(score, 0.0), 1.0),
            )
        return similarities

    def _semantic_scores(
        self,
        asins: Iterable[str],
        constraints: object | None,
    ) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
        """Score product facts against the independent Layer 2 state.

        Each accepted semantic value contributes its retained cosine
        similarity only when the product contains that canonical value. The
        dense track score is the accumulated similarity points; exact
        structured matching is calculated separately.
        """

        if constraints is None:
            return {}, {}

        fields = tuple(
            field_name
            for field_name in (
                "category",
                "color",
                "material",
                "style",
                "feature",
                "use_case",
            )
            if self._constraint_values(constraints, field_name)
        )
        if not fields:
            return {}, {}

        similarities = self._constraint_similarities(constraints)
        eligible_asins = tuple(asins)
        eligible_set = set(eligible_asins)
        scores: dict[str, float] = {}
        labels: dict[str, list[str]] = {}
        for field_name in fields:
            for value in self._constraint_values(constraints, field_name):
                canonical_key = (
                    f"{field_name}:{normalize_text(value).replace(' ', '_')}"
                )
                similarity = similarities.get(canonical_key, 1.0)
                posting = self.inverted_index.get(field_name, {}).get(value, ())
                for asin in posting:
                    if asin not in eligible_set:
                        continue
                    scores[asin] = scores.get(asin, 0.0) + similarity
                    labels.setdefault(asin, []).append(
                        f"{field_name}:{value}@{similarity:.3f}"
                    )

        if not scores:
            # Preserve the fact that semantic constraints were present so the
            # caller does not silently switch to the separate query-level
            # dense fallback merely because no posting list matched.
            return {asin: 0.0 for asin in eligible_asins}, {}

        normalized_labels = {
            asin: tuple(values) for asin, values in labels.items()
        }
        return scores, normalized_labels

    def _matches_field(self, asin: str, field_name: str, constraints: object) -> bool:
        product = self.product_by_asin[asin]
        if field_name == "price":
            price_min, price_max = self._price_bounds(constraints)
            if product.price is None:
                return False
            return (price_min is None or product.price >= price_min) and (
                price_max is None or product.price <= price_max
            )
        requested = self._constraint_values(constraints, field_name)
        return bool(set(product.facts.get(field_name, ())) & set(requested))

    def _matched_labels(self, asin: str, constraints: object) -> tuple[str, ...]:
        labels: list[str] = []
        for field_name in FACT_FIELDS:
            for value in self._constraint_values(constraints, field_name):
                if value in self.product_by_asin[asin].facts.get(field_name, ()):
                    labels.append(f"{field_name}:{value}")
        if "price" in self._constraint_fields(constraints) and self._matches_field(asin, "price", constraints):
            price_min, price_max = self._price_bounds(constraints)
            if price_min is not None:
                labels.append(f"price_min:{price_min:g}")
            if price_max is not None:
                labels.append(f"price_max:{price_max:g}")
        return tuple(labels)

    def _eligible_asins(self, constraints: object) -> list[str]:
        price_min, price_max = self._price_bounds(constraints)
        if price_min is None and price_max is None:
            return list(self._catalog_order)
        return [
            asin
            for asin in self._catalog_order
            if self._matches_price_bounds(asin, price_min, price_max)
        ]

    def _matches_price_bounds(
        self,
        asin: str,
        price_min: float | None,
        price_max: float | None,
    ) -> bool:
        price = self.product_by_asin[asin].price
        if price is None:
            return False
        return (price_min is None or price >= price_min) and (
            price_max is None or price <= price_max
        )

    def _dense_scores(self, query_text: str) -> dict[str, float]:
        if self.query_encoder is None:
            return {}
        if self.layer2_index is not None:
            try:
                query = self._query_embedding(query_text, self.layer2_index.dimension)
                if query is None:
                    return {}
                scores, _ = self.layer2_index.score_all(
                    query,
                    weights=self.layer2_weights,
                )
                return {
                    asin: float(score)
                    for asin, score in zip(self.layer2_index.asins, scores)
                }
            except (TypeError, ValueError, RuntimeError):
                return {}
        if self.embedding_matrix is None or not self.embedding_asins:
            return {}
        try:
            import numpy as np
            query = self._query_embedding(
                query_text,
                int(self.embedding_matrix.shape[1]),
            )
            if query is None:
                return {}
            norm = float(np.linalg.norm(query.astype(np.float64)))
            if not math.isfinite(norm) or norm == 0.0:
                return {}
            scores = (self.embedding_matrix @ query) / (self._embedding_norms * norm)
            if not bool(np.isfinite(scores).all()):
                return {}
            return {asin: float(score) for asin, score in zip(self.embedding_asins, scores)}
        except (ImportError, TypeError, ValueError, RuntimeError):
            return {}

    def _bm25_scores(
        self,
        query_text: str,
        eligible_asins: Collection[str],
        *,
        top_k: int | None = None,
    ) -> dict[str, float]:
        if self.bm25_index is None:
            return {}
        try:
            return self.bm25_index.search(
                query_text,
                allowed_asins=eligible_asins,
                top_k=top_k,
            )
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
            return {}

    def _query_embedding(self, query_text: str, dimension: int) -> Any:
        """Encode one query with the shared runtime encoder and validate it."""
        import numpy as np

        encoder = self.query_encoder
        if encoder is None:
            return None
        if hasattr(encoder, "embed_query"):
            query_value = encoder.embed_query(query_text)
        elif hasattr(encoder, "encode"):
            method = encoder.encode
            try:
                query_value = method(
                    [query_text],
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            except TypeError:
                query_value = method([query_text])
        elif hasattr(encoder, "embed_documents"):
            query_value = encoder.embed_documents([query_text])
        elif hasattr(encoder, "embed"):
            query_value = encoder.embed([query_text])
        elif callable(encoder):
            query_value = encoder(query_text)
        else:
            return None
        query = np.asarray(query_value, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or query.size != dimension:
            return None
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if not math.isfinite(norm) or norm == 0.0:
            return None
        return query

    def _candidate(
        self,
        asin: str,
        mode: str,
        dense_score: float,
        structured_score: float,
        matched_constraints: tuple[str, ...],
        matched_fields: set[str],
        constraint_fields: tuple[str, ...],
        matched_semantic_constraints: tuple[str, ...] = (),
        bm25_score: float = 0.0,
        w_rating: float = 0.0,
    ) -> Candidate:
        violated = tuple(
            f"{field_name}:required"
            for field_name in constraint_fields
            if field_name not in matched_fields
        )
        score = _final_score(mode, structured_score, dense_score, bm25_score)
        rating = self.rating_lookup.get(asin)
        return Candidate(
            parent_asin=asin,
            score=float(score),
            dense_score=float(dense_score),
            constraint_score=float(structured_score),
            matched_constraints=matched_constraints,
            violated_constraints=violated,
            retrieval_mode=mode,
            attributes=self.product_by_asin[asin].facts,
            semantic_score=float(dense_score),
            matched_semantic_constraints=matched_semantic_constraints,
            price=self.product_by_asin[asin].price,
            bm25_score=float(bm25_score),
            rating=rating,
            ranking_score=float(score + w_rating * normalized_rating(rating)),
        )

    def retrieve(
        self,
        mode: str,
        query_text: str,
        constraints: object,
        *,
        semantic_constraints: object | None = None,
        limit: int = 100,
        minimum_candidates: int = 50,
        excluded_asins: Collection[str] | None = None,
        apply_budget: bool = True,
        user_prior_rating: float | None = None,
    ) -> list[Candidate]:
        """Return one deterministic candidate ranking for either mode.

        The shared ranker accumulates exact structured matches from the
        inverted indexes. Non-budget fields are scored softly; an active
        budget is the only eligibility filter.
        """

        del minimum_candidates
        limit = max(0, int(limit))
        if limit == 0 or not self.product_by_asin:
            return []
        mode = "BUYING" if str(mode).upper() == "BUYING" else "BROWSING"
        excluded = {
            str(asin).strip()
            for asin in (excluded_asins or ())
            if str(asin).strip()
        }
        constraint_fields = self._constraint_fields(constraints)
        requested_by_field = {
            field_name: self._constraint_values(constraints, field_name)
            for field_name in constraint_fields
            if field_name != "price"
        }
        price_min, price_max = self._price_bounds(constraints)
        price_eligible_asins = self._eligible_asins(constraints)
        eligible_source = (
            price_eligible_asins if apply_budget else self._catalog_order
        )
        eligible_asins = [
            asin
            for asin in eligible_source
            if asin not in excluded
        ]
        price_match_asins = set(price_eligible_asins)
        eligible_set = set(eligible_asins)

        semantic_scores, semantic_labels = self._semantic_scores(
            eligible_asins,
            semantic_constraints,
        )
        dense_scores = semantic_scores
        if not dense_scores and self.dense_available:
            dense_scores = self._dense_scores(query_text)
        bm25_scores = self._bm25_scores(query_text, eligible_set, top_k=limit)

        constraint_similarities = self._constraint_similarities(constraints)
        matched_weight: dict[str, float] = {}
        matched_fields: dict[str, set[str]] = {}
        matched_labels: dict[str, list[str]] = {}

        for field_name in constraint_fields:
            if field_name == "price":
                field_matches = price_match_asins & eligible_set
                field_match_similarities: dict[str, float] = {}
            else:
                field_matches: set[str] = set()
                field_match_similarities = {}
                for value in requested_by_field[field_name]:
                    indexed = self.inverted_index.get(field_name, {}).get(value, set())
                    canonical_key = (
                        f"{field_name}:{normalize_text(value).replace(' ', '_')}"
                    )
                    similarity = constraint_similarities.get(canonical_key, 1.0)
                    if eligible_set is None:
                        field_matches.update(indexed)
                        for asin in indexed:
                            field_match_similarities[asin] = max(
                                field_match_similarities.get(asin, 0.0), similarity
                            )
                            matched_labels.setdefault(asin, []).append(
                                f"{field_name}:{value}"
                            )
                    else:
                        field_matches.update(indexed & eligible_set)
                        for asin in indexed:
                            if asin in eligible_set:
                                field_match_similarities[asin] = max(
                                    field_match_similarities.get(asin, 0.0), similarity
                                )
                                matched_labels.setdefault(asin, []).append(
                                    f"{field_name}:{value}"
                                )
            field_weight = STRUCTURED_FIELD_WEIGHTS.get(field_name, 0.0)
            for asin in field_matches:
                matched_weight[asin] = matched_weight.get(asin, 0.0) + (
                    field_weight * field_match_similarities.get(asin, 1.0)
                )
                matched_fields.setdefault(asin, set()).add(field_name)

        def structured_score(asin: str) -> float:
            return matched_weight.get(asin, 0.0)

        price_labels: list[str] = []
        if "price" in constraint_fields:
            if price_min is not None:
                price_labels.append(f"price_min:{price_min:g}")
            if price_max is not None:
                price_labels.append(f"price_max:{price_max:g}")

        def labels_for(asin: str) -> tuple[str, ...]:
            labels = list(matched_labels.get(asin, ()))
            if "price" in constraint_fields and asin in price_match_asins:
                labels.extend(price_labels)
            return tuple(labels)

        w_rating = rating_weight(user_prior_rating)

        def _rank_key(
            base: Callable[[str], float]
        ) -> Callable[[str], tuple[float, int]]:
            """The spec's S(x), descending, with catalog order as last resort."""

            def key(asin: str) -> tuple[float, int]:
                bonus = w_rating * normalized_rating(self.rating_lookup.get(asin))
                return (
                    -(base(asin) + bonus),
                    self.product_by_asin[asin].catalog_order,
                )

            return key

        def _select(pool: list[str], base: Callable[[str], float]) -> list[str]:
            """Rank the whole pool *before* slicing to limit (Task 2)."""

            key = _rank_key(base)
            if limit < len(pool):
                return heapq.nsmallest(limit, pool, key=key)
            return sorted(pool, key=key)

        def _zero(_asin: str) -> float:
            return 0.0

        if not dense_scores and not constraint_fields:
            # No constraints and no dense signal: every candidate ties at zero,
            # so the rating is the only thing left to order them by.
            ranked_asins = _select(eligible_asins, _zero)
        elif not dense_scores:
            positive_asins = sorted(matched_weight, key=_rank_key(structured_score))
            ranked_asins = positive_asins[:limit]
            if len(ranked_asins) < limit:
                selected = set(ranked_asins)
                remainder = [
                    asin for asin in eligible_asins if asin not in selected
                ]
                # Zero-match padding is ordered by rating too, not by catalog
                # position, so the pad respects the same preference.
                ranked_asins.extend(
                    heapq.nsmallest(
                        limit - len(ranked_asins), remainder, key=_rank_key(_zero)
                    )
                )
        else:
            final_scores = {
                asin: _final_score(
                    mode,
                    structured_score(asin),
                    dense_scores.get(asin, 0.0),
                    bm25_scores.get(asin, 0.0),
                )
                for asin in eligible_asins
            }
            ranked_asins = _select(eligible_asins, final_scores.__getitem__)
        return [
            self._candidate(
                asin,
                mode,
                dense_scores.get(asin, 0.0),
                structured_score(asin),
                labels_for(asin),
                matched_fields.get(asin, set()),
                constraint_fields,
                semantic_labels.get(asin, ()),
                bm25_scores.get(asin, 0.0),
                w_rating=w_rating,
            )
            for asin in ranked_asins
        ]

    def debug_rank_all(
        self,
        mode: str,
        query_text: str,
        constraints: object,
        *,
        semantic_constraints: object | None = None,
        excluded_asins: Collection[str] | None = None,
        apply_budget: bool = True,
        user_prior_rating: float | None = None,
    ) -> list[Candidate]:
        """Return the complete ranking using the production scorer."""

        return self.retrieve(
            mode,
            query_text,
            constraints,
            semantic_constraints=semantic_constraints,
            limit=len(self.product_by_asin),
            excluded_asins=excluded_asins,
            apply_budget=apply_budget,
            user_prior_rating=user_prior_rating,
        )


InMemoryRetriever = ProductRetriever


__all__ = [
    "CRITICAL_USER_RATING_THRESHOLD",
    "Candidate",
    "BM25_SCORE_WEIGHT",
    "InMemoryRetriever",
    "MODE_SCORE_WEIGHTS",
    "ProductRecord",
    "ProductRetriever",
    "RATING_BOOST_WEIGHT",
    "RATING_DEFAULT_WEIGHT",
    "SharedCandidate",
    "is_critical_user",
    "normalized_rating",
    "rating_weight",
]
