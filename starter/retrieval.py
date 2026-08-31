"""Dependency-light in-memory product retrieval for the Layer 1/2/3 MVP.

The retriever consumes canonical product facts, a BM25 product-text index,
optional BGE canonical-expansion evidence, and an optional V5 product-card
vector index. It does not parse user language. Structured, canonical, product
dense, and lexical scores are kept separate until the mode-specific ranker
combines them.
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
from product_embeddings.v5 import (
    V5_PRODUCT_MODEL,
    V5ProductEmbeddingIndex,
    load_v5_product_embedding_index,
)
from starter.bm25 import BM25Index, BM25QueryCompiler
from starter.browsing import build_browsing_query, format_qwen_query
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
DEFAULT_V5_PRODUCT_ARTIFACT_PATHS = (
    Path("data/derived/product_embeddings_v5"),
    Path("data/derived/annotations/v5/product_embeddings"),
)
DEFAULT_PRODUCT_EMBEDDING_MODEL_PATHS = (
    Path("models/qwen3-embedding-0.6b"),
    Path("models/Qwen3-Embedding-0.6B"),
    Path("model/qwen3-embedding-0.6b"),
)
PRODUCT_EMBEDDING_MODEL_ENV = "SHOPPING_PRODUCT_EMBEDDING_MODEL"
BROWSING_RETRIEVAL_MODE_ENV = "SHOPPING_BROWSING_RETRIEVAL_MODE"
BROWSING_RETRIEVAL_MODES = ("hybrid", "qwen_dense")


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
    # Buying uses structured evidence and BM25 as its primary signals.  The
    # canonical BGE posting-list score is a small supporting signal; it is not
    # product-level dense retrieval.
    "BUYING": {"structured": 1.00, "semantic": 0.20, "dense": 0.00, "bm25": 1.00},
    # Browsing is fused by rank below, not by adding cosine and BM25 values.
    "BROWSING": {"structured": 0.00, "semantic": 0.00, "dense": 1.00, "bm25": 1.00},
}
# Retain the public name for integrations that used it for the former small
# semantic contribution.  Product-level dense retrieval is Browsing-only.
DENSE_SCORE_WEIGHT = MODE_SCORE_WEIGHTS["BUYING"]["semantic"]
BM25_SCORE_WEIGHT = MODE_SCORE_WEIGHTS["BUYING"]["bm25"]

RRF_K = 60
BROWSING_DENSE_TOP_K = 100
BROWSING_BM25_TOP_K = 100
BROWSING_FUSED_POOL_K = 50
BROWSING_MMR_LAMBDA = 0.80

# Rating tie-breaker (INSTRUCTION.md section 1).  The structured matcher scores
# most candidates identically -- 9.6 of every 10 returned candidates share one
# score -- so without a continuous term the top-10 order is catalog position.
# The catalog's own average_rating supplies that term.
RATING_SCALE = 5.0
NEUTRAL_NORMALIZED_RATING = 0.5
CRITICAL_USER_RATING_THRESHOLD = 3.5
RATING_BOOST_WEIGHT = 0.15
RATING_DEFAULT_WEIGHT = 0.02


def _env_flag(name: str, *, default: bool) -> bool:
    """Read a small boolean runtime switch without making configuration mandatory."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _normalise_browsing_retrieval_mode(value: object | None) -> str:
    """Resolve the Browsing retrieval arm without changing the default path."""

    configured = str(
        value
        if value is not None
        else os.environ.get(BROWSING_RETRIEVAL_MODE_ENV, "hybrid")
    ).strip().casefold().replace("-", "_")
    aliases = {
        "dense": "qwen_dense",
        "qwen": "qwen_dense",
        "qwen_dense_only": "qwen_dense",
    }
    mode = aliases.get(configured, configured)
    if mode not in BROWSING_RETRIEVAL_MODES:
        raise ValueError(
            f"unsupported Browsing retrieval mode {value!r}; "
            f"expected one of {', '.join(BROWSING_RETRIEVAL_MODES)}"
        )
    return mode


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
    semantic_score: float | None = None,
    browsing_fusion_score: float | None = None,
) -> float:
    if mode == "BROWSING" and browsing_fusion_score is not None:
        return float(browsing_fusion_score)
    weights = MODE_SCORE_WEIGHTS[mode]
    canonical_score = dense_score if semantic_score is None else semantic_score
    return float(
        weights["structured"] * structured_score
        + weights["dense"] * dense_score
        + weights.get("semantic", 0.0) * canonical_score
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
    fusion_score: float = 0.0
    mmr_score: float | None = None
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
            "fusion_score": self.fusion_score,
            "mmr_score": self.mmr_score,
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
        product_query_encoder: QueryEncoder | None = None,
        layer2_artifact_dir: str | Path | None = None,
        layer2_weights: Mapping[str, float] | None = None,
        product_embedding_artifact_dir: str | Path | None = None,
        browsing_retrieval_mode: str | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.query_encoder = query_encoder
        self.product_query_encoder = product_query_encoder
        self.browsing_retrieval_mode = _normalise_browsing_retrieval_mode(
            browsing_retrieval_mode
        )
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
        self.bm25_index: BM25Index | None = None
        self.bm25_query_compiler = BM25QueryCompiler()
        # The per-slot BM25 path is the default. Keeping an explicit raw-query
        # switch makes before/after lexical comparisons possible without
        # changing the BM25 index itself.
        self.use_slot_bm25_groups = _env_flag(
            "SHOPPING_BM25_SLOT_GROUPS",
            default=True,
        )
        self.bm25_state = "loading"
        self.bm25_error: str | None = None
        self.bm25_build_seconds: float | None = None
        try:
            self.bm25_index = BM25Index(
                self.product_by_asin,
                self._catalog_order,
            )
            self.bm25_state = "ready"
            self.bm25_build_seconds = self.bm25_index.build_seconds
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            self.bm25_state = "unavailable"
            self.bm25_error = f"BM25 index unavailable: {exc}"
            print(f"[retrieval] {self.bm25_error}", flush=True)
        self.embedding_matrix: Any = None
        self.embedding_asins: tuple[str, ...] = ()
        self._embedding_norms: Any = None
        self._load_embeddings(embeddings_path, metadata_path)
        self.layer2_index: Layer2EmbeddingIndex | None = None
        self._layer2_encoder_compatible = False
        self.layer2_compatibility_error: str | None = None
        self._load_layer2(layer2_artifact_dir)
        self.product_embedding_index: V5ProductEmbeddingIndex | None = None
        self._product_embedding_encoder_compatible = False
        self.product_embedding_compatibility_error: str | None = None
        self._load_v5_product_embeddings(product_embedding_artifact_dir)

    @property
    def valid_asins(self) -> frozenset[str]:
        return frozenset(self.product_by_asin)

    @property
    def has_dense_index(self) -> bool:
        return bool(
            self.product_embedding_index is not None
            and self.product_embedding_index.asins
        ) or bool(self.layer2_index is not None and self.layer2_index.asins) or (
            self.embedding_matrix is not None and bool(self.embedding_asins)
        )

    @property
    def product_dense_available(self) -> bool:
        """Whether the V5 product-card query/vector pair is usable."""

        encoder = self.product_query_encoder
        return bool(
            self.product_embedding_index is not None
            and self._product_embedding_encoder_compatible
            and (
                callable(encoder)
                or any(
                    callable(getattr(encoder, name, None))
                    for name in ("embed_query", "encode", "embed_documents", "embed")
                )
            )
        )

    def _load_layer2(self, artifact_dir: str | Path | None) -> None:
        if artifact_dir is None:
            # The legacy direct-catalog Layer 2/Jina artifact is opt-in.  The
            # active product-level semantic path is the V5 product-card index;
            # do not accidentally turn an older artifact into the Browsing
            # dense signal merely because a caller supplied another encoder.
            return
        candidates = (Path(artifact_dir),)
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

    @staticmethod
    def _local_product_model_path() -> str | None:
        configured = os.environ.get(PRODUCT_EMBEDDING_MODEL_ENV, "").strip()
        if configured:
            return configured
        for path in DEFAULT_PRODUCT_EMBEDDING_MODEL_PATHS:
            if path.is_dir():
                return path.as_posix()
        return None

    def _load_v5_product_embeddings(
        self,
        artifact_dir: str | Path | None,
    ) -> None:
        if artifact_dir is None:
            candidates = DEFAULT_V5_PRODUCT_ARTIFACT_PATHS
        else:
            candidates = (Path(artifact_dir),)
        expected_asins = tuple(self._catalog_order)
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                index = load_v5_product_embedding_index(
                    candidate,
                    expected_asins=expected_asins,
                )
            except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.product_embedding_compatibility_error = (
                    f"unable to load V5 product embedding artifact {candidate}: {exc}"
                )
                continue
            self.product_embedding_index = index
            if self.product_query_encoder is None:
                model_path = self._local_product_model_path()
                if model_path is not None:
                    try:
                        from product_embeddings.pipeline import load_local_sentence_transformer

                        self.product_query_encoder = load_local_sentence_transformer(
                            model_path,
                            trust_remote_code=True,
                        )
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                        self.product_embedding_compatibility_error = (
                            "V5 product query encoder could not be loaded locally: "
                            f"{exc}"
                        )
                else:
                    self.product_embedding_compatibility_error = (
                        f"set {PRODUCT_EMBEDDING_MODEL_ENV} to a local "
                        "Qwen3-Embedding-0.6B model directory"
                    )
            self._product_embedding_encoder_compatible = (
                self._validate_product_embedding_encoder(index)
            )
            return

    def _validate_product_embedding_encoder(
        self,
        index: V5ProductEmbeddingIndex,
    ) -> bool:
        encoder = self.product_query_encoder
        if encoder is None:
            if self.product_embedding_compatibility_error is None:
                self.product_embedding_compatibility_error = (
                    "V5 product query encoder is not configured"
                )
            return False
        expected_model = index.manifest.get(
            "embedding_model",
            index.manifest.get("model"),
        )
        if not expected_model or not embedding_models_compatible(
            V5_PRODUCT_MODEL,
            expected_model,
        ):
            self.product_embedding_compatibility_error = (
                "V5 product embedding artifact must use the required Qwen model: "
                f"artifact={expected_model!r}, required={V5_PRODUCT_MODEL!r}"
            )
            return False
        actual_model = _encoder_model_id(encoder)
        if not expected_model or not actual_model or not embedding_models_compatible(
            expected_model,
            actual_model,
        ):
            self.product_embedding_compatibility_error = (
                "V5 product embedding model does not match the query encoder: "
                f"artifact={expected_model!r}, encoder={actual_model!r}"
            )
            return False
        actual_dimension = _encoder_dimension(encoder)
        if actual_dimension is not None and actual_dimension != index.dimension:
            self.product_embedding_compatibility_error = (
                "V5 product embedding dimension does not match the query encoder: "
                f"artifact={index.dimension}, encoder={actual_dimension}"
            )
            return False
        if not any(
            callable(getattr(encoder, name, None))
            for name in ("embed_query", "encode", "embed_documents", "embed")
        ) and not callable(encoder):
            self.product_embedding_compatibility_error = (
                "V5 product query encoder has no supported embedding method"
            )
            return False
        self.product_embedding_compatibility_error = None
        return True

    @property
    def dense_available(self) -> bool:
        if self.product_embedding_index is not None:
            # A V5 product artifact must use its matching product encoder; a
            # BGE canonical encoder or a legacy product encoder is never a
            # substitute for it.
            return self.product_dense_available
        encoder = self.query_encoder
        if self.layer2_index is not None and not self._layer2_encoder_compatible:
            return False
        return self.has_dense_index and (
            callable(encoder)
            or any(
                callable(getattr(encoder, name, None))
                for name in ("embed_query", "encode", "embed_documents", "embed")
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

    def _canonical_scores(
        self,
        asins: Iterable[str],
        constraints: object | None,
    ) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
        """Score product facts against BGE canonical expansion state.

        Each accepted semantic value contributes its retained cosine
        similarity only when the product contains that canonical value. The
        accumulated points are sparse canonical evidence, not product-vector
        dense retrieval. Exact structured matching is calculated separately.
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

    # Compatibility name for integrations that used the old terminology.
    def _semantic_scores(
        self,
        asins: Iterable[str],
        constraints: object | None,
    ) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
        return self._canonical_scores(asins, constraints)

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

    def _dense_scores(
        self,
        query_text: str,
        *,
        browsing_query_text: str | None = None,
    ) -> dict[str, float]:
        if self.product_embedding_index is not None:
            # V5 product cards are built from structured facts.  Pair them
            # with the active structured state at query time instead of the
            # transcript/retrieval history, which can contain overridden or
            # otherwise stale preferences.  The legacy dense branches below
            # intentionally retain their existing query contract.
            product_query_text = (
                browsing_query_text
                if browsing_query_text is not None
                else query_text
            )
            if not self.product_dense_available or not product_query_text.strip():
                return {}
            try:
                query = self._query_embedding(
                    format_qwen_query(product_query_text),
                    self.product_embedding_index.dimension,
                    encoder=self.product_query_encoder,
                )
                if query is None:
                    return {}
                scores = self.product_embedding_index.score_all(query)
                return {
                    asin: float(score)
                    for asin, score in zip(self.product_embedding_index.asins, scores)
                }
            except (TypeError, ValueError, RuntimeError):
                return {}

        if self.query_encoder is None:
            return {}
        if self.layer2_index is not None:
            try:
                query = self._query_embedding(
                    query_text,
                    self.layer2_index.dimension,
                    encoder=self.query_encoder,
                )
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
        constraints: object | None = None,
        semantic_constraints: object | None = None,
    ) -> dict[str, float]:
        """Return BM25 scores with canonical BGE expansions.

        This is the expanded lexical signal for both Buying and Browsing.
        Browsing combines it with product-vector retrieval through rank fusion;
        the lexical expansion remains a separate retrieval signal.
        """

        if self.bm25_index is None:
            return {}
        try:
            if not self.use_slot_bm25_groups:
                return self._raw_bm25_scores(query_text, eligible_asins)

            group_specs = self.bm25_query_compiler.compile_group_specs(
                constraints,
                semantic_constraints,
            )
            raw_scores = self._raw_bm25_scores(query_text, eligible_asins)
            if not groups and not raw_scores:
                return {}

            # Include the raw current-goal query even when no structured slot
            # has been extracted. Expanded slot queries are additional lexical
            # evidence, not a replacement for the user's words.
            query_groups: list[dict[str, float]] = []
            if raw_scores:
                query_groups.append(raw_scores)
            for group in group_specs.values():
                scores = self.bm25_index.search(
                    group.phrases,
                    allowed_asins=eligible_asins,
                    fields=group.fields,
                )
                if scores:
                    query_groups.append(scores)

            # Each query is normalized independently before fusion. SQLite's
            # raw BM25 scores are only comparable within the same query; a
            # per-query peak normalization gives every active query one equal
            # contribution while retaining the BGE-expanded slot groups.
            normalized_by_group: list[dict[str, float]] = []
            for scores in query_groups:
                peak = max(
                    (float(score) for score in scores.values() if math.isfinite(float(score))),
                    default=0.0,
                )
                if peak <= 0.0:
                    normalized_by_group.append({})
                    continue
                normalized_by_group.append(
                    {
                        str(asin): min(1.0, max(0.0, float(score) / peak))
                        for asin, score in scores.items()
                        if math.isfinite(float(score)) and float(score) > 0.0
                    }
                )

            group_count = len(normalized_by_group)
            if group_count == 0:
                return {}
            fused: dict[str, float] = {}
            for group_scores in normalized_by_group:
                for asin, score in group_scores.items():
                    fused[asin] = fused.get(asin, 0.0) + score / group_count
            return fused
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
            return {}

    def _raw_bm25_scores(
        self,
        query_text: str,
        eligible_asins: Collection[str],
    ) -> dict[str, float]:
        """Search the current-goal text without canonical slot expansion."""

        if self.bm25_index is None or not str(query_text).strip():
            return {}
        try:
            return self.bm25_index.search(
                query_text,
                allowed_asins=eligible_asins,
            )
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
            return {}

    def debug_bm25(
        self,
        query_text: str,
        constraints: object | None = None,
        semantic_constraints: object | None = None,
        *,
        eligible_asins: Collection[str] | None = None,
        target_asin: str | None = None,
        expanded: bool = True,
    ) -> dict[str, Any]:
        """Describe raw and per-slot BM25 searches for evaluator diagnostics.

        This method only observes the same search calls used by retrieval. It
        never changes candidate state and is intentionally kept outside the
        Agent path so target ranks cannot affect production ranking.
        """

        if self.bm25_index is None:
            return {
                "available": False,
                "raw": None,
                "constraints": {},
            }
        allowed = set(
            self._catalog_order if eligible_asins is None else eligible_asins
        )
        target = str(target_asin).strip() if target_asin is not None else None

        def describe(
            label: str,
            search_text: str | tuple[str, ...],
            *,
            display_query: str | None = None,
            fields: tuple[str, ...] | None = None,
        ) -> dict[str, Any]:
            scores = self.bm25_index.search(
                search_text,
                allowed_asins=allowed,
                fields=fields,
            )
            rank_map = self._rank_map(
                scores,
                allowed,
                top_k=len(allowed),
                positive_only=True,
            )
            target_score = None
            if target:
                raw_score = scores.get(target)
                if raw_score is not None and math.isfinite(float(raw_score)):
                    target_score = float(raw_score)
            return {
                "label": label,
                "query": display_query if display_query is not None else search_text,
                "expression": self.bm25_index.query_expression(
                    search_text,
                    fields=fields,
                ),
                "fields": list(fields) if fields is not None else None,
                "match_count": len(scores),
                "target_rank": rank_map.get(target) if target else None,
                "target_score": target_score,
            }

        result: dict[str, Any] = {
            "available": True,
            "raw": describe("raw", str(query_text or "")),
            "constraints": {},
        }
        if expanded:
            group_specs = self.bm25_query_compiler.compile_group_specs(
                constraints,
                semantic_constraints,
            )
            result["constraints"] = {
                field_name: describe(
                    field_name,
                    group.phrases,
                    display_query=group.query_text,
                    fields=group.fields,
                )
                for field_name, group in group_specs.items()
            }
        return result

    def _query_embedding(
        self,
        query_text: str,
        dimension: int,
        *,
        encoder: object | None = None,
    ) -> Any:
        """Encode one query with the shared runtime encoder and validate it."""
        import numpy as np

        encoder = self.query_encoder if encoder is None else encoder
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
        semantic_score: float = 0.0,
        fusion_score: float = 0.0,
        mmr_score: float | None = None,
        ranking_override: float | None = None,
    ) -> Candidate:
        violated = tuple(
            f"{field_name}:required"
            for field_name in constraint_fields
            if field_name not in matched_fields
        )
        score = (
            float(ranking_override)
            if ranking_override is not None
            else _final_score(
                mode,
                structured_score,
                dense_score,
                bm25_score,
                semantic_score,
                browsing_fusion_score=fusion_score if mode == "BROWSING" else None,
            )
        )
        rating = self.rating_lookup.get(asin)
        ranking_score = float(score + w_rating * normalized_rating(rating))
        return Candidate(
            parent_asin=asin,
            score=ranking_score,
            dense_score=float(dense_score),
            constraint_score=float(structured_score),
            matched_constraints=matched_constraints,
            violated_constraints=violated,
            retrieval_mode=mode,
            attributes=self.product_by_asin[asin].facts,
            semantic_score=float(semantic_score),
            matched_semantic_constraints=matched_semantic_constraints,
            price=self.product_by_asin[asin].price,
            bm25_score=float(bm25_score),
            fusion_score=float(fusion_score),
            mmr_score=(None if mmr_score is None else float(mmr_score)),
            rating=rating,
            ranking_score=ranking_score,
        )

    def _rank_map(
        self,
        scores: Mapping[str, float],
        eligible_asins: Collection[str],
        *,
        top_k: int,
        positive_only: bool = False,
    ) -> dict[str, int]:
        """Return deterministic one-based ranks for one retrieval signal."""

        eligible = set(eligible_asins)
        scored: list[str] = []
        for asin in eligible:
            try:
                score = float(scores.get(asin, 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or (positive_only and score <= 0.0):
                continue
            scored.append(asin)
        scored.sort(
            key=lambda asin: (
                -float(scores.get(asin, 0.0)),
                self.product_by_asin[asin].catalog_order,
            )
        )
        return {
            asin: rank
            for rank, asin in enumerate(scored[: max(0, int(top_k))], 1)
        }

    @staticmethod
    def _rrf_scores(*rank_maps: Mapping[str, int]) -> dict[str, float]:
        """Fuse ranked lists without comparing their unrelated raw scores."""

        fused: dict[str, float] = {}
        for rank_map in rank_maps:
            for asin, rank in rank_map.items():
                fused[asin] = fused.get(asin, 0.0) + 1.0 / (RRF_K + int(rank))
        return fused

    def _mmr_rank(
        self,
        pool: Collection[str],
        fusion_scores: Mapping[str, float],
        *,
        limit: int,
        w_rating: float,
    ) -> tuple[list[str], dict[str, float]]:
        """Apply Browsing-only MMR using V5 product-card cosine similarity."""

        ordered_pool = list(pool)
        index = self.product_embedding_index
        if index is None or not self.product_dense_available:
            ordered_pool.sort(
                key=lambda asin: (
                    -float(fusion_scores.get(asin, 0.0)),
                    -w_rating * normalized_rating(self.rating_lookup.get(asin)),
                    self.product_by_asin[asin].catalog_order,
                )
            )
            return ordered_pool[:limit], {
                asin: float(fusion_scores.get(asin, 0.0)) for asin in ordered_pool
            }

        peak = max((float(fusion_scores.get(asin, 0.0)) for asin in ordered_pool), default=0.0)
        if peak <= 0.0:
            peak = 1.0
        remaining = set(ordered_pool)
        selected: list[str] = []
        mmr_scores: dict[str, float] = {}
        while remaining and len(selected) < max(0, int(limit)):
            best_asin: str | None = None
            best_key: tuple[float, float, float, int] | None = None
            for asin in remaining:
                relevance = max(0.0, float(fusion_scores.get(asin, 0.0))) / peak
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        0.0,
                        max(
                            (index.similarity(asin, previous) for previous in selected),
                            default=0.0,
                        ),
                    )
                mmr = BROWSING_MMR_LAMBDA * relevance - (
                    1.0 - BROWSING_MMR_LAMBDA
                ) * redundancy
                mmr_scores[asin] = mmr
                effective = mmr + w_rating * normalized_rating(
                    self.rating_lookup.get(asin)
                )
                key = (
                    effective,
                    mmr,
                    float(fusion_scores.get(asin, 0.0)),
                    -self.product_by_asin[asin].catalog_order,
                )
                if best_key is None or key > best_key:
                    best_asin = asin
                    best_key = key
            if best_asin is None:
                break
            remaining.remove(best_asin)
            selected.append(best_asin)
        return selected, mmr_scores

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
        candidate_pool_only: bool = False,
        debug_full_ranking: bool = False,
    ) -> list[Candidate]:
        """Return one deterministic candidate ranking for either mode.

        The shared ranker accumulates exact structured matches from the
        inverted indexes. Non-budget fields are scored softly; an active
        budget is the only eligibility filter.

        ``candidate_pool_only`` is used by clarification analysis. It returns
        a broad ranked pool without applying Browsing's small recommendation
        fusion/MMR pool cap. The normal recommendation path remains unchanged.

        ``debug_full_ranking`` is used only by evaluator diagnostics. It keeps
        the production hybrid/MMR ordering for the normal recommendation pool
        and appends the remaining RRF-ranked tail so a target's diagnostic
        position can be inspected even when it is outside the recommendation
        pool. It must not be enabled by the normal Agent path.
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

        canonical_scores, semantic_labels = self._canonical_scores(
            eligible_asins,
            semantic_constraints,
        )
        dense_scores: dict[str, float] = {}
        browsing_dense_only = (
            mode == "BROWSING" and self.browsing_retrieval_mode == "qwen_dense"
        )
        if mode == "BROWSING" and self.dense_available:
            # Qwen Browsing uses only the current active slot values.  This is
            # deliberately separate from ``retrieval_query_text`` used by
            # the lexical route so old conversational wording cannot leak
            # into the product-card dense query.
            browsing_query_text = build_browsing_query(constraints)
            dense_scores = self._dense_scores(
                query_text,
                browsing_query_text=browsing_query_text,
            )
        # The default hybrid Browsing arm keeps BM25 separate from product
        # dense retrieval and fuses the two by rank. The dense-only experiment
        # deliberately bypasses this lexical route below.
        bm25_scores = (
            {}
            if browsing_dense_only
            else self._bm25_scores(
                query_text,
                eligible_set,
                constraints,
                semantic_constraints,
            )
        )

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
        ranking_rating_weight = 0.0 if browsing_dense_only else w_rating

        def _rank_key(
            base: Callable[[str], float]
        ) -> Callable[[str], tuple[float, int]]:
            """The spec's S(x), descending, with catalog order as last resort."""

            def key(asin: str) -> tuple[float, int]:
                bonus = ranking_rating_weight * normalized_rating(
                    self.rating_lookup.get(asin)
                )
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

        base_scores: dict[str, float] = {}
        fusion_scores: dict[str, float] = {}
        ranking_overrides: dict[str, float] = {}
        mmr_scores: dict[str, float] = {}
        if mode == "BUYING":
            # BM25 is part of the Buying score even when no product-vector
            # signal is available.  Canonical BGE posting points are a small
            # supporting signal; they are deliberately not called dense here.
            base_scores = {
                asin: _final_score(
                    mode,
                    structured_score(asin),
                    0.0,
                    bm25_scores.get(asin, 0.0),
                    canonical_scores.get(asin, 0.0),
                )
                for asin in eligible_asins
            }
            ranked_asins = _select(eligible_asins, base_scores.__getitem__)
        else:
            if browsing_dense_only:
                # Experiment arm: expose the raw Qwen product-card ranking.
                # BM25, RRF, MMR, and rating tie-breaking are intentionally
                # excluded so this arm measures the embedding artifact itself.
                ranking_overrides = {
                    asin: float(dense_scores.get(asin, 0.0))
                    for asin in eligible_asins
                }
                ranked_asins = _select(eligible_asins, ranking_overrides.__getitem__)
            elif candidate_pool_only:
                # Clarification needs a broad distribution of product facts,
                # not the small final recommendation pool. Retrieve the same
                # dense/BM25 RRF union with enough ranks to fill ``limit`` and
                # leave MMR to the normal recommendation path.
                dense_ranks = self._rank_map(
                    dense_scores,
                    eligible_asins,
                    top_k=limit,
                )
                bm25_ranks = self._rank_map(
                    bm25_scores,
                    eligible_asins,
                    top_k=limit,
                    positive_only=True,
                )
                fusion_scores = self._rrf_scores(dense_ranks, bm25_ranks)
                if fusion_scores:
                    pool = [asin for asin in self._catalog_order if asin in fusion_scores]
                    pool.sort(
                        key=lambda asin: (
                            -float(fusion_scores.get(asin, 0.0)),
                            self.product_by_asin[asin].catalog_order,
                        )
                    )
                    ranked_asins = pool[:limit]
                    ranking_overrides.update(fusion_scores)
                else:
                    ranking_overrides = {
                        asin: structured_score(asin)
                        + MODE_SCORE_WEIGHTS["BUYING"].get("semantic", 0.0)
                        * canonical_scores.get(asin, 0.0)
                        for asin in eligible_asins
                    }
                    ranked_asins = _select(eligible_asins, ranking_overrides.__getitem__)
            else:
                dense_top_k = (
                    len(eligible_asins)
                    if debug_full_ranking
                    else BROWSING_DENSE_TOP_K
                )
                bm25_top_k = (
                    len(eligible_asins)
                    if debug_full_ranking
                    else BROWSING_BM25_TOP_K
                )
                dense_ranks = self._rank_map(
                    dense_scores,
                    eligible_asins,
                    top_k=dense_top_k,
                )
                bm25_ranks = self._rank_map(
                    bm25_scores,
                    eligible_asins,
                    top_k=bm25_top_k,
                    positive_only=True,
                )
                fusion_scores = self._rrf_scores(dense_ranks, bm25_ranks)
                if fusion_scores:
                    pool = [asin for asin in self._catalog_order if asin in fusion_scores]
                    pool.sort(
                        key=lambda asin: (
                            -float(fusion_scores.get(asin, 0.0)),
                            self.product_by_asin[asin].catalog_order,
                        )
                    )
                    if debug_full_ranking:
                        # The first production-sized slice is ranked exactly
                        # as it is for normal recommendations. The diagnostic
                        # tail uses the complete RRF order because production
                        # intentionally does not run MMR over the full catalog.
                        production_pool = pool[:BROWSING_FUSED_POOL_K]
                        if self.product_dense_available and dense_ranks:
                            production_ranked, mmr_scores = self._mmr_rank(
                                production_pool,
                                fusion_scores,
                                limit=len(production_pool),
                                w_rating=w_rating,
                            )
                            ranking_overrides.update(mmr_scores)
                        else:
                            production_ranked = self._select(
                                production_pool,
                                fusion_scores.__getitem__,
                            )
                            ranking_overrides.update(
                                {
                                    asin: fusion_scores[asin]
                                    for asin in production_pool
                                }
                            )
                        production_ids = set(production_ranked)
                        ranked_asins = production_ranked + [
                            asin for asin in pool if asin not in production_ids
                        ]
                    else:
                        pool = pool[:BROWSING_FUSED_POOL_K]
                        if self.product_dense_available and dense_ranks:
                            ranked_asins, mmr_scores = self._mmr_rank(
                                pool,
                                fusion_scores,
                                limit=limit,
                                w_rating=w_rating,
                            )
                            ranking_overrides.update(mmr_scores)
                        else:
                            ranking_overrides.update(fusion_scores)
                            ranked_asins = _select(pool, fusion_scores.__getitem__)
                else:
                    # Safe fallback when the product artifact and BM25 are
                    # both unavailable: retain the old structured/canonical
                    # evidence, but do not pretend that it is product dense
                    # retrieval.
                    ranking_overrides = {
                        asin: structured_score(asin)
                        + MODE_SCORE_WEIGHTS["BUYING"].get("semantic", 0.0)
                        * canonical_scores.get(asin, 0.0)
                        for asin in eligible_asins
                    }
                    ranked_asins = _select(eligible_asins, ranking_overrides.__getitem__)
        if not ranked_asins:
            ranked_asins = _select(eligible_asins, _zero)
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
                w_rating=ranking_rating_weight,
                semantic_score=canonical_scores.get(asin, 0.0),
                fusion_score=fusion_scores.get(asin, 0.0),
                mmr_score=mmr_scores.get(asin),
                ranking_override=(
                    ranking_overrides.get(asin)
                    if mode == "BROWSING"
                    else None
                ),
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
        """Return a diagnostic full ranking using the production scorer.

        The explicit debug flag lets normal retrieval retain its bounded
        Browsing recommendation pool while diagnostics can locate targets in
        the unfiltered retrieval ranking.
        """

        return self.retrieve(
            mode,
            query_text,
            constraints,
            semantic_constraints=semantic_constraints,
            limit=len(self.product_by_asin),
            excluded_asins=excluded_asins,
            apply_budget=apply_budget,
            user_prior_rating=user_prior_rating,
            debug_full_ranking=True,
        )


InMemoryRetriever = ProductRetriever


__all__ = [
    "CRITICAL_USER_RATING_THRESHOLD",
    "Candidate",
    "BM25_SCORE_WEIGHT",
    "BROWSING_BM25_TOP_K",
    "BROWSING_DENSE_TOP_K",
    "BROWSING_FUSED_POOL_K",
    "BROWSING_MMR_LAMBDA",
    "BROWSING_RETRIEVAL_MODE_ENV",
    "BROWSING_RETRIEVAL_MODES",
    "InMemoryRetriever",
    "RRF_K",
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
