"""Dependency-light in-memory product retrieval for the Layer 1/2 MVP.

The retriever consumes canonical product facts and optional direct Layer 2
field vectors. It does not parse user language and it does not maintain a
second lexical/BM25 product-search route. User text is only passed to an
injected compatible query encoder when dense artifacts are available; canonical
constraints drive the Buying filters and preference boosts drive Browsing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping

from dictionary.registry import normalize_text

from product_embeddings.layer2 import (
    Layer2EmbeddingIndex,
    load_layer2_embedding_index,
)
from starter.routing.constraints import CATEGORICAL_FIELDS


FACT_FIELDS = tuple(CATEGORICAL_FIELDS)
DEFAULT_FACT_PATHS = (
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
# relative weights, and the final structured score is normalized to [0, 1].
STRUCTURED_FIELD_WEIGHTS: dict[str, float] = {
    "category": 1.00,
    "price": 1.00,
    "brand": 0.90,
    "size": 0.80,
    "color": 0.70,
    "material": 0.70,
    "style": 0.30,
    "feature": 0.30,
    "use_case": 0.30,
}
DENSE_SCORE_WEIGHT = 0.20


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

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_asin": self.parent_asin,
            "score": self.score,
            "dense_score": self.dense_score,
            "constraint_score": self.constraint_score,
            "matched_constraints": list(self.matched_constraints),
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
        self._facts_by_asin, self._annotated_prices = self._load_fact_artifact(facts_path)
        self._load_catalog()
        self.embedding_matrix: Any = None
        self.embedding_asins: tuple[str, ...] = ()
        self._embedding_norms: Any = None
        self._load_embeddings(embeddings_path, metadata_path)
        self.layer2_index: Layer2EmbeddingIndex | None = None
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
            return

    @property
    def dense_available(self) -> bool:
        encoder = self.query_encoder
        return self.has_dense_index and (
            callable(encoder)
            or any(
                callable(getattr(encoder, name, None))
                for name in ("encode", "embed_documents", "embed")
            )
        )

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
                raw=dict(row),
            )
            self.product_by_asin[asin] = product
            self._catalog_order.append(asin)
            self.price_lookup[asin] = price
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

    def _query_embedding(self, query_text: str, dimension: int) -> Any:
        """Encode one query with the shared runtime encoder and validate it."""
        import numpy as np

        encoder = self.query_encoder
        if encoder is None:
            return None
        if hasattr(encoder, "encode"):
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
    ) -> Candidate:
        violated = tuple(
            f"{field_name}:required"
            for field_name in constraint_fields
            if field_name not in matched_fields
        )
        score = structured_score + DENSE_SCORE_WEIGHT * dense_score
        return Candidate(
            parent_asin=asin,
            score=float(score),
            dense_score=float(dense_score),
            constraint_score=float(structured_score),
            matched_constraints=matched_constraints,
            violated_constraints=violated,
            retrieval_mode=mode,
            attributes=self.product_by_asin[asin].facts,
        )

    def retrieve(
        self,
        mode: str,
        query_text: str,
        constraints: object,
        *,
        limit: int = 100,
        minimum_candidates: int = 50,
        excluded_asins: Collection[str] | None = None,
        apply_budget: bool = True,
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
        dense_scores = self._dense_scores(query_text) if self.dense_available else {}
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

        total_weight = sum(
            STRUCTURED_FIELD_WEIGHTS.get(field_name, 0.0)
            for field_name in constraint_fields
        )
        matched_weight: dict[str, float] = {}
        matched_fields: dict[str, set[str]] = {}
        matched_labels: dict[str, list[str]] = {}

        for field_name in constraint_fields:
            if field_name == "price":
                field_matches = price_match_asins & eligible_set
            else:
                field_matches: set[str] = set()
                for value in requested_by_field[field_name]:
                    indexed = self.inverted_index.get(field_name, {}).get(value, set())
                    if eligible_set is None:
                        field_matches.update(indexed)
                        for asin in indexed:
                            matched_labels.setdefault(asin, []).append(
                                f"{field_name}:{value}"
                            )
                    else:
                        field_matches.update(indexed & eligible_set)
                        for asin in indexed:
                            if asin in eligible_set:
                                matched_labels.setdefault(asin, []).append(
                                    f"{field_name}:{value}"
                                )
            field_weight = STRUCTURED_FIELD_WEIGHTS.get(field_name, 0.0)
            for asin in field_matches:
                matched_weight[asin] = matched_weight.get(asin, 0.0) + field_weight
                matched_fields.setdefault(asin, set()).add(field_name)

        def structured_score(asin: str) -> float:
            if total_weight <= 0.0:
                return 0.0
            return matched_weight.get(asin, 0.0) / total_weight

        if not constraint_fields and not dense_scores:
            ranked_asins = eligible_asins[:limit]
        elif not dense_scores:
            positive_asins = sorted(
                matched_weight,
                key=lambda asin: (
                    -structured_score(asin),
                    self.product_by_asin[asin].catalog_order,
                ),
            )
            ranked_asins = positive_asins[:limit]
            if len(ranked_asins) < limit:
                selected = set(ranked_asins)
                ranked_asins.extend(
                    asin for asin in eligible_asins if asin not in selected
                )
                ranked_asins = ranked_asins[:limit]
        else:
            ranked_asins = sorted(
                eligible_asins,
                key=lambda asin: (
                    -structured_score(asin),
                    -dense_scores.get(asin, 0.0),
                    self.product_by_asin[asin].catalog_order,
                ),
            )[:limit]

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

        return [
            self._candidate(
                asin,
                mode,
                dense_scores.get(asin, 0.0),
                structured_score(asin),
                labels_for(asin),
                matched_fields.get(asin, set()),
                constraint_fields,
            )
            for asin in ranked_asins
        ]

    def debug_rank_all(
        self,
        mode: str,
        query_text: str,
        constraints: object,
        *,
        excluded_asins: Collection[str] | None = None,
        apply_budget: bool = True,
    ) -> list[Candidate]:
        """Return the complete ranking using the production scorer."""

        return self.retrieve(
            mode,
            query_text,
            constraints,
            limit=len(self.product_by_asin),
            excluded_asins=excluded_asins,
            apply_budget=apply_budget,
        )


InMemoryRetriever = ProductRetriever


__all__ = [
    "Candidate",
    "InMemoryRetriever",
    "ProductRecord",
    "ProductRetriever",
    "SharedCandidate",
]
