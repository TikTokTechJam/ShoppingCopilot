"""Dependency-light in-memory product retrieval for Issues #13 and #15.

The retriever consumes canonical product facts and optional product vectors. It
does not parse user language and it does not maintain a second lexical/BM25
product-search route. User text is only passed to an injected compatible query
encoder when dense artifacts are available; canonical constraints drive the
Buying filters and preference boosts drive Browsing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from starter.routing.constraints import CATEGORICAL_FIELDS


FACT_FIELDS = tuple(CATEGORICAL_FIELDS)
DEFAULT_FACT_PATHS = (
    Path("data/annotations.jsonl"),
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


def _normalise_value(value: object) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


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

    # Raw catalog categories are useful only as a conservative artifact
    # fallback. The canonical annotation, when present, takes precedence.
    if "category" in facts:
        expanded: list[str] = []
        for value in facts["category"]:
            for item in (value, value[:-1] if value.endswith("s") and not value.endswith("ss") else ""):
                if item and item not in expanded:
                    expanded.append(item)
        facts["category"] = tuple(expanded)
    return facts


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
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.query_encoder = query_encoder
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

    @property
    def valid_asins(self) -> frozenset[str]:
        return frozenset(self.product_by_asin)

    @property
    def has_dense_index(self) -> bool:
        return self.embedding_matrix is not None and bool(self.embedding_asins)

    def _load_fact_artifact(
        self, facts_path: str | Path | None
    ) -> tuple[dict[str, Mapping[str, tuple[str, ...]]], dict[str, float | None]]:
        selected = Path(facts_path) if facts_path is not None else _first_existing(DEFAULT_FACT_PATHS)
        if selected is None:
            return {}, {}
        facts: dict[str, Mapping[str, tuple[str, ...]]] = {}
        prices: dict[str, float | None] = {}
        for row in _read_jsonl(selected):
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
            catalog_facts = _record_facts(row)
            facts = _merge_facts(catalog_facts, self._facts_by_asin.get(asin, {}))
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

    def _strict_pool(self, constraints: object, relaxed: set[str] | None = None) -> list[str]:
        relaxed = relaxed or set()
        active_fields = [
            field_name
            for field_name in self._constraint_fields(constraints)
            if field_name not in relaxed
        ]
        if not active_fields:
            return list(self._catalog_order)

        pool = set(self._catalog_order)
        for field_name in active_fields:
            if field_name == "price":
                pool = {asin for asin in pool if self._matches_field(asin, field_name, constraints)}
                continue
            requested = self._constraint_values(constraints, field_name)
            field_pool: set[str] = set()
            for value in requested:
                field_pool.update(self.inverted_index.get(field_name, {}).get(value, ()))
            pool.intersection_update(field_pool)
            if not pool:
                break
        return [asin for asin in self._catalog_order if asin in pool]

    def _relaxation_order(self, constraints: object) -> tuple[str, ...]:
        evidence_by_field: dict[str, list[tuple[str, float]]] = {}
        for item in getattr(constraints, "evidence", ()) or ():
            field_name = str(getattr(item, "attribute", ""))
            if field_name:
                method = str(getattr(item, "match_method", "exact"))
                confidence = float(getattr(item, "confidence", 1.0))
                evidence_by_field.setdefault(field_name, []).append((method, confidence))
        soft_order = {name: index for index, name in enumerate((
            "style", "use_case", "feature", "material", "color", "size", "brand", "category", "price",
        ))}
        fields = list(self._constraint_fields(constraints))
        return tuple(sorted(
            fields,
            key=lambda field_name: (
                field_name in {"category", "price"},
                min((confidence for _method, confidence in evidence_by_field.get(field_name, (("exact", 1.0),))), default=1.0),
                min((method != "semantic" for method, _confidence in evidence_by_field.get(field_name, (("exact", 1.0),))), default=True),
                soft_order.get(field_name, len(soft_order)),
                field_name,
            ),
        ))

    def _dense_scores(self, query_text: str) -> dict[str, float]:
        if not self.has_dense_index or self.query_encoder is None:
            return {}
        try:
            import numpy as np

            encoder = self.query_encoder
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
            elif callable(encoder):
                query_value = encoder(query_text)
            else:
                return {}
            query = np.asarray(query_value, dtype=np.float32)
            if query.ndim == 2 and query.shape[0] == 1:
                query = query[0]
            query = query.reshape(-1)
            if query.size != int(self.embedding_matrix.shape[1]):
                return {}
            norm = float(np.linalg.norm(query))
            if not math.isfinite(norm) or norm == 0.0:
                return {}
            scores = (self.embedding_matrix @ query) / (self._embedding_norms * norm)
            if not bool(np.isfinite(scores).all()):
                return {}
            return {asin: float(score) for asin, score in zip(self.embedding_asins, scores)}
        except (ImportError, TypeError, ValueError, RuntimeError):
            return {}

    def _candidate(
        self,
        asin: str,
        mode: str,
        dense_score: float,
        constraints: object,
        relaxed: set[str],
        total_fields: int,
    ) -> Candidate:
        matched = self._matched_labels(asin, constraints)
        matched_fields = sum(
            1 for field_name in self._constraint_fields(constraints)
            if self._matches_field(asin, field_name, constraints)
        )
        violated = tuple(
            f"{field_name}:{'required'}"
            for field_name in self._constraint_fields(constraints)
            if field_name not in relaxed and not self._matches_field(asin, field_name, constraints)
        )
        relaxed_labels = tuple(
            field_name
            for field_name in self._constraint_fields(constraints)
            if field_name in relaxed
        )
        constraint_score = matched_fields / total_fields if total_fields else 0.0
        score = dense_score + 0.25 * constraint_score if mode == "BUYING" else 0.85 * dense_score + 0.15 * constraint_score
        return Candidate(
            parent_asin=asin,
            score=float(score),
            dense_score=float(dense_score),
            constraint_score=float(constraint_score),
            matched_constraints=matched,
            violated_constraints=violated,
            relaxed_constraints=relaxed_labels,
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
    ) -> list[Candidate]:
        """Return one deterministic shared candidate pool for either mode."""

        limit = max(0, int(limit))
        if limit == 0 or not self.product_by_asin:
            return []
        mode = "BUYING" if str(mode).upper() == "BUYING" else "BROWSING"
        dense_scores = self._dense_scores(query_text)
        total_fields = len(self._constraint_fields(constraints))

        if mode == "BROWSING":
            asins = list(self._catalog_order)
            candidates = [
                self._candidate(asin, mode, dense_scores.get(asin, 0.0), constraints, set(), total_fields)
                for asin in asins
            ]
            return sorted(
                candidates,
                key=lambda item: (
                    -item.score,
                    -item.constraint_score,
                    self.product_by_asin[item.parent_asin].catalog_order,
                ),
            )[:limit]

        relaxed: set[str] = set()
        asins = self._strict_pool(constraints, relaxed)
        target = min(len(self._catalog_order), max(int(minimum_candidates), limit))
        relaxation_order = self._relaxation_order(constraints)
        for field_name in relaxation_order:
            if len(asins) >= target:
                break
            relaxed.add(field_name)
            asins = self._strict_pool(constraints, relaxed)

        if not asins:
            # A malformed/partial facts artifact must not turn into an invalid
            # response. Dense search is the last fallback and remains optional.
            asins = list(self._catalog_order)
            relaxed = set(relaxation_order)

        candidates = [
            self._candidate(asin, mode, dense_scores.get(asin, 0.0), constraints, relaxed, total_fields)
            for asin in asins
        ]
        return sorted(
            candidates,
            key=lambda item: (
                bool(item.violated_constraints),
                -item.score,
                -item.constraint_score,
                self.product_by_asin[item.parent_asin].catalog_order,
            ),
        )[:limit]


InMemoryRetriever = ProductRetriever


__all__ = [
    "Candidate",
    "InMemoryRetriever",
    "ProductRecord",
    "ProductRetriever",
    "SharedCandidate",
]
