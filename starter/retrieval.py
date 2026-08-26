"""In-memory product retrieval for the Buying and Browsing routes.

The retriever deliberately starts after user-language canonicalisation.  It
does not contain a BM25/FTS product path: structured facts drive Buying, and
optional product vectors drive Browsing.  Everything needed by later Agent
stages is represented by :class:`RetrievalCandidate`.

The catalog is required input.  Canonical facts and product embeddings are
optional local artifacts.  Malformed optional artifacts are ignored so that a
submission remains runnable with only ``data/catalog.jsonl``.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CANONICAL_FIELDS: tuple[str, ...] = (
    "category",
    "brand",
    "color",
    "material",
    "size",
    "style",
    "feature",
    "use_case",
)
PRICE_FIELDS = ("price_min", "price_max")
ALL_CONSTRAINT_FIELDS = CANONICAL_FIELDS + PRICE_FIELDS

# Semantic/soft preferences are relaxed before exact category, brand, and
# numeric budget constraints.  The order is intentionally fixed because the
# retriever has no authority to make a stochastic relaxation decision.
RELAXATION_ORDER: tuple[str, ...] = (
    "use_case",
    "style",
    "feature",
    "material",
    "color",
    "size",
    "brand",
    "category",
    "price_min",
    "price_max",
)

_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
def _valid_asin(value: object) -> str | None:
    """Return a usable catalog identifier, rejecting null/blank metadata."""

    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _canonical_text(value: object) -> str:
    """Normalize a stored canonical value without inventing semantics."""

    if not isinstance(value, str):
        return ""
    value = value.strip().casefold()
    value = re.sub(r"[\s-]+", "_", value)
    return value


def _value_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[object]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    result: list[str] = []
    for item in values:
        normalized = _canonical_text(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


def _price(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _NUMBER_RE.search(str(value).replace(",", ""))
        if match is None:
            return None
        try:
            number = float(match.group(0))
        except ValueError:
            return None
    return number if math.isfinite(number) else None


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    """Yield object records, skipping malformed optional-artifact lines."""

    try:
        handle = path.open(encoding="utf-8")
    except (OSError, UnicodeError):
        return
    try:
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if isinstance(record, Mapping):
                    yield record
    except (OSError, UnicodeError):
        return


def _metadata_rows(payload: object) -> list[Mapping[str, Any]] | None:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        return None
    return [row for row in rows if isinstance(row, Mapping)] if len(rows) == sum(
        isinstance(row, Mapping) for row in rows
    ) else None


@dataclass(frozen=True)
class RetrievalCandidate:
    """Shared candidate contract consumed by later Agent components."""

    parent_asin: str
    retrieval_mode: str
    dense_score: float = 0.0
    constraint_score: float = 0.0
    matched_constraints: tuple[str, ...] = ()
    violated_constraints: tuple[str, ...] = ()
    relaxed_constraints: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_asin": self.parent_asin,
            "retrieval_mode": self.retrieval_mode,
            "dense_score": self.dense_score,
            "constraint_score": self.constraint_score,
            "matched_constraints": list(self.matched_constraints),
            "violated_constraints": list(self.violated_constraints),
            "relaxed_constraints": list(self.relaxed_constraints),
        }

    to_dict = as_dict

    # Mapping-style convenience keeps the representation easy to hand to
    # response builders without making the dataclass itself mutable.
    def __getitem__(self, key: str) -> object:
        return self.as_dict()[key]


class RetrievalIndex:
    """Load catalog/facts once and expose deterministic route-specific search."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        facts_path: str | Path | None = None,
        *,
        canonical_facts_path: str | Path | None = None,
        embeddings_path: str | Path | None = None,
        embedding_metadata_path: str | Path | None = None,
        product_embeddings_path: str | Path | None = None,
        product_embedding_metadata_path: str | Path | None = None,
        query_encoder: Callable[[str], object] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        if facts_path is not None and canonical_facts_path is not None:
            raise TypeError("pass facts_path or canonical_facts_path, not both")
        self.facts_path = Path(facts_path or canonical_facts_path) if (
            facts_path is not None or canonical_facts_path is not None
        ) else self._find_facts_path(self.catalog_path)
        if embeddings_path is not None and product_embeddings_path is not None:
            raise TypeError("pass embeddings_path or product_embeddings_path, not both")
        if embedding_metadata_path is not None and product_embedding_metadata_path is not None:
            raise TypeError(
                "pass embedding_metadata_path or product_embedding_metadata_path, not both"
            )
        embeddings_path = embeddings_path or product_embeddings_path
        embedding_metadata_path = embedding_metadata_path or product_embedding_metadata_path
        self.product_by_asin: dict[str, dict[str, Any]] = {}
        self.catalog_order: tuple[str, ...] = ()
        self.facts_by_asin: dict[str, dict[str, Any]] = {}
        self.inverted_indexes: dict[str, dict[str, set[str]]] = {
            field: defaultdict(set) for field in CANONICAL_FIELDS
        }
        self.price_lookup: dict[str, float | None] = {}
        self._catalog_position: dict[str, int] = {}
        self._embeddings: Any = None
        self._embedding_asins: tuple[str, ...] = ()
        self._embedding_rows: dict[str, int] = {}
        self._query_encoder = query_encoder

        self._load_catalog()
        self._load_facts()
        self._build_indexes()
        self._load_embeddings(embeddings_path, embedding_metadata_path)

    @classmethod
    def load(cls, catalog_path: str | Path = "data/catalog.jsonl", **kwargs: Any) -> "RetrievalIndex":
        """Named constructor for callers that treat the index as an artifact."""

        return cls(catalog_path, **kwargs)

    @staticmethod
    def _find_facts_path(catalog_path: Path) -> Path | None:
        roots = (Path("."), catalog_path.parent)
        relative_paths = (
            Path("data/derived/catalog_facts/catalog_facts.jsonl"),
            Path("data/derived/annotations/v2/annotations.jsonl"),
            Path("data/derived/annotations/v1/annotations.jsonl"),
        )
        for root in roots:
            for relative_path in relative_paths:
                candidate = root / relative_path
                if candidate.is_file():
                    return candidate
        # Also permit a catalog placed directly beside generated artifacts.
        for candidate in (
            catalog_path.parent / "catalog_facts.jsonl",
            catalog_path.parent / "annotations.jsonl",
        ):
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _embedding_candidates(catalog_path: Path) -> tuple[tuple[Path, Path], ...]:
        roots = (Path("."), catalog_path.parent, Path("data"), Path("data/derived"))
        return tuple(
            pair
            for root in roots
            for pair in (
                (root / "product_embeddings.npy", root / "product_embedding_metadata.json"),
                (
                    root / "embeddings" / "product_embeddings.npy",
                    root / "embeddings" / "product_embedding_metadata.json",
                ),
            )
        )

    def _load_catalog(self) -> None:
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"catalog not found: {self.catalog_path}")
        order: list[str] = []
        for record in _read_jsonl(self.catalog_path):
            asin = _valid_asin(record.get("parent_asin"))
            if asin is None or asin in self.product_by_asin:
                continue
            product = dict(record)
            product["parent_asin"] = asin
            self.product_by_asin[asin] = product
            order.append(asin)
        self.catalog_order = tuple(order)
        self._catalog_position = {asin: position for position, asin in enumerate(order)}

    def _load_facts(self) -> None:
        if self.facts_path is None or not self.facts_path.is_file():
            return
        for record in _read_jsonl(self.facts_path):
            asin = _valid_asin(record.get("parent_asin"))
            if asin is None or asin not in self.product_by_asin or asin in self.facts_by_asin:
                continue
            nested = record.get("facts")
            source = nested if isinstance(nested, Mapping) else record
            facts: dict[str, Any] = {}
            for field in CANONICAL_FIELDS:
                if field not in source:
                    continue
                if field == "brand":
                    values = _value_tuple(source.get(field))
                    if values:
                        facts[field] = values[0]
                else:
                    values = _value_tuple(source.get(field))
                    if values:
                        facts[field] = values
            raw_price = record.get("price", source.get("price"))
            if raw_price is not None:
                parsed_price = _price(raw_price)
                if parsed_price is not None:
                    facts["price"] = parsed_price
            if facts:
                self.facts_by_asin[asin] = facts

    def _facts_for(self, asin: str) -> dict[str, Any]:
        facts = self.facts_by_asin.get(asin)
        if facts is not None:
            return facts
        product = self.product_by_asin[asin]
        fallback: dict[str, Any] = {}
        for field in CANONICAL_FIELDS:
            source = product.get(field)
            if source is None and field == "category":
                source = product.get("categories")
            if field == "brand" and source is None:
                source = product.get("store")
            values = _value_tuple(source)
            if values:
                fallback[field] = values[0] if field == "brand" else values
        return fallback

    def _build_indexes(self) -> None:
        for asin in self.catalog_order:
            facts = self._facts_for(asin)
            for field in CANONICAL_FIELDS:
                values = _value_tuple(facts.get(field))
                for value in values:
                    self.inverted_indexes[field][value].add(asin)
            product_price = _price(self.product_by_asin[asin].get("price"))
            self.price_lookup[asin] = _price(facts.get("price"))
            if self.price_lookup[asin] is None:
                self.price_lookup[asin] = product_price

    def _load_embeddings(
        self,
        embeddings_path: str | Path | None,
        embedding_metadata_path: str | Path | None,
    ) -> None:
        if embeddings_path is None and embedding_metadata_path is None:
            for matrix_path, metadata_path in self._embedding_candidates(self.catalog_path):
                if matrix_path.is_file() or metadata_path.is_file():
                    embeddings_path, embedding_metadata_path = matrix_path, metadata_path
                    break
        if embeddings_path is None or embedding_metadata_path is None:
            return
        matrix_path = Path(embeddings_path)
        metadata_path = Path(embedding_metadata_path)
        if not matrix_path.is_file() or not metadata_path.is_file():
            return
        try:
            import numpy as np

            matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
            with metadata_path.open(encoding="utf-8") as handle:
                rows = _metadata_rows(json.load(handle))
            if rows is None or matrix.ndim != 2 or matrix.shape[0] != len(rows):
                return
            asins: list[str] = []
            row_numbers: dict[str, int] = {}
            for position, row in enumerate(rows):
                asin = _valid_asin(row.get("parent_asin", row.get("asin")))
                if asin is None or asin not in self.product_by_asin or asin in row_numbers:
                    return
                declared = row.get("row", row.get("index"))
                if declared is not None:
                    try:
                        if int(declared) != position:
                            return
                    except (TypeError, ValueError):
                        return
                asins.append(asin)
                row_numbers[asin] = position
            if not np.isfinite(matrix).all():
                return
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms <= 0):
                return
            self._embeddings = matrix / norms
            self._embedding_asins = tuple(asins)
            self._embedding_rows = row_numbers
        except (ImportError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            # NumPy and vector files are deliberately optional.  A bad pair
            # must not make structured retrieval unavailable.
            self._embeddings = None
            self._embedding_asins = ()
            self._embedding_rows = {}

    @property
    def products(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.product_by_asin[asin] for asin in self.catalog_order)

    @property
    def has_dense_index(self) -> bool:
        return self._embeddings is not None and bool(self._embedding_asins)

    @property
    def categorical_inverted_indexes(self) -> Mapping[str, Mapping[str, set[str]]]:
        return self.inverted_indexes

    @property
    def price_by_asin(self) -> Mapping[str, float | None]:
        return self.price_lookup

    def set_query_encoder(self, encoder: Callable[[str], object] | None) -> None:
        if encoder is not None and not callable(encoder):
            raise TypeError("query encoder must be callable or None")
        self._query_encoder = encoder

    def _query_vector(self, query_embedding: object, query_text: str | None) -> Any:
        if self._embeddings is None:
            return None
        if query_embedding is None and query_text and self._query_encoder is not None:
            try:
                query_embedding = self._query_encoder(query_text)
            except Exception:  # optional encoder failures should degrade safely
                return None
        if query_embedding is None:
            return None
        try:
            import numpy as np

            vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            if vector.shape[0] != self._embeddings.shape[1] or not np.isfinite(vector).all():
                return None
            norm = float(np.linalg.norm(vector))
            return None if norm <= 0 else vector / norm
        except (ImportError, TypeError, ValueError):
            return None

    @staticmethod
    def _constraints(constraints: object) -> dict[str, tuple[str, ...] | float | None]:
        result: dict[str, tuple[str, ...] | float | None] = {
            field: () for field in CANONICAL_FIELDS
        }
        result.update({"price_min": None, "price_max": None})
        if constraints is None:
            return result
        for field in CANONICAL_FIELDS:
            value = constraints.get(field) if isinstance(constraints, Mapping) else getattr(
                constraints, field, None
            )
            result[field] = _value_tuple(value)
        for field in PRICE_FIELDS:
            value = constraints.get(field) if isinstance(constraints, Mapping) else getattr(
                constraints, field, None
            )
            result[field] = _price(value)
        # A few lightweight Agent callers use a single ``price`` mapping.
        if isinstance(constraints, Mapping) and isinstance(constraints.get("price"), Mapping):
            budget = constraints["price"]
            result["price_min"] = _price(budget.get("min", budget.get("price_min")))
            result["price_max"] = _price(budget.get("max", budget.get("price_max")))
        return result

    def _field_values(self, asin: str, field: str) -> tuple[str, ...]:
        return _value_tuple(self._facts_for(asin).get(field))

    def _field_match(self, asin: str, field: str, values: tuple[str, ...]) -> bool:
        if field in CANONICAL_FIELDS:
            return bool(set(values).intersection(self._field_values(asin, field)))
        price = self.price_lookup.get(asin)
        if price is None:
            return False
        if field == "price_min":
            return price >= float(values[0]) if values else True
        if field == "price_max":
            return price <= float(values[0]) if values else True
        return True

    @staticmethod
    def _labels(field: str, values: tuple[str, ...] | float | None) -> tuple[str, ...]:
        if values is None or values == ():
            return ()
        if field in PRICE_FIELDS:
            return (f"{field}:{float(values):g}",)
        return tuple(f"{field}:{value}" for value in values)

    def _match_details(
        self,
        asin: str,
        constraints: dict[str, tuple[str, ...] | float | None],
        relaxed_fields: set[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], float]:
        matched: list[str] = []
        violated: list[str] = []
        relaxed: list[str] = []
        total = 0
        matched_count = 0
        for field in ALL_CONSTRAINT_FIELDS:
            values = constraints[field]
            labels = self._labels(field, values)
            if not labels:
                continue
            total += 1
            is_match = self._field_match(asin, field, values if isinstance(values, tuple) else (str(values),))
            if field in relaxed_fields:
                relaxed.extend(labels)
            if is_match:
                matched.extend(labels)
                matched_count += 1
            else:
                violated.extend(labels)
        score = 1.0 if total == 0 else matched_count / total
        return tuple(matched), tuple(violated), tuple(relaxed), score

    def _filtered(
        self,
        constraints: dict[str, tuple[str, ...] | float | None],
        active_fields: set[str],
    ) -> list[str]:
        result: list[str] = []
        for asin in self.catalog_order:
            if all(
                field not in active_fields
                or self._field_match(
                    asin,
                    field,
                    values if isinstance(values, tuple) else (str(values),),
                )
                for field, values in constraints.items()
                if values not in (None, ())
            ):
                result.append(asin)
        return result

    def _dense_scores(self, query_embedding: object, query_text: str | None) -> dict[str, float]:
        vector = self._query_vector(query_embedding, query_text)
        if vector is None:
            return {}
        scores = self._embeddings @ vector
        return {asin: float(scores[position]) for position, asin in enumerate(self._embedding_asins)}

    def _candidate(
        self,
        asin: str,
        mode: str,
        constraints: dict[str, tuple[str, ...] | float | None],
        relaxed_fields: set[str],
        dense_scores: Mapping[str, float],
    ) -> RetrievalCandidate:
        matched, violated, relaxed, score = self._match_details(asin, constraints, relaxed_fields)
        return RetrievalCandidate(
            parent_asin=asin,
            retrieval_mode=mode,
            dense_score=dense_scores.get(asin, 0.0),
            constraint_score=score,
            matched_constraints=matched,
            violated_constraints=violated,
            relaxed_constraints=relaxed,
        )

    def buying(
        self,
        constraints: object = None,
        *,
        top_k: int = 100,
        query_embedding: object = None,
        query_text: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve precision-first candidates using canonical hard filters."""

        if top_k <= 0:
            return []
        normalized = self._constraints(constraints)
        requested = {field for field, values in normalized.items() if values not in (None, ())}
        active = set(requested)
        relaxed_fields: set[str] = set()
        survivors = self._filtered(normalized, active)
        if not survivors and requested:
            for field in RELAXATION_ORDER:
                if field not in active:
                    continue
                active.remove(field)
                relaxed_fields.add(field)
                survivors = self._filtered(normalized, active)
                if survivors:
                    break

        # With no valid structured survivor, dense ranking is allowed to use
        # the full valid catalog.  This is a deliberate last resort, not a
        # second lexical retrieval branch.
        if not survivors:
            survivors = list(self.catalog_order)
            relaxed_fields = set(requested)
        dense_scores = self._dense_scores(query_embedding, query_text)
        candidates = [
            self._candidate(asin, "BUYING", normalized, relaxed_fields, dense_scores)
            for asin in survivors
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.constraint_score,
                -candidate.dense_score if dense_scores else 0.0,
                self._catalog_position[candidate.parent_asin],
            )
        )
        return candidates[:top_k]

    def browsing(
        self,
        constraints: object = None,
        *,
        preferences: object = None,
        top_k: int = 100,
        query_embedding: object = None,
        query_text: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve recall-first candidates without turning preferences into filters."""

        if top_k <= 0:
            return []
        normalized = self._constraints(preferences if preferences is not None else constraints)
        dense_scores = self._dense_scores(query_embedding, query_text)
        candidates: list[RetrievalCandidate] = []
        for asin in self.catalog_order:
            candidate = self._candidate(asin, "BROWSING", normalized, set(), dense_scores)
            candidates.append(candidate)

        # With vectors, dense relevance is primary and canonical preferences
        # are a small deterministic boost.  Without vectors, facts/preferences
        # are only a stable boost over catalog order; no BM25 fallback is used.
        def key(candidate: RetrievalCandidate) -> tuple[float, float, int]:
            preference_boost = candidate.constraint_score
            dense = candidate.dense_score if dense_scores else 0.0
            return (
                -(dense + 0.15 * preference_boost),
                -preference_boost,
                self._catalog_position[candidate.parent_asin],
            )

        candidates.sort(key=key)
        return candidates[:top_k]

    retrieve_buying = buying
    retrieve_browsing = browsing

    def retrieve(
        self,
        mode: str,
        constraints: object = None,
        *,
        preferences: object = None,
        top_k: int = 100,
        query_embedding: object = None,
        query_text: str | None = None,
    ) -> list[RetrievalCandidate]:
        """Common Agent entry point for either route."""

        normalized_mode = str(mode).upper()
        if normalized_mode == "BUYING":
            return self.buying(
                constraints,
                top_k=top_k,
                query_embedding=query_embedding,
                query_text=query_text,
            )
        if normalized_mode == "BROWSING":
            return self.browsing(
                constraints,
                preferences=preferences,
                top_k=top_k,
                query_embedding=query_embedding,
                query_text=query_text,
            )
        raise ValueError("mode must be BUYING or BROWSING")


ProductRetriever = RetrievalIndex


def build_retrieval_index(
    catalog_path: str | Path = "data/catalog.jsonl",
    **kwargs: Any,
) -> RetrievalIndex:
    """Small factory for Agent integration and notebook-style callers."""

    return RetrievalIndex(catalog_path, **kwargs)


__all__ = [
    "ALL_CONSTRAINT_FIELDS",
    "CANONICAL_FIELDS",
    "ProductRetriever",
    "RELAXATION_ORDER",
    "RetrievalCandidate",
    "RetrievalIndex",
    "build_retrieval_index",
]
