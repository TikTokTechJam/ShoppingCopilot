"""Slot-guided SQLite FTS5 BM25 retrieval.

The BGE attribute matcher resolves a shopper's current turn into canonical
slot values and semantic evidence.  This module compiles that state into one
bounded lexical query per slot, then routes each query to the product fields
where that slot is meaningful.  SQLite still owns tokenization, FTS matching,
BM25 scoring, and result ordering; Python only builds the query groups.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from dictionary.registry import normalize_text, semantic_query_tokens


# These are the original raw-catalog columns and weights.  Keep this tuple
# stable for callers that use the raw BM25 compatibility path.
BM25_FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
MAX_QUERY_TERMS = 40
MAX_QUERY_NGRAM = 3

# The compiler supplies each active slot with a deliberately small number of
# semantic surface forms.  The retriever searches those slot groups separately
# so one slot cannot dominate the combined BM25 signal merely because it has
# many BGE candidates.
BM25_EXPANSIONS_PER_FIELD = 3
BM25_EXPANSION_MIN_SIMILARITY = 0.82
BM25_EXPANSION_MAX_SCORE_GAP = 0.08
BM25_QUERY_FIELDS = (
    "category",
    "brand",
    "color",
    "material",
    "feature",
    "use_case",
    "style",
)

# The V5 annotation field names are indexed directly as dedicated SQLite
# columns.  They correspond to the actual V5 JSONL keys/files (``category``,
# ``brand``, ...), not to fictional ``v5_*`` files.  The remaining names are
# raw catalog columns.  A group is restricted to this route with an FTS5
# column filter, so a category query cannot receive evidence from (for
# example) description text unless the route explicitly allows it.
BM25_SLOT_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "category": ("category", "categories", "title"),
    "brand": ("brand", "title", "store"),
    "color": ("color", "title", "features"),
    "material": ("material", "features", "details"),
    "feature": ("feature", "features", "details", "description"),
    "use_case": ("use_case", "features", "description", "categories"),
    "style": ("style", "title", "features"),
}

BM25_ANNOTATION_COLUMNS = BM25_QUERY_FIELDS
BM25_RAW_COLUMNS = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)
BM25_INDEX_COLUMNS = BM25_ANNOTATION_COLUMNS + BM25_RAW_COLUMNS

# V5 annotation facts mirror the lexical importance of the corresponding raw
# field.  These are BM25 field weights, not the removed structured score. The
# raw-column weights themselves remain unchanged.
BM25_ANNOTATION_FIELD_WEIGHTS = {
    "category": 4.0,
    "brand": 1.5,
    "color": 2.5,
    "material": 2.5,
    "feature": 2.5,
    "use_case": 2.5,
    "style": 6.0,
}
BM25_INDEX_FIELD_WEIGHTS = (
    0.0,
    *(BM25_ANNOTATION_FIELD_WEIGHTS[column] for column in BM25_ANNOTATION_COLUMNS),
    *BM25_FIELD_WEIGHTS[1:],
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {_text(item)}" for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _query_terms(text: str) -> tuple[str, ...]:
    """Use the existing semantic tokenizer and stopword policy for BM25."""

    return tuple(dict.fromkeys(semantic_query_tokens(text)))[:MAX_QUERY_TERMS]


def _query_ngrams(text: str, *, max_ngram: int = MAX_QUERY_NGRAM) -> tuple[str, ...]:
    """Return cleaned contiguous 1-, 2-, and 3-word phrases for BM25.

    The phrases deliberately overlap.  For example, ``black rain boots``
    produces the unigrams, ``black rain`` and ``rain boots``, and
    ``black rain boots``.  The existing semantic tokenizer removes
    conversational stopwords before the phrase windows are built.
    """

    terms = _query_terms(text)
    if not terms:
        return ()
    width_limit = max(1, min(int(max_ngram), MAX_QUERY_NGRAM))
    phrases: list[str] = []
    for width in range(1, width_limit + 1):
        for start in range(0, len(terms) - width + 1):
            phrases.append(" ".join(terms[start : start + width]))
    return tuple(dict.fromkeys(phrases))


def _phrase_ngrams(phrases: Iterable[str]) -> tuple[str, ...]:
    """Expand each OR alternative without creating phrases across alternatives."""

    result: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        for candidate in _query_ngrams(str(phrase)):
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return tuple(result)


def _match_expression(
    phrases: Iterable[str],
    *,
    fields: Collection[str] | None = None,
) -> str:
    escaped = (
        f'"{str(phrase).replace(chr(34), chr(34) * 2)}"'
        for phrase in phrases
        if str(phrase).strip()
    )
    expression = " OR ".join(escaped)
    if not expression or not fields:
        return expression
    # Field names come only from BM25_SLOT_FIELD_PATHS/BM25_RAW_COLUMNS, never
    # from user input. The column filter keeps every phrase, including a
    # multi-word phrase, inside one selected FTS column.
    field_expression = " ".join(str(field) for field in fields)
    return f"{{{field_expression}}} : ({expression})"


def _field_values(source: object | None, field_name: str) -> tuple[str, ...]:
    """Read a tuple-like constraint field without coupling BM25 to state types."""

    if source is None:
        return ()
    value = getattr(source, field_name, None)
    if value is None and isinstance(source, Mapping):
        value = source.get(field_name)
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = (value,)
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _evidence_items(source: object | None) -> tuple[object, ...]:
    if source is None:
        return ()
    value = getattr(source, "evidence", None)
    if value is None and isinstance(source, Mapping):
        value = source.get("evidence")
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _evidence_value(item: object) -> str:
    """Return the canonical value represented by a canonical-id evidence item."""

    canonical_id = getattr(item, "canonical_id", None)
    if isinstance(item, Mapping):
        canonical_id = item.get("canonical_id", canonical_id)
    if not isinstance(canonical_id, str):
        return ""
    _, separator, value = canonical_id.partition(":")
    return (value if separator else canonical_id).replace("_", " ").strip()


def _evidence_field(item: object) -> str:
    value = getattr(item, "attribute", None)
    if isinstance(item, Mapping):
        value = item.get("attribute", value)
    return str(value or "").strip()


def _evidence_raw_text(item: object) -> str:
    value = getattr(item, "raw_text", None)
    if isinstance(item, Mapping):
        value = item.get("raw_text", value)
    return str(value or "").strip()


def _evidence_score(item: object) -> float:
    value = getattr(item, "confidence", None)
    if isinstance(item, Mapping):
        value = item.get("confidence", value)
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if score == score else 0.0


def _is_semantic_evidence(item: object) -> bool:
    layer = getattr(item, "layer", None)
    method = getattr(item, "match_method", None)
    if isinstance(item, Mapping):
        layer = item.get("layer", layer)
        method = item.get("match_method", method)
    return str(layer or "").casefold() == "layer2" or str(method or "").startswith(
        "semantic_"
    )


@dataclass(frozen=True)
class BM25QueryGroup:
    """One slot's lexical alternatives and its allowed product fields."""

    field_name: str
    phrases: tuple[str, ...]
    fields: tuple[str, ...]

    @property
    def match_phrases(self) -> tuple[str, ...]:
        """Return OR alternatives, including the existing bounded n-grams.

        N-grams are generated independently for each alternative.  This keeps
        an expansion such as ``rainy weather`` separate from the next
        alternative instead of accidentally creating terms like
        ``weather lightweight`` across two BGE surfaces.
        """

        return _phrase_ngrams(self.phrases)

    @property
    def query_text(self) -> str:
        """Human-readable FTS5 expression used by diagnostics."""

        return _match_expression(self.match_phrases, fields=self.fields)


class BM25QueryCompiler:
    """Compile active slot state into bounded lexical BM25 query groups.

    Each returned group represents one information need such as ``feature`` or
    ``use_case``. Canonical values already present in the active state are
    included once. Evidence-derived user surfaces are optional expansions and
    capped per slot.

    The compiler only creates query text. SQLite still owns tokenization, FTS
    matching, BM25 scoring, and result ordering. Score normalization/fusion is
    deliberately left to the retriever because it needs the per-group result
    distributions.
    """

    def __init__(
        self,
        *,
        expansions_per_field: int = BM25_EXPANSIONS_PER_FIELD,
        min_expansion_similarity: float = BM25_EXPANSION_MIN_SIMILARITY,
        max_expansion_score_gap: float = BM25_EXPANSION_MAX_SCORE_GAP,
    ) -> None:
        self.expansions_per_field = max(0, int(expansions_per_field))
        self.min_expansion_similarity = float(min_expansion_similarity)
        self.max_expansion_score_gap = max(0.0, float(max_expansion_score_gap))

    @staticmethod
    def _add_term(
        terms: list[str],
        seen: set[str],
        value: str,
    ) -> None:
        text = str(value).strip()
        if not text:
            return
        # Use the same semantic normalization as the normal BM25 query path
        # for de-duplication, while retaining the readable surface in the
        # compiled query.  BM25Index will apply the final tokenizer cleanup.
        key = " ".join(semantic_query_tokens(text))
        if not key or key in seen:
            return
        seen.add(key)
        terms.append(text)

    def compile_group_specs(
        self,
        constraints: object | None,
        semantic_constraints: object | None = None,
    ) -> dict[str, BM25QueryGroup]:
        """Return one field-routed lexical query group per active normal slot."""

        semantic_evidence = _evidence_items(semantic_constraints)
        # Some callers retain the evidence on the combined constraints object;
        # include it as a fallback without duplicating the semantic view.
        if not semantic_evidence:
            semantic_evidence = tuple(
                item for item in _evidence_items(constraints) if _is_semantic_evidence(item)
            )

        groups: dict[str, BM25QueryGroup] = {}
        for field_name in BM25_QUERY_FIELDS:
            terms: list[str] = []
            seen: set[str] = set()
            for value in _field_values(constraints, field_name):
                self._add_term(terms, seen, value)

            expansions = [
                item
                for item in semantic_evidence
                if _evidence_field(item) == field_name
                and _is_semantic_evidence(item)
                and _evidence_raw_text(item)
            ]
            expansions.sort(
                key=lambda item: (-_evidence_score(item), _evidence_raw_text(item))
            )
            if expansions:
                best_score = _evidence_score(expansions[0])
                added = 0
                for item in expansions:
                    score = _evidence_score(item)
                    if score < self.min_expansion_similarity:
                        continue
                    if best_score - score > self.max_expansion_score_gap:
                        continue
                    before = len(terms)
                    # The canonical value and the user-facing surface are two
                    # lexical realizations of one accepted semantic slot
                    # value. They consume one expansion budget together.
                    self._add_term(terms, seen, _evidence_value(item))
                    self._add_term(terms, seen, _evidence_raw_text(item))
                    if len(terms) == before:
                        continue
                    added += 1
                    if added >= self.expansions_per_field:
                        break

            # A manually constructed/legacy semantic state may not carry
            # evidence. In that case its active values are still useful, but
            # there is no confidence ordering from which to cap them.
            evidenced_values = {
                normalize_text(_evidence_value(item))
                for item in semantic_evidence
                if _evidence_field(item) == field_name
            }
            for value in _field_values(semantic_constraints, field_name):
                if not semantic_evidence or normalize_text(value) not in evidenced_values:
                    self._add_term(terms, seen, value)

            if terms:
                groups[field_name] = BM25QueryGroup(
                    field_name=field_name,
                    phrases=tuple(terms),
                    fields=BM25_SLOT_FIELD_PATHS[field_name],
                )

        return groups

    def compile_groups(
        self,
        constraints: object | None,
        semantic_constraints: object | None = None,
    ) -> dict[str, str]:
        """Return compiled FTS expressions for legacy callers/debuggers.

        Retrieval uses :meth:`compile_group_specs` so it can pass the safe
        field route and OR alternatives to SQLite separately.
        """

        return {
            field_name: group.query_text
            for field_name, group in self.compile_group_specs(
                constraints,
                semantic_constraints,
            ).items()
        }


class BM25Index:
    """SQLite FTS5 BM25 index over the catalog's searchable text fields."""

    def __init__(
        self,
        products: Mapping[str, Any],
        catalog_order: Iterable[str],
    ) -> None:
        started = time.perf_counter()
        self.connection = sqlite3.connect(":memory:")
        self._asins = tuple(str(asin) for asin in catalog_order)
        self.indexed_rows = 0
        self.build_seconds: float | None = None
        print(
            f"[bm25] preprocessing catalog: {len(self._asins):,} products",
            flush=True,
        )
        try:
            self._build(products)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[bm25] preprocessing failed after {elapsed:.1f}s "
                f"at {self.indexed_rows:,}/{len(self._asins):,} products: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        self.build_seconds = time.perf_counter() - started
        print(
            f"[bm25] ready: {self.indexed_rows:,} products indexed "
            f"in {self.build_seconds:.1f}s",
            flush=True,
        )

    def _build(self, products: Mapping[str, Any]) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, "
            "category, brand, color, material, feature, use_case, style, "
            "title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        placeholders = ", ".join("?" for _ in range(len(BM25_INDEX_COLUMNS) + 1))
        insert_sql = f"INSERT INTO products VALUES ({placeholders})"
        batch: list[tuple[str, ...]] = []
        next_log = 5000
        for asin in self._asins:
            product = products.get(asin)
            raw = getattr(product, "raw", None)
            if raw is None and isinstance(product, Mapping):
                raw = product.get("raw", product)
            if not isinstance(raw, Mapping):
                raw = {}
            facts = getattr(product, "facts", None)
            if facts is None and isinstance(product, Mapping):
                facts = product.get("facts", {})
            if not isinstance(facts, Mapping):
                facts = {}
            batch.append(
                (
                    asin,
                    *(
                        _text(facts.get(field_name, ()))
                        for field_name in BM25_QUERY_FIELDS
                    ),
                    _text(raw.get("title")),
                    _text(raw.get("categories")),
                    _text(raw.get("features")),
                    _text(raw.get("details")),
                    _text(raw.get("store")),
                    _text(raw.get("description")),
                )
            )
            if len(batch) >= 1000:
                cursor.executemany(insert_sql, batch)
                self.indexed_rows += len(batch)
                batch.clear()
                if self.indexed_rows >= next_log:
                    print(
                        f"[bm25] indexed {self.indexed_rows:,}/{len(self._asins):,}",
                        flush=True,
                    )
                    next_log += 5000
        if batch:
            cursor.executemany(insert_sql, batch)
            self.indexed_rows += len(batch)
        print(
            f"[bm25] catalog text preprocessing complete: "
            f"{self.indexed_rows:,}/{len(self._asins):,} rows",
            flush=True,
        )
        self.connection.commit()

    def search(
        self,
        query_text: str | Iterable[str],
        *,
        allowed_asins: Collection[str] | None = None,
        fields: Collection[str] | None = None,
    ) -> dict[str, float]:
        # No route means the legacy raw BM25 view.  Slot-guided retrieval
        # always supplies an explicit route that may include V5 fact columns.
        selected_fields = BM25_RAW_COLUMNS if fields is None else fields
        if isinstance(query_text, str):
            phrases = _query_ngrams(query_text)
        else:
            phrases = _phrase_ngrams(query_text)
        expression = _match_expression(phrases, fields=selected_fields)
        if not expression:
            return {}

        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) AS rank "
            "FROM products WHERE products MATCH ? "
            "ORDER BY rank ASC, rowid ASC",
            (*BM25_INDEX_FIELD_WEIGHTS, expression),
        ).fetchall()
        allowed = None if allowed_asins is None else set(allowed_asins)
        scores: dict[str, float] = {}
        for asin, raw_rank in rows:
            asin = str(asin)
            if allowed is not None and asin not in allowed:
                continue
            # SQLite FTS5 exposes bm25 as a lower-is-better negative rank.
            # Convert it once to a higher-is-better score for the shared
            # candidate/reranker contract.
            score = max(0.0, -float(raw_rank))
            scores[asin] = score
        return scores


__all__ = [
    "BM25_EXPANSIONS_PER_FIELD",
    "BM25_EXPANSION_MIN_SIMILARITY",
    "BM25_EXPANSION_MAX_SCORE_GAP",
    "BM25_FIELD_WEIGHTS",
    "BM25_SLOT_FIELD_PATHS",
    "BM25QueryGroup",
    "BM25Index",
    "BM25QueryCompiler",
]
