"""Small in-memory BM25 index for the product-text retrieval path.

The index exposes both native BM25 scores and deterministic one-based ranks.
The retrieval layer uses the ranks for reciprocal-rank fusion, while keeping
the native score available for compatibility and diagnostics.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from dictionary.registry import semantic_query_tokens


# One weight is required for every FTS column, including the UNINDEXED ASIN.
# The ASIN weight is zero so identifiers never influence lexical relevance.
BM25_FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
MAX_QUERY_TERMS = 40


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


def _match_expression(terms: Iterable[str]) -> str:
    escaped = (
        f'"{str(term).replace(chr(34), chr(34) * 2)}"'
        for term in terms
        if str(term).strip()
    )
    return " OR ".join(escaped)


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
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        next_log = 5000
        for asin in self._asins:
            product = products.get(asin)
            raw = getattr(product, "raw", product)
            if not isinstance(raw, Mapping):
                raw = {}
            batch.append(
                (
                    asin,
                    _text(raw.get("title")),
                    _text(raw.get("categories")),
                    _text(raw.get("features")),
                    _text(raw.get("details")),
                    _text(raw.get("store")),
                    _text(raw.get("description")),
                )
            )
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                self.indexed_rows += len(batch)
                batch.clear()
                if self.indexed_rows >= next_log:
                    print(
                        f"[bm25] indexed {self.indexed_rows:,}/{len(self._asins):,}",
                        flush=True,
                    )
                    next_log += 5000
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            self.indexed_rows += len(batch)
        print(
            f"[bm25] catalog text preprocessing complete: "
            f"{self.indexed_rows:,}/{len(self._asins):,} rows",
            flush=True,
        )
        self.connection.commit()

    def search(
        self,
        query_text: str,
        *,
        allowed_asins: Collection[str] | None = None,
    ) -> dict[str, float]:
        return self._search_scores(_query_terms(query_text), allowed_asins=allowed_asins)

    def search_terms(
        self,
        terms: Iterable[str],
        *,
        allowed_asins: Collection[str] | None = None,
        max_results: int | None = None,
        max_query_terms: int | None = MAX_QUERY_TERMS,
    ) -> dict[str, int]:
        """Return one-based BM25 ranks for a token query.

        The caller supplies already-cleaned terms so a raw query and a
        constraint query can share the repository's semantic stopword policy.
        Ranks are assigned after the eligibility filter; an excluded product
        therefore cannot consume a useful rank.
        """

        unique_terms = tuple(dict.fromkeys(terms))
        if max_query_terms is not None:
            unique_terms = unique_terms[:max_query_terms]
        expression = _match_expression(unique_terms)
        if not expression:
            return {}

        rows = self._rows(expression)
        allowed = None if allowed_asins is None else set(allowed_asins)
        ranks: dict[str, int] = {}
        for asin, _raw_rank in rows:
            asin = str(asin)
            if allowed is not None and asin not in allowed:
                continue
            ranks[asin] = len(ranks) + 1
            if max_results is not None and len(ranks) >= max_results:
                break
        return ranks

    def search_ranked(
        self,
        query_text: str,
        *,
        allowed_asins: Collection[str] | None = None,
        max_results: int | None = None,
    ) -> dict[str, int]:
        """Return one-based ranks for a cleaned conversational query."""

        return self.search_terms(
            _query_terms(query_text),
            allowed_asins=allowed_asins,
            max_results=max_results,
        )

    def _rows(self, expression: str) -> list[tuple[object, object]]:
        return self.connection.execute(
            "SELECT parent_asin, bm25(products, ?, ?, ?, ?, ?, ?, ?) AS rank "
            "FROM products WHERE products MATCH ? "
            "ORDER BY rank ASC, rowid ASC",
            (*BM25_FIELD_WEIGHTS, expression),
        ).fetchall()

    def _search_scores(
        self,
        terms: Iterable[str],
        *,
        allowed_asins: Collection[str] | None = None,
    ) -> dict[str, float]:
        expression = _match_expression(terms)
        if not expression:
            return {}

        rows = self._rows(expression)
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


__all__ = ["BM25_FIELD_WEIGHTS", "MAX_QUERY_TERMS", "BM25Index"]
