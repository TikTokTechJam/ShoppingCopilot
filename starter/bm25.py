"""Small in-memory BM25 index for the product-text retrieval path.

This is a third retrieval signal. It does not replace structured or semantic
matching, and it intentionally reuses the semantic path's lexical cleanup so
the three paths see the same conversational query terms.
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
MAX_QUERY_NGRAM = 3


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


def _match_expression(phrases: Iterable[str]) -> str:
    escaped = (
        f'"{str(phrase).replace(chr(34), chr(34) * 2)}"'
        for phrase in phrases
        if str(phrase).strip()
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
        phrases = _query_ngrams(query_text)
        expression = _match_expression(phrases)
        if not expression:
            return {}

        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, ?, ?, ?, ?, ?, ?, ?) AS rank "
            "FROM products WHERE products MATCH ? "
            "ORDER BY rank ASC, rowid ASC",
            (*BM25_FIELD_WEIGHTS, expression),
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


__all__ = ["BM25_FIELD_WEIGHTS", "BM25Index"]
