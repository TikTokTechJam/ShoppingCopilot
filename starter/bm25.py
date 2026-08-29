"""Small in-memory BM25 index for the product-text retrieval path.

This is a third retrieval signal. It does not replace structured or semantic
matching, and it intentionally reuses the semantic path's lexical cleanup so
the three paths see the same conversational query terms.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from dictionary.registry import semantic_query_tokens


BM25_FIELD_WEIGHTS = (6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
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
        self.connection = sqlite3.connect(":memory:")
        self._asins = tuple(str(asin) for asin in catalog_order)
        self._build(products)

    def _build(self, products: Mapping[str, Any]) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
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
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def search(
        self,
        query_text: str,
        *,
        allowed_asins: Collection[str] | None = None,
    ) -> dict[str, float]:
        terms = _query_terms(query_text)
        expression = _match_expression(terms)
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
