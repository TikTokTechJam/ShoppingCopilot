"""Small in-memory field-aware BM25F index for product-text retrieval.

The lexical signal is intentionally kept separate from structured and dense
retrieval.  BM25F combines term evidence from the catalog's text fields while
using the existing field priorities.  The public ``BM25Index`` name remains as
a compatibility alias for callers that already use the lexical component.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from dictionary.registry import semantic_query_tokens


# These are the searchable catalog fields, in the same priority order used by
# the previous SQLite lexical index.  The ASIN is metadata, not searchable
# product text.
BM25F_FIELDS = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)

# Preserve the existing lexical field priorities.  BM25F applies these boosts
# while combining field term frequencies rather than treating the whole
# product as one undifferentiated text blob.
BM25F_FIELD_WEIGHTS = {
    "title": 6.0,
    "categories": 4.0,
    "features": 2.5,
    "details": 2.5,
    "store": 1.5,
    "description": 1.0,
}

# Keep the old tuple available for small integrations that imported it.  Its
# first zero corresponds to the old UNINDEXED ASIN column.
BM25_FIELD_WEIGHTS = (
    0.0,
    *(BM25F_FIELD_WEIGHTS[field] for field in BM25F_FIELDS),
)

BM25F_K1 = 1.2
BM25F_B = 0.75
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
    """Use the shared semantic tokenizer and remove duplicate query terms."""

    return tuple(dict.fromkeys(semantic_query_tokens(text)))[:MAX_QUERY_TERMS]


def _field_terms(value: object) -> tuple[str, ...]:
    """Tokenize one catalog field with the same lexical policy as queries."""

    return semantic_query_tokens(_text(value))


class BM25FIndex:
    """In-memory BM25F index over separately weighted product text fields.

    The implementation stores only postings for terms that actually occur in
    each field.  At query time it scores matching rows, so non-matching
    products do not require a full 50,000-row scan.
    """

    def __init__(
        self,
        products: Mapping[str, Any],
        catalog_order: Iterable[str],
    ) -> None:
        started = time.perf_counter()
        self._asins = tuple(str(asin) for asin in catalog_order)
        self._asin_to_row = {
            asin: row_id for row_id, asin in enumerate(self._asins)
        }
        self.indexed_rows = 0
        self.build_seconds: float | None = None
        self._postings: list[dict[str, dict[int, int]]] = [
            {} for _ in BM25F_FIELDS
        ]
        self._field_lengths = [
            [0] * len(self._asins) for _ in BM25F_FIELDS
        ]
        self._average_field_lengths: tuple[float, ...] = ()
        self._document_frequency: dict[str, int] = {}
        print(
            f"[bm25f] preprocessing catalog: {len(self._asins):,} products",
            flush=True,
        )
        try:
            self._build(products)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"[bm25f] preprocessing failed after {elapsed:.1f}s "
                f"at {self.indexed_rows:,}/{len(self._asins):,} products: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        product_count = max(len(self._asins), 1)
        self._average_field_lengths = tuple(
            sum(lengths) / product_count for lengths in self._field_lengths
        )
        self.build_seconds = time.perf_counter() - started
        posting_count = sum(
            len(field_postings)
            for field_postings in self._postings
        )
        print(
            f"[bm25f] ready: {self.indexed_rows:,} products indexed, "
            f"{posting_count:,} field terms in {self.build_seconds:.1f}s",
            flush=True,
        )

    def _build(self, products: Mapping[str, Any]) -> None:
        next_log = 5000
        for row_id, asin in enumerate(self._asins):
            product = products.get(asin)
            raw = getattr(product, "raw", product)
            if not isinstance(raw, Mapping):
                raw = {}

            terms_in_product: set[str] = set()
            for field_id, field_name in enumerate(BM25F_FIELDS):
                counts = Counter(_field_terms(raw.get(field_name)))
                self._field_lengths[field_id][row_id] = sum(counts.values())
                for term, frequency in counts.items():
                    self._postings[field_id].setdefault(term, {})[row_id] = frequency
                    terms_in_product.add(term)

            for term in terms_in_product:
                self._document_frequency[term] = (
                    self._document_frequency.get(term, 0) + 1
                )

            self.indexed_rows += 1
            if self.indexed_rows >= next_log:
                print(
                    f"[bm25f] indexed {self.indexed_rows:,}/"
                    f"{len(self._asins):,}",
                    flush=True,
                )
                next_log += 5000

        print(
            f"[bm25f] catalog text preprocessing complete: "
            f"{self.indexed_rows:,}/{len(self._asins):,} rows",
            flush=True,
        )

    def search(
        self,
        query_text: str,
        *,
        allowed_asins: Collection[str] | None = None,
    ) -> dict[str, float]:
        terms = _query_terms(query_text)
        if not terms:
            return {}

        allowed_rows = None
        if allowed_asins is not None:
            allowed_rows = {
                self._asin_to_row[asin]
                for asin in (str(value) for value in allowed_asins)
                if asin in self._asin_to_row
            }
            if not allowed_rows:
                return {}

        scores_by_row: dict[int, float] = {}
        document_count = max(len(self._asins), 1)
        for term in terms:
            document_frequency = self._document_frequency.get(term, 0)
            if not document_frequency:
                continue

            # Positive Robertson/Sparck Jones IDF.  A term's document
            # frequency is counted once per product even if it occurs in
            # multiple fields.
            idf = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            weighted_term_frequency: dict[int, float] = {}
            for field_id, field_name in enumerate(BM25F_FIELDS):
                postings = self._postings[field_id].get(term)
                if not postings:
                    continue
                average_length = self._average_field_lengths[field_id]
                if average_length <= 0.0:
                    continue
                field_weight = BM25F_FIELD_WEIGHTS[field_name]
                for row_id, frequency in postings.items():
                    if allowed_rows is not None and row_id not in allowed_rows:
                        continue
                    field_length = self._field_lengths[field_id][row_id]
                    normalization = (1.0 - BM25F_B) + BM25F_B * (
                        field_length / average_length
                    )
                    weighted_term_frequency[row_id] = (
                        weighted_term_frequency.get(row_id, 0.0)
                        + field_weight * frequency / normalization
                    )

            for row_id, weighted_frequency in weighted_term_frequency.items():
                saturated = (
                    (BM25F_K1 + 1.0) * weighted_frequency
                    / (BM25F_K1 + weighted_frequency)
                )
                scores_by_row[row_id] = (
                    scores_by_row.get(row_id, 0.0) + idf * saturated
                )

        return {
            self._asins[row_id]: score
            for row_id, score in scores_by_row.items()
            if score > 0.0
        }


# Preserve the existing import/API while making the implementation explicitly
# field-aware BM25F.
BM25Index = BM25FIndex


__all__ = [
    "BM25F_B",
    "BM25F_FIELDS",
    "BM25F_FIELD_WEIGHTS",
    "BM25FIndex",
    "BM25F_K1",
    "BM25_FIELD_WEIGHTS",
    "BM25Index",
]
