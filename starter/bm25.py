"""Lexical BM25F reference and SQLite/FTS5 implementations.

The repository previously used SQLite's built-in ``bm25()`` auxiliary
function. That function is a single-length BM25 variant, not explicit
field-normalised BM25F. ``PythonBM25FIndex`` is the readable reference;
``SQLiteBM25FIndex`` delegates matching and scoring to the native ``bm25f``
FTS5 extension.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from dictionary.registry import semantic_query_tokens


BM25F_FIELDS = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)

# These are the exact weights used by the existing SQLite BM25 path. The old
# table also had an UNINDEXED ASIN column with weight zero; the new schema puts
# the ASIN in a companion mapping table, so only searchable fields remain.
BM25_FIELD_WEIGHTS = (6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
BM25F_FIELD_B = (0.75,) * len(BM25F_FIELDS)
BM25F_K1 = 1.2
BM25F_IDF_VERSION = "log1p-robertson-walker-v1"
BM25F_TOKENIZER = "unicode61 remove_diacritics 2"
BM25F_DOCUMENT_PREPROCESSING = "semantic_query_tokens-v1"
BM25F_SCHEMA_VERSION = 1
MAX_QUERY_TERMS = 40
BM25F_NGRAMS_ENV = "SHOPPING_BM25F_NGRAMS"

DEFAULT_BM25F_ARTIFACT_DIR = Path("data/derived/bm25f_sqlite")
DEFAULT_BM25F_DB_NAME = "bm25f.db"
DEFAULT_BM25F_EXTENSION_NAMES = {
    "darwin": "bm25f.dylib",
    "win32": "bm25f.dll",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_text(item) for item in value)
    return str(value)


def _query_terms(text: str) -> tuple[str, ...]:
    """Use the existing semantic query cleanup before lexical retrieval."""

    return tuple(dict.fromkeys(semantic_query_tokens(text)))[:MAX_QUERY_TERMS]


def _ngrams_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    configured = os.environ.get(BM25F_NGRAMS_ENV, "1").strip().casefold()
    return configured in {"1", "true", "yes", "on"}


def _query_ngrams(
    text: str,
    *,
    enabled: bool | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Return all overlapping contiguous query n-grams.

    The input tokens are the existing cleaned lexical tokens.  The baseline
    remains available by explicitly disabling n-grams; the runtime default is
    the overlapping 1-to-n-gram experiment.
    """

    tokens = _query_terms(text)
    if not tokens:
        return ()
    if not _ngrams_enabled(enabled):
        return tuple((token,) for token in tokens)
    return tuple(
        tuple(tokens[start : start + size])
        for size in range(1, len(tokens) + 1)
        for start in range(0, len(tokens) - size + 1)
    )


def _ngrams_by_level(
    text: str,
    *,
    enabled: bool | None = None,
) -> dict[int, tuple[tuple[str, ...], ...]]:
    result: dict[int, list[tuple[str, ...]]] = {}
    for gram in _query_ngrams(text, enabled=enabled):
        result.setdefault(len(gram), []).append(gram)
    return {size: tuple(grams) for size, grams in result.items()}


def _match_expression(grams: Iterable[tuple[str, ...]]) -> str:
    """Quote terms/phrases so user punctuation cannot become FTS5 operators."""

    escaped = (
        f'"{" ".join(str(part) for part in gram).replace(chr(34), chr(34) * 2)}"'
        for gram in grams
        if gram and " ".join(str(part) for part in gram).strip()
    )
    return " OR ".join(escaped)


def _field_tokens(value: object) -> tuple[str, ...]:
    """Use the exact document cleanup used by the Python BM25F branch."""

    return semantic_query_tokens(_text(value))


def _product_raw(product: Any) -> Mapping[str, Any]:
    raw = getattr(product, "raw", product)
    return raw if isinstance(raw, Mapping) else {}


def _catalog_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PythonBM25FIndex:
    """Readable BM25F reference implementation and fallback."""

    def __init__(
        self,
        products: Mapping[str, Any],
        catalog_order: Iterable[str],
        *,
        k1: float = BM25F_K1,
        field_weights: Iterable[float] = BM25_FIELD_WEIGHTS,
        field_b: Iterable[float] = BM25F_FIELD_B,
        ngrams_enabled: bool | None = None,
    ) -> None:
        started = time.perf_counter()
        self._asins = tuple(str(asin) for asin in catalog_order)
        self._catalog_position = {
            asin: position for position, asin in enumerate(self._asins)
        }
        self._k1 = float(k1)
        self._weights = tuple(float(value) for value in field_weights)
        self._b = tuple(float(value) for value in field_b)
        self.ngrams_enabled = _ngrams_enabled(ngrams_enabled)
        if len(self._weights) != len(BM25F_FIELDS) or len(self._b) != len(BM25F_FIELDS):
            raise ValueError("BM25F configuration does not match searchable fields")
        self._tokens: dict[str, tuple[tuple[str, ...], ...]] = {}
        self._lengths: dict[str, tuple[int, ...]] = {}
        document_frequency: Counter[str] = Counter()
        total_lengths = [0] * len(BM25F_FIELDS)

        for asin in self._asins:
            raw = _product_raw(products.get(asin))
            fields = tuple(
                _field_tokens(raw.get(field_name)) for field_name in BM25F_FIELDS
            )
            self._tokens[asin] = fields
            lengths = tuple(len(tokens) for tokens in fields)
            self._lengths[asin] = lengths
            for index, length in enumerate(lengths):
                total_lengths[index] += length
            for term in {term for field in fields for term in field}:
                document_frequency[term] += 1

        self.indexed_rows = len(self._asins)
        self._average_lengths = tuple(
            total / self.indexed_rows if self.indexed_rows else 0.0
            for total in total_lengths
        )
        self._idf = {
            term: math.log1p(
                (self.indexed_rows - df + 0.5) / (df + 0.5)
            )
            for term, df in document_frequency.items()
        }
        self.build_seconds = time.perf_counter() - started
        self.backend = "python_reference"

    @staticmethod
    def _count_gram(tokens: tuple[str, ...], gram: tuple[str, ...]) -> int:
        size = len(gram)
        if size == 1:
            return tokens.count(gram[0])
        if size > len(tokens):
            return 0
        return sum(
            tokens[start : start + size] == gram
            for start in range(0, len(tokens) - size + 1)
        )

    def _phrase_document_frequencies(
        self,
        grams: tuple[tuple[str, ...], ...],
    ) -> dict[tuple[str, ...], int]:
        targets = set(grams)
        frequencies: Counter[tuple[str, ...]] = Counter()
        max_size = max((len(gram) for gram in grams), default=1)
        for fields in self._tokens.values():
            found: set[tuple[str, ...]] = set()
            for tokens in fields:
                upper = min(max_size, len(tokens))
                for size in range(1, upper + 1):
                    for start in range(0, len(tokens) - size + 1):
                        gram = tokens[start : start + size]
                        if gram in targets:
                            found.add(gram)
            frequencies.update(found)
        return dict(frequencies)

    def _score_breakdown(
        self,
        asin: str,
        grams: tuple[tuple[str, ...], ...],
        idf: Mapping[tuple[str, ...], float],
    ) -> dict[int, float]:
        fields = self._tokens[asin]
        lengths = self._lengths[asin]
        levels: dict[int, float] = {}
        for gram in grams:
            gram_idf = idf.get(gram)
            if gram_idf is None:
                continue
            weighted_tf = 0.0
            for index, field_tokens in enumerate(fields):
                tf = self._count_gram(field_tokens, gram)
                if not tf:
                    continue
                average_length = self._average_lengths[index]
                normalizer = 1.0
                if average_length > 0.0:
                    normalizer = 1.0 - self._b[index] + self._b[index] * (
                        lengths[index] / average_length
                    )
                weighted_tf += self._weights[index] * tf / normalizer
            if weighted_tf:
                contribution = gram_idf * ((self._k1 + 1.0) * weighted_tf) / (
                    self._k1 + weighted_tf
                )
                levels[len(gram)] = levels.get(len(gram), 0.0) + contribution
        return levels

    def _score_one(
        self,
        asin: str,
        grams: tuple[tuple[str, ...], ...],
        idf: Mapping[tuple[str, ...], float],
    ) -> float:
        return float(sum(self._score_breakdown(asin, grams, idf).values()))

    def breakdown(self, query_text: str, asin: str) -> dict[int, float]:
        grams = _query_ngrams(query_text, enabled=self.ngrams_enabled)
        if not grams or asin not in self._tokens:
            return {}
        if not self.ngrams_enabled:
            idf: Mapping[tuple[str, ...], float] = {
                (term,): value for term, value in self._idf.items()
            }
        else:
            frequencies = self._phrase_document_frequencies(grams)
            idf = {
                gram: math.log1p(
                    (self.indexed_rows - frequency + 0.5) / (frequency + 0.5)
                )
                for gram, frequency in frequencies.items()
            }
        return self._score_breakdown(asin, grams, idf)

    def search(
        self,
        query_text: str,
        *,
        allowed_asins: Collection[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, float]:
        grams = _query_ngrams(query_text, enabled=self.ngrams_enabled)
        if not grams:
            return {}
        if not self.ngrams_enabled:
            idf: Mapping[tuple[str, ...], float] = {
                (term,): value for term, value in self._idf.items()
            }
        else:
            frequencies = self._phrase_document_frequencies(grams)
            idf = {
                gram: math.log1p(
                    (self.indexed_rows - frequency + 0.5) / (frequency + 0.5)
                )
                for gram, frequency in frequencies.items()
            }
        allowed = None if allowed_asins is None else {str(value) for value in allowed_asins}
        ranked = sorted(
            (
                (asin, self._score_one(asin, grams, idf))
                for asin in self._asins
                if allowed is None or asin in allowed
            ),
            key=lambda item: (-item[1], self._catalog_position[item[0]]),
        )
        if top_k is not None:
            ranked = ranked[: max(0, int(top_k))]
        return dict(ranked)


class SQLiteBM25FIndex:
    """Read-only wrapper around the native FTS5 BM25F extension."""

    _CONFIG_PLACEHOLDERS = ", ".join("?" for _ in range(19))
    _SEARCH_SQL = (
        "SELECT product_rows.parent_asin, product_fts.rowid, "
        f"bm25f(product_fts, {_CONFIG_PLACEHOLDERS}) AS score "
        "FROM product_fts JOIN product_rows ON product_rows.rowid = product_fts.rowid "
        "WHERE product_fts MATCH ? "
        "ORDER BY score DESC, product_fts.rowid ASC LIMIT ?"
    )
    _SEARCH_ALLOWED_SQL = (
        "SELECT product_rows.parent_asin, product_fts.rowid, "
        f"bm25f(product_fts, {_CONFIG_PLACEHOLDERS}) AS score "
        "FROM product_fts JOIN product_rows ON product_rows.rowid = product_fts.rowid "
        "JOIN temp.bm25f_allowed ON bm25f_allowed.rowid = product_fts.rowid "
        "WHERE product_fts MATCH ? "
        "ORDER BY score DESC, product_fts.rowid ASC LIMIT ?"
    )
    _SEARCH_EXCLUDED_SQL = (
        "SELECT product_rows.parent_asin, product_fts.rowid, "
        f"bm25f(product_fts, {_CONFIG_PLACEHOLDERS}) AS score "
        "FROM product_fts JOIN product_rows ON product_rows.rowid = product_fts.rowid "
        "WHERE product_fts MATCH ? "
        "AND NOT EXISTS (SELECT 1 FROM temp.bm25f_excluded "
        "WHERE bm25f_excluded.rowid = product_fts.rowid) "
        "ORDER BY score DESC, product_fts.rowid ASC LIMIT ?"
    )
    _BREAKDOWN_SQL = (
        "SELECT bm25f_levels(product_fts, "
        f"{_CONFIG_PLACEHOLDERS}) AS levels "
        "FROM product_fts JOIN product_rows ON product_rows.rowid = product_fts.rowid "
        "WHERE product_fts MATCH ? AND product_rows.parent_asin = ? LIMIT 1"
    )

    def __init__(
        self,
        db_path: str | Path,
        *,
        extension_path: str | Path | None = None,
        expected_catalog_sha256: str | None = None,
        expected_catalog_rows: int | None = None,
        ngrams_enabled: bool | None = None,
    ) -> None:
        started = time.perf_counter()
        self.db_path = Path(db_path)
        self.ngrams_enabled = _ngrams_enabled(ngrams_enabled)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"BM25F SQLite artifact not found: {self.db_path}")
        self.extension_path = Path(extension_path) if extension_path else default_extension_path()
        if not self.extension_path.is_file():
            raise FileNotFoundError(
                "BM25F native extension not found: "
                f"{self.extension_path}. Build it with "
                "python -m scripts.build_bm25f_sqlite_extension."
            )
        uri = f"file:{self.db_path.resolve()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)
        try:
            self._load_extension()
            self.manifest = self._read_manifest()
            self._validate_manifest(expected_catalog_sha256, expected_catalog_rows)
            self._configure_connection()
            self._asins, self._rowids = self._load_mapping()
            self.indexed_rows = len(self._asins)
            self.build_seconds = time.perf_counter() - started
            self.backend = "sqlite_fts5_native_bm25f"
        except Exception:
            self.connection.close()
            raise

    def _load_extension(self) -> None:
        self.connection.enable_load_extension(True)
        try:
            self.connection.load_extension(str(self.extension_path))
        finally:
            self.connection.enable_load_extension(False)

    def _read_manifest(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT value FROM bm25f_metadata WHERE key = 'manifest'"
        ).fetchone()
        if row is None:
            raise ValueError("BM25F SQLite artifact is missing its manifest")
        manifest = json.loads(str(row[0]))
        if not isinstance(manifest, dict):
            raise ValueError("BM25F SQLite manifest must be an object")
        return manifest

    def _validate_manifest(
        self,
        expected_catalog_sha256: str | None,
        expected_catalog_rows: int | None,
    ) -> None:
        if self.manifest.get("schema_version") != BM25F_SCHEMA_VERSION:
            raise ValueError("BM25F SQLite schema version is incompatible")
        if tuple(self.manifest.get("fields", ())) != BM25F_FIELDS:
            raise ValueError("BM25F SQLite searchable fields are incompatible")
        if tuple(float(value) for value in self.manifest.get("field_weights", ())) != BM25_FIELD_WEIGHTS:
            raise ValueError("BM25F SQLite field weights are incompatible")
        if tuple(float(value) for value in self.manifest.get("field_b", ())) != BM25F_FIELD_B:
            raise ValueError("BM25F SQLite field b values are incompatible")
        if float(self.manifest.get("k1", -1.0)) != BM25F_K1:
            raise ValueError("BM25F SQLite k1 is incompatible")
        if self.manifest.get("idf_formula") != BM25F_IDF_VERSION:
            raise ValueError("BM25F SQLite IDF formula is incompatible")
        if self.manifest.get("tokenizer") != BM25F_TOKENIZER:
            raise ValueError("BM25F SQLite tokenizer is incompatible")
        if self.manifest.get("document_preprocessing") != BM25F_DOCUMENT_PREPROCESSING:
            raise ValueError("BM25F SQLite document preprocessing is incompatible")
        if expected_catalog_sha256 and self.manifest.get("catalog_sha256") != expected_catalog_sha256:
            raise ValueError("BM25F SQLite catalog hash does not match the loaded catalog")
        if expected_catalog_rows is not None and int(self.manifest.get("product_count", -1)) != int(expected_catalog_rows):
            raise ValueError("BM25F SQLite product count does not match the loaded catalog")

    def _configure_connection(self) -> None:
        self.connection.execute("PRAGMA temp_store = MEMORY")
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS bm25f_allowed "
            "(rowid INTEGER PRIMARY KEY)"
        )
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS bm25f_excluded "
            "(rowid INTEGER PRIMARY KEY)"
        )
        self.connection.commit()

    def _load_mapping(self) -> tuple[tuple[str, ...], dict[str, int]]:
        rows = self.connection.execute(
            "SELECT rowid, parent_asin FROM product_rows ORDER BY rowid"
        ).fetchall()
        asins = tuple(str(asin) for _rowid, asin in rows)
        return asins, {asin: int(rowid) for rowid, asin in rows}

    def _config_parameters(self) -> tuple[float, ...]:
        parameters: list[float] = [BM25F_K1]
        averages = self.manifest["average_field_lengths"]
        for weight, b, field_name in zip(
            BM25_FIELD_WEIGHTS, BM25F_FIELD_B, BM25F_FIELDS
        ):
            parameters.extend((weight, b, float(averages[field_name])))
        return tuple(parameters)

    def search(
        self,
        query_text: str,
        *,
        allowed_asins: Collection[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, float]:
        grams = _query_ngrams(query_text, enabled=self.ngrams_enabled)
        expression = _match_expression(grams)
        if not expression:
            return {}
        limit = self.indexed_rows if top_k is None else max(0, int(top_k))
        if limit == 0:
            return {}
        parameters = self._config_parameters()
        if allowed_asins is None:
            rows = self.connection.execute(
                self._SEARCH_SQL,
                (*parameters, expression, limit),
            ).fetchall()
        else:
            allowed = {
                str(asin) for asin in allowed_asins if str(asin) in self._rowids
            }
            if not allowed:
                return {}
            if len(allowed) > self.indexed_rows // 2:
                excluded_rows = sorted(
                    self._rowids[asin] for asin in self._asins if asin not in allowed
                )
                self.connection.execute("DELETE FROM temp.bm25f_excluded")
                self.connection.executemany(
                    "INSERT INTO temp.bm25f_excluded(rowid) VALUES (?)",
                    ((rowid,) for rowid in excluded_rows),
                )
                rows = self.connection.execute(
                    self._SEARCH_EXCLUDED_SQL,
                    (*parameters, expression, limit),
                ).fetchall()
            else:
                allowed_rows = sorted(self._rowids[asin] for asin in allowed)
                self.connection.execute("DELETE FROM temp.bm25f_allowed")
                self.connection.executemany(
                    "INSERT INTO temp.bm25f_allowed(rowid) VALUES (?)",
                    ((rowid,) for rowid in allowed_rows),
                )
                rows = self.connection.execute(
                    self._SEARCH_ALLOWED_SQL,
                    (*parameters, expression, limit),
                ).fetchall()
        return {str(asin): float(score) for asin, _rowid, score in rows}

    def query_diagnostics(self, query_text: str) -> dict[str, object]:
        """Return cleaned tokens and generated overlapping query grams."""

        terms = _query_terms(query_text)
        by_level = _ngrams_by_level(query_text, enabled=self.ngrams_enabled)
        return {
            "cleaned_tokens": list(terms),
            "n": len(terms),
            "grams_by_level": {
                str(size): [" ".join(gram) for gram in grams]
                for size, grams in by_level.items()
            },
            "total_grams": sum(len(grams) for grams in by_level.values()),
            "ngrams_enabled": self.ngrams_enabled,
        }

    def breakdown(self, query_text: str, asin: str) -> dict[int, float]:
        """Return native aggregate S_k values for one matching product."""

        grams = _query_ngrams(query_text, enabled=self.ngrams_enabled)
        expression = _match_expression(grams)
        if not expression or str(asin) not in self._rowids:
            return {}
        row = self.connection.execute(
            self._BREAKDOWN_SQL,
            (*self._config_parameters(), expression, str(asin)),
        ).fetchone()
        if row is None:
            return {}
        payload = json.loads(str(row[0]))
        return {
            int(level): float(score)
            for level, score in payload.items()
        }


def default_extension_path() -> Path:
    configured = os.environ.get("SHOPPING_BM25F_EXTENSION")
    if configured:
        return Path(configured)
    name = DEFAULT_BM25F_EXTENSION_NAMES.get(sys.platform, "bm25f.so")
    return Path("native") / "build" / name


# Existing integrations import BM25Index. Keep that name as the reference
# implementation; ProductRetriever selects the native artifact explicitly.
BM25Index = PythonBM25FIndex


__all__ = [
    "BM25F_FIELDS",
    "BM25_FIELD_WEIGHTS",
    "BM25F_FIELD_B",
    "BM25F_K1",
    "BM25F_IDF_VERSION",
    "BM25F_TOKENIZER",
    "BM25F_DOCUMENT_PREPROCESSING",
    "PythonBM25FIndex",
    "SQLiteBM25FIndex",
    "BM25Index",
    "DEFAULT_BM25F_ARTIFACT_DIR",
    "default_extension_path",
    "_catalog_sha256",
]
