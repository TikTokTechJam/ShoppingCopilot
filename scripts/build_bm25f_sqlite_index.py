"""Build the read-only SQLite FTS5 artifact used by native BM25F."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from starter.bm25 import (
    BM25F_FIELDS,
    BM25F_FIELD_B,
    BM25F_DOCUMENT_PREPROCESSING,
    BM25F_IDF_VERSION,
    BM25F_K1,
    BM25F_SCHEMA_VERSION,
    BM25F_TOKENIZER,
    BM25_FIELD_WEIGHTS,
    DEFAULT_BM25F_ARTIFACT_DIR,
    DEFAULT_BM25F_DB_NAME,
    _catalog_sha256,
    _field_tokens,
)


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"catalog row {line_number} must be an object")
            asin = str(value.get("parent_asin", "")).strip()
            if not asin:
                raise ValueError(f"catalog row {line_number} has no parent_asin")
            rows.append(value)
    return rows


def _tokenized_fields(row: dict[str, Any]) -> tuple[str, ...]:
    # Materialize the same conservative tokens used by the Python reference.
    # Feeding token strings to unicode61 keeps native xColumnSize/xInst counts
    # exactly aligned with the reference even for punctuation/Unicode edge
    # cases, while retaining the existing unicode61 token semantics.
    return tuple(
        " ".join(_field_tokens(row.get(field_name)))
        for field_name in BM25F_FIELDS
    )


def _manifest(
    catalog: Path,
    rows: list[dict[str, Any]],
    totals: list[int],
    *,
    batch_size: int,
) -> dict[str, Any]:
    count = len(rows)
    return {
        "schema_version": BM25F_SCHEMA_VERSION,
        "source_path": str(catalog),
        "catalog_sha256": _catalog_sha256(catalog),
        "product_count": count,
        "fields": list(BM25F_FIELDS),
        "field_weights": list(BM25_FIELD_WEIGHTS),
        "field_b": list(BM25F_FIELD_B),
        "k1": BM25F_K1,
        "idf_formula": BM25F_IDF_VERSION,
        "tokenizer": BM25F_TOKENIZER,
        "document_preprocessing": BM25F_DOCUMENT_PREPROCESSING,
        "average_field_lengths": {
            field_name: (total / count if count else 0.0)
            for field_name, total in zip(BM25F_FIELDS, totals)
        },
        "field_token_totals": {
            field_name: total
            for field_name, total in zip(BM25F_FIELDS, totals)
        },
        "row_mapping": "product_rows.rowid equals product_fts.rowid; catalog order is rowid-1",
        "generation_version": "sqlite-bm25f-v1",
        "batch_size": batch_size,
        "created_at_epoch": time.time(),
    }


def build_index(catalog: Path, output_dir: Path, *, batch_size: int, force: bool) -> Path:
    rows = _read_catalog(catalog)
    asins = [str(row["parent_asin"]).strip() for row in rows]
    if len(set(asins)) != len(asins):
        raise ValueError("catalog contains duplicate parent_asin values")
    totals = [0] * len(BM25F_FIELDS)
    for row in rows:
        fields = _tokenized_fields(row)
        for index, field_value in enumerate(fields):
            totals[index] += len(field_value.split())

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / DEFAULT_BM25F_DB_NAME
    manifest_path = output_dir / "manifest.json"
    if not force and (db_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"{output_dir} already contains a BM25F artifact; pass --force to rebuild"
        )
    if db_path.exists():
        db_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    started = time.perf_counter()
    connection = sqlite3.connect(str(db_path))
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF;"
            "PRAGMA synchronous=OFF;"
            "PRAGMA temp_store=MEMORY;"
            "CREATE TABLE product_rows ("
            "rowid INTEGER PRIMARY KEY,"
            "parent_asin TEXT NOT NULL UNIQUE,"
            "catalog_row INTEGER NOT NULL UNIQUE"
            ");"
            "CREATE VIRTUAL TABLE product_fts USING fts5("
            "title, categories, features, details, store, description,"
            "tokenize='unicode61 remove_diacritics 2', detail=full"
            ");"
            "CREATE TABLE bm25f_metadata ("
            "key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL"
            ");"
            "PRAGMA user_version=1;"
        )
        row_batch: list[tuple[int, str, int]] = []
        fts_batch: list[tuple[int, *tuple[str, ...]]] = []
        for catalog_row, (asin, row) in enumerate(zip(asins, rows)):
            rowid = catalog_row + 1
            fields = _tokenized_fields(row)
            row_batch.append((rowid, asin, catalog_row))
            fts_batch.append((rowid, *fields))
            if len(row_batch) >= batch_size:
                connection.executemany(
                    "INSERT INTO product_rows(rowid,parent_asin,catalog_row) VALUES (?,?,?)",
                    row_batch,
                )
                connection.executemany(
                    "INSERT INTO product_fts(rowid,title,categories,features,details,store,description) "
                    "VALUES (?,?,?,?,?,?,?)",
                    fts_batch,
                )
                row_batch.clear()
                fts_batch.clear()
                print(f"[bm25f] indexed {catalog_row + 1:,}/{len(rows):,}", flush=True)
        if row_batch:
            connection.executemany(
                "INSERT INTO product_rows(rowid,parent_asin,catalog_row) VALUES (?,?,?)",
                row_batch,
            )
            connection.executemany(
                "INSERT INTO product_fts(rowid,title,categories,features,details,store,description) "
                "VALUES (?,?,?,?,?,?,?)",
                fts_batch,
            )
        connection.execute("INSERT INTO product_fts(product_fts) VALUES ('optimize')")
        manifest = _manifest(catalog, rows, totals, batch_size=batch_size)
        connection.execute(
            "INSERT INTO bm25f_metadata(key,value) VALUES ('manifest',?)",
            (json.dumps(manifest, sort_keys=True),),
        )
        connection.commit()
        product_count = int(connection.execute("SELECT count(*) FROM product_rows").fetchone()[0])
        fts_count = int(connection.execute("SELECT count(*) FROM product_fts").fetchone()[0])
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if product_count != len(rows) or fts_count != len(rows) or integrity != "ok":
            raise RuntimeError(
                f"artifact validation failed: rows={product_count}, fts={fts_count}, integrity={integrity}"
            )
    finally:
        connection.close()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[bm25f] ready: {len(rows):,} products, "
        f"{time.perf_counter() - started:.1f}s, {db_path}",
        flush=True,
    )
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BM25F_ARTIFACT_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    build_index(args.catalog, args.output_dir, batch_size=args.batch_size, force=args.force)


if __name__ == "__main__":
    main()
