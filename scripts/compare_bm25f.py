"""Compare Python-reference and native SQLite BM25F on fixed queries."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from starter.bm25 import (
    DEFAULT_BM25F_ARTIFACT_DIR,
    DEFAULT_BM25F_DB_NAME,
    PythonBM25FIndex,
    SQLiteBM25FIndex,
)


DEFAULT_QUERIES = (
    "waterproof",
    "hiking shoes",
    "waterproof hiking shoes",
    "machine washable jacket",
    "polarized sunglasses",
    "cotton shirt",
    "everyday wear",
    "stainless steel",
    "non slip shoes",
)


def _load_catalog(path: Path) -> tuple[dict[str, dict], list[str]]:
    rows: dict[str, dict] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            asin = str(row["parent_asin"])
            rows[asin] = row
            order.append(asin)
    return rows, order


def _timing(index: object, query: str, repeats: int) -> tuple[float, float, float]:
    search = index.search  # type: ignore[attr-defined]
    for _ in range(3):
        search(query, top_k=100)
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        search(query, top_k=100)
        values.append((time.perf_counter() - started) * 1000.0)
    values.sort()
    return (
        statistics.fmean(values),
        statistics.median(values),
        values[max(0, int(len(values) * 0.95) - 1)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_BM25F_ARTIFACT_DIR / DEFAULT_BM25F_DB_NAME
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("queries", nargs="*", default=list(DEFAULT_QUERIES))
    args = parser.parse_args()
    rows, order = _load_catalog(args.catalog)
    reference = PythonBM25FIndex(rows, order)
    native = SQLiteBM25FIndex(args.db)
    differences: list[float] = []
    for query in args.queries:
        expected = reference.search(query)
        actual = native.search(query)
        common = set(expected) & set(actual)
        deltas = [abs(expected[asin] - actual[asin]) for asin in common]
        differences.extend(deltas)
        print(
            f"{query!r}: matching={len(common):,} "
            f"top10_same={list(expected)[:10] == list(actual)[:10]} "
            f"max_abs={max(deltas, default=0.0):.12g} "
            f"mean_abs={statistics.fmean(deltas) if deltas else 0.0:.12g}"
        )
    print(
        f"all matching scores: max_abs={max(differences, default=0.0):.12g} "
        f"mean_abs={statistics.fmean(differences) if differences else 0.0:.12g}"
    )
    print("latency_ms (warm, top_k=100):")
    for index in (reference, native):
        mean, median, p95 = _timing(index, "waterproof hiking shoes", args.repeats)
        print(
            f"  {index.backend}: mean={mean:.3f} median={median:.3f} p95={p95:.3f}"
        )


if __name__ == "__main__":
    main()
