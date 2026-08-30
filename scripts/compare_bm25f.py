"""Compare native SQLite BM25F unigram and overlapping n-gram modes."""

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
    _ngrams_by_level,
    _query_terms,
)


DEFAULT_QUERIES = (
    "running",
    "running shoes",
    "waterproof hiking shoes",
    "machine washable jacket",
    "polarized sunglasses",
    "cotton shirt",
    "everyday wear",
    "stainless steel",
    "non slip shoes",
    "black leather hiking shoes",
    "machine washable cotton hiking jacket",
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


def _timing_samples(index: object, query: str, repeats: int) -> list[float]:
    search = index.search  # type: ignore[attr-defined]
    for _ in range(3):
        search(query, top_k=100)
    values: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        search(query, top_k=100)
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def _summary(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return (
        statistics.fmean(ordered),
        statistics.median(ordered),
        ordered[position],
    )


def _format_top10(scores: dict[str, float]) -> str:
    return ", ".join(
        f"{position}:{asin}({score:.4f})"
        for position, (asin, score) in enumerate(list(scores.items())[:10], 1)
    ) or "<none>"


def _rank_map(scores: dict[str, float]) -> dict[str, int]:
    return {asin: position for position, asin in enumerate(scores, 1)}


def _print_expansion(query: str) -> int:
    terms = _query_terms(query)
    by_level = _ngrams_by_level(query, enabled=True)
    print(f"  cleaned_tokens={list(terms)!r} n={len(terms)}")
    for level in range(1, len(terms) + 1):
        grams = [" ".join(gram) for gram in by_level.get(level, ())]
        print(f"  G{level} ({len(grams)}): {grams}")
    print(f"  total_grams={sum(len(grams) for grams in by_level.values())}")
    return len(terms)


def _print_rank_changes(
    baseline: dict[str, float], ngrams: dict[str, float]
) -> None:
    baseline_top = list(baseline)[:10]
    ngram_top = list(ngrams)[:10]
    baseline_ranks = _rank_map(baseline)
    ngram_ranks = _rank_map(ngrams)
    union = list(dict.fromkeys([*baseline_top, *ngram_top]))
    changes = []
    for asin in union:
        old = baseline_ranks.get(asin, "-")
        new = ngram_ranks.get(asin, "-")
        if old != new:
            changes.append(f"{asin}:{old}->{new}")
    overlap = len(set(baseline_top) & set(ngram_top))
    print(f"  top10_overlap={overlap}/10")
    print(f"  top10_rank_changes={', '.join(changes) if changes else '<none>'}")


def _print_breakdown(
    index: SQLiteBM25FIndex,
    query: str,
    scores: dict[str, float],
    limit: int,
) -> None:
    for asin in list(scores)[: max(0, limit)]:
        levels = index.breakdown(query, asin)
        total = sum(levels.values())
        formatted = ", ".join(
            f"S{level}={levels[level]:.6f}" for level in sorted(levels)
        )
        print(f"  {asin}: {formatted}; Final={total:.6f}")


def _print_reference_parity(
    reference: PythonBM25FIndex,
    native: SQLiteBM25FIndex,
    query: str,
) -> None:
    expected = reference.search(query)
    actual = native.search(query)
    common = set(expected) & set(actual)
    deltas = [abs(expected[asin] - actual[asin]) for asin in common]
    print(
        "  python_reference_parity: "
        f"matching={len(common):,} "
        f"top10_same={list(expected)[:10] == list(actual)[:10]} "
        f"max_abs={max(deltas, default=0.0):.12g} "
        f"mean_abs={statistics.fmean(deltas) if deltas else 0.0:.12g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_BM25F_ARTIFACT_DIR / DEFAULT_BM25F_DB_NAME
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "ngrams", "both"),
        default="both",
        help="native scorer mode to inspect (default: both)",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--breakdown-top",
        type=int,
        default=3,
        help="number of n-gram results for which to print S_k values",
    )
    parser.add_argument(
        "--no-reference-parity",
        action="store_true",
        help="skip the Python-reference versus native-unigram comparison",
    )
    parser.add_argument("queries", nargs="*", default=list(DEFAULT_QUERIES))
    args = parser.parse_args()

    rows, order = _load_catalog(args.catalog)
    reference = None
    if not args.no_reference_parity and args.mode in {"baseline", "both"}:
        reference = PythonBM25FIndex(rows, order, ngrams_enabled=False)

    baseline = None
    ngrams = None
    if args.mode in {"baseline", "both"}:
        baseline = SQLiteBM25FIndex(args.db, ngrams_enabled=False)
    if args.mode in {"ngrams", "both"}:
        ngrams = SQLiteBM25FIndex(args.db, ngrams_enabled=True)

    timing_values: dict[str, dict[int, list[float]]] = {}
    for query in args.queries:
        print(f"\nquery={query!r}")
        n = _print_expansion(query)

        baseline_scores = (
            baseline.search(query, top_k=10) if baseline is not None else None
        )
        ngram_scores = ngrams.search(query, top_k=10) if ngrams is not None else None

        if baseline_scores is not None:
            print(f"  baseline_top10={_format_top10(baseline_scores)}")
            if reference is not None and baseline is not None:
                _print_reference_parity(reference, baseline, query)
        if ngram_scores is not None:
            print(f"  ngram_top10={_format_top10(ngram_scores)}")
            _print_breakdown(ngrams, query, ngram_scores, args.breakdown_top)
        if baseline_scores is not None and ngram_scores is not None:
            _print_rank_changes(baseline_scores, ngram_scores)
            if n == 1:
                baseline_all = baseline.search(query)
                ngram_all = ngrams.search(query)
                common = set(baseline_all) & set(ngram_all)
                deltas = [
                    abs(baseline_all[asin] - ngram_all[asin]) for asin in common
                ]
                print(
                    "  single_token_score_delta: "
                    f"max_abs={max(deltas, default=0.0):.12g} "
                    f"mean_abs={statistics.fmean(deltas) if deltas else 0.0:.12g}"
                )

        for label, index in (("baseline", baseline), ("ngrams", ngrams)):
            if index is None:
                continue
            values = _timing_samples(index, query, args.repeats)
            timing_values.setdefault(label, {}).setdefault(n, []).extend(values)
            mean, median, p95 = _summary(values)
            print(
                f"  {label}_latency_ms: mean={mean:.3f} "
                f"median={median:.3f} p95={p95:.3f}"
            )

    print("\nlatency_by_cleaned_query_length_ms (warm, top_k=100):")
    for label, by_length in timing_values.items():
        for length in sorted(by_length):
            mean, median, p95 = _summary(by_length[length])
            print(
                f"  {label} n={length}: mean={mean:.3f} "
                f"median={median:.3f} p95={p95:.3f} "
                f"samples={len(by_length[length])}"
            )


if __name__ == "__main__":
    main()
