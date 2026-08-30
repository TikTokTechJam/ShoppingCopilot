"""Generate a labelled intent dataset with the annotation model.

    python -m tests.generate_intent_dataset --buying 100 --browsing 100 --override 200

Writes one JSON file per intent holding ground truth only -- the utterance and
its label. Nothing our workflow predicts is recorded, so the files stay a
fixed benchmark that can be scored against any future version of the
classifier rather than a snapshot of what it happens to do today.

Cases are generated in batches because a single reply cannot hold a hundred of
them, and each batch is seeded from a fresh catalog sample so the vocabulary
keeps moving. Utterances are de-duplicated across batches, and each batch is
held to the same per-intent constraint band the test suite enforces.

INTENT OVERRIDE utterances are generated *with* prior context, because the
model writes a more coherent pivot when it knows what is being abandoned, but
the context is dropped before writing: the saved record is the override
utterance alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tests.utils.catalog_loader import DEFAULT_CATALOG, sample_catalog_context
from tests.utils.intent_generator import generate_synthetic_intent_cases
from tests.utils.intent_workflow import BROWSING, BUYING, INTENT_OVERRIDE
from tests.utils.llm_client import EndpointUnavailable, build_client, preflight


DEFAULT_OUT_DIR = "data/derived/intent_cases"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CATALOG_SAMPLE = 5
# A batch that yields nothing usable twice in a row is not going to start.
MAX_EMPTY_BATCHES = 5


def _slug(intent: str) -> str:
    return intent.lower().replace(" ", "_")


def generate_dataset(
    intent: str,
    target: int,
    client: Any,
    *,
    catalog: str = DEFAULT_CATALOG,
    batch_size: int = DEFAULT_BATCH_SIZE,
    catalog_sample: int = DEFAULT_CATALOG_SAMPLE,
) -> List[Dict[str, Any]]:
    """Collect ``target`` distinct cases for one intent."""

    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    empty_batches = 0
    started = time.perf_counter()

    while len(records) < target and empty_batches < MAX_EMPTY_BATCHES:
        wanted = min(batch_size, target - len(records))
        # A fresh sample per batch: reusing one sample for a hundred cases
        # would make the model repeat the same handful of products.
        context = sample_catalog_context(catalog, catalog_sample)
        try:
            batch = generate_synthetic_intent_cases(
                intent_type=intent,
                catalog_sample=context,
                count=wanted,
                client=client,
                # The minimum band exists to make suite cases hard; here it
                # would bias the corpus toward what our extractor parses
                # finely. The BROWSING maximum stays enforced: it is the
                # definition of the intent, not a difficulty knob.
                enforce_minimum=False,
            )
        except Exception as exc:  # noqa: BLE001 - one bad batch must not end the run
            print(f"[{intent}] batch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            empty_batches += 1
            continue

        added = 0
        for case in batch:
            key = case["utterance"].casefold()
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "id": f"{_slug(intent)}_{len(records) + 1:04d}",
                    "utterance": case["utterance"],
                    "expected_intent": intent,
                }
            )
            added += 1
            if len(records) >= target:
                break

        empty_batches = 0 if added else empty_batches + 1
        elapsed = time.perf_counter() - started
        print(
            f"[{intent}] {len(records)}/{target} cases "
            f"(+{added} this batch, {elapsed:.0f}s)",
            file=sys.stderr,
        )

    if len(records) < target:
        print(
            f"[{intent}] stopped at {len(records)}/{target}: "
            f"{MAX_EMPTY_BATCHES} consecutive batches produced nothing new",
            file=sys.stderr,
        )
    return records


def write_dataset(
    intent: str,
    records: List[Dict[str, Any]],
    out_dir: str,
) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(intent)}.json"
    path.write_text(
        json.dumps(
            {
                "intent": intent,
                "count": len(records),
                "cases": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a labelled intent dataset with the annotation model",
    )
    parser.add_argument("--buying", type=int, default=100)
    parser.add_argument("--browsing", type=int, default=100)
    parser.add_argument("--override", type=int, default=200)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--catalog-sample", type=int, default=DEFAULT_CATALOG_SAMPLE)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        client = build_client(timeout=args.timeout)
    except EndpointUnavailable as exc:
        print(f"annotation endpoint unavailable: {exc}", file=sys.stderr)
        return 2
    failure = preflight(client)
    if failure:
        print(f"annotation endpoint unreachable: {failure}", file=sys.stderr)
        return 2
    print(f"generating with model {client.model!r}", file=sys.stderr)

    plan = ((BUYING, args.buying), (BROWSING, args.browsing), (INTENT_OVERRIDE, args.override))
    written: List[Path] = []
    for intent, target in plan:
        if target <= 0:
            continue
        records = generate_dataset(
            intent,
            target,
            client,
            catalog=args.catalog,
            batch_size=args.batch_size,
            catalog_sample=args.catalog_sample,
        )
        path = write_dataset(intent, records, args.out_dir)
        written.append(path)
        print(f"[{intent}] wrote {len(records)} cases to {path}", file=sys.stderr)

    print(json.dumps({"files": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
