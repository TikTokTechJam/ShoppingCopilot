"""Catalog dataset reader for the synthetic intent generator (Task 1).

The generator needs *real* product vocabulary so its utterances name things
the catalog actually sells. Sampling is reservoir-based so a 50,000-line,
60 MB catalog costs one streaming pass and a handful of retained rows rather
than holding every record in memory for a five-product sample.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List


DEFAULT_CATALOG = "data/catalog.jsonl"

# Fields worth putting in a prompt. The raw rows carry long descriptions and
# feature lists that would crowd out the instructions without adding
# vocabulary the generator cannot already see in the title.
CONTEXT_FIELDS = ("parent_asin", "title", "categories", "store", "price")


def sample_catalog_context(
    filepath: str = DEFAULT_CATALOG,
    sample_size: int = 5,
    *,
    seed: int | None = None,
) -> List[Dict]:
    """Reads data/catalog.jsonl and returns a random sample of product records.

    ``seed`` makes a run reproducible; leaving it ``None`` keeps the fresh
    sampling the specification asks for, so repeated runs explore different
    product vocabulary.
    """

    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(record)
                continue
            index = rng.randrange(seen)
            if index < sample_size:
                reservoir[index] = record
    return reservoir


def summarize_products(products: List[Dict]) -> List[Dict]:
    """Compact catalog rows to the fields a prompt can actually use."""

    summaries: List[Dict[str, Any]] = []
    for product in products:
        summary: Dict[str, Any] = {}
        for field in CONTEXT_FIELDS:
            value = product.get(field)
            if value in (None, "", [], {}):
                continue
            if field == "title":
                value = str(value)[:140]
            elif field == "categories" and isinstance(value, (list, tuple)):
                value = [str(item) for item in value][:5]
            summary[field] = value
        if summary:
            summaries.append(summary)
    return summaries


__all__ = ["CONTEXT_FIELDS", "DEFAULT_CATALOG", "sample_catalog_context", "summarize_products"]
