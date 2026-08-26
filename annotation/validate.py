from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .runner import iter_catalog
from .schema import (
    CANONICAL_FACT_FIELDS,
    CANONICAL_RECORD_FIELDS,
    normalize_canonical_facts,
    normalize_price,
)


def validate_catalog_facts(
    catalog_path: str | Path,
    facts_path: str | Path,
) -> dict[str, Any]:
    source_asins: set[str] = set()
    source_count = 0
    for product in iter_catalog(catalog_path):
        source_count += 1
        source_asins.add(product["parent_asin"])

    fact_asins: set[str] = set()
    with Path(facts_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{facts_path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{facts_path}:{line_number}: record must be an object")
            if set(record) != set(CANONICAL_RECORD_FIELDS):
                raise ValueError(f"{facts_path}:{line_number}: canonical keys mismatch")

            parent_asin = record["parent_asin"]
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"{facts_path}:{line_number}: missing parent_asin")
            if parent_asin in fact_asins:
                raise ValueError(f"{facts_path}:{line_number}: duplicate parent_asin")
            fact_asins.add(parent_asin)

            price = record["price"]
            if price is not None and (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(float(price))
            ):
                raise ValueError(f"{facts_path}:{line_number}: price must be numeric or null")

            canonical = {field: record[field] for field in CANONICAL_FACT_FIELDS}
            if normalize_canonical_facts(canonical) != canonical:
                raise ValueError(f"{facts_path}:{line_number}: facts are not normalized")
            if normalize_price(price) != price:
                raise ValueError(f"{facts_path}:{line_number}: invalid normalized price")

    missing = sorted(source_asins - fact_asins)
    extra = sorted(fact_asins - source_asins)
    if missing or extra:
        raise ValueError(f"catalog/facts ASIN mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if source_count != len(fact_asins):
        raise ValueError("catalog and facts record counts differ")

    return {
        "source_product_count": source_count,
        "facts_record_count": len(fact_asins),
        "read_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical catalog facts without modifying them.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--facts", default="data/derived/catalog_facts/catalog_facts.jsonl")
    args = parser.parse_args()
    print(json.dumps(validate_catalog_facts(args.catalog, args.facts), indent=2))


if __name__ == "__main__":
    main()
