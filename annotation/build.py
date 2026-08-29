from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runner import iter_catalog
from .schema import canonical_record_from_annotation, normalize_price, validate_annotation_record


def _read_annotations(path: str | Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            record = validate_annotation_record(raw)
            parent_asin = record["parent_asin"]
            if parent_asin in annotations:
                raise ValueError(f"{path}:{line_number}: duplicate annotation {parent_asin}")
            annotations[parent_asin] = record
    return annotations


def build_catalog_facts(
    catalog_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    annotations = _read_annotations(annotations_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    source_count = 0
    written_asins: set[str] = set()

    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for product in iter_catalog(catalog_path):
                source_count += 1
                parent_asin = product["parent_asin"]
                record = annotations.get(parent_asin)
                if record is None:
                    raise ValueError(f"missing annotation for {parent_asin}")
                source_price = normalize_price(product.get("price"))
                if record["price"] != source_price:
                    raise ValueError(f"price mismatch for {parent_asin}")
                canonical = canonical_record_from_annotation(record)
                written_asins.add(parent_asin)
                handle.write(
                    json.dumps(canonical, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

            extra = sorted(set(annotations) - written_asins)
            if extra:
                raise ValueError(f"annotation contains ASINs absent from catalog: {extra[:5]}")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {"source_product_count": source_count, "facts_record_count": len(written_asins)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic canonical catalog facts.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--annotations", default="data/derived/annotations/v5/annotations.jsonl")
    parser.add_argument("--output", default="data/derived/catalog_facts/catalog_facts.jsonl")
    args = parser.parse_args()
    print(json.dumps(build_catalog_facts(args.catalog, args.annotations, args.output), indent=2))


if __name__ == "__main__":
    main()
