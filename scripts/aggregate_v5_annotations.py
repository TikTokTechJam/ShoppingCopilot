"""Aggregate the V5 single-attribute annotation files."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from annotation.runner import iter_catalog
from annotation.schema import normalize_price, validate_annotation_record


V5_ATTRIBUTES = (
    "category",
    "brand",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
)
DEFAULT_CATALOG = Path("data/catalog.jsonl")
DEFAULT_INPUT_DIR = Path("data/derived/annotations/v5")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "annotations.jsonl"
DEFAULT_STYLE_FALLBACK = Path("data/derived/annotations/v4/annotations.jsonl")


def _normalize_value(value: str) -> str:
    """Apply only deterministic display cleanup to an already annotated value."""

    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def _deduplicate_values(
    raw_values: Any,
    *,
    path: Path,
    line_number: int,
    attribute: str,
) -> list[str]:
    if not isinstance(raw_values, list):
        raise ValueError(
            f"{path}:{line_number}: {attribute} must be an array of strings"
        )
    values: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str):
            raise ValueError(
                f"{path}:{line_number}: {attribute} must contain only strings"
            )
        normalized = _normalize_value(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(normalized)
    return values


def _read_attribute_file(
    path: Path,
    attribute: str,
    catalog_asins: set[str],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Read one V5 file and validate its ASIN relationship to the catalog."""

    if not path.exists():
        return {}, {
            "file_present": False,
            "source": "v5",
            "rows_read": 0,
            "unique_asins": 0,
            "missing_records": len(catalog_asins),
            "empty_values": len(catalog_asins),
        }

    records: dict[str, list[str]] = {}
    empty_values = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            if set(record) != {"parent_asin", attribute}:
                raise ValueError(
                    f"{path}:{line_number}: expected keys "
                    f"['parent_asin', '{attribute}']"
                )

            parent_asin = record.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"{path}:{line_number}: missing parent_asin")
            parent_asin = parent_asin.strip()
            if parent_asin not in catalog_asins:
                raise ValueError(
                    f"{path}:{line_number}: parent_asin is absent from catalog: "
                    f"{parent_asin}"
                )
            if parent_asin in records:
                raise ValueError(
                    f"{path}:{line_number}: duplicate parent_asin: {parent_asin}"
                )

            values = _deduplicate_values(
                record[attribute],
                path=path,
                line_number=line_number,
                attribute=attribute,
            )
            if not values:
                empty_values += 1
            records[parent_asin] = values

    missing_asins = catalog_asins - records.keys()
    return records, {
        "file_present": True,
        "source": "v5",
        "rows_read": len(records),
        "unique_asins": len(records),
        "missing_records": len(missing_asins),
        "missing_asin_examples": sorted(missing_asins)[:5],
        "empty_values": empty_values,
    }


def _read_legacy_style_file(
    path: Path,
    catalog_asins: set[str],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Read style from the existing full V4 annotations when needed.

    V5 was originally produced for six single-field files and intentionally
    left style empty. The V4 records already contain a validated style field,
    so they are a safe local migration source until a dedicated V5
    ``style.jsonl`` is produced. An explicit V5 style file always takes
    precedence over this fallback.
    """

    if not path.exists():
        return {}, {
            "file_present": False,
            "source": "v4_fallback",
            "rows_read": 0,
            "unique_asins": 0,
            "missing_records": len(catalog_asins),
            "empty_values": len(catalog_asins),
        }

    records: dict[str, list[str]] = {}
    empty_values = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
                record = validate_annotation_record(raw_record)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid V4 annotation") from exc

            parent_asin = record["parent_asin"]
            if parent_asin not in catalog_asins:
                raise ValueError(
                    f"{path}:{line_number}: parent_asin is absent from catalog: "
                    f"{parent_asin}"
                )
            if parent_asin in records:
                raise ValueError(
                    f"{path}:{line_number}: duplicate annotation {parent_asin}"
                )

            values = _deduplicate_values(
                record["facts"]["style"],
                path=path,
                line_number=line_number,
                attribute="style",
            )
            if not values:
                empty_values += 1
            records[parent_asin] = values

    missing_asins = catalog_asins - records.keys()
    return records, {
        "file_present": True,
        "source": "v4_fallback",
        "rows_read": len(records),
        "unique_asins": len(records),
        "missing_records": len(missing_asins),
        "missing_asin_examples": sorted(missing_asins)[:5],
        "empty_values": empty_values,
    }


def aggregate_v5_annotations(
    catalog_path: str | Path = DEFAULT_CATALOG,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
    style_fallback_path: str | Path | None = DEFAULT_STYLE_FALLBACK,
) -> dict[str, Any]:
    """Join V5 annotations onto every catalog row in catalog order."""

    catalog_products = list(iter_catalog(catalog_path))
    catalog_asins = {str(product["parent_asin"]) for product in catalog_products}
    input_root = Path(input_dir)

    annotations: dict[str, dict[str, list[str]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for attribute in V5_ATTRIBUTES:
        attribute_path = input_root / f"{attribute}.jsonl"
        if attribute == "style" and not attribute_path.exists() and style_fallback_path:
            annotations[attribute], reports[attribute] = _read_legacy_style_file(
                Path(style_fallback_path),
                catalog_asins,
            )
        else:
            annotations[attribute], reports[attribute] = _read_attribute_file(
                attribute_path,
                attribute,
                catalog_asins,
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    written_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for product in catalog_products:
                parent_asin = str(product["parent_asin"])
                facts = {
                    attribute: annotations[attribute].get(parent_asin, [])
                    for attribute in V5_ATTRIBUTES
                }
                record = {
                    "parent_asin": parent_asin,
                    "price": normalize_price(product.get("price")),
                    "facts": facts,
                }
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                written_count += 1
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    if written_count != len(catalog_products):
        raise RuntimeError(
            "aggregation did not emit one output record per catalog product: "
            f"{written_count} != {len(catalog_products)}"
        )

    return {
        "catalog_product_count": len(catalog_products),
        "output_product_count": written_count,
        "attributes": list(V5_ATTRIBUTES),
        "rows_read_by_attribute": {
            attribute: reports[attribute]["rows_read"] for attribute in V5_ATTRIBUTES
        },
        "missing_records_by_attribute": {
            attribute: reports[attribute]["missing_records"]
            for attribute in V5_ATTRIBUTES
        },
        "empty_values_by_attribute": {
            attribute: reports[attribute]["empty_values"]
            for attribute in V5_ATTRIBUTES
        },
        "missing_asin_examples_by_attribute": {
            attribute: reports[attribute].get("missing_asin_examples", [])
            for attribute in V5_ATTRIBUTES
            if reports[attribute]["missing_records"]
        },
        "missing_files": [
            attribute
            for attribute in V5_ATTRIBUTES
            if not reports[attribute]["file_present"]
        ],
        "style_source": reports["style"].get("source"),
        "output_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate catalog-ordered V5 attribute annotations."
    )
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--style-fallback",
        default=str(DEFAULT_STYLE_FALLBACK),
        help="Existing V4 annotation JSONL used only when V5 style.jsonl is absent",
    )
    parser.add_argument(
        "--no-style-fallback",
        action="store_true",
        help="Require style.jsonl instead of reading the V4 fallback",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_v5_annotations(
                args.catalog,
                args.input_dir,
                args.output,
                None if args.no_style_fallback else args.style_fallback,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
