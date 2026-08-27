"""Tier 4 raw catalog-source extraction.

Tier 4 deliberately keeps source text separate from canonical facts. It is a
small derived view of the immutable catalog and is used as recall context when
constructing whole-product embedding documents.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TIER4_ARTIFACT_VERSION = "tier4-raw-text-v1"
TIER4_FIELDS = ("parent_asin", "title", "features", "description", "details")


def _clean_text(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    return " ".join(value.split())


def _clean_text_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise TypeError(f"{field} must be an array of strings, a string, or null")

    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError(f"{field} must contain only strings")
        result.append(" ".join(item.split()))
    return result


def _copy_json_value(value: Any, *, field: str) -> Any:
    """Validate and copy a catalog detail value without interpreting it."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must contain JSON-serializable values") from exc


def normalize_tier4_source(
    product: Mapping[str, Any],
    *,
    require_parent_asin: bool = False,
) -> dict[str, Any]:
    """Return the stable raw-text view used by Tier 4.

    The source catalog is never modified. Text arrays are whitespace
    normalized while retaining source order and duplicates; ``details`` keeps
    its object shape and values so unusual catalog language is not discarded.
    """
    if not isinstance(product, Mapping):
        raise TypeError("product must be an object")

    parent_asin = product.get("parent_asin")
    if require_parent_asin and (
        not isinstance(parent_asin, str) or not parent_asin.strip()
    ):
        raise ValueError("tier 4 record requires a non-empty parent_asin")

    details = product.get("details")
    if details is None:
        normalized_details: dict[str, Any] = {}
    elif isinstance(details, Mapping):
        normalized_details = {
            str(key): _copy_json_value(value, field="details")
            for key, value in details.items()
        }
    else:
        raise TypeError("details must be an object or null")

    result: dict[str, Any] = {
        "parent_asin": "" if parent_asin is None else str(parent_asin).strip(),
        "title": _clean_text(product.get("title"), field="title"),
        "features": _clean_text_list(product.get("features", []), field="features"),
        "description": _clean_text_list(
            product.get("description", []), field="description"
        ),
        "details": normalized_details,
    }
    if not require_parent_asin and not parent_asin:
        result.pop("parent_asin")
    return result


def build_tier4_record(product: Mapping[str, Any]) -> dict[str, Any]:
    """Build one ``parent_asin/title/features/description/details`` record."""
    return normalize_tier4_source(product, require_parent_asin=True)


def _read_catalog_jsonl(path: Path) -> Sequence[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
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
            normalized = build_tier4_record(record)
            asin = normalized["parent_asin"]
            if asin in seen:
                raise ValueError(f"{path}:{line_number}: duplicate parent_asin {asin}")
            seen.add(asin)
            records.append(normalized)
    if not records:
        raise ValueError(f"{path}: catalog contains no products")
    return records


def build_tier4_raw_text(
    catalog_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Extract the Tier 4 raw-text JSONL artifact from the immutable catalog."""
    catalog = Path(catalog_path)
    output = Path(output_path)
    records = _read_catalog_jsonl(catalog)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "artifact_format_version": TIER4_ARTIFACT_VERSION,
        "catalog_path": catalog.as_posix(),
        "output_path": output.as_posix(),
        "product_count": len(records),
    }


__all__ = [
    "TIER4_ARTIFACT_VERSION",
    "TIER4_FIELDS",
    "build_tier4_raw_text",
    "build_tier4_record",
    "normalize_tier4_source",
]
