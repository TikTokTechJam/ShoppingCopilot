"""Deterministic V5 semantic product-card construction for Browsing."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


BROWSING_CARD_FIELDS: tuple[str, ...] = (
    "category",
    "brand",
    "color",
    "material",
    "feature",
    "use_case",
    "style",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    source: Iterable[object]
    if isinstance(value, str):
        source = (value,)
    elif isinstance(value, (list, tuple)):
        source = value
    else:
        return ()
    result: list[str] = []
    for item in source:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _facts(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("facts")
    return nested if isinstance(nested, Mapping) else row


def _asin(row: Mapping[str, Any], *, path: Path, row_number: int) -> str:
    value = row.get("parent_asin")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{row_number}: missing parent_asin")
    return value.strip()


def serialize_product_card(
    product: Mapping[str, Any],
    facts: Mapping[str, Any] | None = None,
) -> str:
    """Serialize one product as a compact plain document for Qwen.

    Product documents deliberately contain no ``Instruct:`` or ``Query:``
    prefix. The query-side instruction is added only by ``format_qwen_query``.
    """

    selected_facts = facts or {}
    lines: list[str] = []
    for field_name in BROWSING_CARD_FIELDS:
        values = _values(selected_facts.get(field_name))
        if values:
            lines.append(f"{field_name}: {', '.join(values)}")
    title = product.get("title")
    if isinstance(title, str):
        title = " ".join(title.split())
        if title:
            lines.append(f"title: {title}")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def build_product_cards(
    catalog_path: str | Path,
    annotations_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build cards in exact catalog order from the V5 annotation artifact."""

    catalog = Path(catalog_path)
    annotations = Path(annotations_path)
    output = Path(output_dir)
    catalog_rows = _read_jsonl(catalog)
    annotation_rows = _read_jsonl(annotations)

    annotation_by_asin: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(annotation_rows, 1):
        asin = _asin(row, path=annotations, row_number=row_number)
        if asin in annotation_by_asin:
            raise ValueError(f"{annotations}:{row_number}: duplicate parent_asin {asin}")
        annotation_by_asin[asin] = _facts(row)

    cards: list[dict[str, str]] = []
    seen_catalog: set[str] = set()
    missing_annotations = 0
    empty_cards = 0
    for row_number, row in enumerate(catalog_rows, 1):
        asin = _asin(row, path=catalog, row_number=row_number)
        if asin in seen_catalog:
            raise ValueError(f"{catalog}:{row_number}: duplicate parent_asin {asin}")
        seen_catalog.add(asin)
        facts = annotation_by_asin.get(asin)
        if facts is None:
            missing_annotations += 1
            facts = {}
        text = serialize_product_card(row, facts)
        if not text:
            empty_cards += 1
        cards.append({"parent_asin": asin, "text": text})

    card_path = output / "product_cards.jsonl"
    card_text = "".join(
        json.dumps(card, ensure_ascii=False, separators=(",", ":")) + "\n"
        for card in cards
    )
    _atomic_write(card_path, card_text)

    manifest: dict[str, Any] = {
        "schema_version": "browsing-product-card-v1",
        "representation": "v5_semantic_product_card",
        "catalog_path": str(catalog),
        "catalog_sha256": _sha256(catalog),
        "annotations_path": str(annotations),
        "annotations_sha256": _sha256(annotations),
        "product_count": len(cards),
        "annotation_field_order": list(BROWSING_CARD_FIELDS),
        "row_order": "catalog.jsonl order",
        "phrase_ordering": "V5 annotation array order; title last",
        "missing_annotation_rows": missing_annotations,
        "empty_card_count": empty_cards,
    }
    _atomic_write(
        output / "product_cards_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return manifest


__all__ = [
    "BROWSING_CARD_FIELDS",
    "build_product_cards",
    "serialize_product_card",
]
