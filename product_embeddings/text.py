from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PRODUCT_TEXT_VERSION = "product-text-v1"
DESCRIPTION_MAX_CHARS = 1_000

# Keep this order stable. It is part of the embedding document contract.
_FACT_TEXT_FIELDS = (
    ("Category", "category"),
    ("Brand", "brand"),
    ("Material", "material"),
    ("Color", "color"),
    ("Style", "style"),
    ("Features", "feature"),
    ("Use cases", "use_case"),
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("product text values must be strings or null")
    return " ".join(value.split())


def _clean_sequence(value: Any, *, field: str, sort_values: bool) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise TypeError(f"{field} must be a string, array, or null")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _clean_text(item)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    if sort_values:
        cleaned.sort(key=lambda item: (item.casefold(), item))
    return cleaned


def _facts_payload(facts_record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept canonical facts and the Issue #5 annotation wrapper."""
    nested = facts_record.get("facts")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("facts must be an object when present")
        return nested
    return facts_record


def _source_description(product: Mapping[str, Any], max_chars: int) -> str:
    descriptions = _clean_sequence(
        product.get("description", []),
        field="description",
        sort_values=False,
    )
    description = " ".join(descriptions)
    return description[:max_chars].rstrip()


def build_product_text(
    product: Mapping[str, Any],
    canonical_facts: Mapping[str, Any],
    *,
    description_max_chars: int = DESCRIPTION_MAX_CHARS,
) -> str:
    """Build the stable semantic document used for one product embedding.

    Only selected canonical facts and the source title/description are used.
    Raw JSON, prices, annotation metadata, and source ``features`` are omitted
    so provenance and noisy marketing text cannot silently change the document
    representation. Values from canonical arrays are de-duplicated and sorted.
    """
    if not isinstance(product, Mapping):
        raise TypeError("product must be an object")
    if not isinstance(canonical_facts, Mapping):
        raise TypeError("canonical_facts must be an object")
    if description_max_chars < 0:
        raise ValueError("description_max_chars must be non-negative")

    facts = _facts_payload(canonical_facts)
    lines = [f"Title: {_clean_text(product.get('title'))}"]
    for label, field in _FACT_TEXT_FIELDS:
        values = _clean_sequence(
            facts.get(field),
            field=field,
            sort_values=field != "brand",
        )
        lines.append(f"{label}: {', '.join(values)}")
    lines.append(
        f"Description: {_source_description(product, description_max_chars)}"
    )
    return "\n".join(lines)
