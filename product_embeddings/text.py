from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .tier4 import normalize_tier4_source

PRODUCT_TEXT_VERSION = "product-text-v2"
DESCRIPTION_MAX_CHARS = 1_000
RAW_FEATURES_MAX_CHARS = 2_000
RAW_DETAILS_MAX_CHARS = 2_000

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
    descriptions = [
        value
        for value in product.get("description", [])
        if isinstance(value, str) and value
    ]
    description = " ".join(descriptions)
    return description[:max_chars].rstrip()


def _detail_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in sorted(value, key=lambda item: (str(item).casefold(), str(item))):
            rendered = _detail_value_text(value[key])
            if rendered:
                parts.append(f"{key}: {rendered}")
        return "; ".join(parts)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ", ".join(
            rendered
            for item in value
            if (rendered := _detail_value_text(item))
        )
    return " ".join(str(value).split())


def _source_details(product: Mapping[str, Any], max_chars: int) -> str:
    details = product.get("details", {})
    if not isinstance(details, Mapping):
        raise TypeError("details must be an object or null")
    parts: list[str] = []
    for key in sorted(details, key=lambda item: (str(item).casefold(), str(item))):
        value = _detail_value_text(details[key])
        if value:
            parts.append(f"{_clean_text(key)}: {value}")
    return "; ".join(parts)[:max_chars].rstrip()


def build_product_text(
    product: Mapping[str, Any],
    canonical_facts: Mapping[str, Any],
    *,
    raw_text: Mapping[str, Any] | None = None,
    description_max_chars: int = DESCRIPTION_MAX_CHARS,
    raw_features_max_chars: int = RAW_FEATURES_MAX_CHARS,
    raw_details_max_chars: int = RAW_DETAILS_MAX_CHARS,
) -> str:
    """Build the stable semantic document used for one product embedding.

    Canonical Tier 1–3 semantic facts are combined with selected Tier 4 source
    text. Prices, sizes, annotation metadata, and arbitrary raw JSON are not
    included. Values from canonical arrays are de-duplicated and sorted, while
    raw source fields retain their source order and are bounded for stability.
    """
    if not isinstance(product, Mapping):
        raise TypeError("product must be an object")
    if not isinstance(canonical_facts, Mapping):
        raise TypeError("canonical_facts must be an object")
    if (
        description_max_chars < 0
        or raw_features_max_chars < 0
        or raw_details_max_chars < 0
    ):
        raise ValueError("text length limits must be non-negative")

    facts = _facts_payload(canonical_facts)
    source = normalize_tier4_source(raw_text if raw_text is not None else product)
    lines = [f"Title: {_clean_text(source.get('title'))}"]
    for label, field in _FACT_TEXT_FIELDS:
        values = _clean_sequence(
            facts.get(field),
            field=field,
            sort_values=field != "brand",
        )
        lines.append(f"{label}: {', '.join(values)}")
    lines.append(f"Description: {_source_description(source, description_max_chars)}")
    raw_features = [
        value
        for value in source.get("features", [])
        if isinstance(value, str) and value
    ]
    lines.append(
        f"Raw features: {'; '.join(raw_features)[:raw_features_max_chars].rstrip()}"
    )
    lines.append(f"Details: {_source_details(source, raw_details_max_chars)}")
    return "\n".join(lines)
