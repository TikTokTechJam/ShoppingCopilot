from __future__ import annotations

import json
import math
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

MODEL_FACT_FIELDS = (
    "color",
    "material",
    "size",
    "style",
    "feature",
    "use_case",
)
CANONICAL_FACT_FIELDS = (
    "category",
    "brand",
    *MODEL_FACT_FIELDS,
)
FACT_LIMITS = {
    "color": 3,
    "material": 4,
    "size": 4,
    "style": 4,
    "feature": 6,
    "use_case": 3,
    "category": 4,
}
CANONICAL_RECORD_FIELDS = (
    "parent_asin",
    "category",
    "brand",
    "price",
    "color",
    "material",
    "size",
    "style",
    "feature",
    "use_case",
)
ANNOTATION_RECORD_FIELDS = {"parent_asin", "price", "facts", "annotation"}
ANNOTATION_METADATA_FIELDS = {"status", "model", "prompt_version"}


def canonicalize_value(value: str) -> str:
    """Apply only lexical normalization; do not infer or merge concepts."""
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return "_".join(tokens)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _normalize_list_value(field: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array of strings")
    normalized_values: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"{field} must contain non-empty strings")
        normalized = canonicalize_value(item)
        if not normalized:
            raise ValueError(f"{field} contains a value with no usable characters")
        if normalized in seen:
            raise ValueError(f"{field} contains duplicate value {normalized!r}")
        seen.add(normalized)
        normalized_values.append(normalized)
    return normalized_values[: FACT_LIMITS[field]]


def _normalize_brand(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError("brand must be a non-empty string or null")
    normalized = canonicalize_value(value)
    if not normalized:
        raise ValueError("brand must contain a usable value")
    return normalized


def normalize_model_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("model facts must be an object")
    _require_exact_keys(payload, set(MODEL_FACT_FIELDS), "model facts")
    return {
        field: _normalize_list_value(field, payload[field])
        for field in MODEL_FACT_FIELDS
    }


def normalize_canonical_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("canonical facts must be an object")
    _require_exact_keys(payload, set(CANONICAL_FACT_FIELDS), "canonical facts")
    result = {
        "category": _normalize_list_value("category", payload["category"]),
        "brand": _normalize_brand(payload["brand"]),
    }
    result.update(
        normalize_model_facts(
            {field: payload[field] for field in MODEL_FACT_FIELDS}
        )
    )
    return result


def normalize_catalog_categories(value: Any) -> list[str]:
    """Copy raw category hierarchy levels using lexical normalization only."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("catalog categories must be an array")
    levels: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError("catalog categories must contain strings")
        for raw_level in item.split(">"):
            normalized = canonicalize_value(raw_level)
            if normalized and normalized not in seen:
                seen.add(normalized)
                levels.append(normalized)
    return levels[-FACT_LIMITS["category"] :]


def deterministic_catalog_brand(product: Mapping[str, Any]) -> str | None:
    """Read brand/manufacturer fields only; never use seller/store text."""
    candidates: list[Any] = [
        product.get("brand"),
        product.get("manufacturer"),
    ]
    details = product.get("details")
    if isinstance(details, Mapping):
        for key, value in details.items():
            if str(key).strip().lower() in {"brand", "manufacturer"}:
                candidates.append(value)
    for value in candidates:
        if isinstance(value, str) and value.strip():
            normalized = canonicalize_value(value)
            if normalized:
                return normalized
    return None


def parse_and_validate_json(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8")
    if isinstance(raw_response, str):
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not valid JSON") from exc
    else:
        payload = raw_response
    return normalize_model_facts(payload)


def normalize_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("price must be numeric or null")

    if isinstance(value, (int, float, Decimal)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        if text[0] in "$£€":
            text = text[1:].strip()
        try:
            number = float(Decimal(text))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid price value: {value!r}") from exc
    else:
        raise TypeError("price must be numeric or null")

    if not math.isfinite(number):
        raise ValueError("price must be finite")
    return number


def validate_annotation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("annotation record must be an object")
    _require_exact_keys(record, ANNOTATION_RECORD_FIELDS, "annotation record")

    parent_asin = record["parent_asin"]
    if not isinstance(parent_asin, str) or not parent_asin.strip():
        raise ValueError("annotation record needs a non-empty parent_asin")

    metadata = record["annotation"]
    if not isinstance(metadata, Mapping):
        raise TypeError("annotation metadata must be an object")
    _require_exact_keys(metadata, ANNOTATION_METADATA_FIELDS, "annotation metadata")
    if metadata["status"] != "success":
        raise ValueError("only successful annotations may be stored")
    for field in ("model", "prompt_version"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"annotation.{field} must be a non-empty string")

    return {
        "parent_asin": parent_asin.strip(),
        "price": normalize_price(record["price"]),
        "facts": normalize_canonical_facts(record["facts"]),
        "annotation": {
            "status": "success",
            "model": metadata["model"].strip(),
            "prompt_version": metadata["prompt_version"].strip(),
        },
    }


def canonical_record_from_annotation(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_annotation_record(record)
    facts = normalized["facts"]
    return {
        "parent_asin": normalized["parent_asin"],
        "category": facts["category"],
        "brand": facts["brand"],
        "price": normalized["price"],
        "color": facts["color"],
        "material": facts["material"],
        "size": facts["size"],
        "style": facts["style"],
        "feature": facts["feature"],
        "use_case": facts["use_case"],
    }
