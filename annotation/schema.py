from __future__ import annotations

import json
import math
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

MODEL_FACT_FIELDS = (
    "category",
    "brand",
    "color",
    "material",
    "size",
    "style",
    "feature",
    "use_case",
)
LIST_FACT_FIELDS = tuple(field for field in MODEL_FACT_FIELDS if field != "brand")
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


def normalize_model_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("model facts must be an object")
    _require_exact_keys(payload, set(MODEL_FACT_FIELDS), "model facts")

    result: dict[str, Any] = {}
    for field in MODEL_FACT_FIELDS:
        value = payload[field]
        if field == "brand":
            if value is None:
                result[field] = None
                continue
            if not isinstance(value, str) or not value.strip():
                raise TypeError("brand must be a non-empty string or null")
            normalized = canonicalize_value(value)
            if not normalized:
                raise ValueError("brand must contain a usable value")
            result[field] = normalized
            continue

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
        result[field] = normalized_values

    return result


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
        "parent_asin": parent_asin,
        "price": normalize_price(record["price"]),
        "facts": normalize_model_facts(record["facts"]),
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
