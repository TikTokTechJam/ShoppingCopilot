from __future__ import annotations

import json
import math
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

# The model-facing V4 contract. Brand is intentionally an array so the model
# can return both the manufacturer brand and a strongly identifying product line.
MODEL_FACT_FIELDS = (
    "brand",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
)

# Category is copied from the catalog by the runner. It is stored alongside the
# model facts, but it is not part of the model response.
ANNOTATION_FACT_FIELDS = (
    "category",
    *MODEL_FACT_FIELDS,
)

# The final Issue #5 catalog-facts contract remains compatible with downstream
# runtime consumers. Its brand remains scalar and its structured size field
# remains present; the V4 builder leaves size empty because the model no longer
# annotates it.
CANONICAL_FACT_FIELDS = (
    "category",
    "brand",
    "color",
    "material",
    "size",
    "style",
    "feature",
    "use_case",
)

ANNOTATION_LIMITS = {
    "brand": 3,
    "color": 3,
    "material": 4,
    "style": 6,
    "feature": 8,
    "use_case": 5,
}
FACT_LIMITS = {
    "category": 4,
    "color": 3,
    "material": 4,
    "size": 4,
    "style": 6,
    "feature": 8,
    "use_case": 5,
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

_NORMALIZATION_ALIASES = {
    "4 way stretch": "four way stretch",
    "high waist": "high waisted",
    "machine wash": "machine washable",
    "moisture wicking": "moisture wicking",
    "no show": "no show",
    "pull on": "pull on",
    "quick dry": "quick drying",
}

_INVALID_COLOR_VALUES = {
    "color block",
    "floral",
    "graphic",
    "iridescent",
    "polka dot",
    "printed",
    "solid",
    "solid color",
    "striped",
}
_INVALID_BRAND_VALUES = {
    "classic",
    "hoodie",
    "leather",
    "premium",
    "running shoe",
    "sneaker",
    "waterproof",
    "women",
}
_INVALID_STYLE_VALUES = {
    "adult",
    "boys",
    "dress",
    "fashion sneaker",
    "girls",
    "jacket",
    "men",
    "professional",
    "shirt",
    "shoe",
    "toddler",
    "unisex",
    "women",
}
_INVALID_FEATURE_VALUES = {
    "amazing",
    "best",
    "excellent",
    "high quality",
    "no closure",
    "premium",
    "stylish",
}
_INVALID_USE_CASE_VALUES = {
    "all occasions",
    "casual wear",
    "daily life",
    "daily wear",
    "everyday use",
    "everyday wear",
    "general use",
    "general wear",
    "lifestyle",
    "normal wear",
}
_SIZE_WORDS = {
    "2xl",
    "3xl",
    "large",
    "l",
    "medium",
    "m",
    "one size",
    "s",
    "small",
    "xl",
    "xs",
    "xxl",
}
_MEASUREMENT_WORDS = {
    "centimeter",
    "centimeters",
    "cm",
    "dimension",
    "dimensions",
    "inch",
    "inches",
    "inseam",
    "measurement",
    "measurements",
    "meter",
    "meters",
    "mm",
    "package",
    "size",
    "weight",
}


def normalize_annotation_value(value: str) -> str:
    """Normalize a V4 annotation value to lowercase natural text with spaces."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    characters: list[str] = []
    for character in normalized:
        if character.isalnum() or character in {"'", "’"}:
            characters.append("'" if character == "’" else character)
        else:
            characters.append(" ")
    normalized = " ".join("".join(characters).split())
    return _NORMALIZATION_ALIASES.get(normalized, normalized)


def canonicalize_value(value: str) -> str:
    """Preserve the legacy snake_case form used by final catalog facts."""
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


def _looks_like_size_or_measurement(value: str) -> bool:
    parts = value.split()
    if value in _SIZE_WORDS:
        return True
    if any(part in _MEASUREMENT_WORDS for part in parts):
        return True
    if any(character.isdigit() for character in value):
        return any(
            part in _MEASUREMENT_WORDS or part in {"size", "sizing"}
            for part in parts
        )
    return False


def _is_invalid_annotation_value(field: str, value: str) -> bool:
    if field == "brand":
        return value in _INVALID_BRAND_VALUES
    if field == "color":
        return value in _INVALID_COLOR_VALUES
    if field == "style" and value in _INVALID_STYLE_VALUES:
        return True
    if field == "feature" and value in _INVALID_FEATURE_VALUES:
        return True
    if field == "use_case" and value in _INVALID_USE_CASE_VALUES:
        return True
    return field != "brand" and _looks_like_size_or_measurement(value)


def _remove_redundant_values(values: list[str]) -> list[str]:
    """Remove only an obvious generic stretch duplicate when specificity exists."""
    has_specific_stretch = any(
        value not in {"stretch", "stretchy"} and value.endswith(" stretch")
        for value in values
    )
    if not has_specific_stretch:
        return values
    return [
        value for value in values if value not in {"stretch", "stretchy"}
    ]


def _normalize_list_value(
    field: str,
    value: Any,
    *,
    annotation: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array of strings")
    normalized_values: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field} must contain strings")
        normalized = (
            normalize_annotation_value(item)
            if annotation
            else canonicalize_value(item)
        )
        if not normalized:
            continue
        if annotation and _is_invalid_annotation_value(field, normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_values.append(normalized)
    if annotation:
        normalized_values = _remove_redundant_values(normalized_values)
        return normalized_values[: ANNOTATION_LIMITS[field]]
    return normalized_values[: FACT_LIMITS[field]]


def _normalize_annotation_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("annotation facts must be an object")
    _require_exact_keys(payload, set(ANNOTATION_FACT_FIELDS), "annotation facts")
    return {
        "category": _normalize_list_value("category", payload["category"]),
        **{
            field: _normalize_list_value(field, payload[field], annotation=True)
            for field in MODEL_FACT_FIELDS
        },
    }


def normalize_model_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the exact six-field V4 model response."""
    if not isinstance(payload, Mapping):
        raise TypeError("model facts must be an object")
    _require_exact_keys(payload, set(MODEL_FACT_FIELDS), "model facts")
    return {
        field: _normalize_list_value(field, payload[field], annotation=True)
        for field in MODEL_FACT_FIELDS
    }


def _normalize_canonical_brand(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError("brand must be a non-empty string or null")
    normalized = canonicalize_value(value)
    if not normalized:
        raise ValueError("brand must contain a usable value")
    return normalized


def normalize_canonical_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("canonical facts must be an object")
    _require_exact_keys(payload, set(CANONICAL_FACT_FIELDS), "canonical facts")
    result = {
        "category": _normalize_list_value("category", payload["category"]),
        "brand": _normalize_canonical_brand(payload["brand"]),
    }
    for field in ("color", "material", "size", "style", "feature", "use_case"):
        result[field] = _normalize_list_value(field, payload[field])
    return result


def normalize_catalog_categories(value: Any) -> list[str]:
    """Copy raw category hierarchy levels using legacy lexical normalization."""
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
        "facts": _normalize_annotation_facts(record["facts"]),
        "annotation": {
            "status": "success",
            "model": metadata["model"].strip(),
            "prompt_version": metadata["prompt_version"].strip(),
        },
    }


def _legacy_values(values: list[str]) -> list[str]:
    return [canonicalize_value(value) for value in values]


def canonical_record_from_annotation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt V4 annotation facts to the unchanged Issue #5 final record shape."""
    normalized = validate_annotation_record(record)
    facts = normalized["facts"]
    brand_values = facts["brand"]
    return {
        "parent_asin": normalized["parent_asin"],
        "category": facts["category"],
        "brand": canonicalize_value(brand_values[0]) if brand_values else None,
        "price": normalized["price"],
        "color": _legacy_values(facts["color"]),
        "material": _legacy_values(facts["material"]),
        "size": [],
        "style": _legacy_values(facts["style"]),
        "feature": _legacy_values(facts["feature"]),
        "use_case": _legacy_values(facts["use_case"]),
    }
