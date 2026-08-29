from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dictionary.registry import (
    ATTRIBUTE_FIELDS,
    NORMALIZATION_VERSION,
    AttributeDictionary,
    canonical_id,
    normalize_text,
)
from dictionary.semantic import (
    ATTRIBUTE_EMBEDDING_DIMENSION,
    ATTRIBUTE_EMBEDDING_MODEL,
    ATTRIBUTE_EMBEDDING_NORMALIZATION,
)


SCHEMA_VERSION = "canonical-attribute-dictionary/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_records(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("values"), Mapping):
        records: list[Mapping[str, Any]] = []
        for value_id, record in payload["values"].items():
            if not isinstance(record, Mapping):
                raise ValueError("canonical value records must be objects")
            if str(record.get("canonical_id")) != str(value_id):
                raise ValueError("canonical value map key disagrees with canonical_id")
            records.append(record)
        return records
    if isinstance(payload, Mapping) and isinstance(payload.get("values"), list):
        return payload["values"]
    raise ValueError("canonical_values.json must contain a values object or array")


def _expected_lookup(records: list[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    lookup: dict[str, dict[str, list[str]]] = {
        attribute: {} for attribute in ATTRIBUTE_FIELDS
    }
    for record in records:
        lookup[str(record["attribute"])].setdefault(
            str(record["normalized"]), []
        ).append(str(record["canonical_id"]))
    for surfaces in lookup.values():
        for ids in surfaces.values():
            ids.sort()
    return lookup


def _load_lookup(root: Path) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(
        (root / "normalized_lookup.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, Mapping):
        raise ValueError("normalized_lookup.json must contain an object")
    attributes = payload.get("attributes", payload)
    if not isinstance(attributes, Mapping):
        raise ValueError("normalized_lookup.json must contain attribute maps")
    return {
        str(attribute): {
            str(surface): [str(value_id) for value_id in ids]
            for surface, ids in surfaces.items()
        }
        for attribute, surfaces in attributes.items()
    }


def _validate_manifest_source(manifest: Mapping[str, Any], root: Path) -> None:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("manifest source metadata is missing")
    source_path_value = manifest.get("source_path", source.get("path"))
    source_hash = manifest.get("source_sha256", source.get("sha256"))
    if not isinstance(source_path_value, str) or not source_path_value:
        raise ValueError("manifest source path is missing")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("manifest source SHA-256 is missing")
    source_path = Path(source_path_value)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if not source_path.exists():
        raise ValueError(f"manifest source path does not exist: {source_path}")
    if _sha256(source_path) != source_hash:
        raise ValueError("manifest source SHA-256 does not match source file")
    if source.get("path") != source_path_value or source.get("sha256") != source_hash:
        raise ValueError("manifest source metadata is inconsistent")

    records_read = manifest.get("records_read")
    records_used = manifest.get("records_used")
    records_skipped = manifest.get("records_skipped")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (records_read, records_used, records_skipped)
    ):
        raise ValueError("manifest record counts must be non-negative integers")
    if records_used > records_read or records_skipped != records_read - records_used:
        raise ValueError("manifest record counts are inconsistent")
    if source.get("record_count") != records_read:
        raise ValueError("manifest source record count is inconsistent")
    if source.get("product_count") != manifest.get("source_product_count"):
        raise ValueError("manifest source product count is inconsistent")


def validate_attribute_dictionary(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    required = ("manifest.json", "canonical_values.json", "normalized_lookup.json")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise ValueError(f"missing dictionary artifacts: {missing}")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest.json must contain an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dictionary schema version")
    if manifest.get("dictionary_version") != SCHEMA_VERSION:
        raise ValueError("manifest dictionary version is inconsistent")
    if manifest.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("manifest normalization version does not match runtime")
    if manifest.get("fields") != list(ATTRIBUTE_FIELDS):
        raise ValueError("manifest fields do not match the seven-field contract")
    if any(field in {"price", "size"} for field in manifest["fields"]):
        raise ValueError("price and size must not be dictionary fields")

    registry_payload = json.loads(
        (root / "canonical_values.json").read_text(encoding="utf-8")
    )
    records = _canonical_records(registry_payload)
    if isinstance(registry_payload, Mapping):
        if registry_payload.get("attributes") != list(ATTRIBUTE_FIELDS):
            raise ValueError("canonical_values attributes do not match the contract")

    seen_ids: set[str] = set()
    seen_surfaces: set[tuple[str, str]] = set()
    expected_order: list[tuple[int, str, str]] = []
    for record in records:
        if set(record) != {
            "canonical_id", "attribute", "value", "normalized", "count"
        }:
            raise ValueError("canonical value has an invalid field set")
        value_id = record["canonical_id"]
        attribute = record["attribute"]
        value = record["value"]
        normalized = record["normalized"]
        count = record["count"]
        if not all(isinstance(item, str) and item.strip() for item in (value_id, attribute, value, normalized)):
            raise ValueError("canonical values and IDs must be non-empty strings")
        if attribute not in ATTRIBUTE_FIELDS:
            raise ValueError(f"unexpected dictionary attribute: {attribute}")
        if attribute in {"price", "size"}:
            raise ValueError("price and size must not occur in dictionary records")
        if value_id in seen_ids:
            raise ValueError(f"duplicate canonical ID: {value_id}")
        if value_id != canonical_id(attribute, value):
            raise ValueError(f"invalid canonical ID: {value_id}")
        if normalize_text(value) != normalized:
            raise ValueError(f"invalid normalized surface for {value_id}")
        surface_key = (attribute, normalized)
        if surface_key in seen_surfaces:
            raise ValueError(f"duplicate attribute/normalized pair: {surface_key}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"invalid count for {value_id}")
        seen_ids.add(value_id)
        seen_surfaces.add(surface_key)
        expected_order.append((ATTRIBUTE_FIELDS.index(attribute), normalized, value_id))

    if expected_order != sorted(expected_order):
        raise ValueError("canonical values are not deterministically ordered")

    lookup = _load_lookup(root)
    if list(lookup) != list(ATTRIBUTE_FIELDS):
        raise ValueError("normalized lookup fields do not match the contract")
    expected_lookup = _expected_lookup(records)
    if lookup != expected_lookup:
        raise ValueError("normalized lookup does not match canonical values")
    for surfaces in lookup.values():
        if list(surfaces) != sorted(surfaces):
            raise ValueError("normalized lookup surfaces are not ordered")
        for value_ids in surfaces.values():
            if value_ids != sorted(value_ids):
                raise ValueError("normalized lookup IDs are not ordered")
            if any(value_id not in seen_ids for value_id in value_ids):
                raise ValueError("normalized lookup references an unknown ID")

    _validate_manifest_source(manifest, root)
    if manifest.get("canonical_value_count") != len(records):
        raise ValueError("manifest canonical value count is inconsistent")
    expected_by_attribute = {
        attribute: sum(1 for record in records if record["attribute"] == attribute)
        for attribute in ATTRIBUTE_FIELDS
    }
    if manifest.get("canonical_value_count_by_attribute") != expected_by_attribute:
        raise ValueError("manifest per-attribute counts are inconsistent")

    embeddings_enabled = manifest.get("embeddings")
    if not isinstance(embeddings_enabled, bool):
        raise ValueError("manifest embeddings must be boolean")
    embedding = manifest.get("embedding")
    if not isinstance(embedding, Mapping):
        raise ValueError("manifest embedding metadata is missing")
    if embeddings_enabled:
        if embedding.get("status") != "generated":
            raise ValueError("generated embeddings must have generated status")
        if embedding.get("model") != ATTRIBUTE_EMBEDDING_MODEL:
            raise ValueError(
                "attribute embeddings must use "
                f"{ATTRIBUTE_EMBEDDING_MODEL}"
            )
        if embedding.get("dimension") != ATTRIBUTE_EMBEDDING_DIMENSION:
            raise ValueError(
                "attribute embedding dimension must be "
                f"{ATTRIBUTE_EMBEDDING_DIMENSION}"
            )
        if embedding.get("normalization") != ATTRIBUTE_EMBEDDING_NORMALIZATION:
            raise ValueError(
                "attribute embeddings must use "
                f"{ATTRIBUTE_EMBEDDING_NORMALIZATION} normalization"
            )
        if embedding.get("query_prefix") is not None:
            raise ValueError("attribute embeddings must not use a query prefix")
        if not (root / "attribute_embeddings.npy").exists():
            raise ValueError("embedding matrix is missing")
        if not (root / "embedding_metadata.json").exists():
            raise ValueError("embedding metadata is missing")
    else:
        if embedding.get("status") != "not_generated":
            raise ValueError("exact-only manifest has an embedding status")
        if manifest.get("embedded_value_count") != 0:
            raise ValueError("exact-only dictionary cannot contain embedded values")
        if (root / "attribute_embeddings.npy").exists() or (root / "embedding_metadata.json").exists():
            raise ValueError("exact-only dictionary must not contain embedding files")

    dictionary = AttributeDictionary.load(root)
    if len(dictionary.values) != len(records):
        raise ValueError("runtime registry count does not match artifact")
    return {
        "canonical_value_count": len(records),
        "canonical_value_count_by_attribute": expected_by_attribute,
        "records_read": manifest["records_read"],
        "records_used": manifest["records_used"],
        "records_skipped": manifest["records_skipped"],
        "normalized_collision_count": manifest.get("normalized_collision_count", 0),
        "ambiguous_normalized_surface_count": manifest.get(
            "ambiguous_normalized_surface_count", 0
        ),
        "has_embedding_matrix": embeddings_enabled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V4 dictionary artifacts.")
    parser.add_argument("--directory", default="data/derived/dictionary")
    args = parser.parse_args()
    print(json.dumps(validate_attribute_dictionary(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
