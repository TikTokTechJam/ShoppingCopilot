from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from dictionary.registry import (
    ATTRIBUTE_FIELDS,
    NORMALIZATION_VERSION,
    SEMANTIC_ATTRIBUTES,
    canonical_id,
    normalize_text,
)
from dictionary.semantic import (
    ATTRIBUTE_EMBEDDING_DIMENSION,
    ATTRIBUTE_EMBEDDING_MODEL,
    ATTRIBUTE_EMBEDDING_NORMALIZATION,
    load_bge_attribute_encoder,
)


SCHEMA_VERSION = "canonical-attribute-dictionary/v2"
DEFAULT_INPUT = "data/derived/annotations/v4/annotations.jsonl"


@dataclass(frozen=True)
class _ReadResult:
    counts: dict[tuple[str, str], int]
    representatives: dict[tuple[str, str], str]
    raw_variants: dict[tuple[str, str], frozenset[str]]
    records_read: int
    records_used: int
    records_skipped: int
    skipped_by_reason: dict[str, int]
    source_product_count: int | None
    prompt_versions: tuple[str, ...]
    input_format: str


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_value(value: str) -> str:
    """Return a lowercase human-readable value while retaining apostrophes."""

    text = unicodedata.normalize("NFKC", value).casefold()
    apostrophes = {"'", "’", "ʼ", "＇"}
    output: list[str] = []
    pending_space = False
    for index, character in enumerate(text):
        if character.isalnum():
            if pending_space and output:
                output.append(" ")
            output.append(character)
            pending_space = False
        elif (
            character in apostrophes
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isalnum()
            and text[index + 1].isalnum()
        ):
            output.append("'")
            pending_space = False
        else:
            pending_space = bool(output)
    return "".join(output).strip()


def _source_product_count(path: Path, records_read: int) -> int | None:
    """Read a nearby annotation manifest when it declares the source size."""

    manifest_path = path.with_name("manifest.json")
    if not manifest_path.exists():
        return records_read
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return records_read
    if not isinstance(payload, Mapping):
        return records_read
    for key in ("source_product_count", "selected_product_count", "catalog_product_count"):
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    source = payload.get("source")
    if isinstance(source, Mapping):
        value = source.get("product_count")
        if isinstance(value, int) and value >= 0:
            return value
    return records_read


def _read_facts(path: Path, *, input_format: str = "auto") -> _ReadResult:
    """Read V4 wrappers or the aggregated V5 nested-facts records."""

    if input_format not in {"auto", "v4", "v5"}:
        raise ValueError("input_format must be one of: auto, v4, v5")

    product_membership: dict[tuple[str, str], set[str]] = defaultdict(set)
    representatives: dict[tuple[str, str], str] = {}
    raw_variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_products: set[str] = set()
    skipped_by_reason: dict[str, int] = defaultdict(int)
    prompt_versions: set[str] = set()
    observed_formats: set[str] = set()
    records_read = 0
    records_used = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            records_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_by_reason["invalid_json"] += 1
                continue
            if not isinstance(record, Mapping):
                skipped_by_reason["record_not_object"] += 1
                continue

            record_format = input_format
            if record_format == "auto":
                record_format = "v4" if "annotation" in record else "v5"
            observed_formats.add(record_format)

            annotation = record.get("annotation")
            if record_format == "v4":
                if not isinstance(annotation, Mapping) or annotation.get("status") != "success":
                    skipped_by_reason["not_successful"] += 1
                    continue
            parent_asin = record.get("parent_asin")
            facts = record.get("facts")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                skipped_by_reason["missing_parent_asin"] += 1
                continue
            if parent_asin in seen_products:
                skipped_by_reason["duplicate_parent_asin"] += 1
                continue
            if not isinstance(facts, Mapping):
                skipped_by_reason["facts_not_object"] += 1
                continue
            required_fields = ATTRIBUTE_FIELDS
            if record_format == "v5":
                required_fields = tuple(field for field in ATTRIBUTE_FIELDS if field != "style")
            if any(field not in facts for field in required_fields):
                skipped_by_reason["missing_dictionary_field"] += 1
                continue

            normalized_record: dict[str, list[tuple[str, str]]] = {}
            malformed = False
            for attribute in ATTRIBUTE_FIELDS:
                # V5 aggregation intentionally omits style. It remains part of
                # the seven-field dictionary contract, but contributes no values.
                if record_format == "v5":
                    raw_values = [] if attribute == "style" else facts.get(attribute, [])
                else:
                    raw_values = facts.get(attribute)
                if not isinstance(raw_values, list):
                    malformed = True
                    break
                values: list[tuple[str, str]] = []
                for raw_value in raw_values:
                    if not isinstance(raw_value, str):
                        malformed = True
                        break
                    display = _natural_value(raw_value)
                    normalized = normalize_text(display)
                    if not normalized:
                        continue
                    values.append((display, normalized))
                if malformed:
                    break
                normalized_record[attribute] = values
            if malformed:
                skipped_by_reason["malformed_dictionary_field"] += 1
                continue

            seen_products.add(parent_asin)
            records_used += 1
            if record_format == "v4":
                prompt_version = annotation.get("prompt_version")
                if isinstance(prompt_version, str) and prompt_version.strip():
                    prompt_versions.add(prompt_version.strip())

            # A product contributes at most once per attribute/surface, even if
            # the model repeats a value or returns separator variants.
            product_surfaces: set[tuple[str, str]] = set()
            for attribute, values in normalized_record.items():
                for display, normalized in values:
                    key = (attribute, normalized)
                    if key in product_surfaces:
                        continue
                    product_surfaces.add(key)
                    product_membership[key].add(parent_asin)
                    raw_variants[key].add(display)
                    representatives[key] = min(
                        representatives.get(key, display), display
                    )

    counts = {key: len(products) for key, products in product_membership.items()}
    detected_formats = sorted(observed_formats)
    detected_format = (
        detected_formats[0]
        if len(detected_formats) == 1
        else "mixed"
        if detected_formats
        else input_format
    )
    return _ReadResult(
        counts=counts,
        representatives=representatives,
        raw_variants={key: frozenset(values) for key, values in raw_variants.items()},
        records_read=records_read,
        records_used=records_used,
        records_skipped=records_read - records_used,
        skipped_by_reason=dict(sorted(skipped_by_reason.items())),
        source_product_count=_source_product_count(path, records_read),
        prompt_versions=tuple(sorted(prompt_versions)),
        input_format=detected_format,
    )


def _registry_records(result: _ReadResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for attribute in ATTRIBUTE_FIELDS:
        keys = sorted(
            (key for key in result.counts if key[0] == attribute),
            key=lambda item: item[1],
        )
        for key in keys:
            _, normalized = key
            value = result.representatives[key]
            records.append(
                {
                    "canonical_id": canonical_id(attribute, value),
                    "attribute": attribute,
                    "value": value,
                    "normalized": normalized,
                    "count": result.counts[key],
                }
            )
    return records


def _normalized_lookup(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    lookup: dict[str, dict[str, list[str]]] = {
        attribute: {} for attribute in ATTRIBUTE_FIELDS
    }
    for record in records:
        attribute = str(record["attribute"])
        normalized = str(record["normalized"])
        lookup[attribute].setdefault(normalized, []).append(str(record["canonical_id"]))
    for surfaces in lookup.values():
        for canonical_ids in surfaces.values():
            canonical_ids.sort()
    return lookup


def _ambiguous_surface_count(lookup: Mapping[str, Mapping[str, Iterable[str]]]) -> int:
    surfaces: dict[str, set[str]] = defaultdict(set)
    for attribute, values in lookup.items():
        for normalized, canonical_ids in values.items():
            if canonical_ids:
                surfaces[normalized].add(attribute)
    return sum(1 for attributes in surfaces.values() if len(attributes) > 1)


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "embedding generation requires NumPy; install requirements-embeddings.txt"
        ) from exc
    return np


def _encode_with_sentence_transformers(
    model_path: str,
    texts: list[str],
) -> tuple[Any, str]:
    encoder = load_bge_attribute_encoder(model_path)
    matrix = encoder.embed_documents(texts)
    return matrix, ATTRIBUTE_EMBEDDING_MODEL


def _normalize_matrix(matrix: Any) -> Any:
    np = _load_numpy()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("embeddings must not contain zero vectors")
    return matrix / norms


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npy(path: Path, matrix: Any) -> None:
    np = _load_numpy()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary.replace(path)


def build_attribute_dictionary(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = "data/derived/dictionary",
    *,
    facts_path: str | Path | None = None,
    embedding_model: str | None = None,
    precomputed_embeddings: str | Path | None = None,
    semantic_attributes: Iterable[str] = SEMANTIC_ATTRIBUTES,
    query_encoder_factory: Callable[[str], Any] | None = None,
    input_format: str = "auto",
) -> dict[str, Any]:
    """Build deterministic exact registry artifacts and optional embeddings."""

    source_path = Path(facts_path if facts_path is not None else input_path)
    output = Path(output_dir)
    selected_attributes = tuple(semantic_attributes)
    unknown = sorted(set(selected_attributes) - set(SEMANTIC_ATTRIBUTES))
    if unknown:
        raise ValueError(f"semantic embeddings are not allowed for attributes: {unknown}")
    if embedding_model and precomputed_embeddings:
        raise ValueError("choose embedding_model or precomputed_embeddings, not both")

    result = _read_facts(source_path, input_format=input_format)
    records = _registry_records(result)
    lookup = _normalized_lookup(records)
    output.mkdir(parents=True, exist_ok=True)

    _write_json(
        output / "canonical_values.json",
        {
            "schema_version": SCHEMA_VERSION,
            "attributes": list(ATTRIBUTE_FIELDS),
            "values": {record["canonical_id"]: record for record in records},
        },
    )
    _write_json(
        output / "normalized_lookup.json",
        {
            "schema_version": SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "attributes": lookup,
        },
    )

    embedded_records = [
        record for record in records if record["attribute"] in selected_attributes
    ]
    embedding_rows = [
        {
            "row": row,
            "canonical_id": record["canonical_id"],
            "attribute": record["attribute"],
            "value": record["value"],
            "embedding_text": record["normalized"],
        }
        for row, record in enumerate(embedded_records)
    ]
    embedding_dimension = None
    embedding_model_name = None
    embedding_status = "not_generated"
    if embedding_model:
        matrix, embedding_model_name = _encode_with_sentence_transformers(
            embedding_model, [str(row["embedding_text"]) for row in embedding_rows]
        )
        normalized_matrix = _normalize_matrix(matrix)
        if int(normalized_matrix.shape[0]) != len(embedding_rows):
            raise ValueError("embedding count does not match canonical values")
        if int(normalized_matrix.shape[1]) != ATTRIBUTE_EMBEDDING_DIMENSION:
            raise ValueError(
                "BGE attribute embeddings must have dimension "
                f"{ATTRIBUTE_EMBEDDING_DIMENSION}, got {normalized_matrix.shape[1]}"
            )
        _write_npy(output / "attribute_embeddings.npy", normalized_matrix)
        embedding_dimension = int(normalized_matrix.shape[1])
        embedding_status = "generated"
    elif precomputed_embeddings:
        np = _load_numpy()
        normalized_matrix = _normalize_matrix(
            np.load(Path(precomputed_embeddings), allow_pickle=False)
        )
        if int(normalized_matrix.shape[0]) != len(embedding_rows):
            raise ValueError(
                "precomputed embedding rows must follow the deterministic "
                "embedded-value order"
            )
        _write_npy(output / "attribute_embeddings.npy", normalized_matrix)
        embedding_model_name = "precomputed"
        embedding_dimension = int(normalized_matrix.shape[1])
        embedding_status = "generated"
    else:
        # Never leave placeholder metadata behind in an exact-only build.
        for optional_path in (
            output / "embedding_metadata.json",
            output / "attribute_embeddings.npy",
        ):
            if optional_path.exists():
                optional_path.unlink()
        embedding_rows = []

    if embedding_status == "generated":
        _write_json(output / "embedding_metadata.json", embedding_rows)

    canonical_value_count_by_attribute = {
        attribute: sum(1 for record in records if record["attribute"] == attribute)
        for attribute in ATTRIBUTE_FIELDS
    }
    normalized_collision_count = sum(
        1 for variants in result.raw_variants.values() if len(variants) > 1
    )
    prompt_version: str | None
    if len(result.prompt_versions) == 1:
        prompt_version = result.prompt_versions[0]
    else:
        prompt_version = None
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dictionary_version": SCHEMA_VERSION,
            "source_path": str(source_path),
            "input_format": result.input_format,
            "source_sha256": _source_sha256(source_path),
            "source_product_count": result.source_product_count,
            "records_read": result.records_read,
            "successful_annotation_records": result.records_used,
            "records_used": result.records_used,
            "records_skipped": result.records_skipped,
            "skipped_by_reason": result.skipped_by_reason,
            "prompt_version": prompt_version,
            "fields": list(ATTRIBUTE_FIELDS),
            "normalization_version": NORMALIZATION_VERSION,
            "canonical_value_count": len(records),
            "canonical_value_count_by_attribute": canonical_value_count_by_attribute,
            "normalized_collision_count": normalized_collision_count,
            "ambiguous_normalized_surface_count": _ambiguous_surface_count(lookup),
            "embeddings": embedding_status == "generated",
            "semantic_attributes": list(selected_attributes),
            "embedding": {
                "status": embedding_status,
                "model": embedding_model_name,
                "dimension": embedding_dimension,
                "normalization": (
                    ATTRIBUTE_EMBEDDING_NORMALIZATION
                    if embedding_status == "generated"
                    else None
                ),
                "query_prefix": None,
            },
            "source": {
                "path": str(source_path),
                "sha256": _source_sha256(source_path),
                "record_count": result.records_read,
                "product_count": result.source_product_count,
            },
            "embedded_value_count": len(embedding_rows),
        },
    )
    return {
        "source_product_count": result.source_product_count,
        "input_format": result.input_format,
        "records_read": result.records_read,
        "successful_annotation_records": result.records_used,
        "records_used": result.records_used,
        "records_skipped": result.records_skipped,
        "skipped_by_reason": result.skipped_by_reason,
        "canonical_value_count": len(records),
        "canonical_value_count_by_attribute": canonical_value_count_by_attribute,
        "normalized_collision_count": normalized_collision_count,
        "ambiguous_normalized_surface_count": _ambiguous_surface_count(lookup),
        "embedding_status": embedding_status,
        "output_dir": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical attribute dictionary artifacts."
    )
    parser.add_argument(
        "--input",
        "--facts",
        dest="input_path",
        default=DEFAULT_INPUT,
        help="V4 annotation or V5 aggregate JSONL containing nested facts",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "v4", "v5"),
        default="auto",
        help="Input record shape; auto detects V4 wrappers or V5 aggregate records",
    )
    parser.add_argument(
        "--output-dir",
        default="data/derived/dictionary",
        help="Derived dictionary artifact directory",
    )
    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument("--embedding-model")
    embedding.add_argument("--precomputed-embeddings")
    embedding.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Build only exact registry artifacts",
    )
    parser.add_argument(
        "--semantic-attributes",
        nargs="+",
        default=list(SEMANTIC_ATTRIBUTES),
        choices=SEMANTIC_ATTRIBUTES,
    )
    args = parser.parse_args()
    if not (args.embedding_model or args.precomputed_embeddings or args.no_embeddings):
        parser.error(
            "choose --embedding-model, --precomputed-embeddings, or --no-embeddings"
        )
    result = build_attribute_dictionary(
        args.input_path,
        args.output_dir,
        embedding_model=args.embedding_model,
        precomputed_embeddings=args.precomputed_embeddings,
        semantic_attributes=args.semantic_attributes,
        input_format=args.input_format,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
