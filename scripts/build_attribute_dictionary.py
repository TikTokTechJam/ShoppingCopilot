from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from dictionary.registry import (
    ATTRIBUTE_FIELDS,
    NORMALIZATION_VERSION,
    SEMANTIC_ATTRIBUTES,
    canonical_id,
    normalize_text,
)


LIST_FIELDS = tuple(field for field in ATTRIBUTE_FIELDS if field != "brand")
SCHEMA_VERSION = "canonical-attribute-dictionary/v1"


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_facts(path: Path) -> tuple[dict[tuple[str, str], int], int, int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    seen_products: set[str] = set()
    record_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number}: fact record must be an object")

            parent_asin = record.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"{path}:{line_number}: missing parent_asin")
            if parent_asin in seen_products:
                raise ValueError(f"{path}:{line_number}: duplicate product {parent_asin}")
            seen_products.add(parent_asin)

            missing = [field for field in ATTRIBUTE_FIELDS if field not in record]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing facts {missing}")

            for attribute in ATTRIBUTE_FIELDS:
                raw_values = record[attribute]
                if attribute == "brand":
                    values: Iterable[Any] = () if raw_values is None else (raw_values,)
                else:
                    if raw_values is None:
                        values = ()
                    elif isinstance(raw_values, list):
                        values = raw_values
                    else:
                        raise ValueError(
                            f"{path}:{line_number}: {attribute} must be an array"
                        )

                product_values: set[str] = set()
                for value in values:
                    if not isinstance(value, str):
                        raise ValueError(
                            f"{path}:{line_number}: {attribute} values must be strings"
                        )
                    if not value.strip():
                        continue
                    if not normalize_text(value):
                        raise ValueError(
                            f"{path}:{line_number}: {attribute} has no searchable text"
                        )
                    product_values.add(value)
                for value in product_values:
                    counts[(attribute, value)] += 1
            record_count += 1
    return dict(counts), record_count, len(seen_products)


def _registry_records(counts: Mapping[tuple[str, str], int]) -> list[dict[str, Any]]:
    records = []
    for (attribute, value), count in sorted(counts.items()):
        records.append(
            {
                "canonical_id": canonical_id(attribute, value),
                "attribute": attribute,
                "value": value,
                "normalized": normalize_text(value),
                "count": count,
            }
        )
    return records


def _normalized_lookup(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
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


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "embedding generation requires NumPy; install requirements-embeddings.txt"
        ) from exc
    return np


def _encode_with_sentence_transformers(
    model_name: str,
    texts: list[str],
) -> tuple[Any, str]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "--embedding-model requires sentence-transformers; "
            "install requirements-embeddings.txt"
        ) from exc
    model = SentenceTransformer(model_name)
    matrix = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return matrix, model_name


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
    facts_path: str | Path = "data/derived/catalog_facts/catalog_facts.jsonl",
    output_dir: str | Path = "data/derived/dictionary",
    *,
    embedding_model: str | None = None,
    precomputed_embeddings: str | Path | None = None,
    semantic_attributes: Iterable[str] = SEMANTIC_ATTRIBUTES,
    query_encoder_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic registry artifacts and optional real embeddings.

    When no embedding source is supplied, this builds the exact normalized
    registry only. Use ``embedding_model`` or ``precomputed_embeddings`` to
    produce the shared semantic matrix; no hash-based pseudo-embeddings are
    generated.
    """

    facts = Path(facts_path)
    output = Path(output_dir)
    selected_attributes = tuple(semantic_attributes)
    unknown = sorted(set(selected_attributes) - set(SEMANTIC_ATTRIBUTES))
    if unknown:
        raise ValueError(f"semantic embeddings are not allowed for attributes: {unknown}")
    if embedding_model and precomputed_embeddings:
        raise ValueError("choose embedding_model or precomputed_embeddings, not both")

    counts, record_count, product_count = _read_facts(facts)
    records = _registry_records(counts)
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
        texts = [str(row["embedding_text"]) for row in embedding_rows]
        matrix, embedding_model_name = _encode_with_sentence_transformers(
            embedding_model, texts
        )
        normalized_matrix = _normalize_matrix(matrix)
        if int(normalized_matrix.shape[0]) != len(embedding_rows):
            raise ValueError("embedding count does not match canonical values")
        _write_npy(output / "attribute_embeddings.npy", normalized_matrix)
        embedding_dimension = int(normalized_matrix.shape[1])
        embedding_status = "generated"
    elif precomputed_embeddings:
        np = _load_numpy()
        matrix = np.load(Path(precomputed_embeddings), allow_pickle=False)
        normalized_matrix = _normalize_matrix(matrix)
        if int(normalized_matrix.shape[0]) != len(embedding_rows):
            raise ValueError(
                "precomputed embedding rows must follow the deterministic embedded "
                "canonical-value order"
            )
        _write_npy(output / "attribute_embeddings.npy", normalized_matrix)
        embedding_model_name = "precomputed"
        embedding_dimension = int(normalized_matrix.shape[1])
        embedding_status = "generated"
    else:
        embedding_rows = []

    _write_json(output / "embedding_metadata.json", embedding_rows)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "path": str(facts),
                "sha256": _source_sha256(facts),
                "record_count": record_count,
                "product_count": product_count,
            },
            "normalization_version": NORMALIZATION_VERSION,
            "canonical_value_count": len(records),
            "embedded_value_count": len(embedding_rows),
            "semantic_attributes": list(selected_attributes),
            "embedding": {
                "status": embedding_status,
                "model": embedding_model_name,
                "dimension": embedding_dimension,
            },
        },
    )
    return {
        "source_record_count": record_count,
        "product_count": product_count,
        "canonical_value_count": len(records),
        "embedded_value_count": len(embedding_rows),
        "embedding_status": embedding_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Issue #8 canonical attribute registry."
    )
    parser.add_argument(
        "--facts",
        default="data/derived/catalog_facts/catalog_facts.jsonl",
        help="Issue #5 canonical facts JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/derived/dictionary",
        help="Derived dictionary artifact directory",
    )
    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument(
        "--embedding-model",
        help="SentenceTransformers model name or local model path",
    )
    embedding.add_argument(
        "--precomputed-embeddings",
        help=".npy matrix in deterministic embedded-value order",
    )
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
        help="Attributes included in the one shared semantic matrix",
    )
    args = parser.parse_args()
    if not (args.embedding_model or args.precomputed_embeddings or args.no_embeddings):
        parser.error(
            "choose --embedding-model, --precomputed-embeddings, or --no-embeddings"
        )
    result = build_attribute_dictionary(
        args.facts,
        args.output_dir,
        embedding_model=args.embedding_model,
        precomputed_embeddings=args.precomputed_embeddings,
        semantic_attributes=args.semantic_attributes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
