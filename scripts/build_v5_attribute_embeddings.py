"""Build separate canonical-value embedding matrices for the V5 dictionary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from dictionary.registry import ATTRIBUTE_FIELDS, AttributeDictionary, normalize_text
from product_embeddings.pipeline import load_local_sentence_transformer


ATTRIBUTE_EMBEDDING_FIELDS = tuple(
    field for field in ATTRIBUTE_FIELDS if field != "brand"
)
DEFAULT_DICTIONARY_DIR = Path("data/derived/annotations/v5/dictionary")
DEFAULT_OUTPUT_DIR = DEFAULT_DICTIONARY_DIR / "attribute_embeddings"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_PATHS = (
    Path("models/bge-small-en-v1.5"),
    Path("model/bge-small-en-v1.5"),
)
EXPECTED_DIMENSION = 384
SCHEMA_VERSION = "canonical-attribute-embeddings/v1"


def _log(message: str) -> None:
    print(f"[build_v5_attribute_embeddings] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npy(path: Path, matrix: Any) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "embedding generation requires NumPy; install "
            "requirements-embeddings.txt"
        ) from exc
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    temporary.replace(path)


def _load_dictionary_rows(
    dictionary_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    """Load canonical values referenced by the V5 normalized lookup."""

    try:
        dictionary = AttributeDictionary.load(dictionary_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"unable to load V5 dictionary from {dictionary_dir}: {exc}"
        ) from exc
    values_by_id = {value.canonical_id: value for value in dictionary.values}

    lookup_path = dictionary_dir / "normalized_lookup.json"
    try:
        lookup_payload = json.loads(lookup_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read V5 normalized lookup: {lookup_path}") from exc
    if not isinstance(lookup_payload, Mapping):
        raise ValueError("normalized_lookup.json must contain an object")
    attribute_lookup = lookup_payload.get("attributes", lookup_payload)
    if not isinstance(attribute_lookup, Mapping):
        raise ValueError("normalized_lookup.json must contain attribute maps")
    if set(attribute_lookup) != set(ATTRIBUTE_FIELDS):
        raise ValueError(
            "V5 normalized lookup fields must match the seven-field contract"
        )

    rows_by_attribute: dict[str, list[dict[str, str]]] = {}
    referenced_ids: set[str] = set()
    for attribute in ATTRIBUTE_FIELDS:
        surfaces = attribute_lookup.get(attribute)
        if not isinstance(surfaces, Mapping):
            raise ValueError(f"normalized lookup for {attribute} must be an object")
        attribute_ids: set[str] = set()
        for surface, value_ids in surfaces.items():
            if not isinstance(surface, str) or not surface.strip():
                raise ValueError(f"{attribute} contains an empty normalized surface")
            if normalize_text(surface) != surface:
                raise ValueError(
                    f"normalized lookup surface is not normalized: {surface!r}"
                )
            if not isinstance(value_ids, list) or not value_ids:
                raise ValueError(
                    f"normalized lookup surface has no canonical IDs: "
                    f"{attribute}:{surface}"
                )
            for value_id in value_ids:
                if not isinstance(value_id, str) or value_id not in values_by_id:
                    raise ValueError(
                        f"normalized lookup references unknown canonical ID: {value_id!r}"
                    )
                value = values_by_id[value_id]
                if value.attribute != attribute or value.normalized != surface:
                    raise ValueError(
                        f"normalized lookup disagrees with canonical value: {value_id}"
                    )
                attribute_ids.add(value_id)
                referenced_ids.add(value_id)

        rows_by_attribute[attribute] = [
            {
                "canonical_id": value_id,
                "attribute": values_by_id[value_id].attribute,
                "value": values_by_id[value_id].value,
                "normalized": values_by_id[value_id].normalized,
                "embedding_text": values_by_id[value_id].normalized,
            }
            for value_id in sorted(attribute_ids)
        ]

    if referenced_ids != set(values_by_id):
        missing = sorted(set(values_by_id) - referenced_ids)
        extra = sorted(referenced_ids - set(values_by_id))
        raise ValueError(
            "normalized lookup and canonical values disagree: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return rows_by_attribute


def _resolve_model_path(model: str | None) -> str:
    configured = (
        model
        or os.environ.get("SHOPPING_ATTRIBUTE_EMBEDDING_MODEL", "")
    ).strip()
    if configured:
        return configured
    for path in DEFAULT_MODEL_PATHS:
        if path.is_dir():
            return path.as_posix()
    return DEFAULT_MODEL_NAME


def _normalise_matrix(matrix: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "embedding generation requires NumPy; install "
            "requirements-embeddings.txt"
        ) from exc
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("encoder output must be a two-dimensional matrix")
    if not bool(np.isfinite(array).all()):
        raise ValueError("encoder output contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool((norms == 0).any()):
        raise ValueError("encoder output contains a zero vector")
    return array / norms


def _encode_rows(
    encoder: Any,
    rows: Sequence[Mapping[str, str]],
    *,
    batch_size: int,
    dimension: int,
    attribute: str,
    started: float,
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "embedding generation requires NumPy; install "
            "requirements-embeddings.txt"
        ) from exc
    embed_documents = getattr(encoder, "embed_documents", None)
    if not callable(embed_documents):
        raise RuntimeError("local attribute encoder does not expose embed_documents()")

    matrices: list[Any] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = _normalise_matrix(
            embed_documents([str(row["embedding_text"]) for row in batch])
        )
        if encoded.shape != (len(batch), dimension):
            raise RuntimeError(
                f"{attribute} encoder returned shape {tuple(encoded.shape)}, "
                f"expected ({len(batch)}, {dimension})"
            )
        matrices.append(encoded)
        end = start + len(batch)
        _log(
            f"{attribute}: encoded {end:,}/{len(rows):,} values "
            f"elapsed={time.perf_counter() - started:.1f}s"
        )
    return np.vstack(matrices)


def build_v5_attribute_embeddings(
    dictionary_dir: str | Path = DEFAULT_DICTIONARY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    model: str | None = None,
    batch_size: int = 32,
    attributes: Sequence[str] = ATTRIBUTE_EMBEDDING_FIELDS,
    device: str | None = None,
    half_precision: bool = False,
) -> dict[str, Any]:
    """Build one normalized matrix per V5 dictionary attribute."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected_attributes = tuple(attributes)
    unknown = sorted(set(selected_attributes) - set(ATTRIBUTE_EMBEDDING_FIELDS))
    if unknown:
        raise ValueError(
            "attributes are not eligible for V5 semantic embedding: "
            f"{unknown}"
        )
    if len(set(selected_attributes)) != len(selected_attributes):
        raise ValueError("attributes must not contain duplicates")

    dictionary = Path(dictionary_dir)
    output = Path(output_dir)
    rows_by_attribute = _load_dictionary_rows(dictionary)
    if "style" in selected_attributes and not rows_by_attribute["style"]:
        raise RuntimeError(
            "style embedding requested but the V5 dictionary has no style values; "
            "rebuild it from data/derived/annotations/v5/annotations.jsonl "
            "after style.jsonl has been aggregated"
        )
    model_path = _resolve_model_path(model)
    _log(f"dictionary: {dictionary}")
    _log(f"model: {model_path} (local_files_only=True)")
    _log(
        "encoding text: canonical normalized surface; "
        "no retrieval prefix or query instruction"
    )
    started = time.perf_counter()
    try:
        encoder = load_local_sentence_transformer(
            model_path,
            batch_size=batch_size,
            device=device,
            half_precision=half_precision,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"unable to load local attribute embedding model {model_path!r}: {exc}"
        ) from exc

    dimension = getattr(encoder, "embedding_dimension", None)
    if dimension is None:
        raise RuntimeError("attribute embedding model did not report its dimension")
    dimension = int(dimension)
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"attribute embedding dimension is {dimension}; "
            f"expected {EXPECTED_DIMENSION} for {DEFAULT_MODEL_NAME}"
        )
    _log(f"model loaded: dimension={dimension}, normalization=l2")

    output.mkdir(parents=True, exist_ok=True)
    stale_brand_matrix = output / "brand_embeddings.npy"
    if stale_brand_matrix.exists():
        stale_brand_matrix.unlink()
        _log("removed stale brand_embeddings.npy; brand is exact Layer 1 only")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": model_path,
        "model_path": model_path,
        "dimension": dimension,
        "normalization": "l2",
        "embedding_text": "normalized",
        "attributes": {},
    }
    total_values = 0
    for attribute in selected_attributes:
        rows = rows_by_attribute[attribute]
        started_attribute = time.perf_counter()
        filename = f"{attribute}_embeddings.npy"
        _log(f"{attribute}: starting {len(rows):,} values")
        if rows:
            matrix = _encode_rows(
                encoder,
                rows,
                batch_size=batch_size,
                dimension=dimension,
                attribute=attribute,
                started=started_attribute,
            )
        else:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "embedding generation requires NumPy; install "
                    "requirements-embeddings.txt"
                ) from exc
            matrix = np.empty((0, dimension), dtype=np.float32)
        _write_npy(output / filename, matrix)
        metadata["attributes"][attribute] = {
            "embedding_file": filename,
            "count": len(rows),
            "rows": rows,
        }
        total_values += len(rows)
        _log(
            f"{attribute}: complete {len(rows):,} values -> {filename} "
            f"elapsed={time.perf_counter() - started_attribute:.1f}s"
        )

    source_files = {
        "canonical_values": dictionary / "canonical_values.json",
        "normalized_lookup": dictionary / "normalized_lookup.json",
    }
    manifest = {
        **metadata,
        "dictionary_path": str(dictionary),
        "dictionary_sha256": {
            name: _sha256(path) for name, path in source_files.items()
        },
        "attributes_selected": list(selected_attributes),
        "attributes_excluded": ["brand"],
        "canonical_value_count": total_values,
        "canonical_value_count_by_attribute": {
            attribute: len(rows_by_attribute[attribute])
            for attribute in selected_attributes
        },
        "complete": True,
    }
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "manifest.json", manifest)
    _log(
        f"COMPLETE: {total_values:,} vectors across {len(selected_attributes)} "
        f"attributes in {time.perf_counter() - started:.1f}s"
    )
    return {
        "output_dir": str(output),
        "model": manifest["model"],
        "dimension": dimension,
        "canonical_value_count": total_values,
        "canonical_value_count_by_attribute": manifest[
            "canonical_value_count_by_attribute"
        ],
        "attributes": list(selected_attributes),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary-dir", default=str(DEFAULT_DICTIONARY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument(
        "--attributes",
        nargs="+",
        choices=ATTRIBUTE_EMBEDDING_FIELDS,
        default=list(ATTRIBUTE_EMBEDDING_FIELDS),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        result = build_v5_attribute_embeddings(
            args.dictionary_dir,
            args.output_dir,
            model=args.model,
            batch_size=args.batch_size,
            attributes=args.attributes,
            device=args.device,
            half_precision=args.half_precision,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
