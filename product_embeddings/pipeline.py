from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .text import PRODUCT_TEXT_VERSION, build_product_text
from .tier4 import TIER4_ARTIFACT_VERSION, build_tier4_record

BUILDER_VERSION = "product-embeddings-v2"
_FACT_FIELDS = (
    "category",
    "brand",
    "color",
    "material",
    "size",
    "style",
    "feature",
    "use_case",
)


class EmbeddingModel(Protocol):
    """Small injection boundary for local embedding implementations."""

    def embed_documents(self, texts: Sequence[str]) -> Any:
        ...


class HashEmbeddingModel:
    """Dependency-light deterministic fallback for pipeline smoke generation.

    This is a reproducible lexical sketch, not a quality semantic model. Pass a
    local model or injected embedder for the benchmark artifact.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_id = "hashing-fallback-v1"

    def embed_documents(self, texts: Sequence[str]) -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "NumPy is required to write .npy embedding artifacts"
            ) from exc

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.blake2b(
                    token.encode("utf-8"), digest_size=16
                ).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vectors[row, index] += sign
        return vectors


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "NumPy is required for product embedding artifacts; install the "
            "base embedding requirements before building"
        ) from exc
    return np


def _read_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    # The supplied annotation example is a regular JSON object, while build
    # outputs are normally JSONL. Accept both at the facts boundary.
    if path.suffix.casefold() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: invalid JSON") from exc
        if isinstance(payload, Mapping):
            yield 1, payload
            return
        if isinstance(payload, list):
            for line_number, record in enumerate(payload, 1):
                if not isinstance(record, Mapping):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                yield line_number, record
            return
        raise ValueError(f"{path}: JSON document must contain an object or array")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            yield line_number, record


def _parent_asin(record: Mapping[str, Any], *, path: Path, line_number: int) -> str:
    value = record.get("parent_asin")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: missing parent_asin")
    return value.strip()


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, record in _read_jsonl(path):
        asin = _parent_asin(record, path=path, line_number=line_number)
        if asin in seen:
            raise ValueError(f"{path}:{line_number}: duplicate parent_asin {asin}")
        seen.add(asin)
        product = dict(record)
        product["parent_asin"] = asin
        products.append(product)
    if not products:
        raise ValueError(f"{path}: catalog contains no products")
    return products


def _normalise_fact_values(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise ValueError(f"canonical fact {field} must be an array or string")

    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"canonical fact {field} contains an invalid value")
        cleaned = " ".join(item.split())
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _normalise_facts(
    record: Mapping[str, Any], *, path: Path, line_number: int
) -> dict[str, Any]:
    nested = record.get("facts")
    if nested is None:
        facts_source: Mapping[str, Any] = record
    elif isinstance(nested, Mapping):
        facts_source = nested
    else:
        raise ValueError(f"{path}:{line_number}: facts must be an object")

    # V4 annotation/test records do not contain structured ``size`` yet. The
    # final catalog-facts artifact does, so make only semantic fields required
    # here and supply an empty size list for the annotation shape.
    required_fields = [field for field in _FACT_FIELDS if field != "size"]
    missing = [field for field in required_fields if field not in facts_source]
    if missing:
        raise ValueError(
            f"{path}:{line_number}: canonical facts missing fields {missing}"
        )

    facts: dict[str, Any] = {}
    for field in _FACT_FIELDS:
        if field == "size" and field not in facts_source:
            facts[field] = []
        elif field == "brand":
            value = facts_source[field]
            if value is None:
                facts[field] = None
            elif isinstance(value, str):
                facts[field] = " ".join(value.split())
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                facts[field] = _normalise_fact_values(value, field=field)
            else:
                raise ValueError(
                    f"{path}:{line_number}: canonical fact brand must be string, array, or null"
                )
        else:
            facts[field] = _normalise_fact_values(facts_source[field], field=field)
    return facts


def _read_facts(path: Path) -> dict[str, dict[str, Any]]:
    facts_by_asin: dict[str, dict[str, Any]] = {}
    for line_number, record in _read_jsonl(path):
        asin = _parent_asin(record, path=path, line_number=line_number)
        if asin in facts_by_asin:
            raise ValueError(f"{path}:{line_number}: duplicate parent_asin {asin}")
        facts_by_asin[asin] = _normalise_facts(
            record, path=path, line_number=line_number
        )
    if not facts_by_asin:
        raise ValueError(f"{path}: facts contain no products")
    return facts_by_asin


def _read_tier4(path: Path) -> dict[str, dict[str, Any]]:
    raw_by_asin: dict[str, dict[str, Any]] = {}
    for line_number, record in _read_jsonl(path):
        try:
            raw = build_tier4_record(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid Tier 4 record") from exc
        asin = raw["parent_asin"]
        if asin in raw_by_asin:
            raise ValueError(f"{path}:{line_number}: duplicate parent_asin {asin}")
        raw_by_asin[asin] = raw
    if not raw_by_asin:
        raise ValueError(f"{path}: Tier 4 source contains no products")
    return raw_by_asin


def _validate_input_asins(
    products: Sequence[Mapping[str, Any]], facts_by_asin: Mapping[str, Any]
) -> None:
    catalog_asins = [str(product["parent_asin"]) for product in products]
    catalog_set = set(catalog_asins)
    facts_set = set(facts_by_asin)
    missing = sorted(catalog_set - facts_set)
    extra = sorted(facts_set - catalog_set)
    if missing or extra:
        raise ValueError(
            "catalog/facts ASIN mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _call_encode(model: Any, texts: Sequence[str], batch_size: int) -> Any:
    method = model.encode
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    for name, value in (
        ("batch_size", batch_size),
        ("show_progress_bar", False),
        ("convert_to_numpy", True),
        ("normalize_embeddings", False),
    ):
        if accepts_kwargs or name in parameters:
            kwargs[name] = value
    return method(list(texts), **kwargs)


def _embed_batch(model: Any, texts: Sequence[str], batch_size: int) -> Any:
    if hasattr(model, "embed_documents"):
        return model.embed_documents(list(texts))
    if hasattr(model, "encode"):
        return _call_encode(model, texts, batch_size)
    if hasattr(model, "embed"):
        return model.embed(list(texts))
    if callable(model):
        return model(list(texts))
    raise TypeError(
        "embedder must expose embed_documents, encode, embed, or be callable"
    )


def _as_numpy_batch(value: Any, expected_rows: int, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 1:
        if expected_rows != 1:
            raise ValueError("embedder returned one vector for multiple texts")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != expected_rows:
        raise ValueError(
            "embedder output must be a 2-D matrix with one row per product"
        )
    return array


def _embed_and_normalise(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> tuple[Any, int]:
    np = _require_numpy()
    batches: list[Any] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batches.append(_as_numpy_batch(
            _embed_batch(model, batch_texts, batch_size),
            len(batch_texts),
            np,
        ))
    raw = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if raw.ndim != 2 or raw.shape[0] != len(texts) or raw.shape[1] < 1:
        raise ValueError("embedder output must have shape (product_count, dimension)")
    if not np.isfinite(raw).all():
        raise ValueError("embedder returned non-finite values")

    norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0.0).any():
        raise ValueError("cannot L2-normalize a zero or non-finite product vector")
    normalized = (raw.astype(np.float64) / norms).astype(np.float32)
    return normalized, int(normalized.shape[1])


def _model_id(model: Any) -> str:
    for name in ("model_id", "model_name", "name"):
        value = getattr(model, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(model).__name__


def _source_version(path: Path) -> str:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return "unknown"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if isinstance(payload, Mapping):
        for key in ("version", "facts_version", "catalog_version"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "unknown"


def _manifest_path(path: Path) -> str:
    return path.as_posix()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_npy(path: Path, array: Any, np: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def build_product_embeddings(
    catalog_path: str | Path,
    facts_path: str | Path,
    output_dir: str | Path,
    model: Any,
    *,
    raw_text_path: str | Path | None = None,
    batch_size: int = 32,
    catalog_version: str | None = None,
    facts_version: str | None = None,
    generated_at_utc: str | None = None,
    description_max_chars: int = 1_000,
) -> dict[str, Any]:
    """Build the three Issue #12 artifacts from catalog and canonical facts."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    catalog = Path(catalog_path)
    facts = Path(facts_path)
    output = Path(output_dir)
    products = _read_catalog(catalog)
    facts_by_asin = _read_facts(facts)
    _validate_input_asins(products, facts_by_asin)
    raw_text = Path(raw_text_path) if raw_text_path is not None else None
    raw_by_asin: dict[str, dict[str, Any]] | None = None
    if raw_text is not None:
        raw_by_asin = _read_tier4(raw_text)
        _validate_input_asins(products, raw_by_asin)

    asins = [str(product["parent_asin"]) for product in products]
    texts = [
        build_product_text(
            product,
            facts_by_asin[asin],
            raw_text=raw_by_asin[asin] if raw_by_asin is not None else None,
            description_max_chars=description_max_chars,
        )
        for product, asin in zip(products, asins)
    ]
    vectors, dimension = _embed_and_normalise(model, texts, batch_size=batch_size)
    np = _require_numpy()
    output.mkdir(parents=True, exist_ok=True)

    metadata = [
        {"row": row, "parent_asin": asin}
        for row, asin in enumerate(asins)
    ]
    resolved_catalog_version = catalog_version or _source_version(catalog)
    resolved_facts_version = facts_version or _source_version(facts)
    if generated_at_utc is None:
        generated_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.strip():
        raise ValueError("generated_at_utc must be a non-empty string when provided")

    manifest = {
        "artifact_format_version": BUILDER_VERSION,
        "generation_version": BUILDER_VERSION,
        "generated_at_utc": generated_at_utc,
        "catalog_path": _manifest_path(catalog),
        "facts_path": _manifest_path(facts),
        "tier4_path": _manifest_path(raw_text) if raw_text is not None else None,
        "tier4_version": TIER4_ARTIFACT_VERSION if raw_text is not None else "catalog-source",
        "catalog_version": resolved_catalog_version,
        "facts_version": resolved_facts_version,
        "product_text_version": PRODUCT_TEXT_VERSION,
        "model": _model_id(model),
        "embedding_model": _model_id(model),
        "dimension": dimension,
        "embedding_dimension": dimension,
        "dtype": "float32",
        "normalization": "l2",
        "row_order": "catalog order",
        "product_count": len(asins),
    }

    _atomic_npy(output / "product_embeddings.npy", vectors, np)
    _atomic_json(output / "product_embedding_metadata.json", metadata)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def load_local_sentence_transformer(model_path: str) -> Any:
    """Load a cached/local SentenceTransformer without permitting downloads."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "--model requires the optional sentence-transformers dependency; "
            "inject a local embedder or use the hashing fallback instead"
        ) from exc
    try:
        return SentenceTransformer(model_path, local_files_only=True)
    except TypeError as exc:  # pragma: no cover - old optional dependency
        raise RuntimeError(
            "the installed sentence-transformers version does not support "
            "local_files_only; refusing to load a potentially network-backed model"
        ) from exc


def load_injected_embedder(spec: str) -> Any:
    """Load ``module:attribute`` as an embedder or zero-argument factory."""
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("embedder must use the module:attribute form")
    target = getattr(importlib.import_module(module_name), attribute_name)
    if any(hasattr(target, name) for name in ("embed_documents", "encode", "embed")):
        return target
    if callable(target):
        instance = target()
        if any(hasattr(instance, name) for name in ("embed_documents", "encode", "embed")):
            return instance
    raise TypeError(
        "injected embedder must be an embedder object or zero-argument factory"
    )
