"""Build evaluator Agents with an explicit Layer 2 configuration."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from product_embeddings import HashEmbeddingModel
from product_embeddings.pipeline import (
    is_jina_v5_text_nano,
    load_local_sentence_transformer,
)
from starter.agent import Agent
from starter.retrieval import DEFAULT_LAYER2_ARTIFACT_PATHS


EMBEDDING_MODEL_ENV = "SHOPPING_EMBEDDING_MODEL"


def _first_layer2_artifact_dir(catalog_path: str | Path) -> Path | None:
    try:
        is_default_catalog = Path(catalog_path).resolve() == Path(
            "data/catalog.jsonl"
        ).resolve()
    except OSError:
        is_default_catalog = False
    if not is_default_catalog:
        return None
    for path in DEFAULT_LAYER2_ARTIFACT_PATHS:
        if path.is_dir():
            return path
    return None


def _manifest_embedding_model(artifact_dir: Path) -> str | None:
    try:
        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    model = manifest.get("embedding_model", manifest.get("model"))
    if not isinstance(model, str) or not model.strip():
        return None
    if model.strip().casefold() == "hashing-fallback-v1":
        return None
    return model.strip()


# Keep the benchmark model outside Git while giving every checkout one stable
# location for the locally downloaded Jina weights.
DEFAULT_JINA_MODEL_PATH = Path("model/jina-embeddings-v5-text-nano")


def build_evaluator_agent(
    catalog_path: str | Path,
    *,
    layer2_artifact_dir: str | Path | None = None,
    embedding_model: str | None = None,
    hash_dimension: int | None = None,
    disable_layer2: bool = False,
    half_precision: bool = False,
    device: str | None = None,
) -> Agent:
    """Build the production Agent for a baseline or Layer 2 evaluation run.

    With the default catalog, a local Layer 2 artifact is discovered and its
    manifest model is loaded when available. Layer 2 requires both an artifact
    directory and a compatible query encoder so a matrix can never be silently
    evaluated without a matching encoder.
    """

    if disable_layer2:
        if (
            layer2_artifact_dir is not None
            or embedding_model is not None
            or hash_dimension is not None
        ):
            raise ValueError("--disable-layer2 cannot be combined with Layer 2 options")
        return Agent(catalog_path)

    configured_model = embedding_model
    if configured_model is None:
        configured_model = os.environ.get(EMBEDDING_MODEL_ENV, "").strip() or None

    selected_artifact_dir = (
        Path(layer2_artifact_dir)
        if layer2_artifact_dir is not None
        else _first_layer2_artifact_dir(catalog_path)
    )

    if selected_artifact_dir is None:
        if configured_model is not None or hash_dimension is not None:
            raise ValueError("Layer 2 options require --layer2-artifact-dir")
        return Agent(catalog_path)

    auto_model = configured_model is None and hash_dimension is None
    if auto_model:
        configured_model = _manifest_embedding_model(selected_artifact_dir)
        if (
            configured_model is not None
            and is_jina_v5_text_nano(configured_model)
            and DEFAULT_JINA_MODEL_PATH.is_dir()
        ):
            configured_model = DEFAULT_JINA_MODEL_PATH.as_posix()

    if configured_model is not None and hash_dimension is not None:
        raise ValueError("choose --embedding-model or --hash-dimension, not both")
    if configured_model is None and hash_dimension is None:
        raise ValueError(
            "Layer 2 artifact does not declare a real embedding model; provide "
            f"--embedding-model or use --hash-dimension only for smoke testing"
        )

    query_encoder: Any
    if hash_dimension is not None:
        query_encoder = HashEmbeddingModel(hash_dimension)
    else:
        jina = is_jina_v5_text_nano(configured_model)
        try:
            query_encoder = load_local_sentence_transformer(
                configured_model,
                task="retrieval" if jina else None,
                document_prompt_name="document" if jina else None,
                query_prompt_name="query" if jina else None,
                trust_remote_code=jina,
                device=device,
                half_precision=half_precision,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            if not auto_model:
                raise
            warnings.warn(
                "Layer 2 disabled because its manifest model could not be loaded "
                f"locally: {configured_model}",
                RuntimeWarning,
                stacklevel=2,
            )
            return Agent(catalog_path, layer2_artifact_dir=selected_artifact_dir)

    return Agent(
        catalog_path,
        query_encoder=query_encoder,
        layer2_artifact_dir=selected_artifact_dir,
    )


__all__ = ["build_evaluator_agent"]
