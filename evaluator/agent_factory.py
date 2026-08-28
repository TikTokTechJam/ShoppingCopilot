"""Build evaluator Agents with an explicit Layer 2 configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from product_embeddings import HashEmbeddingModel
from product_embeddings.pipeline import (
    is_jina_v5_text_nano,
    load_local_sentence_transformer,
)
from starter.agent import Agent


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

    No arguments selects the existing Layer 1-only baseline. Layer 2 requires
    both an artifact directory and a query encoder so a matrix can never be
    silently evaluated without a compatible encoder.
    """

    if disable_layer2:
        if (
            layer2_artifact_dir is not None
            or embedding_model is not None
            or hash_dimension is not None
        ):
            raise ValueError("--disable-layer2 cannot be combined with Layer 2 options")
        return Agent(catalog_path)

    if layer2_artifact_dir is None:
        if embedding_model is not None or hash_dimension is not None:
            raise ValueError("Layer 2 options require --layer2-artifact-dir")
        return Agent(catalog_path)

    if embedding_model is not None and hash_dimension is not None:
        raise ValueError("choose --embedding-model or --hash-dimension, not both")
    if embedding_model is None and hash_dimension is None:
        if DEFAULT_JINA_MODEL_PATH.is_dir():
            embedding_model = DEFAULT_JINA_MODEL_PATH.as_posix()
        else:
            raise ValueError(
                "Layer 2 requires --embedding-model or --hash-dimension; "
                "download jina-embeddings-v5-text-nano into "
                f"{DEFAULT_JINA_MODEL_PATH}"
            )

    query_encoder: Any
    if embedding_model is not None:
        jina = is_jina_v5_text_nano(embedding_model)
        query_encoder = load_local_sentence_transformer(
            embedding_model,
            task="retrieval" if jina else None,
            document_prompt_name="document" if jina else None,
            query_prompt_name="query" if jina else None,
            trust_remote_code=jina,
            device=device,
            half_precision=half_precision,
        )
    else:
        query_encoder = HashEmbeddingModel(hash_dimension)

    return Agent(
        catalog_path,
        query_encoder=query_encoder,
        layer2_artifact_dir=layer2_artifact_dir,
    )


__all__ = ["build_evaluator_agent"]
