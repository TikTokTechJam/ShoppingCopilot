"""Download and validate the local Jina Layer 2 query encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from product_embeddings.pipeline import load_local_sentence_transformer


MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"
EXPECTED_DIMENSION = 768
DEFAULT_MODEL_DIR = Path("model") / "jina-embeddings-v5-text-nano"
VALIDATION_QUERY = "jumpsuits for cosplay"


def _snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Hugging Face setup requires huggingface-hub; install "
            "requirements-embeddings.txt first"
        ) from exc
    return snapshot_download


def _validate_query_embedding(encoder: Any) -> tuple[Any, float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("NumPy is required to validate the query encoder") from exc

    embed_query = getattr(encoder, "embed_query", None)
    if not callable(embed_query):
        raise RuntimeError("the local encoder does not expose embed_query()")
    query = np.asarray(embed_query(VALIDATION_QUERY), dtype=np.float32)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1 or query.size != EXPECTED_DIMENSION:
        raise RuntimeError(
            "Jina query encoder returned dimension "
            f"{tuple(query.shape)}, expected ({EXPECTED_DIMENSION},)"
        )
    if not bool(np.isfinite(query).all()):
        raise RuntimeError("Jina query encoder returned non-finite values")
    norm = float(np.linalg.norm(query.astype(np.float64)))
    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("Jina query encoder returned a zero or invalid vector")
    return query, norm


def setup_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    revision: str | None = None,
    downloader: Callable[..., str] | None = None,
    encoder_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Download the exact model into ``model_dir`` and validate one query.

    ``downloader`` and ``encoder_loader`` are injection points for offline unit
    tests. The normal command uses Hugging Face for this one-time setup only;
    runtime loading remains local-only in ``product_embeddings.pipeline``.
    """

    destination = Path(model_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    download = downloader or _snapshot_download()
    download_kwargs: dict[str, Any] = {
        "repo_id": MODEL_ID,
        "local_dir": str(destination),
    }
    if revision is not None:
        if not revision.strip():
            raise ValueError("revision must be non-empty when provided")
        download_kwargs["revision"] = revision
    download(**download_kwargs)

    if not destination.is_dir():
        raise RuntimeError(f"model download did not create {destination}")
    load = encoder_loader or load_local_sentence_transformer
    encoder = load(
        str(destination),
        task="retrieval",
        document_prompt_name="document",
        query_prompt_name="query",
        trust_remote_code=True,
    )
    query, norm = _validate_query_embedding(encoder)
    return {
        "model_id": MODEL_ID,
        "model_path": destination,
        "query": VALIDATION_QUERY,
        "query_shape": tuple(int(value) for value in query.shape),
        "query_norm": norm,
        "dimension": int(query.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Deterministic local directory for the downloaded model.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face branch, tag, or commit to download.",
    )
    args = parser.parse_args()
    try:
        result = setup_model(args.model_dir, revision=args.revision)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"model: {result['model_id']}")
    print(f"local_model_path: {result['model_path']}")
    print(f"validation_query: {result['query']}")
    print(f"query_shape: {result['query_shape']}")
    print(f"query_dimension: {result['dimension']}")
    print(f"query_norm: {result['query_norm']:.6f}")
    print("validation: PASS")


if __name__ == "__main__":
    main()
