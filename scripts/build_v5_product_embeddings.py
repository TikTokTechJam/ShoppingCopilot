"""Build the V5 semantic product-card embedding artifact.

This is an offline setup command.  It never downloads a model and never changes
the V5 annotations or the catalog; it only writes the requested product-vector
artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_embeddings.pipeline import (
    embedding_models_compatible,
    load_local_sentence_transformer,
)
from product_embeddings.v5 import (
    V5_PRODUCT_MODEL,
    build_v5_product_embeddings,
)


DEFAULT_MODEL_PATHS = (
    Path("models/qwen3-embedding-0.6b"),
    Path("models/Qwen3-Embedding-0.6B"),
    Path("model/qwen3-embedding-0.6b"),
)


def _resolve_model_path(value: str | None) -> str:
    configured = (value or os.environ.get("SHOPPING_PRODUCT_EMBEDDING_MODEL", "")).strip()
    if configured:
        return configured
    for path in DEFAULT_MODEL_PATHS:
        if path.is_dir():
            return path.as_posix()
    return DEFAULT_MODEL_PATHS[0].as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--annotations",
        default="data/derived/annotations/v5/annotations.jsonl",
        help="Aggregated V5 annotations used to build each product card.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/derived/product_embeddings_v5",
        help="Directory for the V5 product matrix, metadata, cards, and manifest.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Local Qwen3-Embedding-0.6B SentenceTransformer directory. "
            f"Expected model: {V5_PRODUCT_MODEL}."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    model_path = _resolve_model_path(args.model)
    print(
        f"[v5-product-embeddings] loading local model {model_path} "
        "(local_files_only=True, trust_remote_code=True)",
        flush=True,
    )
    try:
        model = load_local_sentence_transformer(
            model_path,
            trust_remote_code=True,
            batch_size=args.batch_size,
            device=args.device,
            half_precision=args.half_precision,
            show_progress_bar=args.progress,
        )
        actual_model = getattr(model, "model_id", None)
        if actual_model and not embedding_models_compatible(V5_PRODUCT_MODEL, actual_model):
            raise ValueError(
                "local model does not match the required V5 product model: "
                f"expected={V5_PRODUCT_MODEL!r}, actual={actual_model!r}"
            )
        manifest = build_v5_product_embeddings(
            args.catalog,
            args.annotations,
            args.output_dir,
            model,
            batch_size=args.batch_size,
            progress=args.progress,
            model_id=V5_PRODUCT_MODEL,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"[v5-product-embeddings] complete: {manifest['product_count']:,} "
        f"products, dimension={manifest['dimension']}, "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
