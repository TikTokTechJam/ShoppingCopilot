"""Build the direct catalog-only Layer 2 embedding artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_embeddings import HashEmbeddingModel, build_layer2_embeddings
from product_embeddings.pipeline import (
    load_injected_embedder,
    load_local_sentence_transformer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build direct catalog-only Layer 2 field embeddings."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--output-dir",
        default="data/derived/product_embeddings",
        help="Directory for the four Layer 2 matrices, metadata, and manifest.",
    )
    parser.add_argument(
        "--model",
        help="Local or already-cached SentenceTransformer path; never downloaded.",
    )
    parser.add_argument(
        "--embedder",
        help="Injected Python embedder as module:object or module:zero_arg_factory.",
    )
    parser.add_argument(
        "--hash-dimension",
        type=int,
        default=384,
        help="Dimension for the dependency-light deterministic fallback.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        help="SentenceTransformer device, for example cpu or mps; auto-selected when omitted.",
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="Load the model in float16 for supported accelerators.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show field-level logs and SentenceTransformer batch progress.",
    )
    parser.add_argument("--catalog-version")
    parser.add_argument(
        "--generated-at-utc",
        help="Optional fixed timestamp for byte-stable manifest regeneration.",
    )
    args = parser.parse_args()

    if args.model and args.embedder:
        parser.error("choose only one of --model and --embedder")
    if args.model:
        model = load_local_sentence_transformer(
            args.model,
            batch_size=args.batch_size,
            device=args.device,
            half_precision=args.half_precision,
            show_progress_bar=args.progress,
        )
    elif args.embedder:
        model = load_injected_embedder(args.embedder)
    else:
        model = HashEmbeddingModel(args.hash_dimension)

    manifest = build_layer2_embeddings(
        args.catalog,
        args.output_dir,
        model,
        batch_size=args.batch_size,
        catalog_version=args.catalog_version,
        generated_at_utc=args.generated_at_utc,
        progress=args.progress,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
