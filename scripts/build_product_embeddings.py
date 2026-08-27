from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_embeddings import HashEmbeddingModel, build_product_embeddings
from product_embeddings.pipeline import load_injected_embedder, load_local_sentence_transformer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline product embedding artifacts from catalog facts."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--facts",
        default="data/derived/catalog_facts/catalog_facts.jsonl",
        help="Canonical facts JSONL; Issue #5 annotation wrappers are also accepted.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/derived/product_embeddings",
        help="Directory for the three generated artifacts.",
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
    parser.add_argument("--catalog-version")
    parser.add_argument("--facts-version")
    parser.add_argument(
        "--generated-at-utc",
        help="Optional fixed timestamp for byte-stable manifest regeneration.",
    )
    parser.add_argument("--description-max-chars", type=int, default=1000)
    args = parser.parse_args()

    if args.model and args.embedder:
        parser.error("choose only one of --model and --embedder")
    if args.model:
        model = load_local_sentence_transformer(args.model)
    elif args.embedder:
        model = load_injected_embedder(args.embedder)
    else:
        model = HashEmbeddingModel(args.hash_dimension)

    manifest = build_product_embeddings(
        args.catalog,
        args.facts,
        args.output_dir,
        model,
        batch_size=args.batch_size,
        catalog_version=args.catalog_version,
        facts_version=args.facts_version,
        generated_at_utc=args.generated_at_utc,
        description_max_chars=args.description_max_chars,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
