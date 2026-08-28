"""Run one real query through the local Jina encoder and Layer 2 index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.agent_factory import build_evaluator_agent


QUERY = "jumpsuits for cosplay"
DEFAULT_ARTIFACT_DIR = "data/derived/product_embeddings_jina"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--layer2-artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--embedding-model")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    try:
        agent = build_evaluator_agent(
            args.catalog,
            layer2_artifact_dir=args.layer2_artifact_dir,
            embedding_model=args.embedding_model,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    retriever = agent.retriever
    index = retriever.layer2_index
    query_encoder = retriever.query_encoder
    if index is None:
        parser.error("Layer 2 index did not load")
    if query_encoder is None or not retriever.dense_available:
        reason = getattr(retriever, "layer2_compatibility_error", None)
        parser.error(reason or "Layer 2 dense retrieval is unavailable")

    query = retriever._query_embedding(QUERY, index.dimension)
    if query is None:
        parser.error("query encoder did not return a valid Layer 2 vector")
    matches = index.search(query, top_k=args.top_k, weights=retriever.layer2_weights)
    if not matches or not any(abs(match.score) > 1e-6 for match in matches):
        parser.error("Layer 2 returned no non-zero dense scores")

    top_asins = [match.parent_asin for match in matches]
    catalog_prefix = retriever._catalog_order[: len(top_asins)]
    print(json.dumps({
        "layer2_index_loaded": True,
        "query_encoder_loaded": True,
        "dense_available": True,
        "hash_fallback": getattr(query_encoder, "model_id", "")
        == "hashing-fallback-v1",
        "model": getattr(query_encoder, "model_id", None),
        "dimension": index.dimension,
        "query": QUERY,
        "query_shape": list(query.shape),
        "query_finite": True,
        "query_norm": float(np.linalg.norm(query)),
        "catalog_order_fallback": top_asins == catalog_prefix,
        "top_dense_results": [
            {"parent_asin": match.parent_asin, "score": float(match.score)}
            for match in matches
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
