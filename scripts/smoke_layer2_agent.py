"""Print a small end-to-end trace for the configured Layer 2 Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.agent_factory import build_evaluator_agent


CASES = (
    ("BUYING example", ("I want gold earrings under $50",)),
    ("BROWSING example", ("I want something comfortable for walking around Europe",)),
    (
        "MULTI-TURN example",
        (
            "I need shoes for walking around Europe",
            "black",
            "under $100",
        ),
    ),
)


def _query_breakdown(agent: Any, query_text: str, candidates: list[Any]) -> None:
    retriever = agent.retriever
    index = retriever.layer2_index
    if index is None or not retriever.dense_available:
        print("  Layer 2: inactive")
        return

    query = retriever._query_embedding(query_text, index.dimension)
    if query is None:
        print("  query embedding: unavailable")
        return
    scores, view_scores = index.score_all(query, weights=retriever.layer2_weights)
    score_by_asin = dict(zip(index.asins, scores))
    print(f"  query embedding dimension: {query.shape[0]}")
    print("  top products:")
    for candidate in candidates[:10]:
        row = index.row_for_asin(candidate.parent_asin)
        fields = {
            view: float(view_scores[view][row]) if bool(index.presence[view][row]) else None
            for view in ("categories", "title", "features", "description")
        }
        print(
            "    "
            + json.dumps(
                {
                    "parent_asin": candidate.parent_asin,
                    "structured_score": candidate.constraint_score,
                    "category_similarity": fields["categories"],
                    "title_similarity": fields["title"],
                    "features_similarity": fields["features"],
                    "description_similarity": fields["description"],
                    "layer2_score": float(score_by_asin[candidate.parent_asin]),
                    "final_score": candidate.score,
                },
                sort_keys=True,
            )
        )


def run_case(agent: Any, label: str, messages: tuple[str, ...]) -> None:
    session_id = "layer2-smoke"
    agent.reset(session_id, {})
    print(f"\n=== {label} ===")
    for turn, message in enumerate(messages, 1):
        delta = agent._extract(message)
        response = agent.respond(session_id, message, turn, 10)
        state = agent.sessions.get(session_id)
        candidates = agent.retriever.retrieve(
            state.mode or "BROWSING",
            state.query_text,
            state.constraints,
            limit=100,
            excluded_asins=state.excluded_recommendations,
        )
        print(f"turn: {turn}")
        print(f"user: {message}")
        print(f"mode: {state.mode}")
        print(f"delta_constraints: {delta.as_dict()}")
        print(f"session_constraints: {state.constraints.as_dict()}")
        print(f"query_text: {state.query_text}")
        print(f"candidate_count: {len(candidates)}")
        _query_breakdown(agent, state.query_text, candidates)
        print(f"agent_response: {json.dumps(response, ensure_ascii=False, sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--layer2-artifact-dir",
        default="data/derived/product_embeddings",
    )
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument("--embedding-model")
    embedding_group.add_argument("--hash-dimension", type=int)
    args = parser.parse_args()

    hash_dimension = None
    if args.embedding_model is None:
        hash_dimension = args.hash_dimension or 384

    agent = build_evaluator_agent(
        args.catalog,
        layer2_artifact_dir=args.layer2_artifact_dir,
        embedding_model=args.embedding_model,
        hash_dimension=hash_dimension,
    )
    for label, messages in CASES:
        run_case(agent, label, messages)


if __name__ == "__main__":
    main()
