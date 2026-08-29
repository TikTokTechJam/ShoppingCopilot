"""Interactively inspect BGE semantic matches for canonical attribute values."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from dictionary.registry import (
    SEMANTIC_ATTRIBUTES,
    AttributeDictionary,
    LookupMatch,
)
from dictionary.semantic import (
    ATTRIBUTE_EMBEDDING_DIMENSION,
    ATTRIBUTE_EMBEDDING_MODEL,
    load_bge_attribute_encoder,
    resolve_attribute_model_path,
)


DEFAULT_DICTIONARY = Path("data/derived/dictionary")
TOP_K = 10
ATTRIBUTE_OPTIONS = tuple(SEMANTIC_ATTRIBUTES)


@dataclass(frozen=True)
class SemanticAttributeContext:
    dictionary: AttributeDictionary
    model_id: str
    dimension: int


def _print(message: str) -> None:
    print(message, flush=True)


def load_search_context(
    dictionary_path: str | Path = DEFAULT_DICTIONARY,
    *,
    embedding_model: str | None = None,
    progress_fn: Callable[[str], None] = _print,
) -> SemanticAttributeContext:
    """Load the canonical registry and its compatible local BGE encoder."""

    root = Path(dictionary_path)
    dictionary = AttributeDictionary.load(root)
    if not dictionary.has_semantic_embeddings:
        raise RuntimeError(
            f"canonical attribute embeddings are unavailable in {root}; "
            "build with a local BGE model first"
        )
    if dictionary.embedding_model != ATTRIBUTE_EMBEDDING_MODEL:
        raise RuntimeError(
            "canonical attribute artifact model mismatch: "
            f"{dictionary.embedding_model} != {ATTRIBUTE_EMBEDDING_MODEL}"
        )
    if dictionary.embedding_dimension != ATTRIBUTE_EMBEDDING_DIMENSION:
        raise RuntimeError(
            "canonical attribute artifact dimension mismatch: "
            f"{dictionary.embedding_dimension} != {ATTRIBUTE_EMBEDDING_DIMENSION}"
        )

    model_path = resolve_attribute_model_path(embedding_model)
    progress_fn(f"Loading canonical attribute embeddings: {root}")
    progress_fn(f"Loading local BGE query encoder: {model_path}")
    encoder = load_bge_attribute_encoder(model_path)
    dictionary.set_query_encoder(encoder)
    progress_fn(
        f"Ready: model={ATTRIBUTE_EMBEDDING_MODEL} "
        f"dimension={ATTRIBUTE_EMBEDDING_DIMENSION} normalization=l2 prefix=none"
    )
    return SemanticAttributeContext(
        dictionary=dictionary,
        model_id=ATTRIBUTE_EMBEDDING_MODEL,
        dimension=ATTRIBUTE_EMBEDDING_DIMENSION,
    )


def search_attribute(
    context: SemanticAttributeContext,
    attribute: str,
    query: str,
    *,
    top_k: int = TOP_K,
) -> tuple[LookupMatch, ...]:
    if attribute not in ATTRIBUTE_OPTIONS:
        raise ValueError(f"unknown semantic attribute: {attribute}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    # The console is a ranking diagnostic, so it reports the nearest values
    # even when a result would not meet the runtime resolution threshold.
    return context.dictionary.semantic_match(
        query,
        allowed_attribute=attribute,
        top_k=top_k,
        min_similarity=-1.0,
    )


def print_results(
    context: SemanticAttributeContext,
    matches: Iterable[LookupMatch],
    *,
    output_fn: Callable[[str], None] = _print,
) -> None:
    rows = list(matches)
    if not rows:
        output_fn("No canonical values were returned.")
        return
    output_fn("#  canonical value                 score    canonical_id                 count")
    for rank, match in enumerate(rows, 1):
        value = context.dictionary.get(match.canonical_id)
        count = value.count if value is not None else 0
        output_fn(
            f"{rank:>2}  {match.value:<30} {match.similarity:>7.4f}    "
            f"{match.canonical_id:<28} {count:>6}"
        )


def _parse_attribute_choice(raw: str) -> str | None:
    normalized = raw.strip().casefold()
    if normalized in ATTRIBUTE_OPTIONS:
        return normalized
    try:
        index = int(normalized)
    except ValueError:
        return None
    if 1 <= index <= len(ATTRIBUTE_OPTIONS):
        return ATTRIBUTE_OPTIONS[index - 1]
    return None


def _choose_attribute(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str | None:
    output_fn("Select semantic attribute:")
    for index, attribute in enumerate(ATTRIBUTE_OPTIONS, 1):
        output_fn(f"{index}. {attribute}")
    while True:
        try:
            raw = input_fn("> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return None
        if raw.strip().casefold() == "q":
            return None
        selected = _parse_attribute_choice(raw)
        if selected is not None:
            return selected
        output_fn("Choose an attribute number/name, or type q to quit.")


def run_console(
    dictionary_path: str | Path = DEFAULT_DICTIONARY,
    *,
    embedding_model: str | None = None,
    top_k: int = TOP_K,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = _print,
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    context = load_search_context(
        dictionary_path,
        embedding_model=embedding_model,
        progress_fn=output_fn,
    )
    output_fn("Type attribute to change selection, or q to quit.")

    while True:
        attribute = _choose_attribute(input_fn, output_fn)
        if attribute is None:
            return
        while True:
            output_fn("Query:")
            try:
                query = input_fn("> ").strip()
            except (EOFError, KeyboardInterrupt):
                output_fn("")
                return
            command = query.casefold()
            if command == "q":
                return
            if command == "attribute":
                break
            if not query:
                output_fn("Enter a sentence, attribute, or q.")
                continue
            try:
                matches = search_attribute(context, attribute, query, top_k=top_k)
            except (RuntimeError, ValueError) as exc:
                output_fn(f"Semantic search failed: {exc}")
                continue
            print_results(context, matches, output_fn=output_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", default=str(DEFAULT_DICTIONARY))
    parser.add_argument("--embedding-model")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        run_console(
            args.dictionary,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
