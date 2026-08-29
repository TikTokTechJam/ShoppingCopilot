"""Interactively inspect BGE semantic matches for V5 canonical attributes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from dictionary.registry import ATTRIBUTE_FIELDS, AttributeDictionary, LookupMatch, normalize_text
from dictionary.semantic import (
    ATTRIBUTE_EMBEDDING_DIMENSION,
    ATTRIBUTE_EMBEDDING_MODEL,
    ATTRIBUTE_MODEL_ENV,
    load_bge_attribute_encoder,
)


DEFAULT_DICTIONARY_DIR = Path("data/derived/annotations/v5/dictionary")
DEFAULT_ANNOTATIONS = Path("data/derived/annotations/v5/annotations.jsonl")
TOP_K = 10
SEMANTIC_ATTRIBUTES = tuple(
    attribute for attribute in ATTRIBUTE_FIELDS if attribute != "brand"
)
ATTRIBUTE_OPTIONS = tuple((attribute, attribute) for attribute in SEMANTIC_ATTRIBUTES)


class SemanticSearchContext:
    def __init__(
        self,
        dictionary: AttributeDictionary,
        examples: Mapping[tuple[str, str], tuple[str, ...]],
        model_id: str,
        dimension: int,
    ) -> None:
        self.dictionary = dictionary
        self.examples = examples
        self.model_id = model_id
        self.dimension = dimension


def _print(message: str) -> None:
    print(message, flush=True)


def _read_examples(path: str | Path, progress_fn: Callable[[str], None]) -> dict[tuple[str, str], tuple[str, ...]]:
    examples: dict[tuple[str, str], list[str]] = {}
    source = Path(path)
    try:
        handle = source.open(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"unable to read V5 annotations: {source}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid annotations JSON at {source}:{line_number}") from exc
            if not isinstance(record, Mapping):
                continue
            asin = str(record.get("parent_asin", "")).strip()
            facts = record.get("facts")
            if not asin or not isinstance(facts, Mapping):
                continue
            for attribute in SEMANTIC_ATTRIBUTES:
                values = facts.get(attribute, ())
                if not isinstance(values, (list, tuple)):
                    values = (values,)
                for value in values:
                    surface = normalize_text(str(value))
                    if surface and asin not in examples.setdefault((attribute, surface), []):
                        examples[(attribute, surface)].append(asin)
    progress_fn(f"Annotation examples loaded: {sum(len(values) for values in examples.values()):,} value memberships")
    return {key: tuple(values[:3]) for key, values in examples.items()}


def load_search_context(
    dictionary_dir: str | Path = DEFAULT_DICTIONARY_DIR,
    annotations_path: str | Path = DEFAULT_ANNOTATIONS,
    *,
    embedding_model: str | None = None,
    progress_fn: Callable[[str], None] = _print,
) -> SemanticSearchContext:
    progress_fn(f"Loading V5 canonical dictionary: {dictionary_dir}")
    try:
        dictionary = AttributeDictionary.load(dictionary_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to load V5 dictionary from {dictionary_dir}: {exc}") from exc
    if not dictionary.has_semantic_embeddings:
        raise RuntimeError("BGE attribute embeddings are unavailable in the V5 dictionary")

    configured = (embedding_model or os.environ.get(ATTRIBUTE_MODEL_ENV, "")).strip() or None
    progress_fn(
        f"Loading local BGE attribute encoder: {configured or ATTRIBUTE_EMBEDDING_MODEL}"
    )
    try:
        encoder = load_bge_attribute_encoder(configured)
        dictionary.set_query_encoder(encoder)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"unable to load local BGE attribute encoder: {exc}") from exc

    actual_model = str(getattr(encoder, "model_id", configured or ATTRIBUTE_EMBEDDING_MODEL))
    dimension = int(getattr(encoder, "embedding_dimension", 0) or 0)
    if dimension != ATTRIBUTE_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"BGE attribute encoder dimension is {dimension}; "
            f"expected {ATTRIBUTE_EMBEDDING_DIMENSION}"
        )
    progress_fn(
        f"BGE attribute encoder loaded: model={actual_model}, dimension={dimension}"
    )
    examples = _read_examples(annotations_path, progress_fn)
    return SemanticSearchContext(dictionary, examples, actual_model, dimension)


def search_attribute(
    context: SemanticSearchContext,
    attribute: str,
    query: str,
    *,
    top_k: int = TOP_K,
) -> tuple[LookupMatch, ...]:
    if attribute not in SEMANTIC_ATTRIBUTES:
        raise ValueError(f"attribute has no semantic embedding: {attribute}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    return context.dictionary.semantic_match(
        query,
        allowed_attribute=attribute,
        top_k=top_k,
        min_similarity=-1.0,
    )


def print_results(
    context: SemanticSearchContext,
    matches: Iterable[LookupMatch],
    *,
    output_fn: Callable[[str], None] = _print,
) -> None:
    rows = list(matches)
    if not rows:
        output_fn("No canonical values were returned.")
        return
    output_fn("#  canonical value                 similarity    canonical_id                 product examples")
    for rank, match in enumerate(rows, 1):
        canonical = context.dictionary.get(match.canonical_id)
        surface = canonical.normalized if canonical is not None else match.normalized_text
        examples = ", ".join(context.examples.get((match.attribute, surface), ())) or "-"
        output_fn(
            f"{rank:>2}  {match.value:<32} {match.similarity:>10.4f}    "
            f"{match.canonical_id:<28} {examples}"
        )


def _parse_attribute_choice(raw: str) -> str | None:
    normalized = raw.strip().casefold()
    if normalized in SEMANTIC_ATTRIBUTES:
        return normalized
    try:
        index = int(normalized)
    except ValueError:
        return None
    if 1 <= index <= len(ATTRIBUTE_OPTIONS):
        return ATTRIBUTE_OPTIONS[index - 1][1]
    return None


def _choose_attribute(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str | None:
    output_fn("Select semantic attribute:")
    for index, (label, _attribute) in enumerate(ATTRIBUTE_OPTIONS, 1):
        output_fn(f"{index}. {label}")
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
        output_fn(f"Choose 1-{len(ATTRIBUTE_OPTIONS)} or an attribute name, or type q to quit.")


def run_console(
    dictionary_dir: str | Path = DEFAULT_DICTIONARY_DIR,
    annotations_path: str | Path = DEFAULT_ANNOTATIONS,
    *,
    embedding_model: str | None = None,
    top_k: int = TOP_K,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = _print,
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    context = load_search_context(
        dictionary_dir,
        annotations_path,
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
    parser.add_argument("--dictionary-dir", default=str(DEFAULT_DICTIONARY_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--embedding-model")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        run_console(
            args.dictionary_dir,
            args.annotations,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
