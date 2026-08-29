from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dictionary.registry import LookupMatch
from scripts.console_semantic_attribute_test import (
    SEMANTIC_ATTRIBUTES,
    SemanticSearchContext,
    _parse_attribute_choice,
    print_results,
    search_attribute,
)


class FakeDictionary:
    has_semantic_embeddings = True

    def __init__(self) -> None:
        self.encoder = None
        self.calls: list[dict[str, object]] = []

    def set_query_encoder(self, encoder: object) -> None:
        self.encoder = encoder

    def semantic_match(self, query: str, **kwargs: object):
        self.calls.append({"query": query, **kwargs})
        return (
            LookupMatch(
                canonical_id="feature:non_slip",
                attribute="feature",
                value="non slip",
                raw_text=query,
                normalized_text=query,
                match_method="semantic",
                similarity=0.87,
            ),
        )

    def get(self, canonical_id: str):
        if canonical_id == "feature:non_slip":
            return SimpleNamespace(normalized="non slip")
        return None


class FakeEncoder:
    model_id = "models/bge-small-en-v1.5"
    embedding_dimension = 384


class ConsoleSemanticAttributeTests(unittest.TestCase):
    def test_attribute_choices_use_only_embedded_attributes(self) -> None:
        self.assertEqual(_parse_attribute_choice("1"), SEMANTIC_ATTRIBUTES[0])
        self.assertEqual(_parse_attribute_choice("feature"), "feature")
        self.assertIsNone(_parse_attribute_choice("brand"))
        self.assertIsNone(_parse_attribute_choice("description"))

    def test_search_uses_selected_bge_attribute_matrix(self) -> None:
        dictionary = FakeDictionary()
        context = SemanticSearchContext(dictionary, {}, FakeEncoder.model_id, 384)

        matches = search_attribute(
            context,
            "feature",
            "something that won't slip",
            top_k=10,
        )

        self.assertEqual(matches[0].canonical_id, "feature:non_slip")
        self.assertEqual(dictionary.calls, [{
            "query": "something that won't slip",
            "allowed_attribute": "feature",
            "top_k": 10,
            "min_similarity": -1.0,
        }])

    def test_results_show_canonical_id_similarity_and_examples(self) -> None:
        dictionary = FakeDictionary()
        context = SemanticSearchContext(
            dictionary,
            {("feature", "non slip"): ("A1", "A2")},
            FakeEncoder.model_id,
            384,
        )
        output: list[str] = []
        print_results(
            context,
            search_attribute(context, "feature", "won't slip"),
            output_fn=output.append,
        )

        self.assertIn("feature:non_slip", output[-1])
        self.assertIn("0.8700", output[-1])
        self.assertIn("A1, A2", output[-1])


if __name__ == "__main__":
    unittest.main()
