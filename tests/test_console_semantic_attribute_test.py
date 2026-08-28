from __future__ import annotations

import unittest
from types import SimpleNamespace

from product_embeddings.layer2 import Layer2EmbeddingMatch
from scripts.console_semantic_attribute_test import (
    SemanticSearchContext,
    _parse_view_choice,
    print_results,
    search_view,
    view_weights,
)


class FakeLayer2Index:
    dimension = 768

    def __init__(self) -> None:
        self.query_embedding = None
        self.top_k = None
        self.weights = None

    def search(self, query_embedding, *, top_k, weights):
        self.query_embedding = query_embedding
        self.top_k = top_k
        self.weights = weights
        return []


class FakeEncoder:
    def embed_query(self, query):
        assert query == "something that won't slip in the rain"
        return [1.0] * 768


class ConsoleSemanticAttributeTests(unittest.TestCase):
    def test_view_choices_use_only_current_layer2_views(self) -> None:
        self.assertEqual(_parse_view_choice("1"), "categories")
        self.assertEqual(_parse_view_choice("category"), "categories")
        self.assertEqual(_parse_view_choice("feature"), "features")
        self.assertEqual(_parse_view_choice("features"), "features")
        self.assertEqual(_parse_view_choice("description"), "description")
        self.assertIsNone(_parse_view_choice("brand"))

    def test_view_weights_activate_only_selected_matrix(self) -> None:
        self.assertEqual(
            view_weights("features"),
            {
                "categories": 0.0,
                "title": 0.0,
                "features": 1.0,
                "description": 0.0,
            },
        )

    def test_search_uses_query_encoder_and_selected_view_only(self) -> None:
        index = FakeLayer2Index()
        context = SemanticSearchContext(
            index=index,
            query_encoder=FakeEncoder(),
            products={},
            model_id="jinaai/jina-embeddings-v5-text-nano",
            dimension=768,
        )

        search_view(
            context,
            "features",
            "something that won't slip in the rain",
            top_k=10,
        )

        self.assertEqual(index.top_k, 10)
        self.assertEqual(index.query_embedding, [1.0] * 768)
        self.assertEqual(index.weights, view_weights("features"))

    def test_results_show_asin_score_and_selected_view_text(self) -> None:
        product = {"features": ["non slip", "waterproof"]}
        context = SemanticSearchContext(
            index=SimpleNamespace(),
            query_encoder=SimpleNamespace(),
            products={"A1": product},
            model_id="model",
            dimension=768,
        )
        match = Layer2EmbeddingMatch(
            row=0,
            parent_asin="A1",
            score=0.87,
            view_scores={"features": 0.87},
        )
        output: list[str] = []

        print_results(context, "features", [match], output_fn=output.append)

        self.assertIn("A1", output[-1])
        self.assertIn("0.8700", output[-1])
        self.assertIn("non slip waterproof", output[-1])


if __name__ == "__main__":
    unittest.main()
