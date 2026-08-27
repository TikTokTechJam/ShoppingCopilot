from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from product_embeddings import (
    LAYER2_VIEWS,
    Layer2EmbeddingIndex,
    build_layer2_embeddings,
    build_layer2_view_text,
)
from starter.retrieval import ProductRetriever

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without optional deps
    np = None  # type: ignore[assignment]


class ViewKeywordEmbedder:
    model_id = "test-layer2-embedder"

    def embed_documents(self, texts: list[str]) -> Any:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [1.0, 0.0]
                if any(word in lowered for word in ("blue", "boot", "waterproof"))
                else [0.0, 1.0]
            )
        return vectors


@unittest.skipUnless(np is not None, "NumPy is required for Layer 2 artifacts")
class Layer2EmbeddingTests(unittest.TestCase):
    def _catalog(self, root: Path) -> Path:
        path = root / "catalog.jsonl"
        records = [
            {
                "parent_asin": "A1",
                "title": "Blue hiking boots",
                "features": ["Waterproof outsole", "Reinforced toe"],
                "description": ["Boots for mountain trails"],
                "categories": ["Clothing, Shoes & Jewelry", "Boots"],
                "details": {"Closure Type": "lace-up"},
            },
            {
                "parent_asin": "A2",
                "title": "Red city sneakers",
                "features": ["Lightweight sole"],
                "description": [],
                "categories": ["Clothing, Shoes & Jewelry", "Sneakers"],
                "details": {},
            },
        ]
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_view_text_is_catalog_only_and_preserves_field_order(self) -> None:
        product = {
            "title": "  Blue   boots ",
            "categories": ["Shoes", " Boots "],
            "features": ["Waterproof outsole", "Reinforced toe"],
            "description": ["Made for trails", "Use in rain"],
        }
        self.assertEqual(build_layer2_view_text(product, "categories"), "Shoes Boots")
        self.assertEqual(build_layer2_view_text(product, "title"), "Blue boots")
        self.assertEqual(
            build_layer2_view_text(product, "features"),
            "Waterproof outsole Reinforced toe",
        )
        self.assertEqual(
            build_layer2_view_text(product, "description"),
            "Made for trails Use in rain",
        )

    def test_builds_four_aligned_matrices_without_facts(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root)
            output = root / "product_embeddings"
            manifest = build_layer2_embeddings(
                catalog,
                output,
                ViewKeywordEmbedder(),
                batch_size=1,
                catalog_version="catalog-test-v1",
                generated_at_utc="2026-01-01T00:00:00Z",
            )

            self.assertEqual(manifest["views"], list(LAYER2_VIEWS))
            self.assertEqual(manifest["product_count"], 2)
            self.assertEqual(manifest["source_catalog_version"], "catalog-test-v1")
            metadata = json.loads(
                (output / "product_embedding_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual([row["parent_asin"] for row in metadata], ["A1", "A2"])
            self.assertTrue(metadata[0]["has_description"])
            self.assertFalse(metadata[1]["has_description"])

            for view in LAYER2_VIEWS:
                filename = {
                    "categories": "category_embeddings.npy",
                    "title": "title_embeddings.npy",
                    "features": "features_embeddings.npy",
                    "description": "description_embeddings.npy",
                }[view]
                matrix = np.load(output / filename, allow_pickle=False)
                self.assertEqual(matrix.shape, (2, 2))
                self.assertEqual(matrix.dtype, np.dtype(np.float32))
                present = np.array([row[f"has_{view}"] for row in metadata])
                self.assertTrue(np.allclose(np.linalg.norm(matrix[present], axis=1), 1.0))
                self.assertTrue(np.allclose(matrix[~present], 0.0))

            index = Layer2EmbeddingIndex.load(output, expected_asins=["A1", "A2"])
            matches = index.search(
                np.array([1.0, 0.0], dtype=np.float32),
                top_k=2,
                weights={view: 1.0 for view in LAYER2_VIEWS},
            )
            self.assertEqual([match.parent_asin for match in matches], ["A1", "A2"])
            self.assertEqual(matches[1].view_scores["description"], None)

    def test_missing_views_do_not_lower_score_as_negative_evidence(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root)
            output = root / "product_embeddings"
            build_layer2_embeddings(
                catalog,
                output,
                ViewKeywordEmbedder(),
                generated_at_utc="2026-01-01T00:00:00Z",
            )
            index = Layer2EmbeddingIndex.load(output)
            matches = index.search(np.array([0.0, 1.0], dtype=np.float32), top_k=2)
            self.assertEqual(matches[0].parent_asin, "A2")
            self.assertAlmostEqual(matches[0].score, 1.0)

    def test_product_retriever_uses_layer2_when_artifact_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = self._catalog(root)
            output = root / "product_embeddings"
            build_layer2_embeddings(
                catalog,
                output,
                ViewKeywordEmbedder(),
                generated_at_utc="2026-01-01T00:00:00Z",
            )
            retriever = ProductRetriever(
                catalog,
                query_encoder=ViewKeywordEmbedder(),
                layer2_artifact_dir=output,
            )
            self.assertTrue(retriever.has_dense_index)
            candidates = retriever.retrieve("BROWSING", "blue boots", {}, limit=2)
            self.assertEqual([item.parent_asin for item in candidates], ["A1", "A2"])
            self.assertGreater(candidates[0].dense_score, candidates[1].dense_score)


if __name__ == "__main__":
    unittest.main()
