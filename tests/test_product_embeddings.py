from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from product_embeddings import (
    ProductEmbeddingIndex,
    build_product_embeddings,
    build_product_text,
    build_tier4_raw_text,
    build_tier4_record,
)
from starter.retrieval import ProductRetriever

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without optional deps
    np = None  # type: ignore[assignment]


class KeywordEmbedder:
    model_id = "test-keyword-embedder"

    def embed_documents(self, texts: list[str]) -> Any:
        return [
            [1.0, 0.0] if "blue" in text.casefold() else [0.0, 1.0]
            for text in texts
        ]


class ProductEmbeddingTests(unittest.TestCase):
    def test_tier4_record_uses_the_requested_source_shape(self) -> None:
        product = {
            "parent_asin": "B123",
            "title": " Blue hiking boots ",
            "features": ["Vibram   outsole", "Reinforced toe box"],
            "description": ["Designed   for mountain trails"],
            "details": {"closure": "lace-up"},
            "price": 99.0,
            "categories": ["Shoes"],
        }
        self.assertEqual(
            build_tier4_record(product),
            {
                "parent_asin": "B123",
                "title": "Blue hiking boots",
                "features": ["Vibram outsole", "Reinforced toe box"],
                "description": ["Designed for mountain trails"],
                "details": {"closure": "lace-up"},
            },
        )

    def test_product_text_joins_test_json_facts_and_tier4_source(self) -> None:
        product = {
            "parent_asin": "B123",
            "title": "Blue hiking boots",
            "features": ["Vibram outsole", "Reinforced toe box"],
            "description": ["Designed for mountain trails"],
            "details": {"closure": "lace-up"},
        }
        facts = {
            "parent_asin": "B123",
            "price": None,
            "facts": {
                "category": ["boots"],
                "brand": ["trail maker"],
                "color": ["blue"],
                "material": ["leather"],
                "style": ["hiking"],
                "feature": ["waterproof"],
                "use_case": ["hiking"],
            },
        }
        text = build_product_text(product, facts)
        self.assertIn("Title: Blue hiking boots", text)
        self.assertIn("Brand: trail maker", text)
        self.assertIn("Features: waterproof", text)
        self.assertIn("Description: Designed for mountain trails", text)
        self.assertIn("Raw features: Vibram outsole; Reinforced toe box", text)
        self.assertIn("Details: closure: lace-up", text)
        self.assertNotIn("99.0", text)

    @unittest.skipUnless(np is not None, "NumPy is required for embedding artifacts")
    def test_tier4_artifact_and_product_embedding_artifacts(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog_records = [
                {
                    "parent_asin": "B123",
                    "title": "Blue hiking boots",
                    "features": ["Vibram outsole"],
                    "description": ["Mountain trail boots"],
                    "details": {"closure": "lace-up"},
                },
                {
                    "parent_asin": "B456",
                    "title": "Red city sneakers",
                    "features": ["Rubber sole"],
                    "description": ["Everyday walking shoes"],
                    "details": {"closure": "slip-on"},
                },
            ]
            catalog.write_text(
                "\n".join(json.dumps(record) for record in catalog_records) + "\n",
                encoding="utf-8",
            )
            raw_path = root / "tier4" / "raw_text.jsonl"
            summary = build_tier4_raw_text(catalog, raw_path)
            self.assertEqual(summary["product_count"], 2)
            raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                set(json.loads(raw_lines[0])),
                {"parent_asin", "title", "features", "description", "details"},
            )

            facts_path = root / "facts.jsonl"
            facts = [
                {
                    "parent_asin": "B123",
                    "price": None,
                    "facts": {
                        "category": ["boots"],
                        "brand": ["trail maker"],
                        "color": ["blue"],
                        "material": ["leather"],
                        "style": [],
                        "feature": ["waterproof"],
                        "use_case": ["hiking"],
                    },
                },
                {
                    "parent_asin": "B456",
                    "price": None,
                    "facts": {
                        "category": ["sneakers"],
                        "brand": ["city maker"],
                        "color": ["red"],
                        "material": ["mesh"],
                        "style": [],
                        "feature": [],
                        "use_case": ["walking"],
                    },
                },
            ]
            facts_path.write_text(
                "\n".join(json.dumps(record) for record in facts) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "embeddings"
            manifest = build_product_embeddings(
                catalog,
                facts_path,
                output_dir,
                KeywordEmbedder(),
                raw_text_path=raw_path,
                batch_size=1,
                generated_at_utc="2026-01-01T00:00:00Z",
            )
            self.assertEqual(manifest["product_count"], 2)
            self.assertEqual(manifest["tier4_version"], "tier4-raw-text-v1")

            matrix = np.load(output_dir / "product_embeddings.npy", allow_pickle=False)
            self.assertEqual(matrix.dtype, np.dtype(np.float32))
            self.assertEqual(matrix.shape, (2, 2))
            self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), 1.0))

            index = ProductEmbeddingIndex.load(
                output_dir, expected_asins=["B123", "B456"]
            )
            matches = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
            self.assertEqual([match.parent_asin for match in matches], ["B123", "B456"])
            self.assertAlmostEqual(matches[0].score, 1.0)

            retriever = ProductRetriever(
                catalog,
                facts_path=facts_path,
                embeddings_path=output_dir / "product_embeddings.npy",
                metadata_path=output_dir / "product_embedding_metadata.json",
                query_encoder=KeywordEmbedder(),
            )
            candidates = retriever.retrieve(
                "BROWSING", "blue hiking boots", {}, limit=2
            )
            self.assertEqual([item.parent_asin for item in candidates], ["B123", "B456"])


if __name__ == "__main__":
    unittest.main()
