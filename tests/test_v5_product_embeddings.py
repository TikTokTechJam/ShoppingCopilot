from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from product_embeddings.v5 import (
    V5_PRODUCT_MODEL,
    V5ProductEmbeddingIndex,
    build_v5_product_embeddings,
)
from starter.retrieval import ProductRetriever
from starter.routing.constraints import ShoppingConstraints


class FakeQwenEncoder:
    model_id = V5_PRODUCT_MODEL
    embedding_dimension = 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "cosplay" in lowered or "jumpsuit" in lowered:
            return [0.0, 1.0, 0.0]
        if "boots" in lowered:
            return [1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def fixture(root: Path) -> tuple[Path, Path]:
    catalog = root / "catalog.jsonl"
    annotations = root / "annotations.jsonl"
    write_jsonl(
        catalog,
        [
            {"parent_asin": "A", "title": "Cosplay jumpsuit", "categories": ["clothing"]},
            {"parent_asin": "B", "title": "Hiking boots", "categories": ["shoes"]},
            {"parent_asin": "C", "title": "Plain shirt", "categories": ["clothing"]},
        ],
    )
    write_jsonl(
        annotations,
        [
            {
                "parent_asin": "A",
                "facts": {
                    "category": ["jumpsuit"],
                    "brand": [],
                    "color": [],
                    "material": ["polyester"],
                    "style": ["costume"],
                    "feature": ["zipper closure"],
                    "use_case": ["cosplay"],
                },
            },
            {
                "parent_asin": "B",
                "facts": {
                    "category": ["boots"],
                    "brand": [],
                    "color": [],
                    "material": ["leather"],
                    "feature": [],
                    "use_case": ["hiking"],
                },
            },
            {"parent_asin": "C", "facts": {"category": ["shirt"]}},
        ],
    )
    return catalog, annotations


class V5ProductEmbeddingTests(unittest.TestCase):
    def test_build_and_load_preserves_catalog_order_and_normalizes_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, annotations = fixture(root)
            output = root / "product_embeddings_v5"
            manifest = build_v5_product_embeddings(
                catalog,
                annotations,
                output,
                FakeQwenEncoder(),
                batch_size=2,
            )

            self.assertEqual(manifest["embedding_model"], V5_PRODUCT_MODEL)
            self.assertEqual(manifest["product_count"], 3)
            self.assertEqual(manifest["dimension"], 3)
            self.assertIn("style", manifest["product_card_fields"])
            card_text = (output / "product_cards.jsonl").read_text(encoding="utf-8")
            self.assertIn("style: costume", card_text)
            index = V5ProductEmbeddingIndex.load(
                output,
                expected_asins=("A", "B", "C"),
            )
            self.assertEqual(index.asins, ("A", "B", "C"))
            self.assertEqual(index.search([0.0, 1.0, 0.0], 1)[0].parent_asin, "A")

    def test_product_dense_path_uses_matching_encoder_and_full_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, annotations = fixture(root)
            output = root / "product_embeddings_v5"
            encoder = FakeQwenEncoder()
            build_v5_product_embeddings(catalog, annotations, output, encoder)
            retriever = ProductRetriever(
                catalog,
                facts_path=annotations,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
                product_query_encoder=encoder,
                product_embedding_artifact_dir=output,
            )

            self.assertTrue(retriever.product_dense_available)
            ranked = retriever.retrieve(
                "BROWSING",
                "jumpsuits for cosplay",
                ShoppingConstraints(),
                limit=2,
            )
            self.assertEqual(ranked[0].parent_asin, "A")
            self.assertGreater(ranked[0].dense_score, 0.0)
            self.assertEqual(ranked[0].score, ranked[0].ranking_score)

    def test_model_and_dimension_mismatches_disable_product_dense(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, annotations = fixture(root)
            output = root / "product_embeddings_v5"
            build_v5_product_embeddings(catalog, annotations, output, FakeQwenEncoder())

            class WrongModel(FakeQwenEncoder):
                model_id = "some-other-model"

            wrong_model = ProductRetriever(
                catalog,
                facts_path=annotations,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
                product_query_encoder=WrongModel(),
                product_embedding_artifact_dir=output,
            )
            self.assertFalse(wrong_model.product_dense_available)
            self.assertIn("model", wrong_model.product_embedding_compatibility_error or "")

            class WrongDimension(FakeQwenEncoder):
                embedding_dimension = 2

            wrong_dimension = ProductRetriever(
                catalog,
                facts_path=annotations,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
                product_query_encoder=WrongDimension(),
                product_embedding_artifact_dir=output,
            )
            self.assertFalse(wrong_dimension.product_dense_available)
            self.assertIn(
                "dimension",
                wrong_dimension.product_embedding_compatibility_error or "",
            )


if __name__ == "__main__":
    unittest.main()
