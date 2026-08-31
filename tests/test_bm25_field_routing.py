from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.bm25 import BM25Index, BM25QueryCompiler
from starter.retrieval import ProductRetriever
from starter.routing.constraints import ShoppingConstraints


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BM25FieldRoutingTests(unittest.TestCase):
    def test_raw_query_uses_normalized_unique_terms_without_stopword_filtering(self) -> None:
        self.assertEqual(
            BM25Index.query_phrases("The thermal_underwear and thermal underwear"),
            ("the", "thermal", "underwear", "and"),
        )

    def test_slot_group_preserves_phrases_without_ngram_expansion(self) -> None:
        group = BM25QueryCompiler().compile_group_specs(
            ShoppingConstraints(
                category=(
                    "Panties",
                    "thermal_underwear",
                    "thermal underwear",
                    "for boxer briefs",
                )
            )
        )["category"]

        self.assertEqual(
            group.match_phrases,
            ("panties", "thermal underwear", "for boxer briefs"),
        )
        self.assertEqual(
            group.query_text,
            '{category categories title}: ("panties" OR "thermal underwear" '
            'OR "for boxer briefs")',
        )

    def test_slot_group_search_uses_compiled_attribute_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            facts_path = root / "facts.jsonl"
            _write_jsonl(
                catalog_path,
                [
                    {
                        "parent_asin": "A",
                        "title": "Plain walking shoes",
                    },
                    {
                        "parent_asin": "B",
                        "title": "Everyday shoes",
                        "store": "Waterproof collection",
                    },
                ],
            )
            _write_jsonl(
                facts_path,
                [
                    {
                        "parent_asin": "A",
                        "facts": {"feature": ["waterproof"]},
                        "annotation": {"status": "success"},
                    },
                    {
                        "parent_asin": "B",
                        "facts": {"feature": []},
                        "annotation": {"status": "success"},
                    },
                ],
            )
            retriever = ProductRetriever(
                catalog_path,
                facts_path=facts_path,
                embeddings_path=root / "missing.npy",
                metadata_path=root / "missing.json",
            )

            scores = retriever._bm25_scores(
                "",
                {"A", "B"},
                ShoppingConstraints(feature=("waterproof",)),
            )

            self.assertIn("A", scores)
            self.assertNotIn("B", scores)


if __name__ == "__main__":
    unittest.main()
