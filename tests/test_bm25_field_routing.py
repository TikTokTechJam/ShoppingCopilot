from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.retrieval import ProductRetriever
from starter.routing.constraints import ShoppingConstraints


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BM25FieldRoutingTests(unittest.TestCase):
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
