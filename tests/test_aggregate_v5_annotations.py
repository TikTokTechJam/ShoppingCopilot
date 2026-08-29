from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_v5_annotations import (
    V5_ATTRIBUTES,
    aggregate_v5_annotations,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class AggregateV5AnnotationTests(unittest.TestCase):
    def test_joins_in_catalog_order_and_uses_catalog_price(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            input_dir = root / "v5"
            input_dir.mkdir()
            output = root / "annotations.jsonl"
            _write_jsonl(
                catalog,
                [
                    {"parent_asin": "B", "price": "12.99"},
                    {"parent_asin": "A", "price": "-"},
                ],
            )
            for attribute in V5_ATTRIBUTES:
                values = [
                    {"parent_asin": "A", attribute: []},
                    {"parent_asin": "B", attribute: ["New_Balance", "new balance"]},
                ]
                _write_jsonl(input_dir / f"{attribute}.jsonl", values)

            summary = aggregate_v5_annotations(catalog, input_dir, output)
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary["catalog_product_count"], 2)
            self.assertEqual(summary["output_product_count"], 2)
            self.assertEqual(
                [record["parent_asin"] for record in records],
                ["B", "A"],
            )
            self.assertEqual(records[0]["price"], 12.99)
            self.assertIsNone(records[1]["price"])
            self.assertEqual(records[0]["facts"]["brand"], ["new balance"])
            self.assertEqual(set(records[0]["facts"]), set(V5_ATTRIBUTES))

    def test_duplicate_and_external_asins_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            input_dir = root / "v5"
            input_dir.mkdir()
            _write_jsonl(catalog, [{"parent_asin": "A", "price": 1}])
            for attribute in V5_ATTRIBUTES:
                _write_jsonl(
                    input_dir / f"{attribute}.jsonl",
                    [{"parent_asin": "A", attribute: []}],
                )

            _write_jsonl(
                input_dir / "brand.jsonl",
                [
                    {"parent_asin": "A", "brand": []},
                    {"parent_asin": "A", "brand": []},
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate parent_asin"):
                aggregate_v5_annotations(catalog, input_dir, root / "out.jsonl")

            _write_jsonl(
                input_dir / "brand.jsonl",
                [{"parent_asin": "X", "brand": []}],
            )
            with self.assertRaisesRegex(ValueError, "absent from catalog"):
                aggregate_v5_annotations(catalog, input_dir, root / "out.jsonl")


if __name__ == "__main__":
    unittest.main()
