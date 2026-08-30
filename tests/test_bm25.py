from __future__ import annotations

import unittest

from starter.bm25 import BM25FIndex, BM25Index


class BM25FIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.products = {
            "A": {
                "title": "waterproof hiking boots",
                "categories": ["shoes", "boots"],
                "features": ["waterproof"],
                "description": "boots for mountain trails",
            },
            "B": {
                "title": "casual shoes",
                "categories": ["shoes"],
                "description": "waterproof shoes for rain",
            },
            "C": {
                "title": "gold earrings",
                "categories": ["jewelry"],
                "description": "small hoop earrings",
            },
        }
        self.index = BM25FIndex(self.products, ("A", "B", "C"))

    def test_title_field_boosts_a_matching_product(self) -> None:
        scores = self.index.search("waterproof")

        self.assertGreater(scores["A"], scores["B"])
        self.assertNotIn("C", scores)

    def test_allowed_asins_are_respected(self) -> None:
        scores = self.index.search("waterproof", allowed_asins={"B"})

        self.assertEqual(set(scores), {"B"})

    def test_existing_bm25_name_is_compatible_alias(self) -> None:
        self.assertIs(BM25Index, BM25FIndex)


if __name__ == "__main__":
    unittest.main()
