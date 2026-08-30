from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from starter.bm25 import BM25Index


class BM25ConcurrencyTests(unittest.TestCase):
    def test_search_supports_shared_index_from_worker_threads(self) -> None:
        products = {
            "A": {"title": "Blue running shoe"},
            "B": {"title": "Black winter boot"},
        }
        index = BM25Index(products, products)
        expected = index.search("running shoe")

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(index.search, ["running shoe"] * 8))

        self.assertIn("A", expected)
        self.assertTrue(all(result == expected for result in results))


if __name__ == "__main__":
    unittest.main()
