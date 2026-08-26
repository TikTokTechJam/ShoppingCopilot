from __future__ import annotations

import unittest

from interactive_demo import ProductView, build_profile, format_product, parse_profile_tags


class InteractiveDemoTest(unittest.TestCase):
    def test_profile_tags_are_normalized_and_deduplicated(self) -> None:
        self.assertEqual(
            parse_profile_tags(" Comfort,fit, comfort, DURABILITY "),
            ["comfort", "fit", "durability"],
        )

    def test_empty_profile_has_clear_summary(self) -> None:
        profile = build_profile([])
        self.assertEqual(profile["preference_tags"], [])
        self.assertIn("no saved preference tags", profile["summary"])

    def test_product_formatter_contains_purchase_decision_fields(self) -> None:
        product = ProductView(
            parent_asin="ABC123",
            title="Example shoe",
            price="$49.00",
            rating="4.6/5",
            rating_number=1234,
            categories="Women > Shoes",
            store="Example Store",
        )
        rendered = format_product(2, product)
        self.assertIn("2. Example shoe", rendered)
        self.assertIn("ABC123", rendered)
        self.assertIn("$49.00", rendered)
        self.assertIn("1,234 ratings", rendered)


if __name__ == "__main__":
    unittest.main()
