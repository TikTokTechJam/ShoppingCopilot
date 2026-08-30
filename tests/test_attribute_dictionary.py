from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dictionary.registry import ATTRIBUTE_FIELDS, AttributeDictionary, canonical_id, normalize_text
from scripts.build_attribute_dictionary import build_attribute_dictionary
from scripts.validate_attribute_dictionary import validate_attribute_dictionary
from starter.routing.constraints import extract_constraints


def _record(
    asin: str,
    *,
    brand: list[str],
    category: list[str] | None = None,
    color: list[str] | None = None,
    material: list[str] | None = None,
    style: list[str] | None = None,
    feature: list[str] | None = None,
    use_case: list[str] | None = None,
    status: str = "success",
) -> dict[str, object]:
    return {
        "parent_asin": asin,
        "price": 12.99,
        "facts": {
            "category": category or [],
            "brand": brand,
            "color": color or [],
            "material": material or [],
            "style": style or [],
            "feature": feature or [],
            "use_case": use_case or [],
        },
        "annotation": {"status": status, "prompt_version": "v4"},
    }


class AttributeDictionaryTests(unittest.TestCase):
    def _build(self) -> tuple[Path, Path, dict[str, object]]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "annotations.jsonl"
        source.write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    _record(
                        "A",
                        brand=["New_Balance", "New-Balance", "Air Max", "Air Max 270", "orange", "Levi's"],
                        category=["shoes"],
                        color=["black", "orange"],
                        material=["Stainless_Steel"],
                        style=["High_Waisted", "v-neck", "v neck", "V_Neck"],
                        feature=["Moisture Wicking"],
                        use_case=["Trail_Running"],
                    ),
                    _record(
                        "B",
                        brand=["new balance"],
                        category=["shoes"],
                        color=["red"],
                        style=["high waisted"],
                    ),
                    _record("FAILED", brand=[], status="failed"),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        output = root / "dictionary"
        summary = build_attribute_dictionary(source, output)
        return root, output, summary

    def test_v4_nested_facts_brand_list_and_no_size(self) -> None:
        _, output, summary = self._build()
        self.assertEqual(tuple(ATTRIBUTE_FIELDS), (
            "category", "brand", "color", "material", "style", "feature", "use_case"
        ))
        self.assertEqual(summary["records_read"], 3)
        self.assertEqual(summary["records_used"], 2)
        self.assertEqual(summary["records_skipped"], 1)

        payload = json.loads((output / "canonical_values.json").read_text(encoding="utf-8"))
        records = list(payload["values"].values())
        self.assertNotIn("size", payload["attributes"])
        self.assertTrue(all(record["attribute"] != "size" for record in records))
        self.assertTrue(all(record["attribute"] != "price" for record in records))
        self.assertIn("brand:new_balance", payload["values"])
        self.assertIn("brand:air_max_270", payload["values"])

    def test_v5_aggregate_input_does_not_require_annotation_or_style(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "v5_annotations.jsonl"
        source.write_text(
            json.dumps({
                "parent_asin": "V5-A",
                "price": 49.99,
                "facts": {
                    "category": ["running shoes"],
                    "brand": ["New_Balance", "Air Max 270"],
                    "color": ["black"],
                    "material": ["mesh"],
                    "feature": ["quick drying"],
                    "use_case": ["running"],
                },
            }) + "\n",
            encoding="utf-8",
        )
        output = root / "dictionary"
        summary = build_attribute_dictionary(source, output, input_format="v5")
        self.assertEqual(summary["input_format"], "v5")
        self.assertEqual(summary["records_read"], 1)
        self.assertEqual(summary["records_used"], 1)
        payload = json.loads((output / "canonical_values.json").read_text(encoding="utf-8"))
        self.assertIn("brand:new_balance", payload["values"])
        self.assertIn("brand:air_max_270", payload["values"])
        self.assertEqual(payload["values"]["brand:new_balance"]["count"], 1)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["input_format"], "v5")
        self.assertEqual(summary["canonical_value_count_by_attribute"]["style"], 0)

    def test_natural_values_ids_and_apostrophes(self) -> None:
        _, output, _ = self._build()
        payload = json.loads((output / "canonical_values.json").read_text(encoding="utf-8"))
        values = payload["values"]
        self.assertEqual(values["style:high_waisted"]["value"], "high waisted")
        self.assertEqual(values["style:high_waisted"]["normalized"], "high waisted")
        self.assertEqual(values["style:high_waisted"]["count"], 2)
        self.assertEqual(values["brand:levis"]["value"], "levi's")
        self.assertEqual(values["brand:levis"]["normalized"], "levis")
        self.assertEqual(normalize_text("New_Balance"), "new balance")
        self.assertEqual(normalize_text("New-Balance"), "new balance")
        self.assertEqual(normalize_text("Levi's"), "levis")
        self.assertEqual(normalize_text("O'Neill"), "oneill")
        self.assertEqual(canonical_id("feature", "moisture wicking"), "feature:moisture_wicking")

    def test_same_attribute_variants_are_one_concept_and_counts_are_product_distinct(self) -> None:
        _, output, _ = self._build()
        values = json.loads((output / "canonical_values.json").read_text(encoding="utf-8"))["values"]
        self.assertEqual(
            [record for record in values.values() if record["attribute"] == "style" and record["normalized"] == "v neck"],
            [{
                "canonical_id": "style:v_neck",
                "attribute": "style",
                "value": "v neck",
                "normalized": "v neck",
                "count": 1,
            }],
        )
        self.assertEqual(values["brand:new_balance"]["count"], 2)

    def test_ambiguous_surfaces_are_preserved_and_not_assigned(self) -> None:
        _, output, _ = self._build()
        dictionary = AttributeDictionary.load(output)
        matches = dictionary.exact_match("orange")
        self.assertEqual(
            {match.canonical_id for match in matches},
            {"brand:orange", "color:orange"},
        )
        constraints = extract_constraints("orange", dictionary=dictionary)
        self.assertEqual(constraints.brand, ())
        self.assertEqual(constraints.color, ())
        self.assertIn("orange", constraints.unmapped)

    def test_longest_first_and_word_boundaries(self) -> None:
        _, output, _ = self._build()
        dictionary = AttributeDictionary.load(output)
        constraints = extract_constraints("I want Air Max 270 shoes", dictionary=dictionary)
        self.assertEqual(constraints.brand, ("air max 270",))
        self.assertNotIn("air max", constraints.brand)
        self.assertEqual(extract_constraints("credit", dictionary=dictionary).color, ())

    def test_material_context_wins_over_generic_from_brand_context(self) -> None:
        _, output, _ = self._build()
        dictionary = AttributeDictionary.load(output)

        material = extract_constraints(
            "I'd prefer something made from stainless steel",
            dictionary=dictionary,
        )
        self.assertEqual(material.material, ("stainless steel",))
        self.assertEqual(material.brand, ())

        brand = extract_constraints("something from New Balance", dictionary=dictionary)
        self.assertEqual(brand.brand, ("new balance",))

    def test_exact_only_artifacts_validate_and_are_deterministic(self) -> None:
        root, output_one, summary = self._build()
        output_two = root / "dictionary-two"
        build_attribute_dictionary(root / "annotations.jsonl", output_two)
        self.assertEqual(
            (output_one / "canonical_values.json").read_bytes(),
            (output_two / "canonical_values.json").read_bytes(),
        )
        self.assertEqual(
            (output_one / "normalized_lookup.json").read_bytes(),
            (output_two / "normalized_lookup.json").read_bytes(),
        )
        self.assertFalse((output_one / "attribute_embeddings.npy").exists())
        self.assertFalse((output_one / "embedding_metadata.json").exists())
        validation = validate_attribute_dictionary(output_one)
        self.assertEqual(validation["records_used"], summary["records_used"])
        self.assertFalse(validation["has_embedding_matrix"])
        self.assertEqual(AttributeDictionary.load(output_one).get("style:high_waisted").value, "high waisted")


if __name__ == "__main__":
    unittest.main()
