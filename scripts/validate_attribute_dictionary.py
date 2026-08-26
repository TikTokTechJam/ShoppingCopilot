from __future__ import annotations

import argparse
import json
from pathlib import Path

from dictionary.registry import ATTRIBUTE_FIELDS, AttributeDictionary, NORMALIZATION_VERSION


def validate_attribute_dictionary(directory: str | Path) -> dict[str, int | bool]:
    root = Path(directory)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("manifest normalization version does not match runtime")
    if manifest.get("schema_version") != "canonical-attribute-dictionary/v1":
        raise ValueError("unsupported dictionary schema version")

    dictionary = AttributeDictionary.load(root)
    if tuple(manifest.get("semantic_attributes", ())) != tuple(
        attribute for attribute in manifest.get("semantic_attributes", ())
    ):
        raise ValueError("manifest semantic_attributes must be deterministic")
    if set(ATTRIBUTE_FIELDS) != set(
        json.loads((root / "normalized_lookup.json").read_text(encoding="utf-8"))[
            "attributes"
        ]
    ):
        raise ValueError("normalized lookup must contain every canonical attribute")

    expected_values = int(manifest["canonical_value_count"])
    expected_embeddings = int(manifest["embedded_value_count"])
    if len(dictionary.values) != expected_values:
        raise ValueError("manifest canonical value count does not match registry")
    if len(dictionary._embedding_rows) != expected_embeddings:  # noqa: SLF001
        raise ValueError("manifest embedding count does not match metadata")
    return {
        "canonical_value_count": len(dictionary.values),
        "embedded_value_count": len(dictionary._embedding_rows),  # noqa: SLF001
        "has_embedding_matrix": dictionary._embeddings is not None,  # noqa: SLF001
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Issue #8 dictionary artifacts.")
    parser.add_argument("--directory", default="data/derived/dictionary")
    args = parser.parse_args()
    print(json.dumps(validate_attribute_dictionary(args.directory), indent=2))


if __name__ == "__main__":
    main()
