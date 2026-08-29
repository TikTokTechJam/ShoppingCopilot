from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dictionary.registry import AttributeDictionary
from dictionary.semantic import (
    ATTRIBUTE_EMBEDDING_DIMENSION,
    ATTRIBUTE_EMBEDDING_MODEL,
    load_bge_attribute_encoder,
)
from scripts.build_attribute_dictionary import build_attribute_dictionary
from scripts.console_canonical_attribute_semantic_test import (
    _parse_attribute_choice,
    print_results,
    search_attribute,
)
from scripts.validate_attribute_dictionary import validate_attribute_dictionary


def _record(asin: str, feature: str) -> dict[str, object]:
    return {
        "parent_asin": asin,
        "facts": {
            "category": ["shoes"],
            "brand": ["brand"],
            "color": ["black"],
            "material": ["mesh"],
            "style": ["casual"],
            "feature": [feature],
            "use_case": ["walking"],
        },
        "annotation": {"status": "success", "prompt_version": "v4"},
    }


class FakeBgeEncoder:
    model_id = "models/bge-small-en-v1.5"
    embedding_dimension = ATTRIBUTE_EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.document_texts: list[str] = []
        self.query_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.document_texts.extend(texts)
        return np.asarray(
            [[float(index + 1), 1.0] + [0.0] * 382 for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        self.query_texts.append(text)
        return np.asarray([[1.0, 1.0] + [0.0] * 382], dtype=np.float32)


class BgeAttributeSemanticTests(unittest.TestCase):
    def test_loader_is_local_only_and_has_no_prefix(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        encoder = FakeBgeEncoder()

        def loader(path: str, **kwargs: object) -> FakeBgeEncoder:
            calls.append((path, kwargs))
            return encoder

        with patch("dictionary.semantic.load_local_sentence_transformer", loader):
            loaded = load_bge_attribute_encoder("models/bge-small-en-v1.5")

        self.assertIs(loaded, encoder)
        self.assertEqual(calls, [(
            "models/bge-small-en-v1.5",
            {
                "task": None,
                "document_prompt_name": None,
                "query_prompt_name": None,
                "trust_remote_code": False,
            },
        )])

    def test_loader_rejects_wrong_dimension(self) -> None:
        wrong = SimpleNamespace(
            model_id="models/bge-small-en-v1.5",
            embedding_dimension=385,
            embed_query=lambda _text: [1.0],
            embed_documents=lambda _texts: [[1.0]],
        )
        with patch("dictionary.semantic.load_local_sentence_transformer", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
                load_bge_attribute_encoder("models/bge-small-en-v1.5")

    def test_build_normalizes_and_records_bge_manifest(self) -> None:
        encoder = FakeBgeEncoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "annotations.jsonl"
            source.write_text(
                "\n".join(json.dumps(_record(asin, feature)) for asin, feature in (
                    ("A", "non slip"),
                    ("B", "quick drying"),
                )) + "\n",
                encoding="utf-8",
            )
            output = root / "dictionary"
            with patch(
                "scripts.build_attribute_dictionary.load_bge_attribute_encoder",
                return_value=encoder,
            ):
                summary = build_attribute_dictionary(
                    source,
                    output,
                    embedding_model="models/bge-small-en-v1.5",
                    semantic_attributes=("feature",),
                )

            self.assertEqual(summary["embedding_status"], "generated")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["embedding"]["model"], ATTRIBUTE_EMBEDDING_MODEL)
            self.assertEqual(manifest["embedding"]["dimension"], 384)
            self.assertEqual(manifest["embedding"]["normalization"], "l2")
            self.assertIsNone(manifest["embedding"]["query_prefix"])
            matrix = np.load(output / "attribute_embeddings.npy", allow_pickle=False)
            self.assertEqual(matrix.shape, (2, 384))
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0)
            self.assertEqual(validate_attribute_dictionary(output)["has_embedding_matrix"], True)

            dictionary = AttributeDictionary.load(output)
            dictionary.set_query_encoder(encoder)
            matches = dictionary.semantic_match(
                "won't slip",
                allowed_attribute="feature",
                top_k=2,
                min_similarity=-1.0,
            )
            self.assertEqual(len(matches), 2)
            self.assertEqual(encoder.query_texts, ["won't slip"])

    def test_registry_rejects_non_bge_query_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "annotations.jsonl"
            source.write_text(json.dumps(_record("A", "non slip")) + "\n", encoding="utf-8")
            output = root / "dictionary"
            with patch(
                "scripts.build_attribute_dictionary.load_bge_attribute_encoder",
                return_value=FakeBgeEncoder(),
            ):
                build_attribute_dictionary(
                    source,
                    output,
                    embedding_model="models/bge-small-en-v1.5",
                    semantic_attributes=("feature",),
                )
            dictionary = AttributeDictionary.load(output)
            wrong = SimpleNamespace(
                model_id="jinaai/jina-embeddings-v5-text-nano",
                embedding_dimension=384,
                embed_query=lambda _text: [1.0] * 384,
            )
            with self.assertRaisesRegex(ValueError, "model does not match"):
                dictionary.set_query_encoder(wrong)

    def test_console_selects_one_attribute_space(self) -> None:
        self.assertEqual(_parse_attribute_choice("5"), "feature")
        self.assertEqual(_parse_attribute_choice("6"), "use_case")
        self.assertEqual(_parse_attribute_choice("use_case"), "use_case")
        self.assertIsNone(_parse_attribute_choice("category"))

        class Dictionary:
            def semantic_match(self, query, *, allowed_attribute, top_k, min_similarity):
                self.args = (query, allowed_attribute, top_k, min_similarity)
                return (SimpleNamespace(
                    canonical_id="feature:non_slip",
                    attribute="feature",
                    value="non slip",
                    similarity=0.87,
                ),)

            def get(self, _canonical_id):
                return SimpleNamespace(count=12)

        dictionary = Dictionary()
        context = SimpleNamespace(dictionary=dictionary)
        matches = search_attribute(context, "feature", "won't slip", top_k=10)
        self.assertEqual(dictionary.args, ("won't slip", "feature", 10, -1.0))
        output: list[str] = []
        print_results(context, matches, output_fn=output.append)
        self.assertIn("feature:non_slip", output[-1])
        self.assertIn("0.8700", output[-1])


if __name__ == "__main__":
    unittest.main()
