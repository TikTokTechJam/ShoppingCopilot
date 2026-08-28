from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from product_embeddings.pipeline import load_local_sentence_transformer
from scripts.setup_jina_embedding_model import MODEL_ID, setup_model


class FakeQueryEncoder:
    model_id = MODEL_ID
    embedding_dimension = 768

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[list[float]]:
        self.queries.append(text)
        return [[1.0] + [0.0] * 767]


class JinaSetupTests(unittest.TestCase):
    def test_setup_downloads_fixed_model_and_validates_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            download_calls: list[dict[str, object]] = []
            load_calls: list[tuple[str, dict[str, object]]] = []
            encoder = FakeQueryEncoder()

            def downloader(**kwargs: object) -> str:
                download_calls.append(kwargs)
                Path(str(kwargs["local_dir"])).mkdir(parents=True, exist_ok=True)
                return str(kwargs["local_dir"])

            def loader(path: str, **kwargs: object) -> FakeQueryEncoder:
                load_calls.append((path, kwargs))
                return encoder

            result = setup_model(
                model_dir,
                revision="main",
                downloader=downloader,
                encoder_loader=loader,
            )

            self.assertEqual(download_calls, [{
                "repo_id": MODEL_ID,
                "local_dir": str(model_dir.resolve()),
                "revision": "main",
            }])
            self.assertEqual(load_calls, [(
                str(model_dir.resolve()),
                {
                    "task": "retrieval",
                    "document_prompt_name": "document",
                    "query_prompt_name": "query",
                    "trust_remote_code": True,
                },
            )])
            self.assertEqual(result["model_id"], MODEL_ID)
            self.assertEqual(result["dimension"], 768)
            self.assertEqual(result["query_shape"], (768,))
            self.assertEqual(encoder.queries, ["jumpsuits for cosplay"])

    def test_setup_rejects_wrong_query_dimension(self) -> None:
        class WrongDimensionEncoder:
            def embed_query(self, _text: str) -> list[list[float]]:
                return [[1.0, 0.0]]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "expected \(768,\)"):
                setup_model(
                    Path(directory) / "model",
                    downloader=lambda **kwargs: str(kwargs["local_dir"]),
                    encoder_loader=lambda _path, **_kwargs: WrongDimensionEncoder(),
                )

    def test_sentence_transformer_loader_is_local_only_and_trusted_for_jina(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeSentenceTransformer:
            def __init__(self, path: str, **kwargs: object) -> None:
                calls.append({"path": path, **kwargs})

            def get_sentence_embedding_dimension(self) -> int:
                return 768

        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            encoder = load_local_sentence_transformer(
                "model/jina-embeddings-v5-text-nano",
                task="retrieval",
                document_prompt_name="document",
                query_prompt_name="query",
                trust_remote_code=True,
            )

        self.assertEqual(calls, [{
            "path": "model/jina-embeddings-v5-text-nano",
            "device": None,
            "local_files_only": True,
            "trust_remote_code": True,
        }])
        self.assertEqual(encoder.embedding_dimension, 768)
        self.assertEqual(encoder.model_id, "model/jina-embeddings-v5-text-nano")


if __name__ == "__main__":
    unittest.main()
