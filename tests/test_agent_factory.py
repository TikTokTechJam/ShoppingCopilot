from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.agent_factory import build_evaluator_agent


def write_manifest(directory: Path, model: str) -> None:
    (directory / "manifest.json").write_text(
        json.dumps({"embedding_model": model}),
        encoding="utf-8",
    )


class AgentFactoryTests(unittest.TestCase):
    def test_manifest_model_is_loaded_when_no_model_flag_is_given(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifact"
            artifact_dir.mkdir()
            write_manifest(artifact_dir, "test-semantic-model")
            encoder = object()

            with patch(
                "evaluator.agent_factory.load_local_sentence_transformer",
                return_value=encoder,
            ) as loader, patch("evaluator.agent_factory.Agent") as agent_class:
                build_evaluator_agent(
                    Path(directory) / "catalog.jsonl",
                    layer2_artifact_dir=artifact_dir,
                )

            loader.assert_called_once_with(
                "test-semantic-model",
                task=None,
                document_prompt_name=None,
                query_prompt_name=None,
                trust_remote_code=False,
                device=None,
                half_precision=False,
            )
            agent_class.assert_called_once_with(
                Path(directory) / "catalog.jsonl",
                query_encoder=encoder,
                layer2_artifact_dir=artifact_dir,
            )

    def test_hash_manifest_is_not_auto_selected_for_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifact"
            artifact_dir.mkdir()
            write_manifest(artifact_dir, "hashing-fallback-v1")

            with self.assertRaisesRegex(ValueError, "real embedding model"):
                build_evaluator_agent(
                    Path(directory) / "catalog.jsonl",
                    layer2_artifact_dir=artifact_dir,
                )

    def test_auto_model_failure_falls_back_to_structured_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifact"
            artifact_dir.mkdir()
            write_manifest(artifact_dir, "jinaai/jina-embeddings-v5-text-nano")
            catalog_path = Path(directory) / "catalog.jsonl"

            with patch(
                "evaluator.agent_factory.load_local_sentence_transformer",
                side_effect=RuntimeError("model not installed"),
            ), patch("evaluator.agent_factory.Agent") as agent_class:
                build_evaluator_agent(catalog_path, layer2_artifact_dir=artifact_dir)

            agent_class.assert_called_once_with(
                catalog_path,
                layer2_artifact_dir=artifact_dir,
            )


if __name__ == "__main__":
    unittest.main()
