from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.agent_factory import build_evaluator_agent, warm_evaluator_runtime


class AgentFactoryTests(unittest.TestCase):
    def test_builds_agent_without_product_embedding_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            with patch("evaluator.agent_factory.Agent") as agent_class:
                build_evaluator_agent(catalog_path)

            agent_class.assert_called_once_with(
                catalog_path,
                use_user_profile=True,
            )

    def test_disable_user_profile_builds_the_profile_free_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            with patch("evaluator.agent_factory.Agent") as agent_class:
                build_evaluator_agent(catalog_path, disable_user_profile=True)

            agent_class.assert_called_once_with(
                catalog_path,
                use_user_profile=False,
            )

    def test_warmup_initializes_constraint_resources(self) -> None:
        with patch(
            "evaluator.agent_factory.constraint_module.extract_constraints"
        ) as extract_constraints:
            warm_evaluator_runtime()

        extract_constraints.assert_called_once_with("")


if __name__ == "__main__":
    unittest.main()
