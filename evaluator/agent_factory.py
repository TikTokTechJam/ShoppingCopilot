"""Build evaluator Agents using the active structured/BGE attribute flow."""

from __future__ import annotations

from pathlib import Path

from starter.agent import Agent


def build_evaluator_agent(catalog_path: str | Path) -> Agent:
    """Build the evaluator Agent without product-level model loading.

    Semantic matching is initialized by ``starter.routing.constraints`` from
    the generated BGE canonical-attribute registry. The retired direct
    product-embedding path is intentionally not discovered or configured here.
    """

    return Agent(catalog_path)


__all__ = ["build_evaluator_agent"]
