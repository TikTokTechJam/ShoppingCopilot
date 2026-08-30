"""Build evaluator Agents using the active structured/BGE attribute flow."""

from __future__ import annotations

from pathlib import Path

from starter.agent import Agent
from starter.routing import constraints as constraint_module


def build_evaluator_agent(
    catalog_path: str | Path,
    *,
    disable_user_profile: bool = False,
) -> Agent:
    """Build the evaluator Agent without product-level model loading.

    Semantic matching is initialized by ``starter.routing.constraints`` from
    the generated BGE canonical-attribute registry. The retired direct
    product-embedding path is intentionally not discovered or configured here.

    ``disable_user_profile`` runs the profile-free follow-up policy, which is
    the control arm for measuring what ``user_profile`` preference tags are
    worth.
    """

    return Agent(catalog_path, use_user_profile=not disable_user_profile)


def warm_evaluator_runtime() -> None:
    """Load lazy shared resources once before evaluator workers start."""

    constraint_module.extract_constraints("")


__all__ = ["build_evaluator_agent", "warm_evaluator_runtime"]
