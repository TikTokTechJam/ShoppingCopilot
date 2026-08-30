"""Build evaluator Agents using the active structured/BGE attribute flow."""

from __future__ import annotations

from pathlib import Path

from starter.agent import Agent


def build_evaluator_agent(
    catalog_path: str | Path,
    *,
    disable_user_profile: bool = False,
    disable_evolution: bool = False,
) -> Agent:
    """Build the evaluator Agent without product-level model loading.

    Semantic matching is initialized by ``starter.routing.constraints`` from
    the generated BGE canonical-attribute registry. The retired direct
    product-embedding path is intentionally not discovered or configured here.

    ``disable_user_profile`` runs the profile-free follow-up policy, which is
    the control arm for measuring what ``user_profile`` preference tags are
    worth. ``disable_evolution`` runs the pre-feedback-loop code path, the
    control arm for measuring the runtime belief-reweighting loop.
    """

    return Agent(
        catalog_path,
        use_user_profile=not disable_user_profile,
        enable_evolution=not disable_evolution,
    )


__all__ = ["build_evaluator_agent"]
