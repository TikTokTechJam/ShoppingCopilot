"""Build evaluator Agents using the active structured/BGE attribute flow."""

from __future__ import annotations

from pathlib import Path

from starter.agent import Agent
from starter.evolution import FULL_CONFIG, PHASE_A_CONFIG


def build_evaluator_agent(
    catalog_path: str | Path,
    *,
    disable_user_profile: bool = False,
    disable_evolution: bool = False,
    evolution_full: bool = False,
) -> Agent:
    """Build the evaluator Agent without product-level model loading.

    Semantic matching is initialized by ``starter.routing.constraints`` from
    the generated BGE canonical-attribute registry. The retired direct
    product-embedding path is intentionally not discovered or configured here.

    ``disable_user_profile`` runs the profile-free follow-up policy, the control
    arm for measuring the ``user_profile`` preference tags. ``disable_evolution``
    runs the pre-feedback-loop code path. ``evolution_full`` turns on the gated
    stages (implicit-negative decay, RE-PLAN, cross-session LEARN); the default
    is Phase A (belief reweighting only).
    """

    return Agent(
        catalog_path,
        use_user_profile=not disable_user_profile,
        enable_evolution=not disable_evolution,
        evolution_config=FULL_CONFIG if evolution_full else PHASE_A_CONFIG,
    )


__all__ = ["build_evaluator_agent"]
