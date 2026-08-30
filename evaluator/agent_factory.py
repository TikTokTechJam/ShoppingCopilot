"""Build evaluator Agents using the active structured/BGE attribute flow."""

from __future__ import annotations

from pathlib import Path

from starter.agent import Agent


def _load_project_env() -> None:
    """Load the ignored project environment before building an evaluator Agent."""

    env_path = Path(".env")
    if not env_path.is_file():
        return
    try:
        from annotation.config import load_env_file

        # Existing shell variables remain authoritative because the loader does
        # not overwrite values already present in the process environment.
        load_env_file(env_path)
    except (ImportError, OSError, ValueError):
        # A convenience file must not make the evaluator unusable. The Agent
        # will use its normal environment/fallback behavior instead.
        return


def build_evaluator_agent(
    catalog_path: str | Path,
    *,
    disable_user_profile: bool = False,
) -> Agent:
    """Build the evaluator Agent with local-only optional model loading.

    Semantic matching is initialized by ``starter.routing.constraints`` from
    the generated BGE canonical-attribute registry. If the V5 product-card
    artifact is present, ``ProductRetriever`` independently discovers its
    local Qwen query encoder for Browsing; it never substitutes BGE or a hash
    encoder for that product path.

    ``disable_user_profile`` runs the profile-free follow-up policy, which is
    the control arm for measuring what ``user_profile`` preference tags are
    worth.
    """

    _load_project_env()
    return Agent(catalog_path, use_user_profile=not disable_user_profile)


__all__ = ["build_evaluator_agent"]
