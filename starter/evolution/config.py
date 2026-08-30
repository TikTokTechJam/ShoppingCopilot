"""Tunables for the runtime feedback loop.

Every constant the loop uses lives here so retuning means editing one place.
Values are fixed a priori in the ``ProfileAffinity`` style, not swept on a
benchmark. Phase A only reads ``w_min``, ``w_max``, ``reinforce_bump``,
``churn_eps`` and ``max_trace``; the decay fields are declared inert so the
deferred implicit-negative follow-up is a small, gated addition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionConfig:
    """Bounds and step sizes for OBSERVE + DISTILL + ACT."""

    # A per-value factor is clamped to this interval. The lower bound matches
    # ProfileAffinity; the upper bound has a little headroom over 1.25 so a
    # value restated on three separate turns can still move.
    w_min: float = 0.75
    w_max: float = 1.30

    # Additive bump applied at most once per turn per reinforced value.
    reinforce_bump: float = 0.12

    # Churn below this counts as "no movement" for the progress signal.
    churn_eps: float = 1e-9

    # Longest per-turn trace retained on the session (a session is <= 10 turns).
    max_trace: int = 10

    # --- deferred implicit-negative decay (not wired in Phase A) -------------
    enable_implicit_negative: bool = False
    neg_decay: float = 0.10
    oneoff_extra_penalty: float = 0.08
    neg_common_frac: float = 1.0
    implicit_negative_min_turn: int = 3


PHASE_A_CONFIG = EvolutionConfig()


__all__ = ["EvolutionConfig", "PHASE_A_CONFIG"]
