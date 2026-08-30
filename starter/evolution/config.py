"""Tunables for the runtime feedback loop.

Every constant the loop uses lives here so retuning means editing one place.
Values are fixed a priori in the ``ProfileAffinity`` style, not swept on a
benchmark.

Three stages are gated by their own flag and all default OFF, so
``PHASE_A_CONFIG`` (belief reweighting only) is the shipped default and every
other build is byte-identical until a flag is turned on:

* ``enable_implicit_negative`` -- DISTILL also *decays* an attribute shared by
  every already-shown-and-missed candidate.
* ``enable_replan`` -- a per-turn ``StrategyController`` may override the
  structured / dense / bm25 blend and soften the weakest constraint.
* ``enable_learn`` -- a ``CrossSessionStore`` that survives ``reset()`` seeds
  each field's starting factor from surrogate signals of past sessions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionConfig:
    """Bounds, step sizes and stage switches for the whole loop."""

    # --- DISTILL: per-value factor bounds and reinforcement -----------------
    # The lower bound matches ProfileAffinity; the upper bound has a little
    # headroom over 1.25 so a value restated on three separate turns can move.
    w_min: float = 0.75
    w_max: float = 1.30
    reinforce_bump: float = 0.12          # additive, once per turn per value

    churn_eps: float = 1e-9               # below this = "no movement"
    max_trace: int = 10                   # per-turn records kept on the session

    # --- DISTILL: implicit-negative decay (Stage 1, gated) -----------------
    enable_implicit_negative: bool = False
    neg_decay: float = 0.10              # additive down, once per turn per value
    oneoff_extra_penalty: float = 0.08   # extra for a never-reinforced one-off
    neg_common_frac: float = 1.0         # decay only if in ALL shown-and-missed
    implicit_negative_min_turn: int = 3  # pre-override guard (override is 3/4)

    # --- RE-PLAN: strategy controller (Stage 2, gated) --------------------
    enable_replan: bool = False
    replan_min_pool: int = 12            # below this -> RELAX_WEAKEST
    stuck_churn_max: float = 0.10        # <= this + no progress -> DIVERSIFY
    relax_scale: float = 0.85           # weakest active field's multiplier
    # score_weights overrides as (structured, dense, bm25):
    exploit_dense_weight: float = 0.60
    exploit_bm25_weight: float = 0.10
    recall_dense_weight: float = 1.40
    recall_bm25_weight: float = 0.35
    diversify_bm25_weight: float = 0.45

    # --- LEARN: cross-session priors (Stage 3, gated) --------------------
    enable_learn: bool = False
    learn_rate: float = 0.10             # EWMA rate for a field prior update
    learn_prior_floor: float = 0.85
    learn_prior_ceiling: float = 1.15


# Shipped default: belief reweighting only.
PHASE_A_CONFIG = EvolutionConfig()

# Everything on -- for the ablation CLI flag and the "full loop" tests.
FULL_CONFIG = EvolutionConfig(
    enable_implicit_negative=True,
    enable_replan=True,
    enable_learn=True,
)


__all__ = ["EvolutionConfig", "PHASE_A_CONFIG", "FULL_CONFIG"]
