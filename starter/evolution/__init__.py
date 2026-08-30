"""Runtime feedback loop ("self-evolution") -- Phase A.

OBSERVE cheap per-turn signals, DISTILL them into per-value belief weights, ACT
by scaling the structured retrieval pull. Deterministic, no LLM, no I/O. Phase B
(strategy re-planning) and Phase C (cross-session priors) are deliberately left
as open seams; see the plan in ``.claude/plans``.
"""

from __future__ import annotations

from starter.evolution.config import PHASE_A_CONFIG, EvolutionConfig
from starter.evolution.distiller import distill, field_factors
from starter.evolution.loop import EvolutionLoop
from starter.evolution.observe import TurnObservation, constraint_pairs, observe

__all__ = [
    "EvolutionConfig",
    "EvolutionLoop",
    "PHASE_A_CONFIG",
    "TurnObservation",
    "constraint_pairs",
    "distill",
    "field_factors",
    "observe",
]
