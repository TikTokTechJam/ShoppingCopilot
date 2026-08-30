"""Runtime feedback loop ("self-evolution").

Stage 1 (shipped default, ``PHASE_A_CONFIG``): OBSERVE cheap per-turn signals,
DISTILL them into per-value belief weights, ACT by scaling the structured
retrieval pull for reinforced constraints.

Stages 2 and 3, gated OFF by default (turn on with ``FULL_CONFIG`` or the
individual ``EvolutionConfig`` flags):

* implicit-negative decay in DISTILL,
* a per-turn ``StrategyController`` (RE-PLAN),
* a ``CrossSessionStore`` of learned per-field priors (LEARN).

Deterministic, no LLM, no I/O.
"""

from __future__ import annotations

from starter.evolution.config import FULL_CONFIG, PHASE_A_CONFIG, EvolutionConfig
from starter.evolution.distiller import distill, field_factors
from starter.evolution.loop import EvolutionLoop
from starter.evolution.observe import TurnObservation, constraint_pairs, observe
from starter.evolution.planner import Strategy, StrategyController
from starter.evolution.store import CrossSessionStore

__all__ = [
    "CrossSessionStore",
    "EvolutionConfig",
    "EvolutionLoop",
    "FULL_CONFIG",
    "PHASE_A_CONFIG",
    "Strategy",
    "StrategyController",
    "TurnObservation",
    "constraint_pairs",
    "distill",
    "field_factors",
    "observe",
]
