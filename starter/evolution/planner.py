"""RE-PLAN (Stage 2): pick a per-turn retrieval strategy.

The controller reads only signals the Agent already has *before* retrieval --
this turn's ``TurnObservation`` (new constraints, called-again, override) plus
the previous turn's finalized trace record (pool size, pool delta, churn,
made_progress). It returns a :class:`Strategy` and translates it into two knobs
the existing scorer already understands:

* ``score_weights`` -- an ``(structured, dense, bm25)`` triple that overrides
  ``MODE_SCORE_WEIGHTS[mode]`` for this call (so BUYING/BROWSING finally changes
  retrieval);
* an adjusted ``field_weights`` dict -- RELAX_WEAKEST softens the single weakest
  active constraint so a starved pool can recover.

Everything is deterministic. With ``enable_replan`` off, ``choose`` always
returns ``NEUTRAL`` and ``apply`` is a pass-through, so the retrieval call is
identical to Phase A.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

from starter.evolution.config import EvolutionConfig
from starter.retrieval import STRUCTURED_FIELD_WEIGHTS


class Strategy(str, Enum):
    NEUTRAL = "neutral"
    EXPLOIT_NARROW = "exploit_narrow"
    RECALL_BROAD = "recall_broad"
    DIVERSIFY = "diversify"
    RELAX_WEAKEST = "relax_weakest"


ScoreWeights = tuple[float, float, float]


class StrategyController:
    """Rules-first, deterministic per-turn strategy selection."""

    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config

    def choose(
        self,
        obs: object,
        trace: Sequence[Mapping[str, object]],
        constraint_fields: Sequence[str],
    ) -> Strategy:
        if not self.config.enable_replan:
            return Strategy.NEUTRAL

        turn = int(getattr(obs, "turn", 0))
        if turn <= 1 or not constraint_fields:
            return Strategy.RECALL_BROAD

        previous = trace[-1] if trace else None
        called_again = bool(getattr(obs, "called_again", False))
        new_pairs = tuple(getattr(obs, "new_pairs", ()))

        if previous is not None:
            made_progress = bool(previous.get("made_progress", True))
            churn = float(previous.get("churn", 1.0))
            pool_delta = int(previous.get("pool_delta", 0))
            pool_size = int(previous.get("pool_size", 100))

            if (
                called_again
                and not made_progress
                and churn <= self.config.stuck_churn_max
            ):
                return Strategy.DIVERSIFY
            if pool_size < self.config.replan_min_pool:
                return Strategy.RELAX_WEAKEST
            if new_pairs and pool_delta < 0:
                return Strategy.EXPLOIT_NARROW

        return Strategy.NEUTRAL

    def apply(
        self,
        strategy: Strategy,
        field_weights: Mapping[str, float] | None,
        constraint_fields: Sequence[str],
    ) -> tuple[dict[str, float] | None, ScoreWeights | None]:
        """Translate a strategy into (field_weights, score_weights) overrides."""

        cfg = self.config
        if strategy is Strategy.NEUTRAL:
            return (dict(field_weights) if field_weights is not None else None, None)

        if strategy is Strategy.EXPLOIT_NARROW:
            return (
                dict(field_weights) if field_weights is not None else None,
                (1.0, cfg.exploit_dense_weight, cfg.exploit_bm25_weight),
            )
        if strategy is Strategy.RECALL_BROAD:
            return (
                dict(field_weights) if field_weights is not None else None,
                (1.0, cfg.recall_dense_weight, cfg.recall_bm25_weight),
            )
        if strategy is Strategy.DIVERSIFY:
            return (
                dict(field_weights) if field_weights is not None else None,
                (1.0, 1.0, cfg.diversify_bm25_weight),
            )

        # RELAX_WEAKEST: soften the single weakest active constraint.
        base = dict(field_weights) if field_weights is not None else dict(
            STRUCTURED_FIELD_WEIGHTS
        )
        active = [
            f
            for f in constraint_fields
            if f in STRUCTURED_FIELD_WEIGHTS and f != "price"
        ]
        if active:
            weakest = min(active, key=lambda f: STRUCTURED_FIELD_WEIGHTS[f])
            base[weakest] = base.get(weakest, STRUCTURED_FIELD_WEIGHTS[weakest]) * (
                cfg.relax_scale
            )
            return base, None
        return (dict(field_weights) if field_weights is not None else None, None)


__all__ = ["Strategy", "StrategyController", "ScoreWeights"]
