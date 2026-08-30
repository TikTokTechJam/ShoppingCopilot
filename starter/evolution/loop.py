"""The feedback loop coordinator.

``EvolutionLoop`` is stateless apart from its config: it turns Agent-visible
inputs into a belief-weight sidecar, a per-call retrieval weight vector, an
optional score-weight override, a per-turn trace record, and a telemetry
snapshot. ``SessionManager`` performs every session-state mutation; the
cross-session ``CrossSessionStore`` lives on the Agent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from starter.evolution.config import EvolutionConfig
from starter.evolution.distiller import (
    FactLookup,
    PriorFactor,
    distill,
    field_factors,
)
from starter.evolution.observe import TurnObservation, observe
from starter.evolution.planner import ScoreWeights, Strategy, StrategyController
from starter.retrieval import STRUCTURED_FIELD_WEIGHTS


class EvolutionLoop:
    """OBSERVE -> DISTILL -> (RE-PLAN) -> ACT, plus a finalize/telemetry pass."""

    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config
        self.planner = StrategyController(config)

    # -- OBSERVE ----------------------------------------------------------

    def observe(
        self,
        *,
        turn: int,
        trace: list,
        structured_delta: object,
        prev_pairs: set[tuple[str, str]],
        new_pairs: set[tuple[str, str]],
        no_preference: bool,
        override_kind: str | None,
    ) -> TurnObservation:
        return observe(
            turn=turn,
            trace=trace,
            structured_delta=structured_delta,
            prev_pairs=prev_pairs,
            new_pairs=new_pairs,
            no_preference=no_preference,
            override_kind=override_kind,
        )

    # -- DISTILL --------------------------------------------------------

    def distill(
        self,
        belief: Mapping[str, Mapping[str, float]],
        obs: TurnObservation,
        *,
        constraints: object,
        provenance: Mapping[str, Mapping[str, str]] | None = None,
        replacements: object = (),
        trace: list | None = None,
        fact_lookup: FactLookup | None = None,
        prior_factor: PriorFactor | None = None,
    ) -> dict[str, dict[str, float]]:
        return distill(
            belief,
            obs,
            constraints=constraints,
            provenance=provenance,
            replacements=replacements,
            trace=trace,
            fact_lookup=fact_lookup,
            prior_factor=prior_factor,
            config=self.config,
        )

    # -- ACT ----------------------------------------------------------

    def act_field_weights(
        self,
        belief: Mapping[str, Mapping[str, float]],
        constraints: object,
    ) -> dict[str, float] | None:
        """Return a full structured-weight vector, or ``None`` for the untouched path."""

        factors = field_factors(belief, constraints, self.config)
        if all(abs(value - 1.0) <= 1e-9 for value in factors.values()):
            return None
        return {
            field_name: weight * factors.get(field_name, 1.0)
            for field_name, weight in STRUCTURED_FIELD_WEIGHTS.items()
        }

    # -- RE-PLAN ------------------------------------------------------

    def plan(
        self,
        obs: TurnObservation,
        trace: Sequence[Mapping[str, Any]],
        constraint_fields: Sequence[str],
        field_weights: Mapping[str, float] | None,
    ) -> tuple[dict[str, float] | None, ScoreWeights | None, str]:
        """Return (field_weights, score_weights, strategy_name).

        With ``enable_replan`` off this is a pass-through: the strategy is
        NEUTRAL and both overrides come back unchanged.
        """

        strategy = self.planner.choose(obs, trace, constraint_fields)
        adjusted, score_weights = self.planner.apply(
            strategy, field_weights, constraint_fields
        )
        if strategy is Strategy.NEUTRAL and field_weights is None:
            adjusted = None
        return adjusted, score_weights, strategy.value

    # -- finalize + telemetry ---------------------------------------

    def finalize_turn(
        self,
        obs: TurnObservation,
        *,
        pool_size: int,
        ranked: list[str],
        trace: list,
    ) -> dict[str, Any]:
        """Fill the pool/churn fields on ``obs`` and return the trace record."""

        previous = trace[-1] if trace else None
        obs.pool_size = int(pool_size)
        obs.pool_delta = (
            obs.pool_size - int(previous["pool_size"]) if previous else 0
        )
        previous_shown = set(previous["shown"]) if previous else set()
        current_shown = set(ranked)
        if previous_shown or current_shown:
            union = len(previous_shown | current_shown) or 1
            obs.top_n_churn = 1.0 - len(previous_shown & current_shown) / union
        else:
            obs.top_n_churn = 0.0
        obs.made_progress = (
            bool(obs.new_pairs)
            or obs.pool_delta < 0
            or obs.top_n_churn > self.config.churn_eps
        )
        return {
            "turn": obs.turn,
            "pool_size": obs.pool_size,
            "shown": tuple(ranked),
            "reinforced": sorted(f"{f}:{v}" for f, v in obs.reinforced_pairs),
            "new_pairs": sorted(f"{f}:{v}" for f, v in obs.new_pairs),
            "made_progress": obs.made_progress,
            "churn": round(obs.top_n_churn, 6),
            "pool_delta": obs.pool_delta,
        }

    def telemetry_snapshot(
        self,
        previous: Mapping[str, Any] | None,
        obs: TurnObservation,
        state: Any,
        *,
        reweighted: bool,
        strategy: str = "neutral",
        decayed: bool = False,
        learn_updates: int = 0,
    ) -> dict[str, Any]:
        """Update the cumulative run-level diagnostics dict."""

        diagnostics: dict[str, Any] = dict(previous or {})

        def bump(key: str, amount: int = 1) -> None:
            diagnostics[key] = int(diagnostics.get(key, 0)) + amount

        diagnostics["evolution.enabled"] = 1
        bump("evolution.turns_observed")
        if obs.new_pairs:
            bump("evolution.constraint_absorbed_turns")
        diagnostics["evolution.candidate_pool_last"] = int(obs.pool_size)
        if obs.pool_delta < 0:
            bump("evolution.candidate_pool_shrink_turns")
        if reweighted:
            bump("evolution.retrieval_reweight_turns")
        if obs.reinforced_pairs:
            bump("evolution.reinforce_events", len(obs.reinforced_pairs))
        if decayed:
            bump("evolution.implicit_negative_turns")
        if strategy != "neutral":
            bump(f"evolution.strategy_{strategy}_turns")
            bump("evolution.replan_turns")
        if learn_updates:
            bump("evolution.learn_updates", int(learn_updates))
            bump("evolution.learn_sessions")
        diagnostics["evolution.belief_values_tracked"] = int(
            sum(len(values) for values in state.belief_weights.values())
        )
        diagnostics["evolution.top_n_churn_last"] = round(float(obs.top_n_churn), 6)
        if obs.made_progress:
            bump("evolution.made_progress_turns")
        return diagnostics


__all__ = ["EvolutionLoop"]
