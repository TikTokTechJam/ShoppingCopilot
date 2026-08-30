"""LEARN (Stage 3): cross-session priors that survive ``reset()``.

The Agent never sees a hit, so this cannot learn from the real objective. It
learns from a *surrogate*: after a session ends, for each constraint field that
finished with an above-neutral belief factor, it measures whether the turns
*after* that field was reinforced actually made progress (pool contraction or
top-k churn). Fields that co-occurred with progress get their starting factor
nudged up; fields that stalled decay back toward 1.0. The nudge is a bounded
EWMA, so a single session can only move a prior a little.

``prior_factor`` feeds DISTILL: a value's first appearance is seeded at its
field's learned prior instead of 1.0. With ``enable_learn`` off, every prior is
1.0 and ``observe_session_end`` is a no-op.

State is process-local: it lives on the Agent instance and is intentionally not
persisted to disk. Reproducibility holds because the evaluators iterate
sessions in a fixed order; a run is not session-order independent, which is why
this stage has its own flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from starter.evolution.config import EvolutionConfig


@dataclass
class _FieldPrior:
    mean: float = 1.0
    updates: int = 0


@dataclass
class CrossSessionStore:
    config: EvolutionConfig
    _priors: dict[str, _FieldPrior] = field(default_factory=dict, repr=False)
    sessions_finalized: int = 0

    # -- read -----------------------------------------------------------------

    def prior_factor(self, field_name: str) -> float:
        if not self.config.enable_learn:
            return 1.0
        prior = self._priors.get(field_name)
        return prior.mean if prior is not None else 1.0

    def snapshot(self) -> dict[str, float]:
        return {name: round(p.mean, 6) for name, p in sorted(self._priors.items())}

    # -- write --------------------------------------------------------------

    def observe_session_end(
        self,
        *,
        belief_weights: Mapping[str, Mapping[str, float]],
        trace: Sequence[Mapping[str, Any]],
    ) -> int:
        """Fold one finished session's surrogate signal into the priors.

        Returns the number of field priors touched (0 when the stage is off).
        """

        if not self.config.enable_learn or not trace:
            return 0

        self.sessions_finalized += 1
        touched = 0
        reinforced_turn = _first_reinforced_turn(trace)

        for field_name in _fields_in_trace(trace):
            had_factor = bool(belief_weights.get(field_name))
            first = reinforced_turn.get(field_name)
            if first is None:
                # Field never reinforced this session -> gently forget.
                self._nudge(field_name, target=1.0)
                touched += 1
                continue
            progress = _progress_fraction_after(trace, first)
            # progress in [0, 1] -> target factor in [floor, ceiling], centred
            # at 1.0 for a coin-flip session.
            span = self.config.learn_prior_ceiling - self.config.learn_prior_floor
            target = self.config.learn_prior_floor + span * progress
            if not had_factor:
                target = 0.5 * (target + 1.0)  # weaker evidence -> pull toward 1
            self._nudge(field_name, target=target)
            touched += 1
        return touched

    def _nudge(self, field_name: str, *, target: float) -> None:
        prior = self._priors.setdefault(field_name, _FieldPrior())
        rate = self.config.learn_rate
        prior.mean = (1.0 - rate) * prior.mean + rate * target
        prior.mean = min(
            self.config.learn_prior_ceiling,
            max(self.config.learn_prior_floor, prior.mean),
        )
        prior.updates += 1


def _fields_in_trace(trace: Iterable[Mapping[str, Any]]) -> set[str]:
    fields: set[str] = set()
    for record in trace:
        for key in (*record.get("new_pairs", ()), *record.get("reinforced", ())):
            fields.add(str(key).split(":", 1)[0])
    return fields


def _first_reinforced_turn(
    trace: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    first: dict[str, int] = {}
    for record in trace:
        turn = int(record.get("turn", 0))
        for key in record.get("reinforced", ()):
            field_name = str(key).split(":", 1)[0]
            first.setdefault(field_name, turn)
    return first


def _progress_fraction_after(
    trace: Sequence[Mapping[str, Any]], turn: int
) -> float:
    later = [r for r in trace if int(r.get("turn", 0)) > turn]
    if not later:
        return 0.5
    good = sum(
        1
        for r in later
        if bool(r.get("made_progress", False)) or int(r.get("pool_delta", 0)) < 0
    )
    return good / len(later)


__all__ = ["CrossSessionStore"]
