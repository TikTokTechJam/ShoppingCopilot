"""OBSERVE: cheap, deterministic per-turn signals.

Everything here is derived from information the Agent already has on the current
turn -- the extracted delta, the prior constraint set, the previous turn's
trace. No RNG, no wall-clock, no catalog access, so the whole module is a pure
function of its inputs and trivially reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from starter.routing.constraints import CATEGORICAL_FIELDS


def constraint_pairs(constraints: object) -> set[tuple[str, str]]:
    """Return the ``(field, value)`` pairs currently held in a constraint record."""

    pairs: set[tuple[str, str]] = set()
    for field_name in CATEGORICAL_FIELDS:
        values = getattr(constraints, field_name, ()) or ()
        if isinstance(values, str):
            values = (values,)
        for value in values:
            text = str(value).strip()
            if text:
                pairs.add((field_name, text))
    return pairs


@dataclass
class TurnObservation:
    """One turn's feedback signals.

    ``observe`` fills the constraint-side fields; ``EvolutionLoop.finalize_turn``
    fills the pool/churn fields once retrieval has run.
    """

    turn: int
    called_again: bool
    new_pairs: tuple[tuple[str, str], ...]
    reinforced_pairs: tuple[tuple[str, str], ...]
    no_preference: bool
    override_kind: str | None
    pool_size: int = 0
    pool_delta: int = 0
    top_n_churn: float = 0.0
    made_progress: bool = False


def observe(
    *,
    turn: int,
    trace: list,
    structured_delta: object,
    prev_pairs: set[tuple[str, str]],
    new_pairs: set[tuple[str, str]],
    no_preference: bool,
    override_kind: str | None,
) -> TurnObservation:
    """Build the constraint-side half of a :class:`TurnObservation`."""

    delta_pairs = constraint_pairs(structured_delta)
    reinforced = tuple(sorted(delta_pairs & set(prev_pairs)))
    return TurnObservation(
        turn=int(turn),
        called_again=bool(trace),
        new_pairs=tuple(sorted(new_pairs)),
        reinforced_pairs=reinforced,
        no_preference=bool(no_preference),
        override_kind=override_kind,
    )


__all__ = ["TurnObservation", "constraint_pairs", "observe"]
