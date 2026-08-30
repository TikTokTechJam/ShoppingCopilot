"""DISTILL: turn observations into per-value belief weights (reinforcement only).

The belief is ``dict[field][value] -> factor`` with an absent key meaning 1.0.
Phase A only ever *raises* a factor, for a value the shopper has restated (or
re-added after a correction), and clamps it to ``[w_min, w_max]``. The deferred
implicit-negative decay is a separate branch, gated off by config.

Every function is pure: inputs are never mutated, a fresh nested dict is
returned.
"""

from __future__ import annotations

from collections.abc import Mapping

from starter.evolution.config import EvolutionConfig
from starter.evolution.observe import TurnObservation, constraint_pairs
from starter.retrieval import STRUCTURED_FIELD_WEIGHTS


def _reweightable(field_name: str) -> bool:
    # Budget is an eligibility filter in retrieval, never a soft pull, so it is
    # not a reweighting target. Everything else with a structured weight is.
    return field_name in STRUCTURED_FIELD_WEIGHTS and field_name != "price"


def _prior_mentions(trace: list, key: str) -> int:
    return sum(
        1
        for record in trace
        if key in record.get("new_pairs", ()) or key in record.get("reinforced", ())
    )


def distill(
    belief: Mapping[str, Mapping[str, float]],
    obs: TurnObservation,
    *,
    constraints: object,
    provenance: Mapping[str, Mapping[str, str]] | None = None,
    replacements: object = (),
    trace: list | None = None,
    config: EvolutionConfig,
) -> dict[str, dict[str, float]]:
    """Return the next belief-weight sidecar for this turn."""

    del provenance  # reserved for the deferred implicit-negative branch
    trace = trace or []
    new: dict[str, dict[str, float]] = {
        field_name: dict(values) for field_name, values in belief.items()
    }

    # (a) explicit correction: the field was replaced this turn, so any prior
    # weight on it is stale -- drop the whole entry and let survivors restart.
    for field_name in set(replacements):
        new.pop(field_name, None)

    # A preference override reseeds its new values at neutral: direct evidence
    # outranks a carried-over factor.
    new_set = set(obs.new_pairs)
    if obs.override_kind == "PREFERENCE":
        for field_name, value in new_set:
            if field_name in new:
                new[field_name].pop(value, None)

    reinforced_set = set(obs.reinforced_pairs)
    live = constraint_pairs(constraints)

    # (b) reinforcement: +bump, once per turn per value.
    for field_name, value in sorted(live):
        if not _reweightable(field_name):
            continue
        key = f"{field_name}:{value}"
        reinforced_now = (field_name, value) in reinforced_set or (
            (field_name, value) in new_set and _prior_mentions(trace, key) >= 1
        )
        if not reinforced_now:
            continue
        weight = new.get(field_name, {}).get(value, 1.0) + config.reinforce_bump
        weight = min(config.w_max, max(config.w_min, weight))
        new.setdefault(field_name, {})[value] = weight

    # (c) clamp + prune: keep only live pairs whose factor is off unity, so the
    # sidecar stays a sparse strict subset of the active constraints.
    live_by_field: dict[str, set[str]] = {}
    for field_name, value in live:
        live_by_field.setdefault(field_name, set()).add(value)
    pruned: dict[str, dict[str, float]] = {}
    for field_name, values in new.items():
        kept = {
            value: min(config.w_max, max(config.w_min, weight))
            for value, weight in values.items()
            if value in live_by_field.get(field_name, set())
            and abs(weight - 1.0) > 1e-9
        }
        if kept:
            pruned[field_name] = kept
    return pruned


def field_factors(
    belief: Mapping[str, Mapping[str, float]],
    constraints: object,
    config: EvolutionConfig,
) -> dict[str, float]:
    """Collapse per-value weights to one bounded factor per active field.

    The mean is used so one hesitant value cannot tank a field that also holds
    a strongly reinforced one, while a field whose only value is neutral stays
    at exactly 1.0.
    """

    factors: dict[str, float] = {}
    for field_name, values in _active_values(constraints):
        if not _reweightable(field_name):
            continue
        field_belief = belief.get(field_name, {})
        mean = sum(field_belief.get(value, 1.0) for value in values) / len(values)
        factors[field_name] = min(config.w_max, max(config.w_min, mean))
    return factors


def _active_values(constraints: object) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for field_name, value in sorted(constraint_pairs(constraints)):
        grouped.setdefault(field_name, []).append(value)
    return list(grouped.items())


__all__ = ["distill", "field_factors"]
