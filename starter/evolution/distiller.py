"""DISTILL: turn observations into per-value belief weights.

The belief is ``dict[field][value] -> factor`` with an absent key meaning 1.0.

* **reinforcement** (always on): a value the shopper restated -- or re-added
  after a correction -- gains ``+reinforce_bump``, clamped to ``[w_min, w_max]``.
* **implicit-negative decay** (``enable_implicit_negative``): after a turn that
  was called again and made no progress, a value present in *every*
  already-shown-and-missed candidate is non-discriminating, so it loses
  ``neg_decay`` (plus a penalty if it was a never-reinforced one-off). Guarded
  by ``turn >= implicit_negative_min_turn`` so it cannot fire before an intent
  override.
* **learned prior** (``enable_learn``): a value's first appearance is seeded at
  its field's cross-session prior instead of 1.0.

Every function is pure: inputs are never mutated, a fresh nested dict is
returned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from starter.evolution.config import EvolutionConfig
from starter.evolution.observe import TurnObservation, constraint_pairs
from starter.retrieval import STRUCTURED_FIELD_WEIGHTS

FactLookup = Callable[[str, str], Iterable[str]]
PriorFactor = Callable[[str], float]


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


def _clamp(value: float, config: EvolutionConfig) -> float:
    return min(config.w_max, max(config.w_min, value))


def distill(
    belief: Mapping[str, Mapping[str, float]],
    obs: TurnObservation,
    *,
    constraints: object,
    provenance: Mapping[str, Mapping[str, str]] | None = None,
    replacements: object = (),
    trace: list | None = None,
    fact_lookup: FactLookup | None = None,
    prior_factor: PriorFactor | None = None,
    config: EvolutionConfig,
) -> dict[str, dict[str, float]]:
    """Return the next belief-weight sidecar for this turn."""

    del provenance  # available for future rules; unused today
    trace = trace or []
    new: dict[str, dict[str, float]] = {
        field_name: dict(values) for field_name, values in belief.items()
    }

    # (a) explicit correction: the field was replaced this turn, so any prior
    # weight on it is stale -- drop the whole entry and let survivors restart.
    for field_name in set(replacements):
        new.pop(field_name, None)

    new_set = set(obs.new_pairs)
    reinforced_set = set(obs.reinforced_pairs)
    live = constraint_pairs(constraints)
    live_by_field: dict[str, set[str]] = {}
    for field_name, value in live:
        live_by_field.setdefault(field_name, set()).add(value)

    # A preference override reseeds its new values at neutral: direct evidence
    # outranks a carried-over factor.
    if obs.override_kind == "PREFERENCE":
        for field_name, value in new_set:
            if field_name in new:
                new[field_name].pop(value, None)

    # (0) learned prior: seed a value's first appearance at its field prior.
    if prior_factor is not None:
        for field_name, value in sorted(new_set):
            if not _reweightable(field_name):
                continue
            if value in new.get(field_name, {}):
                continue
            seeded = _clamp(float(prior_factor(field_name)), config)
            if abs(seeded - 1.0) > 1e-9:
                new.setdefault(field_name, {})[value] = seeded

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
        new.setdefault(field_name, {})[value] = _clamp(weight, config)

    # (c) implicit-negative decay: a value common to every shown-and-missed
    # candidate carries no discriminating power.
    if (
        config.enable_implicit_negative
        and fact_lookup is not None
        and obs.called_again
        and obs.turn >= config.implicit_negative_min_turn
        and trace
        and trace[-1].get("made_progress") is False
    ):
        shown = tuple(trace[-1].get("shown", ()))
        if shown:
            for field_name, value in sorted(live):
                if not _reweightable(field_name):
                    continue
                present = sum(
                    1 for asin in shown if value in set(fact_lookup(asin, field_name))
                )
                if present / len(shown) < config.neg_common_frac:
                    continue
                key = f"{field_name}:{value}"
                never_reinforced = (
                    (field_name, value) not in reinforced_set
                    and _prior_mentions(trace, key) == 0
                )
                weight = new.get(field_name, {}).get(value, 1.0) - config.neg_decay
                if never_reinforced:
                    weight -= config.oneoff_extra_penalty
                new.setdefault(field_name, {})[value] = _clamp(weight, config)

    # (d) clamp + prune: keep only live pairs whose factor is off unity, so the
    # sidecar stays a sparse strict subset of the active constraints.
    pruned: dict[str, dict[str, float]] = {}
    for field_name, values in new.items():
        kept = {
            value: _clamp(weight, config)
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
        factors[field_name] = _clamp(mean, config)
    return factors


def _active_values(constraints: object) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for field_name, value in sorted(constraint_pairs(constraints)):
        grouped.setdefault(field_name, []).append(value)
    return list(grouped.items())


__all__ = ["distill", "field_factors", "FactLookup", "PriorFactor"]
