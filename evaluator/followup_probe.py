"""Measure the follow-up strategy's `gamma` (Architecture.md Section 12b.6).

The scoring evaluator stops at the first hit, so the target's rank trajectory
after that turn is unobserved and `gamma` -- the probability that a withheld
item is re-retrieved and ranked first on the next turn -- cannot be read off an
ordinary run.

This replay continues the session past a hit and records the target's rank on
every turn.  From the resulting trajectories it computes the empirical
promotion probability and, by backward induction over the observed ranks, the
value an oracle stopping rule would have achieved.  The gap between the current
always-recommend policy and that oracle is the entire upside of a wait branch.

This is a diagnostic outside the Agent boundary.  It reads hidden targets, so
nothing here may be imported by Agent code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import uuid
from collections import Counter
from pathlib import Path

from evaluator import local_evaluator as le
from starter.followup import MAX_TURNS, promotion_threshold, utility


def _trajectory(
    agent, sample, catalog_ids, categories, products, *, retain_shown: bool = False
) -> dict:
    """Run one full session without stopping at the first hit.

    ``retain_shown`` models the withholding counterfactual.  The Agent excludes
    everything it has already recommended, so an item that was *shown* can
    never reappear -- which would force gamma to zero by construction.  An item
    that is *withheld* was never shown and so was never excluded, and clearing
    the exclusion set each turn is what reproduces that.
    """
    session_id = f"probe_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = le.materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = le.initial_message(
        effective, le.coarse_category(categories.get(target, [])), disclosed
    )

    ranks: list[int | None] = []
    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(session_id, message, turn, le.TOP_K)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict):
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        ranked = le.normalize_recommendations(response.get("recommendations"), catalog_ids)
        rank = ranked.index(target) + 1 if target in ranked else None
        # A hit before the override turn is not scoreable, exactly as the
        # evaluator treats it.
        ranks.append(rank if override_applied else None)
        if retain_shown:
            try:
                state = agent.sessions.get(session_id)
                state.excluded_recommendations.clear()
                # last_recommendations is promoted into the exclusion set at
                # the start of the next turn, so it must be cleared too.
                state.last_recommendations = ()
            except (AttributeError, KeyError):
                pass
        if turn == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, ignore my earlier preference."))
        else:
            message, boundary_used = le.customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )
    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "ranks": ranks,
    }


def _greedy_value(ranks: list[int | None]) -> float:
    """Score of the current policy: stop at the first hit."""
    for turn, rank in enumerate(ranks, start=1):
        if rank is not None:
            return utility(turn, rank)
    return 0.0


def _oracle_value(ranks: list[int | None]) -> tuple[float, int | None]:
    """Best achievable score with perfect foresight over this trajectory."""
    best, best_turn = 0.0, None
    for turn, rank in enumerate(ranks, start=1):
        if rank is None:
            continue
        value = utility(turn, rank)
        if value > best:
            best, best_turn = value, turn
    return best, best_turn


def _promotion_stats(trajectories: list[dict]) -> dict:
    """Empirical gamma: P(rank 1 next turn | shown at rank j this turn)."""
    shown: Counter[int] = Counter()
    promoted: Counter[int] = Counter()
    improved: Counter[int] = Counter()
    for item in trajectories:
        ranks = item["ranks"]
        for turn in range(len(ranks) - 1):
            here, nxt = ranks[turn], ranks[turn + 1]
            if here is None:
                continue
            shown[here] += 1
            if nxt == 1:
                promoted[here] += 1
            if nxt is not None and nxt < here:
                improved[here] += 1
    return {
        str(rank): {
            "observations": shown[rank],
            "gamma_to_rank_1": round(promoted[rank] / shown[rank], 4),
            "any_improvement": round(improved[rank] / shown[rank], 4),
            "threshold_at_turn_1": round(promotion_threshold(1, rank), 4),
            "withholding_justified": promoted[rank] / shown[rank]
            > promotion_threshold(1, rank),
        }
        for rank in sorted(shown)
    }


def run(agent, samples, catalog_ids, categories, products, *, retain_shown=False) -> dict:
    trajectories = [
        _trajectory(
            agent, sample, catalog_ids, categories, products, retain_shown=retain_shown
        )
        for sample in samples
    ]
    greedy = [_greedy_value(item["ranks"]) for item in trajectories]
    oracle = [_oracle_value(item["ranks"]) for item in trajectories]
    gains = [o - g for (o, _), g in zip(oracle, greedy)]
    return {
        "sessions": len(trajectories),
        "retain_shown": retain_shown,
        "always_recommend_score": round(statistics.fmean(greedy), 6),
        "oracle_stopping_score": round(statistics.fmean(value for value, _ in oracle), 6),
        "max_wait_branch_upside": round(statistics.fmean(gains), 6),
        "sessions_where_waiting_helps": sum(1 for gain in gains if gain > 1e-9),
        "promotion_by_rank": _promotion_stats(trajectories),
        "trajectories": trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="followup_probe.json")
    parser.add_argument("--disable-layer2", action="store_true")
    parser.add_argument(
        "--retain-shown",
        action="store_true",
        help="Clear already-shown exclusions each turn to measure the true "
             "withholding counterfactual.",
    )
    parser.add_argument("--layer2-artifact-dir")
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    args = parser.parse_args()

    from evaluator.agent_factory import build_evaluator_agent

    samples = le.load_jsonl(args.dataset)
    catalog_ids, categories, products = le.catalog_index(args.catalog)
    agent = build_evaluator_agent(
        args.catalog,
        layer2_artifact_dir=args.layer2_artifact_dir,
        embedding_model=args.embedding_model,
        disable_layer2=args.disable_layer2,
        device=args.device,
    )
    result = run(
        agent, samples, catalog_ids, categories, products,
        retain_shown=args.retain_shown,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "trajectories"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
