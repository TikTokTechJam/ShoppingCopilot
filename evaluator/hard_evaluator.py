from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from starter.agent import Agent


MAX_TURNS = 10
TOP_K = 10

SCENARIO_COUNTS = {
    "buying": 160,
    "browsing": 160,
    "intent_override": 60,
    "boundary": 20,
}

ALLOWED_ATTRIBUTES = {
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(row)
    return rows


def load_catalog_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product.get("parent_asin", "")).strip()
            if not parent_asin:
                raise ValueError(f"{path}:{line_number}: missing parent_asin")
            ids.add(parent_asin)
    return ids


def fact_id(fact: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if not fact:
        return None
    attribute = str(fact.get("attribute", "")).strip()
    canonical = str(fact.get("canonical", "")).strip()
    if not attribute or not canonical:
        return None
    return attribute, canonical


def parse_fact_id(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid fact id: {value!r}")
    attribute = str(value[0]).strip()
    canonical = str(value[1]).strip()
    if not attribute or not canonical:
        raise ValueError(f"invalid fact id: {value!r}")
    return attribute, canonical


def validate_sessions(
    sessions: Iterable[Mapping[str, Any]],
    catalog_ids: set[str],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in sessions]

    if len(rows) != 400:
        raise ValueError(f"expected exactly 400 sessions, got {len(rows)}")

    scenario_counts = Counter(str(row.get("scenario_type", "")) for row in rows)
    if dict(scenario_counts) != SCENARIO_COUNTS:
        raise ValueError(
            f"scenario counts mismatch: got {dict(scenario_counts)}, "
            f"expected {SCENARIO_COUNTS}"
        )

    sample_ids: set[str] = set()
    targets: set[str] = set()

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        sample_ids.add(sample_id)

        target = str(row.get("target_asin", "")).strip()
        if not target or target in targets:
            raise ValueError(f"{sample_id}: duplicate or empty target_asin: {target!r}")
        if target not in catalog_ids:
            raise ValueError(f"{sample_id}: target_asin not found in catalog: {target}")
        targets.add(target)

        scenario = str(row.get("scenario_type", ""))
        hidden = row.get("hidden_facts")
        if not isinstance(hidden, list) or not 2 <= len(hidden) <= 4:
            raise ValueError(f"{sample_id}: hidden_facts must contain 2-4 facts")

        hidden_ids: set[tuple[str, str]] = set()
        hidden_attributes: set[str] = set()
        for fact in hidden:
            if not isinstance(fact, dict):
                raise ValueError(f"{sample_id}: malformed hidden fact")
            fid = fact_id(fact)
            if fid is None:
                raise ValueError(f"{sample_id}: hidden fact missing attribute/canonical")
            if fid in hidden_ids:
                raise ValueError(f"{sample_id}: duplicate hidden fact {fid}")
            hidden_ids.add(fid)

            attribute = str(fact.get("attribute", ""))
            if attribute not in ALLOWED_ATTRIBUTES - {"other", "category"}:
                raise ValueError(f"{sample_id}: unsupported hidden attribute {attribute!r}")
            if attribute in hidden_attributes:
                raise ValueError(
                    f"{sample_id}: duplicate hidden attribute {attribute!r}"
                )
            hidden_attributes.add(attribute)

            if not str(fact.get("display", "")).strip():
                raise ValueError(f"{sample_id}: hidden fact missing display")
            if not str(fact.get("evidence_field", "")).strip():
                raise ValueError(f"{sample_id}: hidden fact missing evidence_field")
            if not str(fact.get("evidence_text", "")).strip():
                raise ValueError(f"{sample_id}: hidden fact missing evidence_text")

        initial_message = str(row.get("initial_message", "")).strip()
        if not initial_message:
            raise ValueError(f"{sample_id}: missing initial_message")

        initial_id = parse_fact_id(row.get("initial_fact_id"))
        if initial_id is not None and initial_id not in hidden_ids:
            raise ValueError(f"{sample_id}: initial_fact_id is outside hidden_facts")

        if scenario in {"browsing", "boundary"} and initial_id is not None:
            raise ValueError(
                f"{sample_id}: {scenario} session must not silently disclose an initial fact"
            )

        if scenario in {"buying", "intent_override"} and initial_id is None:
            raise ValueError(f"{sample_id}: {scenario} session needs initial_fact_id")

        if scenario == "intent_override":
            override_turn = row.get("override_turn")
            if override_turn not in {3, 4}:
                raise ValueError(f"{sample_id}: override_turn must be 3 or 4")

            override_id = parse_fact_id(row.get("override_fact_id"))
            if override_id is None or override_id not in hidden_ids:
                raise ValueError(f"{sample_id}: invalid override_fact_id")
            if override_id == initial_id:
                raise ValueError(f"{sample_id}: override fact must differ from initial fact")
            if initial_id is not None and override_id[0] == initial_id[0]:
                raise ValueError(
                    f"{sample_id}: override should use a different attribute"
                )
            if not str(row.get("override_message", "")).strip():
                raise ValueError(f"{sample_id}: missing override_message")
        else:
            if row.get("override_turn") is not None:
                raise ValueError(f"{sample_id}: non-override session has override_turn")
            if row.get("override_fact_id") is not None:
                raise ValueError(f"{sample_id}: non-override session has override_fact_id")
            if row.get("override_message") is not None:
                raise ValueError(f"{sample_id}: non-override session has override_message")

    return rows


def validate_agent_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise TypeError("Agent response must be an object")

    required = {"message", "ask_attribute", "recommendations"}
    allowed = required | {"usage"}

    if not required.issubset(response):
        missing = sorted(required - set(response))
        raise ValueError(f"Agent response missing fields: {missing}")

    unknown = set(response) - allowed
    if unknown:
        raise ValueError(f"Agent response has unknown fields: {sorted(unknown)}")

    if not isinstance(response["message"], str):
        raise TypeError("response.message must be a string")

    ask_attribute = response["ask_attribute"]
    if ask_attribute is not None:
        if not isinstance(ask_attribute, str) or ask_attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"invalid response.ask_attribute: {ask_attribute!r}")

    recommendations = response["recommendations"]
    if not isinstance(recommendations, list):
        raise TypeError("response.recommendations must be a list")
    if len(recommendations) > 100:
        raise ValueError("response.recommendations may contain at most 100 items")

    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise TypeError("each recommendation must be an object")

        unknown_rec = set(recommendation) - {"parent_asin", "score"}
        if unknown_rec:
            raise ValueError(
                f"recommendation has unknown fields: {sorted(unknown_rec)}"
            )

        parent_asin = recommendation.get("parent_asin")
        if not isinstance(parent_asin, str) or not parent_asin.strip():
            raise TypeError("each recommendation needs a non-empty parent_asin")

        if "score" in recommendation:
            score = recommendation["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("recommendation.score must be numeric")
            if not math.isfinite(float(score)):
                raise ValueError("recommendation.score must be finite")

    if "usage" in response:
        usage = response["usage"]
        if not isinstance(usage, dict):
            raise TypeError("response.usage must be an object")
        if set(usage) != {"prompt_tokens", "completion_tokens"}:
            raise ValueError(
                "response.usage must contain exactly prompt_tokens and completion_tokens"
            )
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"response.usage.{key} must be a non-negative integer")

    return response


def normalize_recommendations(
    recommendations: Any,
    catalog_ids: set[str],
) -> list[str]:
    if not isinstance(recommendations, list):
        return []

    result: list[str] = []
    seen: set[str] = set()

    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue

        parent_asin = str(recommendation.get("parent_asin", "")).strip()
        if not parent_asin or parent_asin not in catalog_ids or parent_asin in seen:
            continue

        seen.add(parent_asin)
        result.append(parent_asin)

        if len(result) == TOP_K:
            break

    return result


def customer_sentence(fact: Mapping[str, Any]) -> str:
    attribute = str(fact["attribute"])
    value = str(fact["display"]).strip()

    if attribute == "material":
        return f"I'd prefer something made from {value}."
    if attribute == "color":
        return f"I'd prefer {value}."
    if attribute == "size":
        return f"For size or fit, I need {value}."
    if attribute == "style":
        return f"For style, I prefer {value}."
    if attribute == "brand":
        return f"I'd prefer the brand {value}."
    if attribute == "budget":
        return f"For budget, I'd like {value}."
    if attribute == "feature":
        return f"A feature that matters to me is {value}."
    if attribute == "use_case":
        return f"I'd mainly use it for {value}."

    return "I don't have an additional preference there."


def simulate_customer_reply(
    session: Mapping[str, Any],
    ask_attribute: Any,
    state: dict[str, Any],
    rng: random.Random,
) -> str:
    if ask_attribute is None:
        return "Those options are not quite right yet. You can ask me about one specific attribute."

    if not isinstance(ask_attribute, str) or ask_attribute not in ALLOWED_ATTRIBUTES:
        return "I don't have an additional preference there."

    # `other` is intentionally NOT a wildcard in this benchmark.
    if ask_attribute in {"other", "category"}:
        return "I don't have a specific additional preference there."

    # Boundary behavior: only sessions explicitly marked boundary_first reject
    # the first meaningful clarification.
    if (
        session.get("scenario_type") == "boundary"
        and bool(session.get("boundary_first"))
        and not bool(state["boundary_used"])
    ):
        state["boundary_used"] = True
        state["no_preference_attributes"].add(ask_attribute)
        return f"I don't really have a preference for {ask_attribute}."

    disclosed: set[tuple[str, str]] = state["disclosed"]
    stale: set[tuple[str, str]] = state["stale_constraints"]

    candidates: list[dict[str, Any]] = []
    for fact in session["hidden_facts"]:
        fid = fact_id(fact)
        if (
            str(fact["attribute"]) == ask_attribute
            and fid is not None
            and fid not in disclosed
            and fid not in stale
        ):
            candidates.append(fact)

    if not candidates:
        state["no_preference_attributes"].add(ask_attribute)
        return f"I don't have an additional preference for {ask_attribute}."

    # There is normally at most one hidden fact per attribute. Keep deterministic
    # random selection anyway so the evaluator remains robust if that changes.
    chosen = rng.choice(candidates)
    chosen_id = fact_id(chosen)

    if chosen_id is not None:
        disclosed.add(chosen_id)
        state["active_constraints"].add(chosen_id)

    return customer_sentence(chosen)


def metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
        }

    hit_rate = sum(bool(row["hit"]) for row in results) / len(results)
    mrr = statistics.fmean(float(row["reciprocal_rank"]) for row in results)
    mttc = statistics.fmean(
        int(row["first_hit_turn"])
        if row["first_hit_turn"] is not None
        else MAX_TURNS + 1
        for row in results
    )

    return {
        "sample_count": len(results),
        "hit_rate_at_10": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
    }


def add_score_fields(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary["mttc"] is None:
        efficiency = 0.0
    else:
        efficiency = max(
            0.0,
            min(1.0, (11.0 - float(summary["mttc"])) / 10.0),
        )

    technical_score = (
        0.50 * float(summary["hit_rate_at_10"])
        + 0.30 * float(summary["mrr"])
        + 0.20 * efficiency
    )

    return {
        **summary,
        "efficiency": efficiency,
        "technical_score": technical_score,
    }


def evaluate(
    agent: Any,
    sessions: Iterable[Mapping[str, Any]],
    catalog_ids: set[str],
    strict: bool = True,
) -> dict[str, Any]:
    rows = validate_sessions(sessions, catalog_ids)

    results: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0

    for session in rows:
        sample_id = str(session["sample_id"])
        target = str(session["target_asin"])
        scenario = str(session["scenario_type"])
        session_id = f"manual400:{sample_id}"

        agent.reset(session_id, dict(session.get("user_profile") or {}))

        rng = random.Random(f"manual400-reply:{sample_id}:{target}")

        initial_id = parse_fact_id(session.get("initial_fact_id"))

        state: dict[str, Any] = {
            "disclosed": {initial_id} if initial_id is not None else set(),
            "active_constraints": {initial_id} if initial_id is not None else set(),
            "stale_constraints": set(),
            "no_preference_attributes": set(),
            "boundary_used": False,
        }

        user_message = str(session["initial_message"])

        # Non-override sessions are scoreable immediately.
        override_applied = scenario != "intent_override"

        first_hit_turn: int | None = None
        best_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            # The override message is what the Agent sees on the configured
            # override turn (3 or 4). The old initial preference becomes stale.
            if not override_applied and turn == int(session["override_turn"]):
                override_applied = True

                if initial_id is not None:
                    state["stale_constraints"].add(initial_id)
                    state["active_constraints"].discard(initial_id)

                override_id = parse_fact_id(session.get("override_fact_id"))
                if override_id is not None:
                    state["disclosed"].add(override_id)
                    state["active_constraints"].add(override_id)

                user_message = str(session["override_message"])

            try:
                raw_response = agent.respond(
                    session_id,
                    user_message,
                    turn,
                    TOP_K,
                )
                response = validate_agent_response(raw_response)
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"Agent failed validation in {sample_id} on turn {turn}"
                    ) from exc
                response = {
                    "message": "",
                    "ask_attribute": None,
                    "recommendations": [],
                }

            usage = response.get("usage")
            if isinstance(usage, dict):
                prompt_tokens += int(usage["prompt_tokens"])
                completion_tokens += int(usage["completion_tokens"])

            ranked = normalize_recommendations(
                response.get("recommendations"),
                catalog_ids,
            )

            # Intent Override sessions cannot score before the override.
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                first_hit_turn = turn
                break

            if turn == MAX_TURNS:
                break

            # If next turn is the override turn, do not also fabricate a normal
            # clarification response. The override itself becomes the next
            # customer message.
            if (
                not override_applied
                and turn + 1 == int(session["override_turn"])
            ):
                continue

            user_message = simulate_customer_reply(
                session,
                response.get("ask_attribute"),
                state,
                rng,
            )

        results.append(
            {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "target_asin": target,
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": (
                    0.0 if best_rank is None else 1.0 / best_rank
                ),
            }
        )

    overall = add_score_fields(metric_summary(results))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["scenario_type"])].append(row)

    scenario_metrics = {
        scenario: add_score_fields(metric_summary(grouped[scenario]))
        for scenario in SCENARIO_COUNTS
    }

    return {
        **overall,
        "scenario_metrics": scenario_metrics,
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "sessions": results,
    }


def rounded_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    def clean_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "sample_count": metrics["sample_count"],
            "hit_rate_at_10": round(float(metrics["hit_rate_at_10"]), 6),
            "mrr": round(float(metrics["mrr"]), 6),
            "mttc": (
                None
                if metrics["mttc"] is None
                else round(float(metrics["mttc"]), 6)
            ),
            "efficiency": round(float(metrics["efficiency"]), 6),
            "technical_score": round(float(metrics["technical_score"]), 6),
        }

    return {
        **clean_metrics(result),
        "scenario_metrics": {
            scenario: clean_metrics(metrics)
            for scenario, metrics in result["scenario_metrics"].items()
        },
        "reported_token_usage": dict(result["reported_token_usage"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate starter.agent.Agent against the fixed GPTAnnotation sessions."
    )
    parser.add_argument(
        "--catalog",
        default="data/catalog.jsonl",
        help="Path to the full 50k catalog JSONL.",
    )
    parser.add_argument(
        "--sessions",
        default="data/derived/gptannotation/sessions.jsonl",
        help="Path to the fixed 400-session JSONL benchmark.",
    )
    parser.add_argument(
        "--output",
        default="results_gptannotation.json",
        help="Where to write detailed results.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Treat malformed Agent responses as empty instead of failing.",
    )
    args = parser.parse_args()

    sessions = load_jsonl(args.sessions)
    catalog_ids = load_catalog_ids(args.catalog)

    agent = Agent(args.catalog)

    result = evaluate(
        agent=agent,
        sessions=sessions,
        catalog_ids=catalog_ids,
        strict=not args.non_strict,
    )

    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(rounded_summary(result), indent=2))


if __name__ == "__main__":
    main()
