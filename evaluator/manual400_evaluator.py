from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from starter.agent import Agent


ROOT = Path(__file__).parents[1]
DEFAULT_CATALOG = ROOT / "data" / "catalog.jsonl"
DEFAULT_DATASET = ROOT / "data" / "derived" / "manual400" / "sessions.jsonl"
MAX_TURNS = 10
TOP_K = 10
SCENARIO_COUNTS = {"buying": 160, "browsing": 160, "intent_override": 60, "boundary": 20}
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other",
}
VALID_ASK_ATTRIBUTES = ALLOWED_ATTRIBUTES | {None}
FACT_PHRASES = {
    "waterproof_protection": "waterproof protection",
    "water_resistance": "water resistance",
    "moisture_wicking": "moisture-wicking performance",
    "quick_drying": "quick-drying performance",
    "breathability": "good breathability",
    "lightweight": "a lightweight design",
    "hypoallergenic": "hypoallergenic materials",
    "memory_foam": "memory-foam cushioning",
    "arch_support": "arch support",
    "slip_resistance": "slip resistance",
    "thermal_insulation": "thermal insulation",
    "uv_protection": "UV protection",
    "stretch": "stretch fabric",
    "adjustable": "adjustability",
    "removable": "removable components",
    "pockets": "pockets",
    "zipper": "a zipper closure",
    "buckle": "a buckle closure",
    "soft_fabric": "soft fabric",
}
METADATA_RE = re.compile(
    r"date first available|item model number|product dimensions|package weight|asin",
    re.IGNORECASE,
)
ASIN_RE = re.compile(r"\bB0[0-9A-Z]{8}\b", re.IGNORECASE)


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_catalog_ids(path: str | Path) -> set[str]:
    identifiers: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                identifiers.add(str(json.loads(line)["parent_asin"]))
    return identifiers


def fact_id(fact: Mapping[str, object] | None) -> tuple[str, str] | None:
    if not fact:
        return None
    return str(fact.get("attribute", "")), str(fact.get("canonical", ""))


def session_fact_ids(session: Mapping[str, object]) -> set[tuple[str, str]]:
    return {
        key for key in (fact_id(fact) for fact in session.get("hidden_facts", ())) if key is not None
    }


def _id_from_field(value: object) -> tuple[str, str] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return str(value[0]), str(value[1])


def fact_visible_in_message(fact: Mapping[str, object], message: str) -> bool:
    normalized_message = re.sub(r"\s+", " ", message.lower().replace("-", " "))
    terms = {
        str(fact.get("display", "")),
        str(fact.get("canonical", "")).replace("_", " "),
        FACT_PHRASES.get(str(fact.get("canonical")), ""),
    }
    return any(
        term.strip()
        and len(term.strip()) >= 3
        and re.sub(r"\s+", " ", term.lower().replace("-", " ")) in normalized_message
        for term in terms
    )


def effective_initial_fact_id(session: Mapping[str, object]) -> tuple[str, str] | None:
    raw_id = _id_from_field(session.get("initial_fact_id"))
    if raw_id is None or str(session.get("scenario_type", "")) == "browsing":
        return None
    initial_message = str(session.get("initial_message", ""))
    for fact in session.get("hidden_facts", ()):
        if isinstance(fact, dict) and fact_id(fact) == raw_id and fact_visible_in_message(fact, initial_message):
            return raw_id
    return None


def validate_sessions(sessions: Iterable[Mapping[str, object]], expected_count: int | None = 400) -> list[dict[str, object]]:
    rows = [dict(session) for session in sessions]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"manual400 requires {expected_count} sessions, got {len(rows)}")
    counts = defaultdict(int)
    targets: set[str] = set()
    for row in rows:
        scenario = str(row.get("scenario_type", ""))
        counts[scenario] += 1
        target = str(row.get("target_asin", ""))
        if not target or target in targets:
            raise ValueError(f"duplicate or empty target_asin: {target}")
        targets.add(target)
        hidden = row.get("hidden_facts")
        if not isinstance(hidden, list) or not 2 <= len(hidden) <= 4:
            raise ValueError(f"invalid hidden card for {row.get('sample_id')}")
        hidden_ids = session_fact_ids(row)
        if len(hidden_ids) != len(hidden):
            raise ValueError(f"duplicate hidden fact for {row.get('sample_id')}")
        for fact in hidden:
            if not isinstance(fact, dict) or not fact_id(fact):
                raise ValueError(f"malformed hidden fact for {row.get('sample_id')}")
            if not fact.get("evidence_field") or not fact.get("evidence_text"):
                raise ValueError(f"hidden fact lacks evidence for {row.get('sample_id')}")
        initial_id = _id_from_field(row.get("initial_fact_id"))
        if initial_id is not None and initial_id not in hidden_ids:
            raise ValueError(f"initial fact is outside hidden card for {row.get('sample_id')}")
        effective_initial_id = effective_initial_fact_id(row)
        if scenario in {"buying", "intent_override"} and initial_id is not None and effective_initial_id is None:
            raise ValueError(f"initial fact is not disclosed in the initial message for {row.get('sample_id')}")
        if scenario == "boundary" and not isinstance(row.get("boundary_first"), bool):
            raise ValueError(f"boundary_first must be boolean for {row.get('sample_id')}")
        initial_message = str(row.get("initial_message", ""))
        override_message = str(row.get("override_message") or "")
        for visible in (initial_message, override_message):
            if ASIN_RE.search(visible) or METADATA_RE.search(visible):
                raise ValueError(f"raw identifier metadata in visible text for {row.get('sample_id')}")
        if scenario == "intent_override":
            if int(row.get("override_turn", 0)) not in {3, 4}:
                raise ValueError(f"invalid override turn for {row.get('sample_id')}")
            override_id = _id_from_field(row.get("override_fact_id"))
            if override_id is None or override_id not in hidden_ids or override_id == initial_id:
                raise ValueError(f"invalid override fact for {row.get('sample_id')}")
            if not override_message:
                raise ValueError(f"missing override message for {row.get('sample_id')}")
        elif row.get("override_turn") is not None or row.get("override_fact_id") is not None:
            raise ValueError(f"non-override session has override state: {row.get('sample_id')}")
    if expected_count == 400 and dict(counts) != SCENARIO_COUNTS:
        raise ValueError(f"scenario counts are {dict(counts)}, expected {SCENARIO_COUNTS}")
    return rows


def validate_response(response: object) -> dict[str, object]:
    if not isinstance(response, dict):
        raise TypeError("Agent response must be an object")
    required = {"message", "ask_attribute", "recommendations"}
    allowed = required | {"usage"}
    if not required.issubset(response) or not set(response).issubset(allowed):
        raise ValueError("Agent response does not match the required response shape")
    if not isinstance(response["message"], str):
        raise TypeError("response.message must be a string")
    ask_attribute = response["ask_attribute"]
    if ask_attribute is not None and (not isinstance(ask_attribute, str) or ask_attribute not in ALLOWED_ATTRIBUTES):
        raise ValueError(f"invalid response.ask_attribute: {ask_attribute}")
    recommendations = response["recommendations"]
    if not isinstance(recommendations, list) or len(recommendations) > TOP_K:
        raise TypeError(f"response.recommendations must be a list with at most {TOP_K} items")
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise TypeError("each recommendation must be an object")
        if set(recommendation) - {"parent_asin", "score"}:
            raise ValueError("recommendation has an unknown field")
        if not isinstance(recommendation.get("parent_asin"), str) or not recommendation["parent_asin"].strip():
            raise TypeError("each recommendation needs a non-empty string parent_asin")
        if "score" in recommendation:
            score = recommendation["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("recommendation.score must be numeric")
            if not math.isfinite(float(score)):
                raise ValueError("recommendation.score must be finite")
    if "usage" in response:
        usage = response["usage"]
        if not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens"}:
            raise ValueError("response.usage must contain exactly prompt_tokens and completion_tokens")
        for name in ("prompt_tokens", "completion_tokens"):
            if isinstance(usage[name], bool) or not isinstance(usage[name], int) or usage[name] < 0:
                raise TypeError(f"response.usage.{name} must be a non-negative integer")
    return response


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        parent_asin = str(item.get("parent_asin", "")).strip() if isinstance(item, dict) else ""
        if parent_asin and parent_asin in catalog_ids and parent_asin not in seen:
            result.append(parent_asin)
            seen.add(parent_asin)
        if len(result) == TOP_K:
            break
    return result


def _fact_value(fact: Mapping[str, object]) -> str:
    return FACT_PHRASES.get(str(fact.get("canonical")), str(fact.get("display", "")))


def _customer_sentence(fact: Mapping[str, object]) -> str:
    value = _fact_value(fact)
    attribute = str(fact.get("attribute"))
    if attribute == "material":
        return f"I'd prefer something made from {value}."
    if attribute == "color":
        return f"I'd like it in {value}."
    if attribute == "style":
        return f"I prefer a {value} look."
    if attribute == "use_case":
        return f"I'd mainly use it for {value}."
    if attribute == "size":
        return f"I need a {value} fit."
    if attribute == "budget":
        return f"I'd like to stay around {value}."
    if attribute == "brand":
        return f"I'd prefer {value}."
    if attribute == "feature":
        return f"I'd like something with {value}."
    return "I don't have a specific preference for another attribute."


def simulate_customer_reply(
    session: Mapping[str, object],
    ask_attribute: object,
    state: dict[str, object],
    rng: random.Random,
) -> str:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if attribute == "other" or attribute == "category":
        return "I don't have a specific preference there; please use your judgment."
    if attribute not in ALLOWED_ATTRIBUTES:
        return "I don't have an additional preference yet."
    if (
        session.get("scenario_type") == "boundary"
        and session.get("boundary_first")
        and not state.get("boundary_used")
    ):
        state["boundary_used"] = True
        state.setdefault("no_preference_attributes", set()).add(attribute)
        return f"I don't really have a preference for {attribute} there."
    disclosed = state.setdefault("disclosed", set())
    stale = state.setdefault("stale_constraints", set())
    if not isinstance(disclosed, set) or not isinstance(stale, set):
        raise TypeError("evaluator state sets are malformed")
    candidates = [
        fact for fact in session.get("hidden_facts", ())
        if isinstance(fact, dict)
        and str(fact.get("attribute")) == attribute
        and fact_id(fact) not in disclosed
        and fact_id(fact) not in stale
    ]
    if not candidates:
        state.setdefault("no_preference_attributes", set()).add(attribute)
        return f"I don't have an additional preference for {attribute}."
    chosen = rng.choice(candidates)
    chosen_id = fact_id(chosen)
    if chosen_id is not None:
        disclosed.add(chosen_id)
        state.setdefault("active_constraints", set()).add(chosen_id)
    return _customer_sentence(chosen)


def metric_summary(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
    hit_rate = sum(int(result["hit"]) for result in results) / len(results)
    mrr = statistics.fmean(float(result["reciprocal_rank"]) for result in results)
    mttc = statistics.fmean(
        int(result["first_hit_turn"]) if result["first_hit_turn"] is not None else MAX_TURNS + 1
        for result in results
    )
    return {"sample_count": len(results), "hit_rate_at_10": hit_rate, "mrr": mrr, "mttc": mttc}


def score_metrics(summary: Mapping[str, object]) -> tuple[float, float]:
    mttc = float(summary["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * float(summary["hit_rate_at_10"]) + 0.30 * float(summary["mrr"]) + 0.20 * efficiency
    return efficiency, technical_score


def evaluate(
    agent: object,
    sessions: Iterable[Mapping[str, object]],
    catalog_ids: set[str],
    strict: bool = True,
    expected_count: int | None = 400,
) -> dict[str, object]:
    rows = validate_sessions(sessions, expected_count=expected_count)
    missing_targets = {str(row["target_asin"]) for row in rows} - catalog_ids
    if missing_targets:
        raise ValueError(f"target ASINs are absent from catalog: {sorted(missing_targets)[:3]}")
    results: list[dict[str, object]] = []
    prompt_tokens = 0
    completion_tokens = 0
    for session in rows:
        sample_id = str(session["sample_id"])
        target = str(session["target_asin"])
        session_id = f"manual400:{sample_id}"
        agent.reset(session_id, dict(session.get("user_profile") or {}))
        rng = random.Random(f"reply:{sample_id}:{target}")
        initial_id = effective_initial_fact_id(session)
        state: dict[str, object] = {
            "disclosed": {initial_id} if initial_id else set(),
            "active_constraints": {initial_id} if initial_id else set(),
            "stale_constraints": set(),
            "no_preference_attributes": set(),
            "boundary_used": False,
        }
        user_message = str(session["initial_message"])
        override_applied = str(session["scenario_type"]) != "intent_override"
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            if not override_applied and turn == int(session["override_turn"]):
                override_applied = True
                if initial_id:
                    state["stale_constraints"].add(initial_id)
                override_id = _id_from_field(session.get("override_fact_id"))
                state["active_constraints"] = {override_id} if override_id else set()
                if override_id:
                    state["disclosed"].add(override_id)
                user_message = str(session["override_message"])
            try:
                response = validate_response(agent.respond(session_id, user_message, turn, TOP_K))
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"{sample_id}, turn={turn}") from exc
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            usage = response.get("usage")
            if isinstance(usage, dict):
                prompt_tokens += int(usage["prompt_tokens"])
                completion_tokens += int(usage["completion_tokens"])
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            if not override_applied and turn + 1 == int(session["override_turn"]):
                continue
            user_message = simulate_customer_reply(session, response.get("ask_attribute"), state, rng)
        results.append({
            "sample_id": sample_id,
            "scenario_type": session["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    overall = metric_summary(results)
    efficiency, technical_score = score_metrics(overall)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["scenario_type"])].append(result)
    return {
        **overall,
        "efficiency": efficiency,
        "recommended_technical_score": technical_score,
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "scenario_metrics": {name: metric_summary(items) for name, items in sorted(grouped.items())},
        "sessions": results,
    }


def display_summary(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "sample_count": result["sample_count"],
        "HitRate@10": round(float(result["hit_rate_at_10"]), 6),
        "MRR": round(float(result["mrr"]), 6),
        "MTTC": round(float(result["mttc"]), 6),
        "Efficiency": round(float(result["efficiency"]), 6),
        "TechnicalScore": round(float(result["recommended_technical_score"]), 6),
        "scenario_metrics": result["scenario_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the evidence-backed TechJam manual400 benchmark")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="results_manual400.json")
    parser.add_argument("--lenient", action="store_true", help="replace malformed Agent responses with empty responses")
    args = parser.parse_args()
    sessions = load_jsonl(args.dataset)
    catalog_ids = load_catalog_ids(args.catalog)
    result = evaluate(Agent(args.catalog), sessions, catalog_ids, strict=not args.lenient)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(display_summary(result), indent=2))


if __name__ == "__main__":
    main()
