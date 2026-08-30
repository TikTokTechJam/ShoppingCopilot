from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluator.agent_factory import build_evaluator_agent, warm_evaluator_runtime
from starter.agent import Agent
from starter.retrieval import MODE_SCORE_WEIGHTS, STRUCTURED_FIELD_WEIGHTS


MAX_TURNS = 10
TOP_K = 10

# How often the no-tqdm fallback reports, in sessions.
PROGRESS_INTERVAL = 20
DEFAULT_CONCURRENCY = 4

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


def select_sessions(
    sessions: Iterable[Mapping[str, Any]],
    *,
    override_only: bool = False,
) -> list[dict[str, Any]]:
    """Select the benchmark rows to execute after full-set validation."""

    rows = [dict(row) for row in sessions]
    if not override_only:
        return rows
    return [
        row for row in rows if str(row.get("scenario_type", "")) == "intent_override"
    ]


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

    if ask_attribute == "category":
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
            (ask_attribute == "other" or str(fact["attribute"]) == ask_attribute)
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


class _Progress:
    """Session counter for long runs.

    Writes to stderr only, so the result JSON on stdout stays machine-readable
    and can still be piped. ``tqdm`` is optional: a missing dependency degrades
    to periodic lines rather than failing an evaluation that was going to take
    minutes anyway.
    """

    def __init__(self, total: int, enabled: bool = False) -> None:
        self.total = int(total)
        self.enabled = bool(enabled)
        self.done = 0
        self.hits = 0
        self.bar: Any | None = None
        if not self.enabled:
            return
        try:
            from tqdm import tqdm
        except ImportError:
            print(
                f"[hard_evaluator] tqdm not installed; reporting every "
                f"{PROGRESS_INTERVAL} sessions",
                file=sys.stderr,
                flush=True,
            )
        else:
            self.bar = tqdm(
                total=self.total,
                desc="sessions",
                unit="session",
                file=sys.stderr,
                dynamic_ncols=True,
            )

    def advance(self, hit: bool) -> None:
        if not self.enabled:
            return
        self.done += 1
        self.hits += bool(hit)
        rate = self.hits / self.done
        if self.bar is not None:
            self.bar.set_postfix_str(f"hit@10={rate:.3f}", refresh=False)
            self.bar.update(1)
        elif self.done % PROGRESS_INTERVAL == 0 or self.done == self.total:
            print(
                f"[hard_evaluator] {self.done}/{self.total} sessions  "
                f"hit@10={rate:.3f}",
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class Manual400SessionRunner:
    """Run one fixed benchmark session one evaluator turn at a time.

    This is the shared execution path for the batch evaluator and local debug
    tooling.  The target is retained here only for evaluator-side scoring; it
    is never passed to the Agent.
    """

    def __init__(
        self,
        agent: Any,
        session: Mapping[str, Any],
        catalog_ids: set[str],
        *,
        strict: bool = True,
    ) -> None:
        self.agent = agent
        self.session = dict(session)
        self.catalog_ids = catalog_ids
        self.strict = strict
        self.sample_id = str(self.session["sample_id"])
        self.target = str(self.session["target_asin"])
        self.scenario = str(self.session["scenario_type"])
        self.session_id = f"manual400:{self.sample_id}"
        self.agent.reset(
            self.session_id,
            dict(self.session.get("user_profile") or {}),
        )

        self.rng = random.Random(
            f"manual400-reply:{self.sample_id}:{self.target}"
        )
        initial_id = parse_fact_id(self.session.get("initial_fact_id"))
        self.simulator_state: dict[str, Any] = {
            "disclosed": {initial_id} if initial_id is not None else set(),
            "active_constraints": {initial_id} if initial_id is not None else set(),
            "stale_constraints": set(),
            "no_preference_attributes": set(),
            "boundary_used": False,
        }
        self.user_message = str(self.session["initial_message"])
        self.override_applied = self.scenario != "intent_override"
        self.next_turn_number = 1
        self.first_hit_turn: int | None = None
        self.best_rank: int | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.events: list[dict[str, Any]] = []
        self.done = False

    def _apply_benchmark_override(self, turn: int) -> None:
        if self.override_applied or turn != int(self.session["override_turn"]):
            return

        self.override_applied = True
        initial_id = parse_fact_id(self.session.get("initial_fact_id"))
        if initial_id is not None:
            self.simulator_state["stale_constraints"].add(initial_id)
            self.simulator_state["active_constraints"].discard(initial_id)

        override_id = parse_fact_id(self.session.get("override_fact_id"))
        if override_id is not None:
            self.simulator_state["disclosed"].add(override_id)
            self.simulator_state["active_constraints"].add(override_id)

        self.user_message = str(self.session["override_message"])

    def next_turn(
        self,
        *,
        before_turn_callback: Any | None = None,
        after_turn_callback: Any | None = None,
    ) -> dict[str, Any] | None:
        """Execute exactly one evaluator turn and return its raw turn record."""

        if self.done:
            return None

        turn = self.next_turn_number
        self._apply_benchmark_override(turn)
        user_message = self.user_message

        callback_args = {
            "session": self.session,
            "session_id": self.session_id,
            "turn": turn,
            "user_message": user_message,
            "agent": self.agent,
            "override_applied": self.override_applied,
        }
        if before_turn_callback is not None:
            before_turn_callback(**callback_args)

        try:
            raw_response = self.agent.respond(
                self.session_id,
                user_message,
                turn,
                TOP_K,
            )
            response = validate_agent_response(raw_response)
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    f"Agent failed validation in {self.sample_id} on turn {turn}"
                ) from exc
            response = {
                "message": "",
                "ask_attribute": None,
                "recommendations": [],
            }

        usage = response.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens += int(usage["prompt_tokens"])
            self.completion_tokens += int(usage["completion_tokens"])

        ranked = normalize_recommendations(
            response.get("recommendations"),
            self.catalog_ids,
        )
        target_in_top10 = self.target in ranked
        scoreable_hit = bool(self.override_applied and target_in_top10)
        session_complete = scoreable_hit or turn == MAX_TURNS
        event = {
            "session_id": self.session_id,
            "sample_id": self.sample_id,
            "scenario_type": self.scenario,
            "target_asin": self.target,
            "turn": turn,
            "user_message": user_message,
            "response": response,
            "ranked": ranked,
            "override_applied": self.override_applied,
            "pre_override_hit": bool(target_in_top10 and not self.override_applied),
            "scoreable_hit": scoreable_hit,
            "session_complete": session_complete,
        }

        if after_turn_callback is not None:
            after_turn_callback(
                **callback_args,
                response=response,
                ranked=ranked,
                session_complete=session_complete,
            )

        self.events.append(event)
        if scoreable_hit:
            self.best_rank = ranked.index(self.target) + 1
            self.first_hit_turn = turn
            self.done = True
        elif turn == MAX_TURNS:
            self.done = True
        else:
            if (
                not self.override_applied
                and turn + 1 == int(self.session["override_turn"])
            ):
                # The configured override becomes the next user message.
                pass
            else:
                self.user_message = simulate_customer_reply(
                    self.session,
                    response.get("ask_attribute"),
                    self.simulator_state,
                    self.rng,
                )
            self.next_turn_number += 1

        return event

    def run_to_end(
        self,
        *,
        before_turn_callback: Any | None = None,
        after_turn_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Run remaining turns sequentially and return only new events."""

        start = len(self.events)
        while not self.done:
            self.next_turn(
                before_turn_callback=before_turn_callback,
                after_turn_callback=after_turn_callback,
            )
        return self.events[start:]

    def result(self) -> dict[str, Any]:
        """Return the batch-evaluator result for this session."""

        return {
            "sample_id": self.sample_id,
            "scenario_type": self.scenario,
            "target_asin": self.target,
            "hit": self.first_hit_turn is not None,
            "first_hit_turn": self.first_hit_turn,
            "best_rank": self.best_rank,
            "reciprocal_rank": (
                0.0
                if self.best_rank is None
                else 1.0 / self.best_rank
            ),
        }


def evaluate(
    agent: Any,
    sessions: Iterable[Mapping[str, Any]],
    catalog_ids: set[str],
    strict: bool = True,
    before_turn_callback: Any | None = None,
    after_turn_callback: Any | None = None,
    session_callback: Any | None = None,
    validate: bool = True,
    progress: bool = False,
    concurrency: int = 1,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    rows = (
        validate_sessions(sessions, catalog_ids)
        if validate
        else [dict(row) for row in sessions]
    )

    results_by_index: list[dict[str, Any] | None] = [None] * len(rows)
    completed_results: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    reporter = _Progress(len(rows), progress)

    def run_session(session: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
        runner = Manual400SessionRunner(
            agent,
            session,
            catalog_ids,
            strict=strict,
        )
        runner.run_to_end(
            before_turn_callback=before_turn_callback,
            after_turn_callback=after_turn_callback,
        )
        result = runner.result()
        return result, runner.prompt_tokens, runner.completion_tokens

    def record_result(
        index: int,
        session: Mapping[str, Any],
        outcome: tuple[dict[str, Any], int, int],
    ) -> None:
        nonlocal prompt_tokens, completion_tokens
        result, session_prompt_tokens, session_completion_tokens = outcome
        results_by_index[index] = result
        completed_results.append(result)
        prompt_tokens += session_prompt_tokens
        completion_tokens += session_completion_tokens

        reporter.advance(result["hit"])

        if session_callback is not None:
            session_callback(
                session=session,
                result=result,
                completed_results=completed_results,
            )

    try:
        if concurrency == 1:
            for index, session in enumerate(rows):
                record_result(index, session, run_session(session))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(run_session, session): (index, session)
                    for index, session in enumerate(rows)
                }
                for future in as_completed(futures):
                    index, session = futures[future]
                    record_result(index, session, future.result())
    finally:
        reporter.close()

    results = [result for result in results_by_index if result is not None]

    overall = add_score_fields(metric_summary(results))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["scenario_type"])].append(row)

    scenario_metrics = {
        scenario: add_score_fields(metric_summary(grouped[scenario]))
        for scenario in SCENARIO_COUNTS
        if grouped.get(scenario)
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


DEBUG_FACT_FIELDS = (
    "category",
    "brand",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
)


def _debug_values(constraints: Any, field_name: str) -> list[str]:
    value = getattr(constraints, field_name, ())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def _debug_constraints(state: Any) -> dict[str, Any]:
    constraints = getattr(state, "constraints", None)
    if constraints is None:
        return {}

    payload: dict[str, Any] = {}
    for field_name in DEBUG_FACT_FIELDS + ("size",):
        values = _debug_values(constraints, field_name)
        if values:
            payload[field_name] = values

    price_min = getattr(constraints, "price_min", None)
    price_max = getattr(constraints, "price_max", None)
    budget: dict[str, float] = {}
    if price_min is not None:
        budget["min"] = float(price_min)
    if price_max is not None:
        budget["max"] = float(price_max)
    if budget:
        payload["budget"] = budget
    return payload


def _debug_semantic_constraints(state: Any) -> dict[str, Any]:
    constraints = getattr(state, "semantic_constraints", state)
    if constraints is None:
        return {}

    payload: dict[str, Any] = {}
    for field_name in DEBUG_FACT_FIELDS:
        values = _debug_values(constraints, field_name)
        if values:
            payload[field_name] = values

    evidence = getattr(constraints, "evidence", ())
    similarities: dict[str, float] = {}
    for item in evidence if isinstance(evidence, (list, tuple)) else ():
        canonical_id = getattr(item, "canonical_id", None)
        confidence = getattr(item, "confidence", None)
        if not isinstance(canonical_id, str):
            continue
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            continue
        similarities[canonical_id] = score
    if similarities:
        payload["similarities"] = similarities
    return payload


def _debug_state_snapshot(agent: Any, session_id: str) -> dict[str, Any]:
    sessions = getattr(agent, "sessions", None)
    getter = getattr(sessions, "get", None)
    if not callable(getter):
        return {}
    state = getter(session_id)
    return {
        "constraints": _debug_constraints(state),
        "semantic_constraints": _debug_semantic_constraints(state),
        "mode": getattr(state, "mode", None),
        "clarification_cycle": int(getattr(state, "clarification_cycle", 1)),
        "attribute_call_count": {
            str(field_name): int(count)
            for field_name, count in sorted(
                getattr(state, "attribute_call_count", {}).items()
            )
        },
        "no_preference_attributes": sorted(
            str(value)
            for value in getattr(state, "no_preference_attributes", set())
        ),
        "clarification_stopped": bool(
            getattr(state, "clarification_stopped", False)
        ),
        "override_kind": getattr(state, "last_override_kind", None),
        "override_delta": _debug_constraints(
            getattr(state, "last_override_delta", None)
        ),
        "override_semantic_delta": _debug_semantic_constraints(
            getattr(
                getattr(state, "last_override_delta", None),
                "semantic_constraints",
                None,
            )
        ),
        "query_text": getattr(state, "query_text", ""),
        "excluded": sorted(
            str(value)
            for value in getattr(state, "excluded_recommendations", set())
        ),
        "last": list(getattr(state, "last_recommendations", ())),
    }


def _debug_target_facts(agent: Any, target: str) -> dict[str, Any]:
    retriever = getattr(agent, "retriever", None)
    products = getattr(retriever, "product_by_asin", {})
    product = products.get(target)
    if product is None:
        return {"available": False}

    facts = {
        field_name: list(product.facts.get(field_name, ()))
        for field_name in DEBUG_FACT_FIELDS
    }
    facts["price"] = product.price

    annotation_facts = getattr(retriever, "_facts_by_asin", {})
    if target not in annotation_facts:
        facts["annotation_note"] = (
            "V4 semantic annotation missing; showing available catalog facts only."
        )
    return facts


def _debug_ranking_snapshot(agent: Any, session_id: str, target: str) -> dict[str, Any]:
    state = agent.sessions.get(session_id)
    retriever = agent.retriever
    rank_all = getattr(retriever, "debug_rank_all", None)
    if not callable(rank_all):
        raise RuntimeError("ProductRetriever.debug_rank_all is required in debug mode")

    mode = getattr(state, "mode", None) or "BROWSING"
    query_text = getattr(state, "query_text", "")
    constraints = getattr(state, "constraints", None)
    semantic_constraints = getattr(state, "semantic_constraints", None)
    exclusions = getattr(state, "excluded_recommendations", set())

    eligible = rank_all(
        mode,
        query_text,
        constraints,
        semantic_constraints=semantic_constraints,
        excluded_asins=exclusions,
        apply_budget=True,
    )
    global_ranking = rank_all(
        mode,
        query_text,
        constraints,
        semantic_constraints=semantic_constraints,
        excluded_asins=None,
        apply_budget=False,
    )

    def sort_by(candidates: list[Any], score_name: str) -> list[Any]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -float(getattr(candidate, score_name)),
                retriever.product_by_asin[candidate.parent_asin].catalog_order,
            ),
        )

    structured = sort_by(eligible, "constraint_score")
    dense = sort_by(eligible, "dense_score")
    bm25 = sort_by(eligible, "bm25_score")
    hybrid = sort_by(eligible, "score")
    global_structured = sort_by(global_ranking, "constraint_score")
    global_dense = sort_by(global_ranking, "dense_score")
    global_bm25 = sort_by(global_ranking, "bm25_score")
    global_hybrid = sort_by(global_ranking, "score")
    return {
        "eligible": eligible,
        "global": global_ranking,
        "structured": structured,
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
        "global_structured": global_structured,
        "global_dense": global_dense,
        "global_bm25": global_bm25,
        "global_hybrid": global_hybrid,
        "target_eligible": next(
            (candidate for candidate in eligible if candidate.parent_asin == target),
            None,
        ),
        "target_global": next(
            (candidate for candidate in global_ranking if candidate.parent_asin == target),
            None,
        ),
    }


def _debug_rank(candidates: list[Any], target: str) -> int | None:
    for index, candidate in enumerate(candidates, 1):
        if candidate.parent_asin == target:
            return index
    return None


def _debug_print_breakdown(constraints: Any, candidate: Any) -> None:
    print("TARGET MATCH BREAKDOWN")
    if candidate is None:
        print("No target candidate was available.")
        return

    labels = set(candidate.matched_constraints)
    active_fields = [
        field_name
        for field_name in STRUCTURED_FIELD_WEIGHTS
        if (
            field_name == "price"
            and (
                getattr(constraints, "price_min", None) is not None
                or getattr(constraints, "price_max", None) is not None
            )
        )
        or field_name != "price" and _debug_values(constraints, field_name)
    ]
    total_weight = sum(STRUCTURED_FIELD_WEIGHTS[field] for field in active_fields)

    for field_name in active_fields:
        weight = STRUCTURED_FIELD_WEIGHTS[field_name]
        if field_name == "price":
            price_min = getattr(constraints, "price_min", None)
            price_max = getattr(constraints, "price_max", None)
            bounds: list[str] = []
            if price_min is not None:
                bounds.append(f"min={float(price_min):g}")
            if price_max is not None:
                bounds.append(f"max={float(price_max):g}")
            matched = "price:required" not in candidate.violated_constraints
            status = "PASS" if matched else "MISS"
            contribution = weight if matched else 0.0
            print(
                f"budget:{', '.join(bounds)}    {status}    "
                f"+{contribution:.2f}"
            )
            continue

        field_matched = False
        for value in _debug_values(constraints, field_name):
            label = f"{field_name}:{value}"
            matched = label in labels
            contribution = weight if matched and not field_matched else 0.0
            if matched:
                field_matched = True
            status = "MATCH" if matched else "MISS"
            print(f"{label}    {status}    +{contribution:.2f}")

    raw_score = float(candidate.constraint_score) * total_weight
    print(f"Structured raw total: {raw_score:.4f}")
    print(f"Structured normalized: {float(candidate.constraint_score):.4f}")


def _debug_print_metrics(results: list[Mapping[str, Any]]) -> None:
    metrics = add_score_fields(metric_summary(results))
    print(f"Completed sessions: {len(results)}")
    print(f"HitRate@10:      {float(metrics['hit_rate_at_10']):.4f}")
    print(f"MRR:             {float(metrics['mrr']):.4f}")
    print(f"MTTC:            {float(metrics['mttc']):.4f}")
    print(f"Efficiency:      {float(metrics['efficiency']):.4f}")
    print(f"TechnicalScore:  {float(metrics['technical_score']):.4f}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["scenario_type"])].append(dict(result))
    print("Scenario running metrics:")
    for scenario in SCENARIO_COUNTS:
        scenario_results = grouped.get(scenario, [])
        if not scenario_results:
            continue
        scenario_metrics = add_score_fields(metric_summary(scenario_results))
        print(
            f"  {scenario}: HitRate@10={float(scenario_metrics['hit_rate_at_10']):.4f} "
            f"MRR={float(scenario_metrics['mrr']):.4f} "
            f"MTTC={float(scenario_metrics['mttc']):.4f} "
            f"TechnicalScore={float(scenario_metrics['technical_score']):.4f}"
        )


class InteractiveDebugPrinter:
    def __init__(
        self,
        agent: Any,
        catalog_ids: set[str],
        input_fn: Any | None = None,
    ) -> None:
        self.agent = agent
        self.catalog_ids = catalog_ids
        self.input_fn = input_fn
        self.before_states: dict[tuple[str, int], dict[str, Any]] = {}
        self.first_hits: dict[str, int] = {}

    def _pause(self, prompt: str) -> None:
        if self.input_fn is None:
            input(prompt)
        else:
            self.input_fn(prompt)

    def before_turn(self, **kwargs: Any) -> None:
        self.before_states[
            (str(kwargs["session_id"]), int(kwargs["turn"]))
        ] = _debug_state_snapshot(
            kwargs["agent"],
            str(kwargs["session_id"]),
        )

    def after_turn(self, **kwargs: Any) -> None:
        session = kwargs["session"]
        session_id = str(kwargs["session_id"])
        turn = int(kwargs["turn"])
        user_message = str(kwargs["user_message"])
        response = kwargs["response"]
        ranked = list(kwargs["ranked"])
        agent = kwargs["agent"]
        target = str(session["target_asin"])
        override_applied = bool(kwargs["override_applied"])
        state = agent.sessions.get(session_id)
        scenario = str(session["scenario_type"])
        override_turn = session.get("override_turn")
        override_kind = getattr(state, "last_override_kind", None)
        override_detected = override_kind in {"FULL_GOAL", "PREFERENCE"}

        print()
        print("=" * 60)
        print(f"Session: {session['sample_id']}")
        print(f"Scenario: {scenario.upper()}")
        print(f"Turn: {turn}")
        print("=" * 60)
        print("USER:")
        print(user_message)
        print()
        print("STRUCTURED CONSTRAINTS SO FAR")
        print(json.dumps(_debug_constraints(state), indent=2, ensure_ascii=False))
        print("DENSE SEMANTIC CONSTRAINTS SO FAR")
        print(
            json.dumps(
                _debug_semantic_constraints(state),
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"Intent mode: {getattr(state, 'mode', None) or 'BROWSING'}")
        print(f"Override kind: {override_kind or 'NONE'}")
        if override_detected:
            print("CURRENT-TURN OVERRIDE DELTA")
            print(
                json.dumps(
                    _debug_constraints(getattr(state, "last_override_delta", None)),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        print(f"Semantic query: {getattr(state, 'query_text', '')}")
        retriever = agent.retriever
        layer2_index = getattr(retriever, "layer2_index", None)
        query_encoder = getattr(retriever, "query_encoder", None)
        artifact_manifest = getattr(layer2_index, "manifest", {}) if layer2_index else {}
        artifact_model = artifact_manifest.get(
            "embedding_model", artifact_manifest.get("model", "N/A")
        )
        encoder_model = getattr(query_encoder, "model_id", None) or "N/A"
        artifact_dimension = getattr(layer2_index, "dimension", None)
        print()
        print("LAYER 2")
        print(f"Enabled: {'YES' if bool(getattr(retriever, 'dense_available', False)) else 'NO'}")
        print(f"Artifact model: {artifact_model}")
        print(f"Query model: {encoder_model}")
        print(f"Dimension: {artifact_dimension if artifact_dimension is not None else 'N/A'}")
        compatibility_error = getattr(retriever, "layer2_compatibility_error", None)
        if compatibility_error and not bool(getattr(retriever, "dense_available", False)):
            print(f"Dense status: {compatibility_error}")
        mode = str(getattr(state, "mode", None) or "BROWSING").upper()
        score_weights = MODE_SCORE_WEIGHTS.get(mode, MODE_SCORE_WEIGHTS["BROWSING"])
        print(
            "Score weights: "
            f"structured={score_weights['structured']:.2f}, "
            f"dense={score_weights['dense']:.2f}, "
            f"bm25={score_weights.get('bm25', 0.0):.2f}"
        )
        print(
            "BM25: "
            f"{'AVAILABLE' if bool(getattr(retriever, 'bm25_available', False)) else 'UNAVAILABLE'}"
        )
        print()
        snapshot = _debug_state_snapshot(agent, session_id)
        excluded = snapshot.get("excluded", [])
        print("EXCLUDED FROM PREVIOUS FAILED TURNS:")
        print(f"Count: {len(excluded)}")
        if len(excluded) <= 20:
            print(f"Excluded products: {excluded}")
        print(f"Last recommendations: {snapshot.get('last', [])}")
        print()
        print("TARGET PRODUCT")
        print(f"ASIN: {target}")
        print("TARGET FACTS")
        print(json.dumps(_debug_target_facts(agent, target), indent=2, ensure_ascii=False))
        print()
        print(f"OVERRIDE DETECTED: {'YES' if override_detected else 'NO'}")
        if override_detected:
            before = self.before_states.get((session_id, turn), {})
            print("Constraints before reset:")
            print(json.dumps(before.get("constraints", {}), indent=2, ensure_ascii=False))
            print("Constraints after reset:")
            print(json.dumps(snapshot.get("constraints", {}), indent=2, ensure_ascii=False))
            print(f"Recommendation exclusions before: {before.get('excluded', [])}")
            print(f"Recommendation exclusions after: {snapshot.get('excluded', [])}")
        ranking = _debug_ranking_snapshot(agent, session_id, target)
        eligible = ranking["eligible"]
        global_ranking = ranking["global"]
        structured_ranking = ranking["structured"]
        dense_ranking = ranking["dense"]
        bm25_ranking = ranking["bm25"]
        hybrid_ranking = ranking["hybrid"]
        target_eligible = ranking["target_eligible"]
        target_global = ranking["target_global"]
        eligible_rank = _debug_rank(eligible, target)
        global_rank = _debug_rank(global_ranking, target)
        budget_active = bool(
            getattr(state.constraints, "price_min", None) is not None
            or getattr(state.constraints, "price_max", None) is not None
        )
        print()
        print("TARGET RANKING")
        if budget_active and target_eligible is None:
            print("Global rank: N/A")
            print("Reason: violates active budget")
        elif global_rank is None:
            print("Global rank: N/A")
        else:
            print(f"Global rank: {global_rank} / {len(global_ranking)}")
        if eligible_rank is None:
            print("Eligible rank: N/A")
            if not budget_active and target_global is not None:
                print("Reason: excluded by current-goal recommendation history")
        else:
            print(f"Eligible rank: {eligible_rank} / {len(eligible)}")
        target_candidate = target_eligible or target_global
        if target_candidate is not None:
            print(f"Structured score: {target_candidate.constraint_score:.4f}")
            print(f"Dense score: {target_candidate.dense_score:.4f}")
            print(f"BM25 score: {target_candidate.bm25_score:.4f}")
            print(f"Final score: {target_candidate.score:.4f}")
        else:
            print("Target score: N/A")
        print("Target ranks (eligible products):")
        print(f"  Structured rank: {_debug_rank(structured_ranking, target) or 'MISS'}")
        print(f"  Dense rank: {_debug_rank(dense_ranking, target) or 'MISS'}")
        print(f"  BM25 rank: {_debug_rank(bm25_ranking, target) or 'MISS'}")
        print(f"  Hybrid rank: {_debug_rank(hybrid_ranking, target) or 'MISS'}")
        top10_rank = ranked.index(target) + 1 if target in ranked else None
        print(f"Top10 rank: {top10_rank if top10_rank is not None else 'MISS'}")

        print()
        print("TOP 10")
        candidate_by_asin = {candidate.parent_asin: candidate for candidate in eligible}
        for index, asin in enumerate(ranked[:TOP_K], 1):
            candidate = candidate_by_asin.get(asin)
            if candidate is None:
                print(f"{index}. {asin} (score unavailable)")
                continue
            print(
                f"{index}. {asin} score={candidate.score:.4f} "
                f"structured={candidate.constraint_score:.4f} "
                f"dense={candidate.dense_score:.4f} "
                f"bm25={candidate.bm25_score:.4f}"
            )
            print(f"   matched={list(candidate.matched_constraints)}")

        print()
        _debug_print_breakdown(state.constraints, target_candidate)
        print()
        print("TURN RESULT")
        scoreable_hit = bool(override_applied and top10_rank is not None)
        if top10_rank is not None:
            suffix = "" if scoreable_hit else " (not scoreable before override)"
            print(f"Target in Top10: YES{suffix}")
            print(f"Target Top10 rank: {top10_rank}")
        else:
            print("Target in Top10: NO")
            print("Target Top10 rank: MISS")
        if scoreable_hit:
            if session_id not in self.first_hits:
                self.first_hits[session_id] = turn
            print(f"Reciprocal rank: {1.0 / top10_rank:.4f}")
            print(f"First successful turn: {self.first_hits[session_id]}")
        else:
            print("Reciprocal rank: 0.0000")
            print(
                f"First successful turn: "
                f"{self.first_hits.get(session_id, '-')}"
            )
        print(f"Turn number: {turn}")
        self._pause("[Press ENTER for next turn]")

    def session_complete(self, **kwargs: Any) -> None:
        print()
        print("SESSION RESULT")
        print(json.dumps(kwargs["result"], indent=2, ensure_ascii=False))
        print()
        print("RUNNING DEBUG BENCHMARK")
        _debug_print_metrics(list(kwargs["completed_results"]))
        self._pause("[Press ENTER for next random session]")


def debug_session_order(
    sessions: Iterable[Mapping[str, Any]],
    *,
    seed: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("debug session limit must be positive")
    rows = [dict(session) for session in sessions]
    random.Random(seed).shuffle(rows)
    return rows if limit is None else rows[:limit]


def debug_evaluate(
    agent: Any,
    sessions: Iterable[Mapping[str, Any]],
    catalog_ids: set[str],
    *,
    seed: int | None = None,
    debug_sessions: int | None = None,
    strict: bool = True,
    input_fn: Any | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    rows = (
        validate_sessions(sessions, catalog_ids)
        if validate
        else [dict(row) for row in sessions]
    )
    selected = debug_session_order(rows, seed=seed, limit=debug_sessions)
    seed_label = str(seed) if seed is not None else "random"
    print(
        f"Interactive hard-benchmark debug mode: "
        f"{len(selected)} session(s), seed={seed_label}"
    )
    printer = InteractiveDebugPrinter(agent, catalog_ids, input_fn=input_fn)
    return evaluate(
        agent=agent,
        sessions=selected,
        catalog_ids=catalog_ids,
        strict=strict,
        before_turn_callback=printer.before_turn,
        after_turn_callback=printer.after_turn,
        session_callback=printer.session_complete,
        validate=False,
        concurrency=1,
    )


def main() -> None:
    started_at = time.perf_counter()
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
        "--disable-user-profile",
        action="store_true",
        help="Ignore user_profile preference_tags when choosing the follow-up "
             "attribute to ask.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the per-session progress bar on stderr.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of sessions to evaluate in parallel (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Treat malformed Agent responses as empty instead of failing.",
    )
    parser.add_argument(
        "--override-only",
        action="store_true",
        help="Evaluate only the 60 intent_override sessions.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Interactively inspect randomly ordered benchmark sessions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for debug session order; only valid with --debug.",
    )
    parser.add_argument(
        "--debug-sessions",
        type=int,
        help="Number of random sessions to inspect; only valid with --debug.",
    )
    args = parser.parse_args()

    sessions = load_jsonl(args.sessions)
    catalog_ids = load_catalog_ids(args.catalog)

    try:
        validated_sessions = validate_sessions(sessions, catalog_ids)
    except ValueError as exc:
        parser.error(str(exc))
    selected_sessions = select_sessions(
        validated_sessions,
        override_only=args.override_only,
    )
    if args.override_only:
        print(
            f"Override-only mode: evaluating {len(selected_sessions)} "
            "intent_override sessions."
        )

    if not args.debug and (args.seed is not None or args.debug_sessions is not None):
        parser.error("--seed and --debug-sessions require --debug")
    if args.debug_sessions is not None and args.debug_sessions <= 0:
        parser.error("--debug-sessions must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")

    try:
        agent = build_evaluator_agent(
            args.catalog,
            disable_user_profile=args.disable_user_profile,
        )
        if not args.debug and args.concurrency > 1:
            warm_evaluator_runtime()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.debug:
        try:
            debug_evaluate(
                agent=agent,
                sessions=selected_sessions,
                catalog_ids=catalog_ids,
                seed=args.seed,
                debug_sessions=args.debug_sessions,
                strict=not args.non_strict,
                validate=False,
            )
        except (KeyboardInterrupt, EOFError):
            print("\nInteractive debug mode stopped.")
        return

    result = evaluate(
        agent=agent,
        sessions=selected_sessions,
        catalog_ids=catalog_ids,
        strict=not args.non_strict,
        validate=False,
        progress=not args.no_progress,
        concurrency=args.concurrency,
    )

    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(rounded_summary(result), indent=2))
    elapsed_seconds = time.perf_counter() - started_at
    elapsed_minutes, remaining_seconds = divmod(elapsed_seconds, 60.0)
    print(
        "Total evaluation time: "
        f"{elapsed_seconds:.2f} seconds "
        f"({int(elapsed_minutes)}m {remaining_seconds:.2f}s)"
    )


if __name__ == "__main__":
    main()
