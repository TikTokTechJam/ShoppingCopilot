"""Write reproducible evaluator-side diagnostics for the fixed Manual400 benchmark.

The official simulation and scoring remain in evaluator.hard_evaluator. This
module observes the Agent boundary and writes analysis artifacts without
changing the response passed to the official evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluator import hard_evaluator
from evaluator.agent_factory import build_evaluator_agent

SCHEMA_VERSION = 1
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TELEMETRY_ATTRIBUTES = (
    "last_diagnostics",
    "diagnostics",
    "last_metrics",
    "metrics",
    "telemetry",
    "last_trace",
)
COUNTER_GROUPS = {
    "canonicalization": ("canonical", "deterministic", "exact", "normalized", "semantic", "unresolved", "constraint"),
    "retrieval": ("retrieval", "search", "candidate", "relax", "dense", "fallback", "top_k"),
    "clarification": ("clarif", "ask_attribute", "attribute", "question", "pool_shrink"),
}


def json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item, depth + 1) for key, item in list(value.items())[:200]}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, depth + 1) for item in value[:200]]
    if isinstance(value, set):
        return sorted((json_safe(item, depth + 1) for item in value), key=str)[:200]
    return str(value)


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numbers(item, name))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return {prefix: number} if math.isfinite(number) else {}
    return {}


def telemetry(agent: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in TELEMETRY_ATTRIBUTES:
        try:
            value = getattr(agent, name)
        except AttributeError:
            continue
        if value is not None and not callable(value):
            safe = json_safe(value)
            if safe not in ({}, []):
                result[name] = safe
    return result


def telemetry_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    previous = flatten_numbers(before)
    result: dict[str, float] = {}
    for name, value in flatten_numbers(after).items():
        amount = value if name not in previous else value - previous[name]
        if amount >= 0:
            result[name] = round(amount, 9)
    return result


def counter_group(name: str) -> str | None:
    lowered = name.lower()
    for group, terms in COUNTER_GROUPS.items():
        if any(term in lowered for term in terms):
            return group
    return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 6)


def metric_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(metrics["sample_count"]),
        "hit_rate_at_10": round(float(metrics["hit_rate_at_10"]), 6),
        "mrr": round(float(metrics["mrr"]), 6),
        "mttc": None if metrics["mttc"] is None else round(float(metrics["mttc"]), 6),
        "efficiency": round(float(metrics["efficiency"]), 6),
        "technical_score": round(float(metrics["technical_score"]), 6),
    }


def git_metadata(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip() or None

    status = git("status", "--short", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "worktree_dirty": bool(status),
    }


def file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


class ObservedAgent:
    def __init__(self, agent: Any, catalog_ids: set[str]) -> None:
        self._agent = agent
        self._catalog_ids = catalog_ids
        self.sessions: dict[str, dict[str, Any]] = {}

    def reset(self, session_id: str, user_profile: dict[str, Any]) -> None:
        started = time.perf_counter()
        result = self._agent.reset(session_id, user_profile)
        self.sessions[session_id] = {
            "session_id": session_id,
            "reset_latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
            "turns": [],
        }
        return result

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> Any:
        before = telemetry(self._agent)
        started = time.perf_counter()
        response: Any = None
        error: dict[str, str] | None = None
        try:
            response = self._agent.respond(session_id, user_message, turn, top_k)
            return response
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record: dict[str, Any] = {
                "turn": turn,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
                "telemetry_delta": telemetry_delta(before, telemetry(self._agent)),
            }
            if error:
                record["error"] = error
            if isinstance(response, Mapping):
                record["ask_attribute"] = response.get("ask_attribute")
                record["recommendation_ids"] = hard_evaluator.normalize_recommendations(
                    response.get("recommendations"), self._catalog_ids
                )
                if isinstance(response.get("usage"), Mapping):
                    record["usage"] = json_safe(response["usage"])
            self.sessions.setdefault(session_id, {"session_id": session_id, "turns": []})["turns"].append(record)


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "miss"
    if rank == 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_to_3"
    if rank <= 5:
        return "rank_4_to_5"
    return "rank_6_to_10"


def session_reports(
    result: Mapping[str, Any],
    observed: ObservedAgent,
    inputs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for scored in result["sessions"]:
        sample_id = str(scored["sample_id"])
        source = inputs[sample_id]
        observed_session = observed.sessions.get(f"manual400:{sample_id}", {"turns": []})
        override_turn = source.get("override_turn")
        cumulative_hit = False
        turns: list[dict[str, Any]] = []
        for observed_turn in observed_session.get("turns", []):
            recommendations = observed_turn.get("recommendation_ids", [])
            target = str(scored["target_asin"])
            raw_rank = recommendations.index(target) + 1 if target in recommendations else None
            eligible = not (
                source.get("scenario_type") == "intent_override"
                and override_turn is not None
                and int(observed_turn["turn"]) < int(override_turn)
            )
            target_rank = raw_rank if eligible else None
            hit = target_rank is not None
            cumulative_hit = cumulative_hit or hit
            turns.append({
                **observed_turn,
                "eligible_for_scoring": eligible,
                "raw_target_rank": raw_rank,
                "target_rank": target_rank,
                "hit": hit,
                "cumulative_hit": cumulative_hit,
                "rank_bucket": rank_bucket(target_rank),
            })
        latencies = [float(turn["latency_ms"]) for turn in turns]
        reports.append({
            "sample_id": sample_id,
            "scenario_type": scored["scenario_type"],
            "target_asin": scored["target_asin"],
            "hit": scored["hit"],
            "first_hit_turn": scored["first_hit_turn"],
            "best_rank": scored["best_rank"],
            "reciprocal_rank": scored["reciprocal_rank"],
            "latency_ms": {
                "reset": observed_session.get("reset_latency_ms"),
                "respond_total": round(sum(latencies), 6),
            },
            "turns": turns,
        })
    return reports


def diagnostics(
    reports: list[Mapping[str, Any]],
    startup_ms: float,
    simulation_ms: float,
) -> dict[str, Any]:
    by_turn: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    first_hits: Counter[str] = Counter()
    ranks: Counter[str] = Counter()
    counters: dict[str, Counter[str]] = {group: Counter() for group in COUNTER_GROUPS}
    asks: Counter[str] = Counter()
    response_latencies: list[float] = []
    reset_latencies: list[float] = []

    for session in reports:
        first_hits[str(session["first_hit_turn"]) if session["first_hit_turn"] is not None else "miss"] += 1
        ranks[rank_bucket(session["best_rank"])] += 1
        if session["latency_ms"].get("reset") is not None:
            reset_latencies.append(float(session["latency_ms"]["reset"]))
        for turn in session["turns"]:
            by_turn[int(turn["turn"])].append(turn)
            response_latencies.append(float(turn["latency_ms"]))
            attribute = turn.get("ask_attribute")
            if isinstance(attribute, str) and attribute:
                asks[attribute] += 1
            for name, value in turn.get("telemetry_delta", {}).items():
                group = counter_group(name)
                if group:
                    counters[group][name] += value

    turn_level: dict[str, Any] = {}
    for turn_number in range(1, hard_evaluator.MAX_TURNS + 1):
        rows = by_turn[turn_number]
        cumulative = sum(bool(row["cumulative_hit"]) for row in rows)
        turn_level[str(turn_number)] = {
            "sample_count": len(rows),
            "cumulative_hit_count": cumulative,
            "cumulative_hit_rate_at_10": round(cumulative / len(reports), 6) if reports else 0.0,
            "turn_hit_count": sum(bool(row["hit"]) for row in rows),
            "mean_latency_ms": round(statistics.fmean(float(row["latency_ms"]) for row in rows), 6) if rows else None,
        }

    def group_view(group: str) -> dict[str, Any]:
        values = {name: round(value, 6) for name, value in sorted(counters[group].items())}
        return {"available": bool(values), "counters": values, "total": round(sum(values.values()), 6)}

    clarification = group_view("clarification")
    clarification["ask_attribute_frequency"] = dict(sorted(asks.items()))
    clarification["turns_with_ask_attribute"] = sum(asks.values())
    clarification["ask_attribute_turn_rate"] = (
        round(sum(asks.values()) / len(response_latencies), 6) if response_latencies else 0.0
    )
    return {
        "turn_level": turn_level,
        "first_hit_turn_distribution": dict(sorted(first_hits.items())),
        "first_hit_rank_buckets": dict(ranks),
        "canonicalization": group_view("canonicalization"),
        "retrieval": group_view("retrieval"),
        "clarification": clarification,
        "latency_ms": {
            "agent_startup": round(startup_ms, 6),
            "simulation": round(simulation_ms, 6),
            "respond_total": round(sum(response_latencies), 6),
            "respond_mean": round(statistics.fmean(response_latencies), 6) if response_latencies else None,
            "respond_p50": percentile(response_latencies, 0.50),
            "respond_p95": percentile(response_latencies, 0.95),
            "reset_mean": round(statistics.fmean(reset_latencies), 6) if reset_latencies else None,
            "turn_count": len(response_latencies),
        },
        "observed_failure_categories": {
            "session_misses": sum(not bool(row["hit"]) for row in reports),
            "turns_with_agent_error": sum(
                1 for session in reports for turn in session["turns"] if "error" in turn
            ),
            "sessions_with_agent_error": sum(
                any("error" in turn for turn in session["turns"]) for session in reports
            ),
        },
    }


def parse_versions(values: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or not name.strip() or not version.strip():
            raise ValueError(f"--artifact-version must use NAME=VALUE, got {value!r}")
        versions[name.strip()] = version.strip()
    return versions


def run(args: argparse.Namespace) -> Path:
    root = Path.cwd()
    report_dir = Path(args.report_root) / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = Path(args.sessions)
    catalog_path = Path(args.catalog)
    sessions = hard_evaluator.load_jsonl(sessions_path)
    inputs = {str(row["sample_id"]): row for row in sessions}

    startup = time.perf_counter()
    agent = build_evaluator_agent(
        catalog_path,
        disable_evolution=getattr(args, "disable_evolution", False),
        evolution_full=getattr(args, "evo_full", False),
    )
    startup_ms = (time.perf_counter() - startup) * 1000.0
    catalog_ids = hard_evaluator.load_catalog_ids(catalog_path)
    observed = ObservedAgent(agent, catalog_ids)

    simulation = time.perf_counter()
    result = hard_evaluator.evaluate(
        agent=observed,
        sessions=sessions,
        catalog_ids=catalog_ids,
        strict=not args.non_strict,
    )
    simulation_ms = (time.perf_counter() - simulation) * 1000.0
    reports = session_reports(result, observed, inputs)
    agent_module = Path(inspect.getfile(hard_evaluator.Agent)).resolve()
    metadata = {
        "run_name": args.run_name,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": " ".join(["python", "-m", "evaluator.manual400_report", "--run-name", args.run_name]),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_metadata(root),
        "artifact_versions": parse_versions(args.artifact_version),
        "artifacts": {
            "catalog": file_metadata(catalog_path.resolve()),
            "sessions": file_metadata(sessions_path.resolve()),
            "agent_module": file_metadata(agent_module),
        },
        "benchmark": {
            "name": "Manual400",
            "evaluator": "evaluator.hard_evaluator",
            "max_turns": hard_evaluator.MAX_TURNS,
            "top_k": hard_evaluator.TOP_K,
            "scenario_counts": hard_evaluator.SCENARIO_COUNTS,
            "strict": not args.non_strict,
        },
    }
    summary = {
        "metadata": metadata,
        "metrics": {
            "overall": metric_view(result),
            "by_scenario": {
                scenario: metric_view(metrics)
                for scenario, metrics in result["scenario_metrics"].items()
            },
            "reported_token_usage": result["reported_token_usage"],
        },
        "diagnostics": diagnostics(reports, startup_ms, simulation_ms),
    }

    serialization = time.perf_counter()
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "sessions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n")
    serialization_ms = (time.perf_counter() - serialization) * 1000.0
    summary["diagnostics"]["latency_ms"]["serialization"] = round(serialization_ms, 6)
    summary["diagnostics"]["latency_ms"]["total"] = round(
        startup_ms + simulation_ms + serialization_ms,
        6,
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrap the fixed Manual400 evaluator with diagnostics.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--sessions", default="data/derived/gptannotation/sessions.jsonl")
    parser.add_argument("--run-name", default="manual400")
    parser.add_argument("--report-root", default="reports/manual400")
    parser.add_argument("--artifact-version", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Pass the existing hard evaluator's non-strict mode through unchanged.",
    )
    parser.add_argument(
        "--disable-evolution",
        action="store_true",
        help="Run the pre-feedback-loop code path (no runtime belief reweighting).",
    )
    parser.add_argument(
        "--evo-full",
        action="store_true",
        help="Turn on the gated loop stages (decay, RE-PLAN, LEARN).",
    )
    args = parser.parse_args()
    if not RUN_NAME.fullmatch(args.run_name):
        parser.error("--run-name must be path-safe using letters, digits, . _ or -")
    output = run(args)
    print(f"Manual400 report written to {output}")


if __name__ == "__main__":
    main()
