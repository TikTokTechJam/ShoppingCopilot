"""Write evaluator-side diagnostics for the fixed Manual400 benchmark.

This module deliberately composes :mod:`evaluator.hard_evaluator` instead of
copying its simulator or scoring code.  ``TracingAgent`` forwards the exact
Agent API calls and returns responses unchanged; the hard evaluator therefore
remains the only source of benchmark semantics and scores.

Run from the repository root with::

    python -m tools.manual400_report --run-name baseline-20260827

The command writes ``reports/manual400/<run-name>/summary.json`` and
``sessions.jsonl``.  It does not include hidden facts, simulated customer
replies, or target product IDs in the diagnostic artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluator import hard_evaluator  # noqa: E402
from starter.agent import Agent  # noqa: E402


REPORT_SCHEMA_VERSION = 1
RUN_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _validate_run_name(value: str) -> str:
    if not value or value in {".", ".."} or any(char not in RUN_NAME_CHARS for char in value):
        raise ValueError(
            "run name must contain only ASCII letters, digits, '.', '_' or '-': "
            f"{value!r}"
        )
    return value


def _sha256(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    metadata: dict[str, Any] = {
        "path": str(candidate),
        "sha256": _sha256(candidate),
    }
    if candidate.is_file():
        metadata["bytes"] = candidate.stat().st_size
    return metadata


def _git_command(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    status = _git_command("status", "--porcelain")
    return {
        "commit_sha": _git_command("rev-parse", "HEAD"),
        "branch": _git_command("branch", "--show-current"),
        "worktree_clean": None if status is None else not bool(status),
    }


def _source_metadata(module: Any, class_name: str | None = None) -> dict[str, Any]:
    source = getattr(module, "__file__", None)
    result: dict[str, Any] = {
        "module": getattr(module, "__name__", None),
        "source": _file_metadata(source) if source else None,
    }
    if class_name is not None:
        result["class"] = class_name
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "total_ms": round(sum(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(_percentile(values, 0.50) or 0.0, 3),
        "p95_ms": round(_percentile(values, 0.95) or 0.0, 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def _response_event(response: Any, turn: int, latency_ms: float) -> dict[str, Any]:
    event: dict[str, Any] = {
        "turn": int(turn),
        "latency_ms": round(latency_ms, 3),
        "ask_attribute": None,
        "recommendation_count": None,
    }
    if not isinstance(response, dict):
        event["response_type"] = type(response).__name__
        return event

    ask_attribute = response.get("ask_attribute")
    event["ask_attribute"] = ask_attribute if isinstance(ask_attribute, str) else None

    recommendations = response.get("recommendations")
    if isinstance(recommendations, list):
        event["recommendation_count"] = len(recommendations)

    usage = response.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                event[key] = value
    return event


class TracingAgent:
    """Transparent Agent proxy that records evaluator-observable responses."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self.respond_latencies_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict[str, Any]) -> None:
        self.sessions[session_id] = []
        self.delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> Any:
        started = time.perf_counter_ns()
        try:
            response = self.delegate.respond(session_id, user_message, turn, top_k)
        except BaseException as exc:
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            self.respond_latencies_ms.append(latency_ms)
            self.sessions.setdefault(session_id, []).append(
                {
                    "turn": int(turn),
                    "latency_ms": round(latency_ms, 3),
                    "error_type": type(exc).__name__,
                }
            )
            raise

        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        self.respond_latencies_ms.append(latency_ms)
        self.sessions.setdefault(session_id, []).append(
            _response_event(response, turn, latency_ms)
        )
        return response


def _metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "technical_score",
    )
    return {field: result.get(field) for field in fields}


def _turn_metrics(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    return [
        {
            "turn": turn,
            "cumulative_hits": sum(
                row.get("first_hit_turn") is not None
                and int(row["first_hit_turn"]) <= turn
                for row in rows
            ),
            "cumulative_hit_rate_at_10": (
                sum(
                    row.get("first_hit_turn") is not None
                    and int(row["first_hit_turn"]) <= turn
                    for row in rows
                )
                / total
                if total
                else 0.0
            ),
        }
        for turn in range(1, hard_evaluator.MAX_TURNS + 1)
    ]


def _sessions_with_traces(
    result: Mapping[str, Any],
    traces: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in result.get("sessions", []):
        sample_id = str(row["sample_id"])
        output.append(
            {
                "sample_id": sample_id,
                "scenario_type": str(row["scenario_type"]),
                "hit": bool(row["hit"]),
                "first_hit_turn": row.get("first_hit_turn"),
                "best_rank": row.get("best_rank"),
                "reciprocal_rank": row.get("reciprocal_rank"),
                "turns": traces.get(f"manual400:{sample_id}", []),
            }
        )
    return output


def _diagnostics(session_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    first_hit_turns = Counter(
        str(row["first_hit_turn"])
        if row.get("first_hit_turn") is not None
        else "miss"
        for row in session_rows
    )
    rank_buckets = Counter(
        "rank_1"
        if row.get("best_rank") == 1
        else "rank_2_3"
        if row.get("best_rank") in {2, 3}
        else "rank_4_5"
        if row.get("best_rank") in {4, 5}
        else "rank_6_10"
        if row.get("best_rank") in {6, 7, 8, 9, 10}
        else "miss"
        for row in session_rows
    )

    all_turns = [turn for row in session_rows for turn in row.get("turns", [])]
    asked = [
        turn
        for turn in all_turns
        if isinstance(turn.get("ask_attribute"), str)
        and turn.get("ask_attribute")
    ]
    asks_by_attribute = Counter(str(turn["ask_attribute"]) for turn in asked)

    scenario_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in session_rows:
        scenario_rows[str(row["scenario_type"])].append(row)

    return {
        "turn_level": {
            "overall": _turn_metrics(session_rows),
            "by_scenario": {
                scenario: _turn_metrics(rows)
                for scenario, rows in sorted(scenario_rows.items())
            },
        },
        "first_hit_turn_distribution": {
            str(turn): first_hit_turns.get(str(turn), 0)
            for turn in range(1, hard_evaluator.MAX_TURNS + 1)
        }
        | {"miss": first_hit_turns.get("miss", 0)},
        "rank_distribution": {
            bucket: rank_buckets.get(bucket, 0)
            for bucket in ("rank_1", "rank_2_3", "rank_4_5", "rank_6_10", "miss")
        },
        "canonicalization": {
            "status": "unavailable",
            "deterministic_dictionary_matches": None,
            "semantic_fallback_matches": None,
            "unresolved_phrases_or_values": None,
            "canonical_constraints_per_turn": None,
            "reason": "The Agent API exposes no canonicalization telemetry.",
        },
        "retrieval": {
            "status": "unavailable",
            "candidate_count_before_buying_filters": None,
            "candidate_count_after_buying_filters": None,
            "controlled_relaxation_count": None,
            "dense_full_catalog_fallback_count": None,
            "top_k_retrieval_latency": None,
            "reason": "The Agent API exposes no retrieval or candidate-pool telemetry.",
        },
        "clarification": {
            "status": "partial",
            "total_executed_turns": len(all_turns),
            "turns_with_ask_attribute": len(asked),
            "ask_attribute_rate": len(asked) / len(all_turns) if all_turns else 0.0,
            "sessions_with_ask_attribute": sum(
                any(turn in asked for turn in row.get("turns", []))
                for row in session_rows
            ),
            "ask_attribute_frequency": dict(sorted(asks_by_attribute.items())),
            "candidate_pool_shrinks_after_reply": None,
            "candidate_pool_shrink_status": "unavailable",
            "reason": "ask_attribute is observable; candidate-pool changes are not.",
        },
    }


def _report_directory(report_root: str | Path, run_name: str) -> Path:
    root = Path(report_root)
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    return root / _validate_run_name(run_name)


def _write_report(
    report_dir: Path,
    summary: Mapping[str, Any],
    session_rows: list[Mapping[str, Any]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "sessions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in session_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed Manual400 evaluator and write evaluator-side diagnostics."
    )
    parser.add_argument("--run-name", required=True, help="Unique name for the report directory.")
    parser.add_argument(
        "--report-root",
        default="reports/manual400",
        help="Report root relative to the repository root (default: reports/manual400).",
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--sessions", default="data/derived/gptannotation/sessions.jsonl")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Forward the hard evaluator's non-strict response handling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_name = _validate_run_name(args.run_name)
    report_dir = _report_directory(args.report_root, run_name)
    if report_dir.exists():
        raise FileExistsError(f"report directory already exists: {report_dir}")

    input_metadata = {
        "catalog": _file_metadata(args.catalog),
        "sessions": _file_metadata(args.sessions),
    }
    runtime_started = time.perf_counter_ns()
    startup_started = time.perf_counter_ns()
    delegate = Agent(args.catalog)
    startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000
    tracer = TracingAgent(delegate)

    sessions = hard_evaluator.load_jsonl(args.sessions)
    catalog_ids = hard_evaluator.load_catalog_ids(args.catalog)
    evaluation_started = time.perf_counter_ns()
    result = hard_evaluator.evaluate(
        agent=tracer,
        sessions=sessions,
        catalog_ids=catalog_ids,
        strict=not args.non_strict,
    )
    evaluation_ms = (time.perf_counter_ns() - evaluation_started) / 1_000_000
    total_runtime_ms = (time.perf_counter_ns() - runtime_started) / 1_000_000

    session_rows = _sessions_with_traces(result, tracer.sessions)
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "ok",
        "run": {
            "name": run_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_directory": str(report_dir),
        },
        "benchmark": {
            "name": "manual400",
            "max_turns": hard_evaluator.MAX_TURNS,
            "top_k": hard_evaluator.TOP_K,
            "expected_session_count": sum(hard_evaluator.SCENARIO_COUNTS.values()),
            "scenario_counts": dict(hard_evaluator.SCENARIO_COUNTS),
        },
        "commit": _git_metadata(),
        "artifacts": {
            "hard_evaluator": _source_metadata(hard_evaluator),
            "agent": _source_metadata(sys.modules[Agent.__module__], "Agent"),
            "inputs": input_metadata,
        },
        "overall_metrics": _metrics(result),
        "scenario_metrics": result.get("scenario_metrics", {}),
        "reported_token_usage": result.get("reported_token_usage", {}),
        "diagnostics": _diagnostics(session_rows),
        "latency": {
            "agent_startup": _latency_stats([startup_ms]),
            "respond": _latency_stats(tracer.respond_latencies_ms),
            "evaluation": _latency_stats([evaluation_ms]),
            "total_runtime_ms": round(total_runtime_ms, 3),
        },
        "execution": {
            "catalog": str(args.catalog),
            "sessions": str(args.sessions),
            "strict": not args.non_strict,
            "reports_do_not_change_benchmark_semantics": True,
        },
    }
    _write_report(report_dir, summary, session_rows)
    print(json.dumps(summary["overall_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

