from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.random_catalog_benchmark import percentile_95
from starter.agent import Agent


NEUTRAL_FILLER = "I don't have an additional preference; please show a different set."


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: object) -> str:
    return " ".join(str(value).lower().split())


def metric_summary(sessions: list[dict]) -> dict:
    if not sessions:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
            "efficiency": 0.0,
            "technical_score": 0.0,
        }
    hit_rate = statistics.fmean(float(session["hit"]) for session in sessions)
    mrr = statistics.fmean(session["reciprocal_rank"] for session in sessions)
    mttc = statistics.fmean(
        session["first_hit_turn"] if session["first_hit_turn"] is not None else 11
        for session in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(technical_score, 6),
    }


def validate_fixture(
    fixture: dict,
    catalog: dict[str, dict],
    public_targets: set[str],
    prior_targets: set[str],
) -> None:
    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("The human benchmark must contain exactly 20 cases")
    case_ids = [str(case.get("case_id") or "") for case in cases]
    targets = [str(case.get("target_parent_asin") or "") for case in cases]
    if len(set(case_ids)) != len(case_ids) or not all(case_ids):
        raise ValueError("case_id values must be non-empty and unique")
    if len(set(targets)) != len(targets):
        raise ValueError("Target ASIN values must be unique")
    if set(targets) & public_targets:
        raise ValueError("Human benchmark overlaps the public evaluator targets")
    if set(targets) & prior_targets:
        raise ValueError("Human benchmark overlaps the earlier random benchmark")

    eligible = sorted(
        asin
        for asin, product in catalog.items()
        if asin not in public_targets | prior_targets
        and product.get("title")
        and product.get("categories")
    )
    expected = random.Random(int(fixture["selection_seed"])).sample(eligible, 20)
    if targets != expected:
        raise ValueError("Targets do not match the declared uniform fixed-seed sample")

    actual_mix = Counter(str(case.get("scenario_type") or "") for case in cases)
    declared_mix = Counter({key: int(value) for key, value in fixture["scenario_mix"].items()})
    if actual_mix != declared_mix:
        raise ValueError(f"Scenario mix mismatch: actual={actual_mix}, declared={declared_mix}")

    for case in cases:
        target = str(case["target_parent_asin"])
        if target not in catalog:
            raise ValueError(f"Unknown target ASIN: {target}")
        messages = case.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 10:
            raise ValueError(f"{case['case_id']} must contain 1-10 messages")
        if not all(isinstance(message, str) and message.strip() for message in messages):
            raise ValueError(f"{case['case_id']} contains an empty/non-string message")
        joined = normalize_text(" ".join(messages))
        title = normalize_text(catalog[target].get("title") or "")
        if target.lower() in joined:
            raise ValueError(f"{case['case_id']} leaks the target ASIN")
        if title and title in joined:
            raise ValueError(f"{case['case_id']} copies the complete target title")
        allowed_turn = int(case.get("conversion_allowed_turn") or 1)
        if not 1 <= allowed_turn <= len(messages):
            raise ValueError(f"Invalid conversion_allowed_turn for {case['case_id']}")


def run_benchmark(
    catalog_path: str | Path,
    public_set_path: str | Path,
    prior_benchmark_path: str | Path,
    fixture_path: str | Path,
) -> dict:
    benchmark_started = time.perf_counter()
    catalog_rows = load_jsonl(catalog_path)
    catalog = {str(product["parent_asin"]): product for product in catalog_rows}
    public_targets = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in load_jsonl(public_set_path)
    }
    prior_result = json.loads(Path(prior_benchmark_path).read_text(encoding="utf-8"))
    prior_targets = {
        str(session["target_parent_asin"])
        for session in prior_result.get("sessions") or []
    }
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    validate_fixture(fixture, catalog, public_targets, prior_targets)

    index_started = time.perf_counter()
    agent = Agent(catalog_path)
    index_seconds = time.perf_counter() - index_started

    response_seconds: list[float] = []
    ask_attributes: Counter[str] = Counter()
    sessions: list[dict] = []
    for case in fixture["cases"]:
        target = str(case["target_parent_asin"])
        session_id = str(case["case_id"])
        agent.reset(
            session_id,
            {
                "purchase_frequency": "human benchmark",
                "average_prior_rating": None,
                "rating_style": "not provided",
                "preference_tags": ["fit", "comfort", "durability"],
                "summary": "Neutral profile; the hand-authored current request takes priority.",
            },
        )
        human_messages = list(case["messages"])
        allowed_turn = int(case["conversion_allowed_turn"])
        hit_turn: int | None = None
        target_rank: int | None = None
        transcript: list[dict] = []
        for turn in range(1, 11):
            is_human_message = turn <= len(human_messages)
            message = human_messages[turn - 1] if is_human_message else NEUTRAL_FILLER
            response_started = time.perf_counter()
            response = agent.respond(session_id, message, turn, 10)
            response_seconds.append(time.perf_counter() - response_started)
            attribute = response.get("ask_attribute")
            if isinstance(attribute, str):
                ask_attributes[attribute] += 1
            ranked = [
                str(item.get("parent_asin") or "")
                for item in response.get("recommendations") or []
                if isinstance(item, dict)
            ][:10]
            transcript.append(
                {
                    "turn": turn,
                    "message_source": "human_authored" if is_human_message else "neutral_filler",
                    "customer": message,
                    "ask_attribute": attribute,
                    "top_10": ranked,
                }
            )
            if turn >= allowed_turn and target in ranked:
                hit_turn = turn
                target_rank = ranked.index(target) + 1
                break

        product = catalog[target]
        sessions.append(
            {
                "case_id": session_id,
                "scenario_type": str(case["scenario_type"]),
                "target_parent_asin": target,
                "target_title": str(product.get("title") or ""),
                "human_message_count": len(human_messages),
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "target_rank": target_rank,
                "reciprocal_rank": 0.0 if target_rank is None else 1.0 / target_rank,
                "transcript": transcript,
            }
        )

    overall_metrics = metric_summary(sessions)
    scenario_metrics = {
        scenario: metric_summary([item for item in sessions if item["scenario_type"] == scenario])
        for scenario in sorted({item["scenario_type"] for item in sessions})
    }
    return {
        "benchmark": fixture["benchmark"],
        "selection_seed": fixture["selection_seed"],
        "authoring_method": fixture["authoring_method"],
        "sample_count": len(sessions),
        "validation": {
            "uniform_seeded_targets_verified": True,
            "public_target_overlap": 0,
            "prior_random_target_overlap": 0,
            "complete_title_copies": 0,
            "target_asin_leaks": 0,
            "evaluator_generated_messages": 0,
        },
        "scenario_mix": dict(Counter(session["scenario_type"] for session in sessions)),
        "metrics": overall_metrics,
        "scenario_metrics": scenario_metrics,
        "clarification_attributes": dict(ask_attributes),
        "timing": {
            "index_build_seconds": round(index_seconds, 6),
            "response_count": len(response_seconds),
            "mean_response_ms": round(statistics.fmean(response_seconds) * 1000.0, 3),
            "p95_response_ms": round(percentile_95(response_seconds) * 1000.0, 3),
            "max_response_ms": round(max(response_seconds, default=0.0) * 1000.0, 3),
            "total_benchmark_seconds": round(time.perf_counter() - benchmark_started, 6),
        },
        "sessions": sessions,
    }


def print_summary(result: dict) -> None:
    metrics = result["metrics"]
    timing = result["timing"]
    print(f"Human-authored benchmark (n={result['sample_count']}, seed={result['selection_seed']})")
    print(
        "HitRate@10={hit_rate_at_10:.6f}  MRR={mrr:.6f}  MTTC={mttc:.6f}  "
        "Efficiency={efficiency:.6f}  TechnicalScore={technical_score:.6f}".format(**metrics)
    )
    print(
        f"Index={timing['index_build_seconds']:.3f}s  "
        f"Mean response={timing['mean_response_ms']:.3f}ms  "
        f"P95={timing['p95_response_ms']:.3f}ms  Responses={timing['response_count']}"
    )
    print("Clarifications: " + ", ".join(
        f"{name}={count}" for name, count in sorted(result["clarification_attributes"].items())
    ))
    print()
    print("#  Scenario         Turn  Rank  ASIN        Title")
    for index, session in enumerate(result["sessions"], start=1):
        turn = session["first_hit_turn"] if session["first_hit_turn"] is not None else "MISS"
        rank = session["target_rank"] if session["target_rank"] is not None else "-"
        title = session["target_title"]
        if len(title) > 70:
            title = title[:67] + "..."
        print(
            f"{index:>2} {session['scenario_type']:<16} {str(turn):>4}  {str(rank):>4}  "
            f"{session['target_parent_asin']:<10}  {title}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed human-authored 20-query benchmark")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--prior-benchmark", default="docs/random_benchmark_10.json")
    parser.add_argument("--fixture", default="benchmarks/human_queries_20.json")
    parser.add_argument("--output", default="docs/human_benchmark_20_results.json")
    args = parser.parse_args()

    result = run_benchmark(
        args.catalog,
        args.public_set,
        args.prior_benchmark,
        args.fixture,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_summary(result)
    print(f"\nFull transcripts written to {output_path}")


if __name__ == "__main__":
    main()
