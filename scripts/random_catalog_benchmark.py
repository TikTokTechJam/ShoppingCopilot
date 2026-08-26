from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.agent import Agent


SCENARIO_WEIGHTS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scenario_mix(sample_count: int) -> list[str]:
    if sample_count == 10:
        # Closest useful integer approximation of the official 40/40/15/5 mix,
        # while retaining at least one override and one boundary case.
        return ["buying"] * 4 + ["browsing"] * 4 + ["intent_override", "boundary"]
    raw = {name: sample_count * weight for name, weight in SCENARIO_WEIGHTS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    for name in ("intent_override", "boundary"):
        if sample_count >= 4 and counts[name] == 0:
            counts[name] = 1
    while sum(counts.values()) > sample_count:
        reducible = max(
            (name for name in counts if counts[name] > 1),
            key=lambda name: counts[name] - raw[name],
        )
        counts[reducible] -= 1
    while sum(counts.values()) < sample_count:
        name = max(raw, key=lambda item: raw[item] - counts[item])
        counts[name] += 1
    return [name for name in SCENARIO_WEIGHTS for _ in range(counts[name])]


def raw_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(item) for item in value.values() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def category_label(product: dict) -> str:
    raw_categories = product.get("categories") or []
    values = raw_categories if isinstance(raw_categories, list) else [raw_categories]
    labels: list[str] = []
    for value in values:
        for part in str(value).split(","):
            cleaned = " ".join(part.split()).strip()
            if cleaned:
                labels.append(cleaned)
    return " ".join(labels[-2:]) if labels else "clothing item"


def independent_clues(product: dict, seed: int) -> list[str]:
    """Sample raw metadata clues without reproducing evaluator intent-card order."""
    candidates = [
        *raw_values(product.get("features")),
        *raw_values(product.get("details")),
        *raw_values(product.get("description")),
    ]
    if product.get("price") not in (None, ""):
        candidates.append(f"priced near ${product['price']}")
    cleaned = list(
        dict.fromkeys(
            " ".join(value.split()).strip(" -;,.\t\n")[:140]
            for value in candidates
            if " ".join(value.split()).strip(" -;,.\t\n")
        )
    )
    rng = random.Random(f"{seed}\0{product.get('parent_asin', '')}")
    rng.shuffle(cleaned)
    fallback = category_label(product)
    while len(cleaned) < 4:
        cleaned.append(fallback)
    return cleaned[:4]


def free_form_conversation(product: dict, scenario: str, seed: int = 0) -> tuple[list[str], int]:
    category = category_label(product)
    c0, c1, c2, c3 = independent_clues(product, seed)
    filler = "I don't have an additional preference; show me more options."

    if scenario == "buying":
        opening = [
            f"I need {category}. The key requirement is {c0}.",
            f"Please also prioritize {c1} and {c2}.",
            f"One more deciding detail is {c3}.",
        ]
        conversion_allowed_turn = 1
    elif scenario == "browsing":
        opening = [
            f"Help me explore {category}; I am open to ideas.",
            f"I care about {c0} and {c1}.",
            f"It should also have {c2} and {c3}.",
        ]
        conversion_allowed_turn = 1
    elif scenario == "intent_override":
        opening = [
            f"I am browsing {category}, and initially I like {c3}.",
            filler,
            f"Actually, ignore that and switch to {category} with {c0} and {c1} instead.",
            f"The remaining details are {c2} and {c3}.",
        ]
        conversion_allowed_turn = 3
    else:
        opening = [
            f"Help me browse {category}.",
            "I don't have a preference for material; please use your judgment.",
            f"What matters after all is {c0} and {c1}.",
            f"You can also prioritize {c2} and {c3}.",
        ]
        conversion_allowed_turn = 1
    return (opening + [filler] * (10 - len(opening)))[:10], conversion_allowed_turn


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def run_benchmark(
    catalog_path: str | Path,
    public_set_path: str | Path,
    sample_count: int,
    seed: int,
) -> dict:
    benchmark_started = time.perf_counter()
    catalog = load_jsonl(catalog_path)
    public_targets = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in load_jsonl(public_set_path)
    }
    eligible = sorted(
        (
            product
            for product in catalog
            if str(product.get("parent_asin") or "") not in public_targets
            and product.get("title")
            and product.get("categories")
        ),
        key=lambda product: str(product["parent_asin"]),
    )
    if sample_count < 1 or sample_count > len(eligible):
        raise ValueError(f"sample_count must be between 1 and {len(eligible)}")

    rng = random.Random(seed)
    selected = rng.sample(eligible, sample_count)
    scenarios = scenario_mix(sample_count)
    rng.shuffle(scenarios)

    index_started = time.perf_counter()
    agent = Agent(catalog_path)
    index_seconds = time.perf_counter() - index_started

    response_seconds: list[float] = []
    sessions: list[dict] = []
    for index, (product, scenario) in enumerate(zip(selected, scenarios, strict=True), start=1):
        target = str(product["parent_asin"])
        session_id = f"random_{seed}_{index:02d}"
        profile = {
            "purchase_frequency": "independent benchmark",
            "average_prior_rating": None,
            "rating_style": "not provided",
            "preference_tags": ["fit", "comfort", "durability"],
            "summary": "Neutral synthetic profile for independent catalog sampling.",
        }
        agent.reset(session_id, profile)
        messages, conversion_allowed_turn = free_form_conversation(product, scenario, seed)
        hit_turn: int | None = None
        target_rank: int | None = None
        transcript: list[dict] = []
        for turn, message in enumerate(messages, start=1):
            response_started = time.perf_counter()
            response = agent.respond(session_id, message, turn, 10)
            response_seconds.append(time.perf_counter() - response_started)
            ranked = [
                str(item.get("parent_asin") or "")
                for item in response.get("recommendations") or []
                if isinstance(item, dict)
            ][:10]
            transcript.append(
                {
                    "turn": turn,
                    "customer": message,
                    "ask_attribute": response.get("ask_attribute"),
                    "top_10": ranked,
                }
            )
            if turn >= conversion_allowed_turn and target in ranked:
                hit_turn = turn
                target_rank = ranked.index(target) + 1
                break
        sessions.append(
            {
                "sample_id": session_id,
                "scenario_type": scenario,
                "target_parent_asin": target,
                "target_title": str(product.get("title") or ""),
                "target_category": category_label(product),
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "target_rank": target_rank,
                "reciprocal_rank": 0.0 if target_rank is None else 1.0 / target_rank,
                "transcript": transcript,
            }
        )

    hit_rate = statistics.fmean(float(session["hit"]) for session in sessions)
    mrr = statistics.fmean(session["reciprocal_rank"] for session in sessions)
    mttc = statistics.fmean(
        session["first_hit_turn"] if session["first_hit_turn"] is not None else 11
        for session in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    total_seconds = time.perf_counter() - benchmark_started
    return {
        "benchmark": "independent random catalog free-form benchmark",
        "seed": seed,
        "sample_count": sample_count,
        "selection": {
            "method": "uniform random sample from sorted eligible catalog products",
            "eligible_products": len(eligible),
            "excluded_public_targets": len(public_targets),
            "scenario_mix": {name: scenarios.count(name) for name in SCENARIO_WEIGHTS},
            "uses_public_session_targets": False,
            "clue_generation": "fixed-seed shuffle of raw features, detail values, descriptions, and price",
            "reconstructs_evaluator_intent_card": False,
        },
        "metrics": {
            "hit_rate_at_10": round(hit_rate, 6),
            "mrr": round(mrr, 6),
            "mttc": round(mttc, 6),
            "efficiency": round(efficiency, 6),
            "technical_score": round(technical_score, 6),
        },
        "timing": {
            "index_build_seconds": round(index_seconds, 6),
            "response_count": len(response_seconds),
            "mean_response_ms": round(statistics.fmean(response_seconds) * 1000.0, 3),
            "p95_response_ms": round(percentile_95(response_seconds) * 1000.0, 3),
            "max_response_ms": round(max(response_seconds, default=0.0) * 1000.0, 3),
            "total_benchmark_seconds": round(total_seconds, 6),
        },
        "sessions": sessions,
    }


def print_summary(result: dict) -> None:
    metrics = result["metrics"]
    timing = result["timing"]
    print(f"Random catalog benchmark (seed={result['seed']}, n={result['sample_count']})")
    print(
        "HitRate@10={hit_rate_at_10:.6f}  MRR={mrr:.6f}  MTTC={mttc:.6f}  "
        "Efficiency={efficiency:.6f}  TechnicalScore={technical_score:.6f}".format(**metrics)
    )
    print(
        f"Index={timing['index_build_seconds']:.3f}s  "
        f"Mean response={timing['mean_response_ms']:.3f}ms  "
        f"P95={timing['p95_response_ms']:.3f}ms  Responses={timing['response_count']}"
    )
    print()
    print("#  Scenario         Turn  Rank  ASIN        Title")
    for index, session in enumerate(result["sessions"], start=1):
        turn = session["first_hit_turn"] if session["first_hit_turn"] is not None else "MISS"
        rank = session["target_rank"] if session["target_rank"] is not None else "-"
        title = session["target_title"]
        if len(title) > 76:
            title = title[:73] + "..."
        print(
            f"{index:>2} {session['scenario_type']:<16} {str(turn):>4}  {str(rank):>4}  "
            f"{session['target_parent_asin']:<10}  {title}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark random non-public catalog targets")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", default="docs/random_benchmark_10.json")
    args = parser.parse_args()

    result = run_benchmark(args.catalog, args.public_set, args.samples, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_summary(result)
    print(f"\nFull transcripts written to {output_path}")


if __name__ == "__main__":
    main()
