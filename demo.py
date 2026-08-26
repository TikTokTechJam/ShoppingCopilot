from __future__ import annotations

import argparse

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent


def run_demo(catalog_path: str, dataset_path: str, sample_id: str) -> int:
    samples = load_jsonl(dataset_path)
    try:
        sample = next(item for item in samples if item["sample_id"] == sample_id)
    except StopIteration as error:
        available = ", ".join(item["sample_id"] for item in samples[:5])
        raise SystemExit(f"Unknown sample {sample_id!r}. Examples: {available}") from error

    catalog_ids, categories, products = catalog_index(catalog_path)
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective_sample,
        coarse_category(categories.get(target, [])),
        disclosed,
    )

    agent = Agent(catalog_path)
    session_id = f"demo_{sample_id}"
    agent.reset(session_id, sample["user_profile"])

    print(f"Scenario: {sample['scenario_type']} ({sample_id})")
    print(f"Profile: {sample['user_profile']['summary']}")
    for turn in range(1, MAX_TURNS + 1):
        print(f"\nTURN {turn}")
        print(f"Customer: {user_message}")
        response = agent.respond(session_id, user_message, turn, TOP_K)
        print(f"Copilot: {response['message']}")
        print(f"Requested attribute: {response['ask_attribute']}")
        ranked = normalize_recommendations(response["recommendations"], catalog_ids)
        for rank, parent_asin in enumerate(ranked[:3], start=1):
            title = products[parent_asin].get("title") or "Untitled product"
            print(f"  {rank}. {parent_asin} — {title}")

        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            print(f"\nCONVERTED on turn {turn} at rank {rank}: {target}")
            return 0
        if turn == MAX_TURNS:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective_sample,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )

    print(f"\nNo conversion. Hidden target was {target}.")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one visible multi-turn public-set demo")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-id", default="public_0002")
    args = parser.parse_args()
    raise SystemExit(run_demo(args.catalog, args.dataset, args.sample_id))


if __name__ == "__main__":
    main()
