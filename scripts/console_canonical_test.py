"""Interactive diagnostic for exact intent, constraints, and intersections."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from dictionary.registry import ATTRIBUTE_FIELDS, normalize_text
from starter.routing.constraints import ShoppingConstraints, extract_constraints
from starter.routing.intent_router import TwoPhaseIntentRouter


DEFAULT_ANNOTATIONS = Path("data/derived/annotations/v4/annotations.jsonl")


@dataclass(frozen=True)
class ProductTagIndex:
    """Normalized inverted indexes for successful V4 product annotations."""

    postings: Mapping[str, Mapping[str, frozenset[str]]]
    product_count: int
    prices: Mapping[str, float | None] = field(default_factory=dict)


def _fact_values(value: object) -> Iterable[object]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return value
    return ()


def build_product_index(path: str | Path = DEFAULT_ANNOTATIONS) -> ProductTagIndex:
    """Build the V4 inverted index once, without changing the source JSONL."""

    postings: dict[str, dict[str, set[str]]] = {
        attribute: defaultdict(set) for attribute in ATTRIBUTE_FIELDS
    }
    asins: set[str] = set()
    prices: dict[str, float | None] = {}

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{source}:{line_number}: record must be an object")

            annotation = record.get("annotation")
            if not isinstance(annotation, Mapping) or annotation.get("status") != "success":
                continue

            asin = record.get("parent_asin")
            if not isinstance(asin, str) or not asin.strip():
                raise ValueError(f"{source}:{line_number}: missing parent_asin")
            asin = asin.strip()
            if asin in asins:
                raise ValueError(f"{source}:{line_number}: duplicate parent_asin {asin}")
            asins.add(asin)

            raw_price = record.get("price")
            if raw_price is not None and (
                isinstance(raw_price, bool) or not isinstance(raw_price, (int, float))
            ):
                raise ValueError(f"{source}:{line_number}: price must be numeric or null")
            prices[asin] = None if raw_price is None else float(raw_price)

            facts = record.get("facts")
            if not isinstance(facts, Mapping):
                raise ValueError(f"{source}:{line_number}: facts must be an object")
            for attribute in ATTRIBUTE_FIELDS:
                for raw_value in _fact_values(facts.get(attribute)):
                    if not isinstance(raw_value, str):
                        continue
                    value = normalize_text(raw_value)
                    if value:
                        postings[attribute][value].add(asin)

    frozen = {
        attribute: {
            value: frozenset(product_ids)
            for value, product_ids in values.items()
        }
        for attribute, values in postings.items()
    }
    return ProductTagIndex(postings=frozen, product_count=len(asins), prices=prices)


def _constraint_values(constraints: object, attribute: str) -> tuple[str, ...]:
    value: Any = getattr(constraints, attribute, None)
    if value is None and isinstance(constraints, Mapping):
        value = constraints.get(attribute)
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def canonical_tags(constraints: ShoppingConstraints) -> dict[str, list[str]]:
    """Return only the seven displayed categorical fields, in fixed order."""

    return {
        attribute: list(_constraint_values(constraints, attribute))
        for attribute in ATTRIBUTE_FIELDS
        if _constraint_values(constraints, attribute)
    }


def _price_bounds(constraints: object) -> tuple[float | None, float | None]:
    def value_for(name: str) -> float | None:
        value: Any = getattr(constraints, name, None)
        if value is None and isinstance(constraints, Mapping):
            value = constraints.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    return value_for("price_min"), value_for("price_max")


def _budget_payload(constraints: object) -> dict[str, int | float]:
    price_min, price_max = _price_bounds(constraints)
    budget: dict[str, int | float] = {}
    if price_min is not None:
        budget["min"] = int(price_min) if price_min.is_integer() else price_min
    if price_max is not None:
        budget["max"] = int(price_max) if price_max.is_integer() else price_max
    return budget


def _price_matches(
    price: float | None,
    price_min: float | None,
    price_max: float | None,
) -> bool:
    if price is None:
        return False
    if price_min is not None and price < price_min:
        return False
    if price_max is not None and price > price_max:
        return False
    return True


def count_products_with_all_tags(
    constraints: ShoppingConstraints | Mapping[str, object],
    index: ProductTagIndex,
) -> int | None:
    """Count products matching categorical values and an optional budget.

    A price constraint filters the categorical intersection to products with a
    known price satisfying the existing inclusive min/max convention. Without
    a price constraint, null-priced products remain eligible.
    """

    tag_sets: list[frozenset[str]] = []
    for attribute in ATTRIBUTE_FIELDS:
        for raw_value in _constraint_values(constraints, attribute):
            normalized = normalize_text(raw_value)
            if not normalized:
                continue
            matches = index.postings.get(attribute, {}).get(normalized)
            if not matches:
                return 0
            tag_sets.append(matches)

    price_min, price_max = _price_bounds(constraints)
    has_price = price_min is not None or price_max is not None
    if not tag_sets:
        if not has_price:
            return None
        result = set(index.prices)
    else:
        result = set(min(tag_sets, key=len))
        for matches in tag_sets:
            result.intersection_update(matches)

    if has_price:
        result = {
            asin
            for asin in result
            if _price_matches(index.prices.get(asin), price_min, price_max)
        }
    return len(result)


def analyze_utterance(
    utterance: str,
    index: ProductTagIndex,
    router: TwoPhaseIntentRouter | None = None,
) -> tuple[str, dict[str, object], int | None]:
    """Run the existing router/canonicalizer and then the exact index."""

    active_router = router or TwoPhaseIntentRouter()
    intent_result = active_router.classify(utterance)
    constraints = intent_result.constraints or extract_constraints(utterance)
    tags: dict[str, object] = dict(canonical_tags(constraints))
    budget = _budget_payload(constraints)
    if budget:
        tags["budget"] = budget
    count = count_products_with_all_tags(constraints, index)
    intent = "BUYING" if str(intent_result.intent).upper() == "BUYING" else "BROWSING"
    return intent, tags, count


def print_analysis(intent: str, tags: Mapping[str, object], count: int | None) -> None:
    print(f"Intent: {intent}")
    print()
    print(json.dumps(dict(tags), indent=2, ensure_ascii=False))
    print()
    if count is None:
        print("No constraints matched.")
    else:
        print(f"Products matching all constraints: {count}")


def run_console(annotations_path: str | Path = DEFAULT_ANNOTATIONS) -> None:
    router = TwoPhaseIntentRouter()
    index = build_product_index(annotations_path)

    print("Loaded intent router.")
    print("Loaded canonical dictionary.")
    print(f"Indexed {index.product_count:,} V4 products.")
    print("Type `exit` to quit.")

    try:
        while True:
            try:
                utterance = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if utterance.casefold() in {"exit", "quit"}:
                return
            intent, tags, count = analyze_utterance(utterance, index, router)
            print_analysis(intent, tags, count)
    except KeyboardInterrupt:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect exact canonical constraints and V4 product intersections."
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANNOTATIONS),
        help="Successful V4 annotation JSONL path.",
    )
    args = parser.parse_args()
    run_console(args.annotations)


if __name__ == "__main__":
    main()
