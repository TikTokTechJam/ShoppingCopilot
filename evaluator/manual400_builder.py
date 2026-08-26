from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).parents[1]
DEFAULT_CATALOG = ROOT / "data" / "catalog.jsonl"
DEFAULT_PUBLIC = ROOT / "data" / "public_set.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "derived" / "manual400"
SEED = 20260826
TARGET_COUNT = 400
BATCH_SIZE = 20
SCENARIO_COUNTS = {"buying": 160, "browsing": 160, "intent_override": 60, "boundary": 20}
ALLOWED_ATTRIBUTES = {"category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other"}
GENERIC_STORES = {"", "amazon", "amazon.com", "unknown", "unbranded", "generic", "various"}
NOISE_RE = re.compile(r"\b(?:more than|longer|his company in|more details|great gift|imported|manufacturer)\b", re.I)
METADATA_RE = re.compile(r"date first available|item model number|product dimensions|package weight|\basin\b", re.I)
CATEGORY_NOISE_RE = re.compile(r"asin|test color|no title match|mfn only|\bprod\b|%|\boff\b|under\s*\$|over\s*\$|prime|sale|savings|discount|exclusive|most-loved|top brands|new arrivals|editor.?s picks|shop by|shop your style|coin\s*star|fashion-forward|veterans day|markdown|outlet|top rated|\btop\s+\d+\b|\b\d+\+\s*stars?\b|frequent|gift guide|fashion capital|clearance|testing|amazon fashion|\btest\b|edit|splurgeworthy|warm[- ]weather|summer-ready|fall-ready|fan favorites", re.I)
GENERIC_CATEGORY_NAMES = {"clothing, shoes & jewelry", "clothing", "fashion", "men", "women", "boys", "girls", "kids", "baby", "unisex"}

RAW_PATTERNS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "material": (
        ("stainless_steel", "stainless steel", r"\bstainless[- ]steel\b"),
        ("cotton", "cotton", r"\bcotton\b"),
        ("polyester", "polyester", r"\bpolyester\b"),
        ("nylon", "nylon", r"\bnylon\b"),
        ("leather", "leather", r"\bleather\b"),
        ("wool", "wool", r"\bwool\b"),
        ("spandex", "spandex", r"\bspandex\b"),
        ("silk", "silk", r"\bsilk\b"),
        ("rayon", "rayon", r"\brayon\b"),
        ("denim", "denim", r"\bdenim\b"),
        ("satin", "satin", r"\bsatin\b"),
        ("canvas", "canvas", r"\bcanvas\b"),
        ("mesh", "mesh", r"\bmesh\b"),
        ("rubber", "rubber", r"\brubber\b"),
    ),
    "color": tuple((value, value, rf"\b{value}\b") for value in (
        "black", "white", "blue", "red", "pink", "green", "brown", "purple", "yellow", "orange"
    )) + (("gray", "gray", r"\b(?:gray|grey)\b"),),
    "style": (
        ("casual", "casual", r"\bcasual\b"),
        ("classic", "classic", r"\bclassic\b"),
        ("formal", "formal", r"\bformal\b"),
        ("vintage", "vintage", r"\bvintage\b"),
        ("boho", "boho", r"\bboho\b"),
        ("athletic", "athletic", r"\bathletic\b"),
        ("sporty", "sporty", r"\bsporty\b"),
        ("relaxed_fit", "relaxed fit", r"\brelaxed[- ]fit\b"),
        ("regular_fit", "regular fit", r"\bregular[- ]fit\b"),
        ("slim_fit", "slim fit", r"\bslim[- ]fit\b"),
        ("wide_leg", "wide-leg fit", r"\bwide[- ]leg\b"),
        ("long_sleeve", "long-sleeve style", r"\blong[- ]sleeve\b"),
        ("short_sleeve", "short-sleeve style", r"\bshort[- ]sleeve\b"),
        ("crew_neck", "crew-neck style", r"\bcrew[- ]neck\b|\bcrew neckline\b"),
        ("v_neck", "V-neck style", r"\bv[- ]neck\b"),
        ("high_waisted", "high-waisted fit", r"\bhigh[- ]waisted\b"),
        ("straight_leg", "straight-leg fit", r"\bstraight[- ]leg\b"),
        ("statement", "statement style", r"\bstatement\b"),
    ),
    "use_case": (
        ("hiking", "hiking", r"\bhiking\b"),
        ("running", "running", r"\brunning\b"),
        ("trail", "trail use", r"\btrail\b"),
        ("gym", "gym workouts", r"\bgym\b"),
        ("workout", "workouts", r"\bworkouts?\b"),
        ("outdoor", "outdoor use", r"\boutdoor\b"),
        ("travel", "travel", r"\btravel\b"),
        ("beach", "beach use", r"\bbeach\b"),
        ("swimming", "swimming", r"\bswimm(?:ing|er)\b"),
        ("winter", "winter wear", r"\bwinter\b"),
        ("everyday_wear", "everyday wear", r"\beveryday\b"),
        ("formal_event", "formal events", r"\bformal events?\b"),
        ("party", "parties", r"\bpart(?:y|ies)\b"),
        ("office", "office wear", r"\boffice wear\b|\bfor the office\b"),
    ),
    "feature": (
        ("waterproof_protection", "waterproof protection", r"\bwater[- ]?proof\b"),
        ("water_resistance", "water resistance", r"\bwater[- ]?resistant\b|\bwater resistance\b"),
        ("moisture_wicking", "moisture-wicking performance", r"\bmoisture[- ]w?icking\b|\bwicking technology\b"),
        ("quick_drying", "quick-drying performance", r"\bquick[- ]dry(?:ing)?\b"),
        ("breathability", "breathability", r"\bbreathable\b|\bbreathability\b"),
        ("lightweight", "a lightweight design", r"\blightweight\b"),
        ("hypoallergenic", "hypoallergenic materials", r"\bhypoallergenic\b"),
        ("memory_foam", "memory-foam cushioning", r"\bmemory[- ]foam\b"),
        ("arch_support", "arch support", r"\barch support\b"),
        ("slip_resistance", "slip resistance", r"\b(?:non[- ]slip|slip[- ]resistant|slip resistance)\b"),
        ("thermal_insulation", "thermal insulation", r"\b(?:thermal insulation|insulated|insulation)\b"),
        ("uv_protection", "UV protection", r"\b(?:UV|UPF)\b.{0,35}\b(?:protect|block|shade)\w*\b"),
        ("stretch", "stretch fabric", r"\bstretch(?:y|able)?\b"),
        ("adjustable", "adjustability", r"\badjustable\b"),
        ("removable", "removable components", r"\bremovable\b"),
        ("pockets", "pockets", r"\bpockets?\b"),
        ("zipper", "zipper closure", r"\bzipper(?:ed)?\b"),
        ("buckle", "buckle closure", r"\bbuckles?\b"),
        ("soft_fabric", "soft fabric", r"\bsoft (?:fabric|material|on the skin)\b"),
    ),
    "size": (
        ("plus_size", "plus-size fit", r"\bplus[- ]size\b"),
        ("petite", "petite fit", r"\bpetite\b"),
        ("tall", "tall fit", r"\btall fit\b|\btall sizes?\b"),
        ("wide_fit", "wide fit", r"\bwide[- ]fit\b|\bwide width\b"),
        ("narrow_fit", "narrow fit", r"\bnarrow[- ]fit\b|\bnarrow width\b"),
        ("one_size", "one-size fit", r"\bone[- ]size\b"),
        ("small", "small size", r"\b(?:size|sizing|sizes|available in)\s+small\b"),
        ("medium", "medium size", r"\b(?:size|sizing|sizes|available in)\s+medium\b"),
        ("large", "large size", r"\b(?:size|sizing|sizes|available in)\s+large\b"),
        ("extra_small", "extra-small size", r"\b(?:size|sizing|sizes|available in)\s+(?:xxs|xs)\b"),
        ("extra_large", "extra-large size", r"\b(?:size|sizing|sizes|available in)\s+(?:xl|xxl|xxxl)\b"),
    ),
}
PATTERNS = {
    attribute: tuple((canonical, display, re.compile(pattern, re.I)) for canonical, display, pattern in values)
    for attribute, values in RAW_PATTERNS.items()
}
PHRASES = {
    "waterproof_protection": "waterproof protection",
    "water_resistance": "water resistance",
    "moisture_wicking": "moisture-wicking performance",
    "quick_drying": "quick-drying performance",
    "breathability": "breathability",
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


def clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(clean(item) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items() if item not in (None, ""))
    return re.sub(r"\s+", " ", str(value)).strip()


def leaf_category(product: Mapping[str, object]) -> str:
    values = [str(value).strip() for value in product.get("categories", ()) if str(value).strip()]
    for value in reversed(values):
        if (
            len(value) <= 80
            and value.lower() not in GENERIC_CATEGORY_NAMES
            and not CATEGORY_NOISE_RE.search(value)
        ):
            return value
    return "clothing item"

def evidence_entries(product: Mapping[str, object]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for field in ("title", "features", "description"):
        values = product.get(field)
        values = values if isinstance(values, list) else [values]
        for item in values:
            text = clean(item)
            if text:
                entries.append((field, text))
    return entries


def add_fact(
    facts: list[dict[str, object]],
    seen: set[tuple[str, str]],
    attribute: str,
    canonical: str,
    display: str,
    field: str,
    evidence: str,
    confidence: float,
) -> None:
    key = (attribute, canonical)
    if key in seen or not evidence:
        return
    seen.add(key)
    facts.append({
        "attribute": attribute,
        "canonical": canonical,
        "display": display,
        "evidence_field": field,
        "evidence_text": evidence,
        "confidence": round(confidence, 3),
    })


def add_pattern_facts(
    facts: list[dict[str, object]],
    seen: set[tuple[str, str]],
    attribute: str,
    entries: list[tuple[str, str]],
) -> None:
    for canonical, display, pattern in PATTERNS[attribute]:
        for field, evidence in entries:
            if pattern.search(evidence):
                confidence = 0.995 if field == "title" else 0.98 if field == "features" else 0.96
                add_fact(facts, seen, attribute, canonical, display, field, evidence, confidence)
                break


def budget_fact(product: Mapping[str, object]) -> dict[str, object] | None:
    raw = product.get("price")
    match = re.search(r"\d+(?:\.\d+)?", str(raw)) if raw not in (None, "") else None
    if not match:
        return None
    value = float(match.group(0))
    if not math.isfinite(value):
        return None
    if value < 25:
        canonical, display = "under_25", "under $25"
    elif value < 50:
        canonical, display = "25_to_50", "$25-$50"
    elif value < 100:
        canonical, display = "50_to_100", "$50-$100"
    elif value < 150:
        canonical, display = "100_to_150", "$100-$150"
    else:
        canonical, display = "over_150", "over $150"
    return {"attribute": "budget", "canonical": canonical, "display": display, "evidence_field": "price", "evidence_text": str(raw), "confidence": 0.995}


def label_product(product: Mapping[str, object]) -> dict[str, object]:
    facts: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    categories = [str(value).strip() for value in product.get("categories", ()) if str(value).strip()]
    add_fact(facts, seen, "category", leaf_category(product).lower().replace(" ", "_"), leaf_category(product), "categories", " > ".join(categories) or "clothing item", 0.995)
    entries = evidence_entries(product)
    for attribute in ("material", "color", "style", "use_case", "feature", "size"):
        add_pattern_facts(facts, seen, attribute, entries)
    price = budget_fact(product)
    if price:
        facts.append(price)
    store = clean(product.get("store"))
    if store and store.lower() not in GENERIC_STORES and len(store) <= 80:
        canonical = re.sub(r"[^a-z0-9]+", "_", store.lower()).strip("_")
        add_fact(facts, seen, "brand", canonical, store, "store", store, 0.97)
    priority = {"feature": 0, "material": 1, "use_case": 2, "style": 3, "color": 4, "size": 5, "budget": 6, "brand": 7, "category": 8}
    facts.sort(key=lambda item: (priority.get(str(item["attribute"]), 20), str(item["canonical"])))
    return {"parent_asin": str(product["parent_asin"]), "category": leaf_category(product), "validated_facts": facts[:9], "review_status": "curated_evidence_pass_v2"}


def useful_facts(label: Mapping[str, object]) -> list[dict[str, object]]:
    return [fact for fact in label.get("validated_facts", ()) if str(fact.get("attribute")) != "category"]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def selected_product(product: Mapping[str, object], rank: int) -> dict[str, object]:
    return {
        "selection_version": "manual400_v6",
        "selection_rank": rank,
        "parent_asin": str(product["parent_asin"]),
        "title": product.get("title") or "",
        "categories": product.get("categories") or [],
        "features": product.get("features") or [],
        "description": product.get("description") or [],
        "details": product.get("details") or {},
        "store": product.get("store") or "",
        "price": product.get("price"),
    }


def select_products(
    products: list[dict[str, object]],
    count: int,
    seed: int,
    frequencies: Mapping[tuple[str, str], int] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for product in products:
        label = label_product(product)
        candidates = hidden_candidates(label, frequencies) if frequencies is not None else useful_facts(label)
        category = leaf_category(product)
        if category != "clothing item" and len(candidates) >= 2 and len({str(fact["attribute"]) for fact in candidates}) >= 2:
            groups[category].append(product)
    rng = random.Random(seed)
    categories = sorted(groups)
    rng.shuffle(categories)
    for category in categories:
        groups[category].sort(key=lambda item: str(item["parent_asin"]))
        rng.shuffle(groups[category])
    chosen: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    chosen_ids: set[str] = set()
    for category in categories:
        if len(chosen) == count:
            break
        product = groups[category][0]
        chosen.append(product)
        category_counts[category] += 1
        chosen_ids.add(str(product["parent_asin"]))
    remaining = [item for category in categories for item in groups[category][1:]]
    rng.shuffle(remaining)
    for product in remaining:
        if len(chosen) == count:
            break
        category = leaf_category(product)
        if category_counts[category] >= 3 or str(product["parent_asin"]) in chosen_ids:
            continue
        chosen.append(product)
        category_counts[category] += 1
        chosen_ids.add(str(product["parent_asin"]))
    if len(chosen) != count:
        raise RuntimeError(f"could only select {len(chosen)} eligible products; need {count}")
    return (
        [selected_product(product, rank) for rank, product in enumerate(chosen, 1)],
        {
            "seed": seed,
            "eligible_products": sum(len(items) for items in groups.values()),
            "eligible_categories": len(groups),
            "selected_products": len(chosen),
            "selected_unique_parent_asin": len(chosen_ids),
            "selected_category_count": len(category_counts),
            "selected_category_distribution": dict(sorted(category_counts.items())),
            "skipped_poor_metadata": len(products) - sum(len(items) for items in groups.values()),
        },
    )


def field_contains(product: Mapping[str, object], field: str, evidence: str) -> bool:
    if field == "price":
        return clean(product.get("price")) == evidence
    if field == "categories":
        return evidence.lower() == " > ".join(str(value).strip() for value in product.get("categories", ())).lower()
    if field == "store":
        return clean(product.get("store")).lower() == evidence.lower()
    value = product.get(field)
    if isinstance(value, list):
        return any(clean(item).lower() == evidence.lower() for item in value)
    return clean(value).lower() == evidence.lower()


def audit_label(label: Mapping[str, object], product: Mapping[str, object]) -> list[str]:
    issues: list[str] = []
    if str(label.get("parent_asin")) != str(product.get("parent_asin")):
        issues.append("parent_asin_mismatch")
    if str(label.get("category")) != leaf_category(product):
        issues.append("category_mismatch")
    seen: set[tuple[str, str]] = set()
    for fact in label.get("validated_facts", ()):
        attribute = str(fact.get("attribute", ""))
        canonical = str(fact.get("canonical", ""))
        key = (attribute, canonical)
        if attribute not in ALLOWED_ATTRIBUTES:
            issues.append(f"invalid_attribute:{attribute}")
        if key in seen:
            issues.append(f"duplicate_fact:{attribute}:{canonical}")
        seen.add(key)
        field = str(fact.get("evidence_field", ""))
        evidence = str(fact.get("evidence_text", ""))
        if not field or not evidence or not field_contains(product, field, evidence):
            issues.append(f"evidence_not_verified:{attribute}:{canonical}")
        if METADATA_RE.search(evidence) or METADATA_RE.search(canonical):
            issues.append(f"metadata_fingerprint:{attribute}:{canonical}")
        if NOISE_RE.search(canonical) or NOISE_RE.search(str(fact.get("display", ""))):
            issues.append(f"noisy_fact:{attribute}:{canonical}")
        if float(fact.get("confidence", 0)) < 0.9:
            issues.append(f"low_confidence:{attribute}:{canonical}")
    if len(useful_facts(label)) < 2:
        issues.append("fewer_than_two_useful_facts")
    return sorted(set(issues))


def audit_and_repair(labels: list[dict[str, object]], products: Mapping[str, Mapping[str, object]], audit_path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    repaired: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    removed = 0
    for original in labels:
        product = products[str(original["parent_asin"])]
        valid: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for fact in original.get("validated_facts", ()):
            key = (str(fact.get("attribute")), str(fact.get("canonical")))
            if key in seen or str(fact.get("attribute")) not in ALLOWED_ATTRIBUTES or METADATA_RE.search(str(fact.get("evidence_text", ""))):
                removed += 1
                continue
            seen.add(key)
            valid.append(dict(fact))
        label = {**original, "validated_facts": valid[:9]}
        issues = audit_label(label, product)
        audit_rows.append({"parent_asin": label["parent_asin"], "status": "pass" if not issues else "needs_fix", "issues": issues})
        repaired.append(label)
    write_jsonl(audit_path, audit_rows)
    return repaired, {"facts_removed": removed, "questionable_labels_remaining": sum(row["status"] != "pass" for row in audit_rows)}


def fact_id(fact: Mapping[str, object]) -> tuple[str, str]:
    return str(fact.get("attribute")), str(fact.get("canonical"))


def fact_value(fact: Mapping[str, object]) -> str:
    return PHRASES.get(str(fact.get("canonical")), str(fact.get("display", "")))


def fact_sentence(fact: Mapping[str, object]) -> str:
    value = fact_value(fact)
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
    if attribute == "feature":
        if str(fact.get("canonical")) == "lightweight":
            return "A lightweight design would be ideal."
        if str(fact.get("canonical")) in {"breathability", "moisture_wicking", "quick_drying", "soft_fabric"}:
            return f"I would value {value}."
        return f"I'd like something with {value}."
    if attribute == "budget":
        return f"I'd like to stay {value}."
    if attribute == "brand":
        return f"I'd prefer {value}."
    return f"I'd like {value} to be important."


def override_sentence(fact: Mapping[str, object]) -> str:
    value = fact_value(fact)
    attribute = str(fact.get("attribute"))
    if attribute == "material":
        return f"Actually, my priority changed. I'd now prioritize {value}."
    if attribute == "color":
        return f"Actually, my priority changed. I'd now prefer {value}."
    if attribute == "style":
        return f"Actually, my priority changed. I now prefer a {value} look."
    if attribute == "use_case":
        return f"Actually, my priority changed. I'll mainly use it for {value}."
    if attribute == "size":
        return f"Actually, my priority changed. I now need a {value} fit."
    if attribute == "budget":
        return f"Actually, my priority changed. I'd like to stay within {value}."
    if attribute == "brand":
        return f"Actually, my priority changed. I'd now prefer {value}."
    return f"Actually, my priority changed. I'd like to prioritize {value}."

def fact_frequency(products: Iterable[Mapping[str, object]]) -> Counter[tuple[str, str]]:
    frequencies: Counter[tuple[str, str]] = Counter()
    for product in products:
        seen: set[tuple[str, str]] = set()
        for fact in useful_facts(label_product(product)):
            key = fact_id(fact)
            if key not in seen:
                frequencies[key] += 1
                seen.add(key)
    return frequencies


def choose_hidden_facts(label: Mapping[str, object], frequencies: Mapping[tuple[str, str], int], seed: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for original in label.get("validated_facts", ()):
        attribute = str(original.get("attribute"))
        key = fact_id(original)
        matches = int(frequencies.get(key, 0))
        if attribute == "category" or len(str(original.get("display", ""))) > 80:
            continue
        if matches < 8 or (attribute == "brand" and matches < 15):
            continue
        candidate = dict(original)
        candidate["catalog_match_count"] = matches
        candidates.append(candidate)
    if len(candidates) < 2:
        raise RuntimeError(f"{label['parent_asin']} has fewer than two non-unique hidden facts")
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda item: (-int(item["catalog_match_count"]), str(item["attribute"]), str(item["canonical"])))
    chosen: list[dict[str, object]] = []
    attributes: set[str] = set()
    for fact in candidates:
        if len(chosen) >= 4:
            break
        attribute = str(fact["attribute"])
        if attribute in attributes:
            continue
        chosen.append(fact)
        attributes.add(attribute)
    if len(chosen) < 2:
        raise RuntimeError(f"{label['parent_asin']} has fewer than two diverse hidden facts")
    return chosen


def profile_distribution(public: Iterable[Mapping[str, object]]) -> dict[str, object]:
    combos: Counter[tuple[str, ...]] = Counter()
    ratings: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    purchases: Counter[str] = Counter()
    for session in public:
        profile = session.get("user_profile", {})
        combos[tuple(sorted(str(tag) for tag in profile.get("preference_tags", ()) ))] += 1
        ratings[str(profile.get("average_prior_rating"))] += 1
        styles[str(profile.get("rating_style", "balanced"))] += 1
        purchases[str(profile.get("purchase_frequency", "occasional shopper"))] += 1
    return {
        "tag_combinations": [(list(tags), count) for tags, count in combos.items()],
        "ratings": dict(ratings),
        "rating_styles": dict(styles),
        "purchase_frequencies": dict(purchases),
    }


def weighted_choice(rng: random.Random, values: list[object], weights: list[int]) -> object:
    return rng.choices(values, weights=weights, k=1)[0] if values else None


def sample_profile(rng: random.Random, distribution: Mapping[str, object]) -> dict[str, object]:
    combos = list(distribution.get("tag_combinations", ()))
    combo = weighted_choice(rng, combos, [int(item[1]) for item in combos])
    tags = list(combo[0]) if combo else []
    ratings = distribution.get("ratings", {})
    rating_values = list(ratings)
    rating = weighted_choice(rng, rating_values, [int(ratings[value]) for value in rating_values])
    styles = distribution.get("rating_styles", {})
    style_values = list(styles)
    style = weighted_choice(rng, style_values, [int(styles[value]) for value in style_values])
    purchases = distribution.get("purchase_frequencies", {})
    purchase_values = list(purchases)
    purchase = weighted_choice(rng, purchase_values, [int(purchases[value]) for value in purchase_values])
    emphasis = ", ".join(tags) if tags else "general product quality"
    rating_text = style or "balanced"
    return {
        "average_prior_rating": None if rating in (None, "", "None") else float(rating),
        "preference_tags": tags,
        "purchase_frequency": purchase or "occasional shopper",
        "rating_style": rating_text,
        "summary": f"Prior purchases emphasize {emphasis}; ratings are {rating_text}.",
    }


def scenario_list(seed: int) -> list[str]:
    values = [scenario for scenario, count in SCENARIO_COUNTS.items() for _ in range(count)]
    random.Random(seed).shuffle(values)
    return values


def choose_fact(facts: list[dict[str, object]], allowed: set[str], excluded: set[tuple[str, str]] = frozenset()) -> dict[str, object] | None:
    return next((fact for fact in facts if str(fact["attribute"]) in allowed and fact_id(fact) not in excluded), None)


def build_sessions(labels: list[dict[str, object]], public: list[Mapping[str, object]], frequencies: Mapping[tuple[str, str], int], seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ordered = list(labels)
    random.Random(seed).shuffle(ordered)
    scenarios = scenario_list(seed)
    profiles = profile_distribution(public)
    sessions: list[dict[str, object]] = []
    debug: list[dict[str, object]] = []
    for index, (label, scenario) in enumerate(zip(ordered, scenarios), 1):
        hidden = choose_hidden_facts(label, frequencies, f"hidden:{seed}:{label['parent_asin']}")
        initial = choose_fact(hidden, {"material", "feature", "use_case", "style", "color", "size", "budget"})
        override = None
        if scenario == "intent_override":
            initial = choose_fact(hidden, {"style", "color", "material", "size", "use_case"}) or hidden[0]
            override = next((fact for fact in hidden if fact_id(fact) != fact_id(initial) and str(fact["attribute"]) != str(initial["attribute"])), None)
            override = override or next((fact for fact in hidden if fact_id(fact) != fact_id(initial)), None)
            if override is None:
                raise RuntimeError(f"{label['parent_asin']} cannot form a distinct override")
        category = str(label["category"]).lower()
        initial_message = f"I'm looking for {category}. {fact_sentence(initial)}" if scenario in {"buying", "intent_override"} and initial else f"I'm exploring {category} and would like to compare options."
        override_turn = 3 + random.Random(f"turn:{seed}:{label['parent_asin']}").randrange(2) if scenario == "intent_override" else None
        override_message = override_sentence(override) if override else None
        profile = sample_profile(random.Random(f"profile:{seed}:{label['parent_asin']}"), profiles)
        session = {
            "sample_id": f"manual400_{index:04d}",
            "scenario_type": scenario,
            "target_asin": str(label["parent_asin"]),
            "category": str(label["category"]),
            "user_profile": profile,
            "hidden_facts": hidden[:4],
            "initial_message": initial_message,
            "initial_fact_id": list(fact_id(initial)) if initial else None,
            "override_turn": override_turn,
            "override_message": override_message,
            "override_fact_id": list(fact_id(override)) if override else None,
            "boundary_first": scenario == "boundary" and random.Random(f"boundary:{seed}:{label['parent_asin']}").random() < 0.8,
        }
        sessions.append(session)
        hidden_ids = {fact_id(fact) for fact in hidden}
        debug.append({
            "sample_id": session["sample_id"],
            "target_asin": session["target_asin"],
            "scenario_type": scenario,
            "category": session["category"],
            "user_profile": profile,
            "hidden_facts": hidden,
            "initial_message": initial_message,
            "override_turn": override_turn,
            "override_fact": override,
            "fact_evidence": [fact for fact in label["validated_facts"] if fact_id(fact) in hidden_ids],
        })
    return sessions, debug


def hidden_candidates(
    label: Mapping[str, object],
    frequencies: Mapping[tuple[str, str], int],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for fact in label.get("validated_facts", ()):
        attribute = str(fact.get("attribute"))
        matches = int(frequencies.get(fact_id(fact), 0))
        if attribute == "category" or len(str(fact.get("display", ""))) > 80:
            continue
        if matches >= 8 and not (attribute == "brand" and matches < 15):
            result.append({**fact, "catalog_match_count": matches})
    return result

def build_manual400(catalog_path: str | Path = DEFAULT_CATALOG, public_path: str | Path = DEFAULT_PUBLIC, output_dir: str | Path = DEFAULT_OUTPUT, count: int = TARGET_COUNT, seed: int = SEED, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    catalog = read_jsonl(Path(catalog_path))
    public = read_jsonl(Path(public_path))
    output = Path(output_dir)
    frequencies = fact_frequency(catalog)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_products.jsonl"
    selected = read_jsonl(selected_path)
    selection_valid = len(selected) == count and len({str(row.get("parent_asin")) for row in selected}) == count and all(row.get("selection_version") == "manual400_v6" for row in selected)
    if selection_valid:
        for row in selected:
            candidates = hidden_candidates(label_product(row), frequencies)
            if len(candidates) < 2 or len({str(fact["attribute"]) for fact in candidates}) < 2:
                selection_valid = False
                break
    if not selection_valid:
        selected, sampling = select_products(catalog, count, seed, frequencies)
        write_jsonl(selected_path, selected)
    else:
        _, sampling = select_products(catalog, count, seed, frequencies)
        sampling["resumed_existing_selection"] = True
    products = {str(row["parent_asin"]): row for row in selected}
    labels_path = output / "labeled_products.jsonl"
    labels_by_asin = {str(row["parent_asin"]): row for row in read_jsonl(labels_path) if str(row.get("parent_asin")) in products and row.get("review_status") == "curated_evidence_pass_v2"}
    pending = [row for row in selected if str(row["parent_asin"]) not in labels_by_asin]
    if pending:
        with labels_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                for product in batch:
                    label = label_product(product)
                    labels_by_asin[str(label["parent_asin"])] = label
                    handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                print(f"labeled {min(start + len(batch), len(pending))}/{len(pending)} pending products")
    labels = [labels_by_asin[str(row["parent_asin"])] for row in selected]
    labels, audit_stats = audit_and_repair(labels, products, output / "label_audit.jsonl")
    write_jsonl(labels_path, labels)
    sessions, debug = build_sessions(labels, public, frequencies, seed)
    write_jsonl(output / "sessions.jsonl", sessions)
    write_jsonl(output / "session_debug.jsonl", debug)
    scenario_counts = Counter(str(session["scenario_type"]) for session in sessions)
    fact_counts = Counter(str(fact["attribute"]) for label in labels for fact in label["validated_facts"])
    categories = Counter(str(label["category"]) for label in labels)
    hidden_sizes = Counter(str(len(session["hidden_facts"])) for session in sessions)
    audit_rows = read_jsonl(output / "label_audit.jsonl")
    report = {
        "seed": seed,
        "total_products": len(labels),
        "category_distribution": dict(categories.most_common()),
        "facts_per_attribute": dict(fact_counts),
        "average_validated_facts_per_product": round(sum(len(label["validated_facts"]) for label in labels) / len(labels), 3),
        "sessions_per_scenario": dict(scenario_counts),
        "hidden_card_size_distribution": dict(hidden_sizes),
        "products_skipped_or_replaced": sampling.get("skipped_poor_metadata", 0),
        "sampling": sampling,
        "facts_removed_during_audit": audit_stats["facts_removed"],
        "questionable_labels_remaining": audit_stats["questionable_labels_remaining"],
        "duplicate_asin_check": len({str(label["parent_asin"]) for label in labels}) == count,
        "exact_scenario_counts": dict(scenario_counts) == SCENARIO_COUNTS,
        "public_profile_source_count": len(public),
        "audit_pass_count": sum(row.get("status") == "pass" for row in audit_rows),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for scenario in SCENARIO_COUNTS:
        print(f"\n[{scenario}]")
        shown = 0
        for session in sessions:
            if session["scenario_type"] != scenario or shown >= 5:
                continue
            print(json.dumps({"sample_id": session["sample_id"], "target_asin": session["target_asin"], "initial_message": session["initial_message"], "override_message": session["override_message"], "hidden_facts": [f"{fact['attribute']}={fact_value(fact)}" for fact in session["hidden_facts"]]}, ensure_ascii=False))
            shown += 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the evidence-backed TechJam manual400 benchmark")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--public", default=str(DEFAULT_PUBLIC))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    build_manual400(args.catalog, args.public, args.output, args.count, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
