from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from starter.agent import Agent


MAX_TURNS = 10
TOP_K = 10
SCENARIO_FRACTIONS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other", None,
}

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|modal|"
    r"acrylic|linen|canvas|denim|cashmere|rubber|mesh|foam|stainless steel|"
    r"steel|wood|gold|silver|silicone|alloy)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|"
    r"navy|beige|cream|ivory|khaki|teal|gold|silver|multicolor|multi[- ]color)\b",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"\b(xxs|xs|small|medium|large|xl|xxl|xxxl|wide|narrow|petite|tall|one size)\b",
    re.IGNORECASE,
)

EXTRA_MATERIALS = (
    "stainless steel", "faux leather", "genuine leather", "organic cotton",
    "polyurethane", "polyester", "cotton", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric", "modal", "acrylic", "linen",
    "canvas", "denim", "cashmere", "rubber", "mesh", "foam", "steel",
    "wood", "gold", "silver", "silicone", "alloy",
)
FEATURE_RULES = (
    (r"memory\s*[- ]?foam", "memory-foam cushioning", 6),
    (r"moisture\s*[- ]?wick(?:ing|s)?", "moisture wicking", 6),
    (r"water\s*[- ]?resistant", "water resistance", 6),
    (r"waterproof", "waterproof protection", 6),
    (r"polarized", "polarized glare reduction", 6),
    (r"\b(?:uv|uva|uvb)\b.{0,20}(?:protect|block)|(?:protect|block).{0,20}\b(?:uv|uva|uvb)\b", "UV protection", 5),
    (r"latex\s+grip", "latex grip", 6),
    (r"anti\s*[- ]?skid|non\s*[- ]?slip", "anti-skid grip", 6),
    (r"thermal|thermolite|insulat(?:ed|ion)", "thermal insulation", 5),
    (r"healing\s+gemstone", "healing gemstone detail", 6),
    (r"wide\s+leg", "wide-leg fit", 5),
    (r"breathable|breathability", "breathability", 5),
    (r"lightweight", "lightweight design", 4),
    (r"quick\s*[- ]?dry(?:ing)?", "quick-drying performance", 5),
    (r"stretch(?:y|able)?", "stretch comfort", 4),
    (r"adjustable", "adjustable fit", 3),
    (r"pockets?", "pockets", 2),
)
STYLE_MARKERS = (
    "casual", "formal", "classic", "athletic", "sporty", "slim", "relaxed fit",
    "loose fit", "regular fit", "vintage", "boho", "elegant", "short sleeve",
    "long sleeve", "crew neck", "v neck", "wide leg", "straight leg", "skinny",
)
USE_CASE_MARKERS = (
    "hiking", "running", "gym", "winter", "outdoor", "work", "trail", "travel",
    "everyday", "beach", "swim", "yoga", "pilates", "walking", "sport", "indoor",
)
WEAK_FACT_RE = re.compile(
    r"^(?:imported|made in (?:the )?usa|machine wash(?: only)?|hand wash(?: only)?|"
    r"no closure|pull on closure|button closure|zip closure|department\b|"
    r"date first available|manufacturer\b|item model number|product dimensions|"
    r"weight\b|closure\b|one size fits all)$",
    re.IGNORECASE,
)
SHORT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "this", "to", "with", "your", "you", "made",
    "great", "high", "quality", "features", "feature", "comfortable", "product",
}
GENERIC_CATEGORIES = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry", "shoes",
}
GENERIC_STORES = {"amazon", "amazon.com", "unknown", "unbranded"}


@dataclass(frozen=True)
class Fact:
    attribute: str
    key: str
    display: str
    priority: int = 1
    numeric: float | None = None


@dataclass
class TargetRecord:
    parent_asin: str
    category: str
    facts: dict[str, tuple[Fact, ...]]
    title_tokens: tuple[str, ...]
    ambiguity_score: float = 0.0


@dataclass
class StressSession:
    sample_id: str
    scenario_type: str
    target_asin: str
    category: str
    user_profile: dict[str, object]
    facts: dict[str, tuple[Fact, ...]]
    initial_message: str
    initial_fact_key: str | None = None
    override_turn: int | None = None
    override_message: str | None = None
    override_fact_key: str | None = None
    boundary_first: bool = False


def _string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [] if value in (None, "") else [str(value)]


def _clean(value: str, limit: int = 80) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\r\n")[:limit].rstrip()


def _key(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def _is_weak(value: str) -> bool:
    lowered = _clean(value).lower()
    if not lowered or WEAK_FACT_RE.fullmatch(lowered):
        return True
    return any(
        phrase in lowered
        for phrase in ("date first available", "item model number", "product dimensions")
    )


def _category_leaf(product: dict[str, object]) -> str:
    parts: list[str] = []
    for raw in _values(product.get("categories")):
        parts.extend(piece.strip() for piece in raw.split(",") if piece.strip())
    for value in reversed(parts):
        if value.lower() not in GENERIC_CATEGORIES:
            return _clean(value, 60)
    return "clothing item"


def _title_tokens(title: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in TOKEN_RE.findall(title)
        if len(token) > 2 and token.lower() not in SHORT_STOPWORDS
    )


def _add_fact(
    fact_map: defaultdict[str, dict[str, Fact]],
    attribute: str,
    display: str,
    priority: int = 1,
    numeric: float | None = None,
) -> None:
    display = _clean(display)
    if not display or _is_weak(display):
        return
    normalized = _key(display)
    if not normalized:
        return
    fact_map[attribute].setdefault(
        normalized,
        Fact(attribute, normalized, display, priority, numeric),
    )


def _price(value: object) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return None if match is None else float(match.group(0))


def _budget_phrase(value: float) -> str:
    if value <= 20:
        return "around $20 or below"
    lower = int(math.floor(value / 10.0) * 10)
    upper = max(lower + 10, int(math.ceil(value / 10.0) * 10))
    return f"around ${lower}-${upper}"


def _short_fact_phrase(raw: str) -> str:
    line = _clean(raw, 180)
    if ":" in line:
        line = line.split(":", 1)[1].strip()
    tokens = [token.lower() for token in TOKEN_RE.findall(line) if token.lower() not in SHORT_STOPWORDS]
    if len(tokens) < 2:
        return ""
    phrase = " ".join(tokens[:5])
    if _is_weak(phrase) or any(token in phrase.split() for token in ("percent", "imported")):
        return ""
    return phrase


def _feature_display(value: str) -> str:
    return value.replace("-", " ")


def normalize_product(product: dict[str, object]) -> TargetRecord:
    fact_map: defaultdict[str, dict[str, Fact]] = defaultdict(dict)
    category = _category_leaf(product)
    _add_fact(fact_map, "category", category, priority=3)

    title = _string(product.get("title"))
    features = _values(product.get("features"))
    descriptions = _values(product.get("description"))
    details = _values(product.get("details"))
    store = _string(product.get("store"))
    corpus = " ".join([title, *features, *descriptions, *details, store]).lower()

    for material in EXTRA_MATERIALS:
        if re.search(rf"\b{re.escape(material)}\b", corpus, re.IGNORECASE):
            _add_fact(fact_map, "material", material, priority=4)
    for color in sorted(set(match.lower() for match in COLOR_RE.findall(corpus))):
        _add_fact(fact_map, "color", color, priority=3)
    for size in sorted(set(match.lower() for match in SIZE_RE.findall(corpus))):
        _add_fact(fact_map, "size", size, priority=2)
    for marker in STYLE_MARKERS:
        if marker in corpus:
            _add_fact(fact_map, "style", marker, priority=3)
    for marker in USE_CASE_MARKERS:
        if marker in corpus:
            _add_fact(fact_map, "use_case", marker, priority=3)

    if store.strip() and store.strip().lower() not in GENERIC_STORES:
        _add_fact(fact_map, "brand", store, priority=2)
    for detail in details:
        if detail.lower().startswith("manufacturer:"):
            manufacturer = detail.split(":", 1)[1].strip()
            if manufacturer.lower() not in GENERIC_STORES:
                _add_fact(fact_map, "brand", manufacturer, priority=2)

    price = _price(product.get("price"))
    if price is not None:
        _add_fact(fact_map, "budget", _budget_phrase(price), priority=2, numeric=price)

    for pattern, display, priority in FEATURE_RULES:
        if re.search(pattern, corpus, re.IGNORECASE):
            _add_fact(fact_map, "feature", _feature_display(display), priority=priority)

    # Extract short normalized phrases from source fields for reusable coverage.
    # The source sentence itself is never retained or returned to the customer.
    for raw in [*features, *descriptions]:
        phrase = _short_fact_phrase(raw)
        if not phrase:
            continue
        if any(re.search(rf"\b{re.escape(value)}\b", phrase, re.IGNORECASE) for value in EXTRA_MATERIALS):
            continue
        if any(marker in phrase for marker in STYLE_MARKERS + USE_CASE_MARKERS):
            continue
        _add_fact(fact_map, "feature", phrase, priority=1)

    # Keep a small, non-wildcard pool for facts that do not fit a clean field.
    for raw in details:
        phrase = _short_fact_phrase(raw)
        if phrase and not any(marker in phrase for marker in STYLE_MARKERS + USE_CASE_MARKERS):
            if not any(phrase == fact.key for fact in fact_map["feature"].values()):
                _add_fact(fact_map, "other", phrase, priority=1)

    feature_keys = set(fact_map["feature"])
    if {"polarized glare reduction", "uv protection"}.issubset(feature_keys):
        _add_fact(fact_map, "feature", "glare reduction and UV protection outdoors", priority=7)
    if "thermal insulation" in feature_keys and any(
        marker in corpus for marker in ("winter", "outdoor", "cold weather")
    ):
        _add_fact(fact_map, "feature", "cold-weather outdoor warmth", priority=7)

    facts = {
        attribute: tuple(sorted(values.values(), key=lambda fact: (-fact.priority, fact.key))[:8])
        for attribute, values in fact_map.items()
        if values
    }
    return TargetRecord(
        parent_asin=str(product["parent_asin"]),
        category=category,
        facts=facts,
        title_tokens=_title_tokens(title),
    )


def load_catalog_records(catalog_path: str | Path) -> list[TargetRecord]:
    records: list[TargetRecord] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(normalize_product(json.loads(line)))
    return records


def _useful_facts(record: TargetRecord) -> list[Fact]:
    attributes = ("feature", "material", "color", "use_case", "style", "size", "budget", "brand")
    return [fact for attribute in attributes for fact in record.facts.get(attribute, ())]


def _prepare_ambiguity(records: list[TargetRecord]) -> None:
    category_counts = Counter(record.category for record in records)
    material_counts = Counter(
        fact.key for record in records for fact in record.facts.get("material", ())
    )
    color_counts = Counter(fact.key for record in records for fact in record.facts.get("color", ()))
    budget_counts = Counter(fact.key for record in records for fact in record.facts.get("budget", ()))
    title_counts = Counter(token for record in records for token in set(record.title_tokens))

    for record in records:
        category_score = math.log1p(category_counts[record.category])
        material_score = max(
            (math.log1p(material_counts[fact.key]) for fact in record.facts.get("material", ())),
            default=0.0,
        )
        color_score = max(
            (math.log1p(color_counts[fact.key]) for fact in record.facts.get("color", ())),
            default=0.0,
        )
        budget_score = max(
            (math.log1p(budget_counts[fact.key]) for fact in record.facts.get("budget", ())),
            default=0.0,
        )
        title_score = sum(math.log1p(title_counts[token]) for token in record.title_tokens) / max(
            1, len(record.title_tokens)
        )
        record.ambiguity_score = (
            1.5 * category_score
            + 0.6 * material_score
            + 0.4 * color_score
            + 0.5 * budget_score
            + 0.3 * title_score
            + min(0.8, 0.12 * len(_useful_facts(record)))
        )


def select_hard_targets(records: list[TargetRecord], count: int = 200) -> list[TargetRecord]:
    if not records or count <= 0:
        return []
    _prepare_ambiguity(records)
    category_counts = Counter(record.category for record in records)
    eligible = [
        record
        for record in records
        if category_counts[record.category] >= 8 and len(_useful_facts(record)) >= 2
    ]
    eligible.sort(key=lambda record: (-record.ambiguity_score, record.parent_asin))
    if len(eligible) < count:
        eligible = sorted(records, key=lambda record: (-record.ambiguity_score, record.parent_asin))

    selected: list[TargetRecord] = []
    selected_ids: set[str] = set()
    category_limit = max(2, math.ceil(count / 12))
    category_selected: Counter[str] = Counter()
    for record in eligible:
        if len(selected) >= count:
            break
        if category_selected[record.category] >= category_limit:
            continue
        selected.append(record)
        selected_ids.add(record.parent_asin)
        category_selected[record.category] += 1
    if len(selected) < count:
        for record in eligible:
            if len(selected) >= count:
                break
            if record.parent_asin not in selected_ids:
                selected.append(record)
                selected_ids.add(record.parent_asin)
    return selected[:count]


def _scenario_counts(count: int) -> dict[str, int]:
    raw = {name: count * fraction for name, fraction in SCENARIO_FRACTIONS.items()}
    result = {name: math.floor(value) for name, value in raw.items()}
    for name in sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))[: count - sum(result.values())]:
        result[name] += 1
    return result


def _profile(rng: random.Random) -> dict[str, object]:
    tags = ["comfort", "fit", "durability", "quality", "style", "value"]
    rng.shuffle(tags)
    return {
        "purchase_frequency": rng.choice(("1-2 prior purchases", "3-4 prior purchases", "frequent shopper")),
        "average_prior_rating": rng.choice((3.5, 4.0, 4.5, 5.0)),
        "rating_style": rng.choice(("usually positive", "balanced", "critical")),
        "preference_tags": tags[:2],
        "summary": "Prior shopping preferences are anonymized and intentionally non-target-specific.",
    }


def _fact_phrase(fact: Fact, rng: random.Random) -> str:
    value = fact.display
    templates = {
        "material": (
            "I'd prefer something made primarily from {value}.",
            "{value} is important to me.",
            "I'd like a {value}-based material.",
        ),
        "feature": (
            "I'd prefer something with {value}.",
            "That {value} detail matters to me.",
            "I care about {value}.",
        ),
        "color": (
            "I'd like it in {value}.",
            "The {value} color appeals to me.",
            "Please prioritize {value}.",
        ),
        "size": (
            "I need a {value} size or fit.",
            "A {value} fit would work best.",
            "Please keep {value} sizing in mind.",
        ),
        "style": (
            "I'd prefer a {value} style.",
            "The {value} look is what I want.",
            "Please prioritize a {value} fit or style.",
        ),
        "brand": (
            "I'd prefer the {value} brand if possible.",
            "The {value} store or brand would be good.",
            "Please keep {value} in mind.",
        ),
        "budget": (
            "I'd like to stay {value}.",
            "A budget of {value} would work for me.",
            "Please keep the price {value}.",
        ),
        "use_case": (
            "I'd mainly use it for {value}.",
            "It should work well for {value}.",
            "My main use would be {value}.",
        ),
        "other": (
            "One extra detail I care about is {value}.",
            "Please keep the {value} detail in mind.",
        ),
        "category": (
            "I'm mainly considering {value}.",
            "The {value} category is what I need.",
        ),
    }
    return rng.choice(templates.get(fact.attribute, ("{value} matters to me.",))).format(value=value)


def _choose_fact(record: TargetRecord, attributes: Iterable[str], excluded: set[str] | None = None) -> Fact | None:
    excluded = excluded or set()
    for attribute in attributes:
        for fact in record.facts.get(attribute, ()):
            if fact.key not in excluded:
                return fact
    return None


def build_stress_sessions(
    catalog: str | Path | list[TargetRecord],
    count: int = 200,
    seed: int = 20260826,
) -> list[StressSession]:
    records = load_catalog_records(catalog) if not isinstance(catalog, list) else catalog
    selected = select_hard_targets(records, count)
    rng = random.Random(seed)
    rng.shuffle(selected)
    scenario_labels = [
        scenario
        for scenario, scenario_count in _scenario_counts(len(selected)).items()
        for _ in range(scenario_count)
    ]
    rng.shuffle(scenario_labels)

    sessions: list[StressSession] = []
    for index, (record, scenario) in enumerate(zip(selected, scenario_labels), start=1):
        session_rng = random.Random(f"{seed}:{record.parent_asin}:{index}")
        useful = _useful_facts(record)
        initial_fact: Fact | None = None
        override_fact: Fact | None = None
        if scenario == "buying":
            initial_fact = _choose_fact(record, ("feature", "material", "color", "use_case", "style", "budget", "brand"))
            initial_message = f"I'm looking for {record.category}."
            if initial_fact:
                initial_message += " " + _fact_phrase(initial_fact, session_rng)
        elif scenario == "browsing":
            initial_fact = _choose_fact(record, ("use_case",))
            initial_message = f"I'm exploring {record.category}"
            if initial_fact:
                initial_message += f" for {initial_fact.display}."
            else:
                initial_message += " and would like to compare options."
        elif scenario == "intent_override":
            initial_fact = _choose_fact(record, ("style", "use_case", "feature", "material", "color", "budget"))
            override_fact = _choose_fact(
                record,
                ("feature", "material", "color", "use_case", "style", "size", "budget", "brand"),
                {initial_fact.key} if initial_fact else set(),
            )
            if override_fact is None and len(useful) > 1:
                override_fact = useful[1]
            initial_message = f"I'm looking for {record.category}."
            if initial_fact:
                initial_message += " " + _fact_phrase(initial_fact, session_rng)
            override_value = override_fact.display if override_fact else "a more important requirement"
            override_templates = (
                "Actually, that earlier preference isn't important anymore. I care much more about {value}.",
                "I've changed my mind about that. {value} is the requirement I need.",
                "Forget my earlier preference; {value} is what matters now.",
            )
            override_message = session_rng.choice(override_templates).format(value=override_value)
        else:
            initial_message = f"I'm exploring {record.category} and want to compare options."
            override_message = None

        if scenario != "intent_override":
            override_message = None
        override_turn = 3 + session_rng.randrange(2) if scenario == "intent_override" else None
        sessions.append(
            StressSession(
                sample_id=f"hard_{index:04d}",
                scenario_type=scenario,
                target_asin=record.parent_asin,
                category=record.category,
                user_profile=_profile(session_rng),
                facts=record.facts,
                initial_message=initial_message,
                initial_fact_key=initial_fact.key if initial_fact else None,
                override_turn=override_turn,
                override_message=override_message,
                override_fact_key=override_fact.key if override_fact else None,
                boundary_first=(scenario == "boundary" and session_rng.random() < 0.8),
            )
        )
    return sessions


def _no_preference(attribute: str | None, rng: random.Random) -> str:
    if attribute == "budget":
        choices = ("I don't have a budget preference.", "I'm flexible on price.", "Any reasonable price is fine.")
    else:
        choices = (
            "I don't really have a preference there.",
            "That isn't important to me.",
            "I'm flexible there.",
        )
    return rng.choice(choices)


def simulate_customer_reply(
    session: StressSession,
    ask_attribute: object,
    state: dict[str, object],
    rng: random.Random,
) -> str:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other" if attribute is not None else None
    asked = state.setdefault("asked_attributes", set())
    disclosed = state.setdefault("disclosed", set())
    stale = state.setdefault("stale_constraints", set())
    no_preference = state.setdefault("no_preference_attributes", set())
    active_constraints = state.setdefault("active_constraints", set())
    assert isinstance(asked, set)
    assert isinstance(disclosed, set)
    assert isinstance(stale, set)
    assert isinstance(no_preference, set)
    assert isinstance(active_constraints, set)
    if attribute is not None:
        asked.add(attribute)

    if (
        session.boundary_first
        and not bool(state.get("boundary_used"))
        and attribute is not None
    ):
        state["boundary_used"] = True
        no_preference.add(attribute)
        return _no_preference(attribute, rng)

    if attribute is None or attribute in no_preference:
        return _no_preference(attribute, rng)

    available = [
        fact
        for fact in session.facts.get(attribute, ())
        if fact.key not in disclosed and fact.key not in stale
    ]
    if not available:
        no_preference.add(attribute)
        return _no_preference(attribute, rng)

    fact = available[0]
    disclosed.add(fact.key)
    active_constraints.add(fact.key)
    return _fact_phrase(fact, rng)


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= TOP_K:
            break
    return result


def metric_summary(sessions: list[dict[str, object]]) -> dict[str, object]:
    if not sessions:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
    hit_rate = sum(int(item["hit"]) for item in sessions) / len(sessions)
    mrr = statistics.fmean(float(item["reciprocal_rank"]) for item in sessions)
    mttc = statistics.fmean(
        int(item["first_hit_turn"]) if item["first_hit_turn"] is not None else MAX_TURNS + 1
        for item in sessions
    )
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
    }


def score_metrics(overall: dict[str, object]) -> tuple[float, float, float]:
    mttc = float(overall["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = (
        0.50 * float(overall["hit_rate_at_10"])
        + 0.30 * float(overall["mrr"])
        + 0.20 * efficiency
    )
    return round(efficiency, 6), round(technical_score, 6), mttc


def evaluate(
    agent: Agent,
    sessions: list[StressSession],
    catalog_ids: set[str] | None = None,
) -> dict[str, object]:
    catalog_ids = catalog_ids or {session.target_asin for session in sessions}
    results: list[dict[str, object]] = []
    for session in sessions:
        session_id = f"hard_{session.sample_id}"
        agent.reset(session_id, session.user_profile)
        seed = f"reply:{session.sample_id}:{session.target_asin}"
        rng = random.Random(seed)
        state: dict[str, object] = {
            "disclosed": {session.initial_fact_key} if session.initial_fact_key else set(),
            "active_constraints": {session.initial_fact_key} if session.initial_fact_key else set(),
            "stale_constraints": set(),
            "asked_attributes": set(),
            "no_preference_attributes": set(),
            "boundary_used": False,
        }
        user_message = session.initial_message
        override_applied = session.scenario_type != "intent_override"
        hit_turn: int | None = None
        best_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            if session.scenario_type == "intent_override" and turn == session.override_turn:
                override_applied = True
                if session.initial_fact_key:
                    state["stale_constraints"].add(session.initial_fact_key)
                state["active_constraints"] = (
                    {session.override_fact_key} if session.override_fact_key else set()
                )
                if session.override_fact_key:
                    state["disclosed"].add(session.override_fact_key)
                user_message = session.override_message or "I've changed my mind about the earlier preference."

            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and session.target_asin in ranked:
                best_rank = ranked.index(session.target_asin) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            if session.scenario_type == "intent_override" and turn + 1 == session.override_turn:
                continue
            user_message = simulate_customer_reply(session, response.get("ask_attribute"), state, rng)

        results.append(
            {
                "sample_id": session.sample_id,
                "scenario_type": session.scenario_type,
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )

    overall = metric_summary(results)
    efficiency, technical_score, _ = score_metrics(overall)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["scenario_type"])].append(result)
    return {
        **overall,
        "efficiency": efficiency,
        "recommended_technical_score": technical_score,
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
        "sessions": results,
    }


def _display_summary(result: dict[str, object]) -> dict[str, object]:
    return {
        "sample_count": result["sample_count"],
        "HitRate@10": result["hit_rate_at_10"],
        "MRR": result["mrr"],
        "MTTC": result["mttc"],
        "Efficiency": result["efficiency"],
        "TechnicalScore": result["recommended_technical_score"],
        "scenario_metrics": result["scenario_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic hard TechJam stress evaluator")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    records = load_catalog_records(args.catalog)
    sessions = build_stress_sessions(records, args.sample_count, args.seed)
    result = evaluate(Agent(args.catalog), sessions, {record.parent_asin for record in records})
    print(json.dumps(_display_summary(result), indent=2))


if __name__ == "__main__":
    main()
