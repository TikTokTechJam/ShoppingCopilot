"""Dynamic synthetic test-case generator (Task 2).

Utterances are generated per intent type from real catalog vocabulary, and the
prompt carries our own canonical constraint fields so the generator writes
cases that exercise the extractor rather than generic e-commerce chatter. The
specification asks for three to five constraints per utterance; that is the
band where an intent decision is genuinely contested, because Phase 1 of
``TwoPhaseIntentRouter`` routes BUYING on two or more filled fields while the
signal ledger can still read the same sentence as exploratory.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from dictionary.registry import SEMANTIC_QUERY_STOPWORDS
from starter.routing.constraints import CATEGORICAL_FIELDS, extract_constraints
from starter.routing.lexicon import BUYING_TAG_THRESHOLD, TAG_COUNT_EXCLUDE
from tests.utils.catalog_loader import summarize_products
from tests.utils.intent_workflow import (
    BROWSING,
    BUYING,
    INTENT_OVERRIDE,
    normalize_intent,
)
from tests.utils.llm_client import parse_json


MIN_CONSTRAINTS = 3
MAX_CONSTRAINTS = 5

# BROWSING cannot carry the same constraint load as the other two intents.
# Phase 1 of ``TwoPhaseIntentRouter`` routes BUYING as soon as an utterance
# fills ``BUYING_TAG_THRESHOLD`` canonical fields, counting everything except
# ``TAG_COUNT_EXCLUDE`` -- naming a product category says which shelf the
# shopper is at, not that they have decided. So an utterance with three
# counted constraints is BUYING by our own definition, and asking the
# generator for a three-constraint BROWSING case asks for a case that cannot
# exist. The browsing band is derived from the threshold rather than written
# down, so retuning the router retunes the generator with it.
BROWSING_MAX_CONSTRAINTS = max(1, BUYING_TAG_THRESHOLD - 1)

# Fields that do not count toward the buying threshold, and so may appear in a
# BROWSING utterance freely.
UNCOUNTED_FIELDS = tuple(TAG_COUNT_EXCLUDE)

# Read from the workflow rather than restated, so a change to the canonical
# attribute set reaches the generator without editing this file.
CONSTRAINT_FIELDS = (*CATEGORICAL_FIELDS, "price")

INTENT_GUIDANCE = {
    BUYING: (
        "High-intent, specific purchase queries: explicit constraints such as "
        "budget, size, or product specifications, urgency, or direct purchase "
        "signals like 'looking to buy' or 'under $X'."
    ),
    BROWSING: (
        "Low-intent, exploratory, open-ended queries: broad category "
        "exploration, feature inquiries, or soft preference matching with no "
        "hard purchasing commitment, such as 'what kinds of...' or "
        "'show me options for...'. Less than 3 preference tags introduced."
    ),
    INTENT_OVERRIDE: (
        "A mid-session pivot: the shopper abruptly shifts category, changes a "
        "constraint they already gave, or cancels a previous preference, such "
        "as 'Actually, forget shoes, show me jackets' or 'Never mind, change "
        "the budget to $200'."
    ),
}


def intent_constraint_band(intent_type: str) -> tuple[int, int]:
    """How many counted constraints an utterance of this intent may carry."""

    if normalize_intent(intent_type) == BROWSING:
        return 0, BROWSING_MAX_CONSTRAINTS
    return MIN_CONSTRAINTS, MAX_CONSTRAINTS


def _is_conversational(surface: str) -> bool:
    """Whether every token of a matched surface is a query stopword."""

    tokens = [token for token in str(surface).casefold().split() if token]
    return bool(tokens) and all(token in SEMANTIC_QUERY_STOPWORDS for token in tokens)


def counted_constraints(utterance: str) -> tuple[str, ...]:
    """The fields our extractor reads, excluding the ones that do not count.

    Each entry is ``field:surface`` where the surface is the text that matched,
    because the count is only as trustworthy as the extractor: a rejection
    reading ``brand:you`` is a dictionary false positive, not a shopper who
    named a brand, and the log should say so rather than leave it to be
    guessed.
    """

    try:
        constraints = extract_constraints(utterance or "")
        fields = constraints.populated_fields(exclude=UNCOUNTED_FIELDS)
        surfaces: Dict[str, str] = {}
        for item in getattr(constraints, "evidence", ()):
            attribute = str(getattr(item, "attribute", ""))
            raw = str(getattr(item, "raw_text", "") or "").strip()
            if not attribute or not raw or attribute in surfaces:
                continue
            # A dictionary hit on a conversational word is not a commitment the
            # shopper made. "you" and "show" are registered as a brand and a
            # use_case, so counting them would make almost any polite question
            # look over-constrained. The repo's own query stopword list is the
            # existing definition of a word that carries no product meaning.
            if _is_conversational(raw):
                continue
            surfaces[attribute] = raw
        return tuple(
            (f"{field}:{surfaces[field]}" if field in surfaces else field)
            for field in fields
            if field in surfaces or field == "price"
        )
    except Exception:  # noqa: BLE001 - a generator check must not break a run
        return ()


def build_prompt(
    intent_type: str,
    catalog_sample: Sequence[Dict],
    count: int,
) -> str:
    intent = normalize_intent(intent_type)
    needs_prior = intent == INTENT_OVERRIDE
    prior_rule = (
        "Every case MUST include 'prior_context': one or two earlier shopper "
        "turns, newline separated, that establish the preference the override "
        "then replaces. The override utterance must contradict or cancel "
        "something stated in prior_context."
        if needs_prior
        else "Set 'prior_context' to an empty string: these are opening turns "
        "of a session with no earlier state."
    )
    low, high = intent_constraint_band(intent)
    if intent == BROWSING:
        constraint_rule = f"""
The assistant extracts these canonical constraint fields:
{', '.join(CONSTRAINT_FIELDS)}.

A BROWSING utterance is defined by how FEW of them it commits to. Each
utterance must name AT MOST {high} of these fields:
{', '.join(f for f in CONSTRAINT_FIELDS if f not in UNCOUNTED_FIELDS)}.
Naming a product type does not count against that limit, so say what kind of
product you are looking at ({', '.join(UNCOUNTED_FIELDS)}) and then stop.

Concretely: do NOT name a brand or store, a price or budget, or a size. Use at
most ONE soft attribute (a colour, a material, a style, an occasion) and only
if the sentence still sounds undecided.

Do NOT state a budget, a size, a brand and a colour in the same sentence: an
utterance carrying {BUYING_TAG_THRESHOLD} or more of the counted fields is a
BUYING utterance by definition, not a BROWSING one, no matter how casually it
is phrased. What makes these cases hard is exploratory or hesitant wording,
open questions, and soft preferences -- not a stack of requirements.

Vary the phrasing across the {count} cases: open questions, window-shopping,
vague quality words, undecided hedging, requests for options or ideas."""
    else:
        constraint_rule = f"""
The assistant extracts these canonical constraint fields:
{', '.join(CONSTRAINT_FIELDS)}.
Each utterance must mention between {low} and {high} of them, so the case
genuinely exercises constraint extraction. Vary which fields you use across
the {count} cases, and vary the phrasing: some direct, some conversational,
some with hedging."""

    return f"""
You are generating benchmark test cases for an e-commerce shopping assistant's
intent classifier. Write {count} DISTINCT shopper utterances whose intent is
exactly {intent}.

{intent} means: {INTENT_GUIDANCE.get(intent, '')}

Ground every utterance in this real catalog sample; do not invent products
outside this domain:
{json.dumps(summarize_products(list(catalog_sample)), indent=2, ensure_ascii=False)}
{constraint_rule}

{prior_rule}

Return ONLY a JSON object of this exact shape, with no commentary:
{{"cases": [{{"utterance": "...", "prior_context": "...", "expected_intent": "{intent}"}}]}}
""".strip()


def _coerce_cases(payload: Any, intent: str) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        raw_cases = payload.get("cases", payload.get("test_cases", []))
    elif isinstance(payload, list):
        raw_cases = payload
    else:
        raw_cases = []

    cases: List[Dict[str, Any]] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        utterance = str(item.get("utterance", "")).strip()
        if not utterance:
            continue
        cases.append(
            {
                "utterance": utterance,
                "prior_context": str(item.get("prior_context", "") or "").strip(),
                "expected_intent": intent,
                "constraint_fields": list(counted_constraints(utterance)),
            }
        )
    return cases


def band_violation(
    case: Dict[str, Any],
    intent: str,
    *,
    enforce_minimum: bool = True,
) -> str:
    """Why this case does not match its intent's constraint band, or ''.

    The maximum is definitional and always enforced: a BROWSING utterance that
    fills two counted fields is BUYING under our own Phase 1 rule, so scoring
    the classifier against it would penalise the correct answer.

    The minimum is not definitional -- it exists to make generated cases hard
    -- and it measures our extractor's granularity rather than the utterance's.
    "the wool jacket ... under $100" reads as one category plus a price because
    "wool jacket" resolves to a single category value, so a perfectly rich
    sentence scores 1. Callers building a ground-truth dataset pass
    ``enforce_minimum=False`` so the corpus is not biased toward the phrasings
    the current extractor happens to parse finely.
    """

    low, high = intent_constraint_band(intent)
    fields = case.get("constraint_fields") or []
    if len(fields) > high:
        return (
            f"{len(fields)} counted constraints {tuple(fields)} exceeds the "
            f"{intent} maximum of {high}"
        )
    if enforce_minimum and len(fields) < low:
        return (
            f"{len(fields)} counted constraints is below the {intent} minimum "
            f"of {low}"
        )
    return ""


def generate_synthetic_intent_cases(
    intent_type: str,
    catalog_sample: List[Dict],
    count: int = 10,
    *,
    client: Any = None,
    enforce_minimum: bool = True,
) -> List[Dict]:
    """Uses LLM to generate test cases containing (sentence, ground_truth_intent, context).

    For INTENT OVERRIDE, it must generate both the prior session state and the
    override utterance.
    """

    intent = normalize_intent(intent_type)
    if client is None:
        from tests.utils.llm_client import build_client

        client = build_client()

    kept: List[Dict[str, Any]] = []
    over_band: List[Dict[str, Any]] = []
    seen: set[str] = set()
    rejected: List[str] = []

    # One top-up attempt: a discarded case leaves the suite short, and asking
    # again is cheaper than lowering the bar for what counts as its intent.
    for attempt in range(2):
        if len(kept) >= count:
            break
        payload = parse_json(
            client.annotate(build_prompt(intent, catalog_sample, count - len(kept)))
        )
        for case in _coerce_cases(payload, intent):
            if case["utterance"].casefold() in seen:
                continue
            seen.add(case["utterance"].casefold())
            if intent == INTENT_OVERRIDE and not case["prior_context"]:
                # An override with no prior turn is not an override; the Agent
                # never tests the opening utterance of a session for one.
                rejected.append(f"{case['utterance']!r} -> no prior_context")
                continue
            problem = band_violation(case, intent, enforce_minimum=enforce_minimum)
            if problem:
                rejected.append(f"{case['utterance']!r} -> {problem}")
                over_band.append(case)
                continue
            kept.append(case)

    if rejected:
        print(
            f"[{intent}] {len(rejected)} generated case(s) discarded for not "
            f"matching the intent definition:"
        )
        for item in rejected:
            print(f"  - {item}")

    if not kept and over_band:
        # Every candidate broke the band. Returning nothing would abort the run
        # with no evidence, which is the least useful outcome: the least
        # constrained candidates are kept, flagged, and left for the judge to
        # adjudicate, so the report shows what the generator actually produced.
        over_band.sort(key=lambda case: len(case.get("constraint_fields") or []))
        kept = over_band[:count]
        for case in kept:
            case["band_warning"] = band_violation(
                case, intent, enforce_minimum=enforce_minimum
            )
        print(
            f"[{intent}] no case met the constraint band; keeping the "
            f"{len(kept)} least constrained and flagging them"
        )
    return kept[:count]


__all__ = [
    "BROWSING_MAX_CONSTRAINTS",
    "CONSTRAINT_FIELDS",
    "band_violation",
    "counted_constraints",
    "intent_constraint_band",
    "MAX_CONSTRAINTS",
    "MIN_CONSTRAINTS",
    "build_prompt",
    "generate_synthetic_intent_cases",
]
