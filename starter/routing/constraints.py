"""Extract canonical shopping constraints from one user utterance (issue #7).

Produces the record shape issue #7 specifies:

    {"category": [], "brand": [], "price_min": None, "price_max": None,
     "color": [], "material": [], "size": [], "style": [], "feature": [],
     "use_case": []}

Two properties matter for the intent router that consumes this:

**Values, not topics.** A tag is only recorded when the customer names an
actual value. "I'm not sure what style I want yet" mentions style but commits
to nothing, so it yields no style tag. This is the difference between this
module and the signal ledger in `lexicon.py`, which deliberately fires on the
bare attribute word as weak evidence.

**Nothing is invented.** Values that look like constraints but map to no
canonical entry are preserved in `unmapped` rather than being forced onto the
nearest vocabulary item, as issue #7 requires.

The vocabulary here is a curated starting point. Issue #5 (canonical catalog
facts) and issue #8 (canonical attribute dictionary) are the intended owners;
`CANONICAL_VOCAB` is deliberately a plain data structure so it can be replaced
wholesale by a generated dictionary without touching the extraction code.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache


# Field -> ((canonical_value, alias_pattern), ...). Order matters: the first
# match wins, so more specific aliases must precede more general ones.
CANONICAL_VOCAB: dict[str, tuple[tuple[str, str], ...]] = {
    "category": (
        ("hiking_boots", r"hiking boots?|walking boots?"),
        ("running_shoes", r"running shoes?|running sneakers?|trainers?"),
        ("walking_shoes", r"walking shoes?"),
        ("sneakers", r"sneakers?|plimsolls?"),
        ("boots", r"boots?"),
        ("sandals", r"sandals?"),
        ("slippers", r"slippers?"),
        ("heels", r"heels?|pumps?"),
        ("handbag", r"handbags?|purses?|totes?"),
        ("backpack", r"backpacks?|rucksacks?"),
        ("jacket", r"jackets?|windbreakers?|parkas?"),
        ("coat", r"coats?|peacoats?|overcoats?"),
        ("hoodie", r"hoodies?|sweatshirts?"),
        ("cardigan", r"cardigans?"),
        ("sweater", r"sweaters?|jumpers?|pullovers?"),
        ("shirt", r"shirts?|blouses?|tees?|t-shirts?|tops?"),
        ("dress", r"dresses|dress"),
        ("skirt", r"skirts?"),
        ("trousers", r"trousers|pants|chinos|slacks"),
        ("jeans", r"jeans|denims"),
        ("shorts", r"shorts"),
        ("swimsuit", r"swimsuits?|bikinis?|swimwear|bathing suits?"),
        ("socks", r"socks?"),
        ("hat", r"hats?|caps?|beanies?"),
        ("scarf", r"scarves|scarf"),
        ("gloves", r"gloves?|mittens?"),
        ("belt", r"belts?"),
        ("watch", r"watch(?:es)?"),
        ("necklace", r"necklaces?"),
        ("bracelet", r"bracelets?"),
        ("ring", r"rings?"),
        ("earrings", r"earrings?"),
        ("blazer", r"blazers?"),
        ("vest", r"vests?|waistcoats?"),
        ("pyjamas", r"pyjamas|pajamas|nightgowns?"),
        ("bra", r"bras?"),
    ),
    "color": (
        ("navy", r"navy|dark blue|midnight blue"),
        ("black", r"black|jet black"),
        ("white", r"white|off[- ]white"),
        ("grey", r"gr[ae]y|charcoal|slate"),
        ("blue", r"blue|cobalt|sky blue"),
        ("red", r"red|crimson|scarlet"),
        ("pink", r"pink|blush|rose"),
        ("green", r"green|olive|emerald|sage"),
        ("brown", r"brown|chocolate|chestnut"),
        ("beige", r"beige|tan|khaki|camel|nude"),
        ("cream", r"cream|ivory|off white"),
        ("purple", r"purple|violet|lilac|lavender"),
        ("yellow", r"yellow|mustard"),
        ("orange", r"orange|rust"),
        ("burgundy", r"burgundy|maroon|wine"),
        ("gold", r"gold|golden"),
        ("silver", r"silver"),
        ("teal", r"teal|turquoise|aqua"),
    ),
    "material": (
        ("faux_leather", r"faux leather|vegan leather|pleather"),
        ("leather", r"leather|suede|nubuck"),
        ("cotton", r"cotton"),
        ("polyester", r"polyester"),
        ("nylon", r"nylon"),
        ("wool", r"wool|merino"),
        ("cashmere", r"cashmere"),
        ("silk", r"silk|satin"),
        ("denim", r"denim"),
        ("linen", r"linen"),
        ("rayon", r"rayon|viscose"),
        ("spandex", r"spandex|elastane|lycra"),
        ("fleece", r"fleece|sherpa"),
        ("rubber", r"rubber"),
        ("mesh", r"mesh"),
        ("canvas", r"canvas"),
        ("corduroy", r"corduroy|cord"),
        ("flannel", r"flannel"),
        ("velvet", r"velvet"),
        ("lace", r"lace(?!s)"),
        ("down", r"goose down|duck down|down[- ](?:filled|insulated)"),
        ("stainless_steel", r"stainless steel"),
        ("sterling_silver", r"sterling silver"),
        ("gore_tex", r"gore[- ]?tex"),
    ),
    "feature": (
        ("waterproof", r"waterproof|water[- ]resistant|water[- ]repellent|weatherproof"),
        ("windproof", r"windproof|protection from wind|wind protection"),
        ("breathable", r"breathable|breathability"),
        ("quick_dry", r"quick[- ]?dry(?:ing)?|dries quickly|dry quickly"),
        ("moisture_wicking", r"moisture[- ]wicking|moisture management|handles? sweat"),
        ("insulated", r"insulated|insulation|thermal|warmth"),
        ("lightweight", r"lightweight|light[- ]weight"),
        ("machine_washable", r"machine washable|machine wash"),
        ("memory_foam", r"memory foam"),
        ("arch_support", r"arch support"),
        ("ankle_support", r"ankle support"),
        ("non_slip", r"non[- ]slip|anti[- ]slip|slip[- ]resistant|good grip|grip|traction"),
        ("adjustable", r"adjustable"),
        ("hypoallergenic", r"hypoallergenic"),
        ("pockets", r"pockets?"),
        ("hooded", r"hooded|with a hood"),
        ("reversible", r"reversible"),
        ("padded", r"padded|cushioning|cushioned"),
        ("long_sleeve", r"long sleeves?"),
        ("short_sleeve", r"short sleeves?"),
        ("uv_protection", r"uv protection|sun protection"),
        ("stretchy", r"stretchy|stretch"),
        ("wrinkle_free", r"wrinkle[- ]free"),
        ("touchscreen", r"touchscreen"),
    ),
    "size": (
        ("xxs", r"\bxxs\b"),
        ("xs", r"\bxs\b|extra small"),
        ("small", r"\bsmall\b|\bsize s\b"),
        ("medium", r"\bmedium\b|\bsize m\b"),
        ("large", r"\blarge\b|\bsize l\b"),
        ("xl", r"\bxl\b|extra large"),
        ("xxl", r"\bxxl\b"),
        ("petite", r"\bpetite\b"),
        ("plus_size", r"plus[- ]size"),
        ("wide_fit", r"wide (?:fit|width)"),
        ("narrow_fit", r"narrow (?:fit|width)"),
        ("big_and_tall", r"big and tall"),
    ),
    "style": (
        ("casual", r"casual"),
        ("formal", r"formal"),
        ("classic", r"classic"),
        ("vintage", r"vintage|retro"),
        ("bohemian", r"boho|bohemian"),
        ("sporty", r"sporty|athletic"),
        ("elegant", r"elegant"),
        ("minimalist", r"minimalist"),
        ("slim_fit", r"slim fit"),
        ("relaxed_fit", r"relaxed fit|loose fit"),
        ("oversized", r"oversized"),
        ("preppy", r"preppy"),
    ),
    "use_case": (
        ("hiking", r"hiking|trekking"),
        ("running", r"running|jogging"),
        ("walking", r"walking"),
        ("gym", r"the gym|gym|workouts?|working out"),
        ("swimming", r"swimming|swim laps"),
        ("tennis", r"tennis"),
        ("golf", r"golf"),
        ("yoga", r"yoga"),
        ("camping", r"camping"),
        ("skiing", r"skiing|snowboarding"),
        ("cycling", r"cycling|biking"),
        ("work", r"the office|at work|for work|office wear"),
        ("wedding", r"weddings?"),
        ("party", r"parties|a party"),
        ("travel", r"travel(?:ling|ing)?|vacation|holiday|trip"),
        ("beach", r"the beach|beach"),
        ("winter", r"winter"),
        ("summer", r"summer"),
        ("halloween", r"halloween"),
        ("christmas", r"christmas"),
        ("everyday", r"everyday (?:wear|use)|daily wear|every day"),
        ("outdoor", r"outdoors?|outdoor activities"),
    ),
    "brand": (
        ("nike", r"nike"),
        ("adidas", r"adidas"),
        ("puma", r"puma"),
        ("reebok", r"reebok"),
        ("new_balance", r"new balance"),
        ("under_armour", r"under armou?r"),
        ("columbia", r"columbia"),
        ("carhartt", r"carhartt"),
        ("levis", r"levi'?s"),
        ("skechers", r"skechers"),
        ("crocs", r"crocs"),
        ("timberland", r"timberland"),
        ("clarks", r"clarks"),
        ("vans", r"vans"),
        ("converse", r"converse"),
        ("asics", r"asics"),
        ("merrell", r"merrell"),
        ("patagonia", r"patagonia"),
        ("north_face", r"north face"),
        ("dr_martens", r"dr\.? martens|doc martens"),
    ),
}

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "category", "brand", "color", "material", "size", "style", "feature", "use_case",
)

def _bounded(alias: str) -> str:
    """Wrap one alias so it cannot match inside a longer word."""
    return rf"(?<![a-z]){alias}(?![a-z])"


def alias_pattern(field: str, *extra: str) -> re.Pattern[str]:
    """A compiled union of every alias for `field`, plus any extra fragments.

    This is the single source of attribute vocabulary in the package. The
    signal ledger in `lexicon.py` builds its brand / budget / size / colour /
    material / feature / use-case / style patterns from here rather than
    restating them, so the two components can never disagree about what
    "navy" or "water resistant" means.
    """
    alternatives = [_bounded(alias) for _canonical, alias in CANONICAL_VOCAB[field]]
    return re.compile("|".join([*extra, *alternatives]), re.IGNORECASE)


_COMPILED: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    name: tuple(
        (canonical, re.compile(_bounded(alias), re.IGNORECASE))
        for canonical, alias in entries
    )
    for name, entries in CANONICAL_VOCAB.items()
}

_NUMBER = r"\$?\s?(\d[\d,]*(?:\.\d+)?)"

PRICE_MAX = re.compile(
    rf"(?:under|below|less than|no more than|cheaper than|up to|within|max(?:imum)?(?:\s+of)?|nothing over|not more than)\s*{_NUMBER}",
    re.IGNORECASE,
)
PRICE_MIN = re.compile(
    rf"(?:over|above|more than|at least|min(?:imum)?(?:\s+of)?|starting (?:at|from)|from)\s*{_NUMBER}",
    re.IGNORECASE,
)
PRICE_RANGE = re.compile(
    rf"(?:between\s+)?{_NUMBER}\s*(?:-|–|to|and)\s*{_NUMBER}\s*(?:dollars|usd|bucks)?",
    re.IGNORECASE,
)
PRICE_AROUND = re.compile(rf"(?:around|about|approximately|roughly)\s*{_NUMBER}", re.IGNORECASE)
# A price only counts as a price when it is written as one.
CURRENCY = re.compile(r"\$|\bdollars?\b|\busd\b|\bbucks\b|\bprice\b|\bbudget\b|\bcost", re.IGNORECASE)

# Any expression that states a price at all. The ledger uses this as its
# budget signal; the extractor above uses the individual bounds to normalise.
PRICE_EXPRESSION = re.compile(
    "|".join(
        [
            PRICE_MAX.pattern,
            PRICE_MIN.pattern,
            PRICE_RANGE.pattern,
            PRICE_AROUND.pattern,
            r"\$\s?\d[\d,]*(?:\.\d+)?",
        ]
    ),
    re.IGNORECASE,
)

# Numeric sizes are values, not vocabulary entries, so they are matched
# separately. Anchored on the word "size" so a bare number is never mistaken
# for one.
SIZE_NUMERIC = re.compile(r"\bsizes?\s*[:\-]?\s*(\d+(?:\.\d)?)\b", re.IGNORECASE)

# Words that look like a commitment but name no value. Recorded as unmapped so
# a later component can ask about them, never counted as a constraint.
ATTRIBUTE_TOPIC = re.compile(
    r"\b(?:colou?r|material|fabric|size|sizing|style|fit|brand|feature|budget|price)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ShoppingConstraints:
    """Canonical constraints from one utterance, in issue #7's shape."""

    category: tuple[str, ...] = ()
    brand: tuple[str, ...] = ()
    price_min: float | None = None
    price_max: float | None = None
    color: tuple[str, ...] = ()
    material: tuple[str, ...] = ()
    size: tuple[str, ...] = ()
    style: tuple[str, ...] = ()
    feature: tuple[str, ...] = ()
    use_case: tuple[str, ...] = ()
    unmapped: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, object]:
        return {
            "category": list(self.category),
            "brand": list(self.brand),
            "price_min": self.price_min,
            "price_max": self.price_max,
            "color": list(self.color),
            "material": list(self.material),
            "size": list(self.size),
            "style": list(self.style),
            "feature": list(self.feature),
            "use_case": list(self.use_case),
        }

    def has_price(self) -> bool:
        return self.price_min is not None or self.price_max is not None

    def populated_fields(self, *, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Which constraint fields the customer actually filled in.

        Price counts once however many bounds were given: "between $50 and
        $100" is one decision, not two.
        """
        found = [
            name
            for name in CATEGORICAL_FIELDS
            if name not in exclude and getattr(self, name)
        ]
        if self.has_price() and "price" not in exclude:
            found.append("price")
        return tuple(found)

    def tag_count(self, *, exclude: tuple[str, ...] = ()) -> int:
        return len(self.populated_fields(exclude=exclude))


def _extract_prices(text: str) -> tuple[float | None, float | None]:
    if not CURRENCY.search(text):
        # "size 10" and "10 to 12 years" are not prices.
        return None, None

    def number(raw: str) -> float:
        return float(raw.replace(",", ""))

    match = PRICE_RANGE.search(text)
    if match is not None:
        low, high = sorted((number(match.group(1)), number(match.group(2))))
        return low, high

    price_min = price_max = None
    match = PRICE_MAX.search(text)
    if match is not None:
        price_max = number(match.group(1))
    match = PRICE_MIN.search(text)
    if match is not None:
        price_min = number(match.group(1))

    if price_min is None and price_max is None:
        match = PRICE_AROUND.search(text)
        if match is not None:
            # "around $60" is a soft bound in both directions.
            centre = number(match.group(1))
            return centre * 0.8, centre * 1.2
        match = re.search(rf"\${_NUMBER[2:]}", text)
        if match is not None:
            price_max = number(match.group(1))

    return price_min, price_max


def extract_constraints(message: str) -> ShoppingConstraints:
    """Map one utterance onto the canonical vocabulary. Never invents values."""
    text = message or ""

    # Collect every candidate match, then award overlapping spans to the
    # longest one. Without this, "dark blue" records both navy and blue, and
    # "running shoes" records the category *and* a use case the customer never
    # stated -- both of which inflate the tag count the router keys off.
    candidates: list[tuple[int, int, str, str]] = []
    for name in CATEGORICAL_FIELDS:
        for canonical, pattern in _COMPILED[name]:
            for match in pattern.finditer(text):
                candidates.append((match.start(), match.end(), name, canonical))

    candidates.sort(key=lambda item: (item[0] - item[1], item[0]))
    claimed: list[tuple[int, int]] = []
    values: dict[str, list[str]] = {name: [] for name in CATEGORICAL_FIELDS}
    for start, end, name, canonical in candidates:
        if any(start < taken_end and end > taken_start for taken_start, taken_end in claimed):
            continue
        claimed.append((start, end))
        if canonical not in values[name]:
            values[name].append(canonical)
    values = {name: tuple(found) for name, found in values.items()}

    numeric_sizes = tuple(dict.fromkeys(SIZE_NUMERIC.findall(text)))
    if numeric_sizes:
        values["size"] = values["size"] + numeric_sizes

    price_min, price_max = _extract_prices(text)

    # An attribute named without a value is a topic, not a constraint. Keep it
    # so a later component can ask about it; never count it as commitment.
    unmapped = tuple(
        sorted(
            {
                word.lower()
                for word in ATTRIBUTE_TOPIC.findall(text)
                if not values.get(_TOPIC_FIELD.get(word.lower(), ""), ())
            }
        )
    )

    return ShoppingConstraints(
        price_min=price_min,
        price_max=price_max,
        unmapped=unmapped,
        **values,
    )


_TOPIC_FIELD = {
    "color": "color", "colour": "color", "material": "material",
    "fabric": "material", "size": "size", "sizing": "size", "style": "style",
    "fit": "style", "brand": "brand", "feature": "feature",
    "budget": "price", "price": "price",
}

# Issue #7/#8 integration.  The original extractor remains available as an
# offline compatibility fallback until Issue #5 facts have produced a
# dictionary artifact.
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path
from typing import Callable as _Callable, Iterable as _Iterable, Mapping as _Mapping

from dictionary.registry import AttributeDictionary as _AttributeDictionary
from dictionary.registry import ATTRIBUTE_FIELDS as _DICTIONARY_ATTRIBUTES
from dictionary.registry import CanonicalValue as _CanonicalValue
from dictionary.registry import LookupMatch as _LookupMatch
from dictionary.registry import normalize_text as _normalize_dictionary_text


@_dataclass(frozen=True)
class ConstraintEvidence:
    """Internal provenance for one resolved constraint."""

    canonical_id: str
    attribute: str
    raw_text: str
    match_method: str
    confidence: float


@_dataclass(frozen=True)
class CanonicalShoppingConstraints(ShoppingConstraints):
    """Issue #7 output with optional resolver provenance."""

    evidence: tuple[ConstraintEvidence, ...] = _field(default=())

    def evidence_dict(self) -> list[dict[str, object]]:
        return [
            {
                "canonical_id": item.canonical_id,
                "attribute": item.attribute,
                "raw_text": item.raw_text,
                "match_method": item.match_method,
                "confidence": item.confidence,
            }
            for item in self.evidence
        ]


@_dataclass(frozen=True)
class _SpanMatch:
    start: int
    end: int
    raw_text: str
    canonical_ids: tuple[str, ...]


@_dataclass(frozen=True)
class _SemanticCandidate:
    canonical_id: str
    score: float


SemanticMatcher = _Callable[[str], _Iterable[object]]


# Ambiguous exact surfaces are resolved only when the evidence is strong. The
# values are centralized so the policy is easy to audit and test.
AMBIGUITY_MIN_TOP_SHARE = 0.75
AMBIGUITY_MIN_COUNT_RATIO = 3.0
# Context is recognized only when a cue is directly attached to the matched
# dictionary phrase. The cue itself is evidence, never part of the match span.
_EXPLICIT_CONTEXT_BEFORE: dict[str, tuple[tuple[str, ...], ...]] = {
    "brand": (("brand",), ("from",), ("made", "by"), ("by",)),
    "color": (("color",), ("colour",), ("in",)),
    "material": (("made", "of"), ("made", "from"), ("made", "with")),
    "style": (("style",), ("fit",)),
    "feature": (("feature",), ("features",)),
    "use_case": (("for",), ("for", "use"), ("use", "for"), ("good", "for")),
}
_EXPLICIT_CONTEXT_AFTER: dict[str, tuple[tuple[str, ...], ...]] = {
    "brand": (("brand",),),
    "color": (("color",), ("colour",)),
    "material": (("material",), ("fabric",)),
    "style": (("style",), ("fit",)),
    "feature": (("feature",), ("features",)),
}

_SHOPPING_FOR_CONTEXT_BLOCKERS = frozenset(
    {"find", "looking", "need", "searching", "shopping", "show", "want"}
)

# Keep this catalog-derived set limited to obvious single-word brand/query
# collisions. Multi-word brands and all non-brand attributes are unaffected.
COMMON_BRAND_COLLISION_TERMS = frozenset({"find"})


_RESIDUAL_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
        "from", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
        "some", "that", "the", "this", "to", "want", "with", "would", "you",
        "looking", "need", "like", "something", "show", "find", "under", "below",
        "less", "than", "more", "over", "between", "around", "about", "within",
    }
)


@lru_cache(maxsize=1)
def _load_default_dictionary() -> _AttributeDictionary | None:
    directory = _Path("data/derived/dictionary")
    if not (directory / "canonical_values.json").exists():
        return None
    if not (directory / "normalized_lookup.json").exists():
        return None
    return _AttributeDictionary.load(directory)


def _dictionary_surface_matches(
    dictionary: _AttributeDictionary,
    text: str,
) -> tuple[_SpanMatch, ...]:
    """Find non-overlapping dictionary phrases with longest-first matching.

    Matching is token based rather than substring based. The registry and
    utterance use the same lexical normalization, while the token spans retain
    the original text for evidence and overlap accounting.
    """

    def tokens(value: str) -> tuple[tuple[str, int, int], ...]:
        folded = unicodedata.normalize("NFKC", value).casefold()
        apostrophes = {"'", "’", "ʼ", "＇"}
        found_tokens: list[tuple[str, int, int]] = []
        current: list[str] = []
        start: int | None = None

        def finish(end: int) -> None:
            nonlocal current, start
            if current and start is not None:
                token = _normalize_dictionary_text("".join(current))
                if token:
                    found_tokens.append((token, start, end))
            current = []
            start = None

        for index, character in enumerate(folded):
            if character.isalnum():
                if start is None:
                    start = index
                current.append(character)
            elif (
                character in apostrophes
                and start is not None
                and index + 1 < len(folded)
                and folded[index + 1].isalnum()
            ):
                # Keep a word such as Levi's as one lexical token.
                continue
            else:
                finish(index)
        finish(len(folded))
        return tuple(found_tokens)

    utterance_tokens = tokens(text)
    found: dict[tuple[int, int], tuple[int, set[str]]] = {}
    for offset, (first_token, _, _) in enumerate(utterance_tokens):
        for parts, value_id in dictionary.phrase_index.get(first_token, ()):
            width = len(parts)
            if tuple(token[0] for token in utterance_tokens[offset : offset + width]) != parts:
                continue
            start = utterance_tokens[offset][1]
            end = utterance_tokens[offset + width - 1][2]
            key = (start, end)
            if key not in found:
                found[key] = (width, set())
            found[key][1].add(value_id)

    candidates = [
        (start, end, ids, width, end - start)
        for (start, end), (width, ids) in found.items()
    ]
    candidates.sort(key=lambda item: (-item[3], -item[4], item[0], item[1]))
    accepted: list[tuple[int, int, set[str]]] = []
    for start, end, ids, _, _ in candidates:
        if any(start < taken_end and end > taken_start for taken_start, taken_end, _ in accepted):
            continue
        accepted.append((start, end, ids))
    return tuple(
        _SpanMatch(start, end, text[start:end], tuple(sorted(ids)))
        for start, end, ids in sorted(accepted, key=lambda item: (item[0], item[1]))
    )


def _utterance_tokens(value: str) -> tuple[tuple[str, int, int], ...]:
    """Tokenize a message with dictionary normalization and source offsets."""

    folded = unicodedata.normalize("NFKC", value).casefold()
    apostrophes = {"'", "’", "ʼ", "＇"}
    found_tokens: list[tuple[str, int, int]] = []
    current: list[str] = []
    start: int | None = None

    def finish(end: int) -> None:
        nonlocal current, start
        if current and start is not None:
            token = _normalize_dictionary_text("".join(current))
            if token:
                found_tokens.append((token, start, end))
        current = []
        start = None

    for index, character in enumerate(folded):
        if character.isalnum():
            if start is None:
                start = index
            current.append(character)
        elif (
            character in apostrophes
            and start is not None
            and index + 1 < len(folded)
            and folded[index + 1].isalnum()
        ):
            # Keep a word such as Levi's as one lexical token.
            continue
        else:
            finish(index)
    finish(len(folded))
    return tuple(found_tokens)


def _context_attributes(
    text: str,
    match: _SpanMatch,
    candidates: tuple[_CanonicalValue, ...],
) -> tuple[str, ...]:
    """Return attributes requested by a directly attached lexical pattern.

    Context cues are directional and must touch the matched dictionary span.
    The cue words are not part of the dictionary match and cannot claim any
    other match elsewhere in the utterance.
    """

    tokens = _utterance_tokens(text)
    span_indices = [
        index
        for index, (_, start, end) in enumerate(tokens)
        if start >= match.start and end <= match.end
    ]
    if not span_indices:
        return ()

    first_span_index = span_indices[0]
    last_span_index = span_indices[-1]
    token_values = tuple(token[0] for token in tokens)

    def directly_before(pattern: tuple[str, ...]) -> bool:
        width = len(pattern)
        start = first_span_index - width
        if start < 0:
            return False
        if token_values[start:first_span_index] != pattern:
            return False
        if (
            pattern == ("for",)
            and start > 0
            and token_values[start - 1] in _SHOPPING_FOR_CONTEXT_BLOCKERS
        ):
            return False
        return True

    def directly_after(pattern: tuple[str, ...]) -> bool:
        width = len(pattern)
        end = last_span_index + 1 + width
        return token_values[last_span_index + 1:end] == pattern

    found: list[str] = []
    candidate_attributes = {candidate.attribute for candidate in candidates}
    for attribute in _DICTIONARY_ATTRIBUTES:
        before_patterns = _EXPLICIT_CONTEXT_BEFORE.get(attribute, ())
        after_patterns = _EXPLICIT_CONTEXT_AFTER.get(attribute, ())

        if any(directly_before(pattern) for pattern in before_patterns):
            if (
                attribute == "color"
                and directly_before(("in",))
                and "color" not in candidate_attributes
            ):
                continue
            found.append(attribute)
            continue

        if any(directly_after(pattern) for pattern in after_patterns):
            found.append(attribute)

    return tuple(found)
def _is_suppressed_common_brand(
    entry: _CanonicalValue,
    context_attributes: tuple[str, ...],
) -> bool:
    """Suppress only uncontextualized, single-token query-word brands."""

    return (
        entry.attribute == "brand"
        and len(entry.normalized.split()) == 1
        and entry.normalized in COMMON_BRAND_COLLISION_TERMS
        and "brand" not in context_attributes
    )


def _resolve_dictionary_match(
    text: str,
    match: _SpanMatch,
    dictionary: _AttributeDictionary,
) -> _CanonicalValue | None:
    candidates = dictionary.get_candidates(match.raw_text)
    context_attributes = _context_attributes(text, match, candidates)
    if context_attributes:
        contextual_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.attribute in context_attributes
        )
        if len(context_attributes) != 1 or len(contextual_candidates) != 1:
            return None
        return contextual_candidates[0]
    if len(candidates) == 1:
        if _is_suppressed_common_brand(candidates[0], context_attributes):
            return None
        return candidates[0]

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.count,
            _DICTIONARY_ATTRIBUTES.index(candidate.attribute),
            candidate.canonical_id,
        ),
    )
    if len(ranked) < 2:
        return None
    total = sum(candidate.count for candidate in ranked)
    top, second = ranked[0], ranked[1]
    top_share = top.count / total
    count_ratio = top.count / second.count
    if (
        top_share >= AMBIGUITY_MIN_TOP_SHARE
        and count_ratio >= AMBIGUITY_MIN_COUNT_RATIO
    ):
        if _is_suppressed_common_brand(top, context_attributes):
            return None
        return top
    return None


def _residual_phrase(text: str, claimed: list[tuple[int, int]]) -> str:
    remaining = list(text)
    for start, end in claimed:
        for index in range(max(0, start), min(len(remaining), end)):
            remaining[index] = " "
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']*", "".join(remaining).lower())
    return " ".join(token for token in tokens if token not in _RESIDUAL_STOPWORDS)


def _semantic_items(
    matcher: SemanticMatcher,
    phrase: str,
) -> tuple[_SemanticCandidate, ...]:
    result = matcher(phrase)
    if isinstance(result, (_LookupMatch, _Mapping)):
        result = (result,)
    items: list[_SemanticCandidate] = []
    for item in result or ():
        if isinstance(item, _LookupMatch):
            items.append(_SemanticCandidate(item.canonical_id, item.similarity))
        elif isinstance(item, _Mapping):
            canonical_id = str(item.get("canonical_id", "")).strip()
            if canonical_id:
                score = item.get("score", item.get("similarity", 0.0))
                items.append(_SemanticCandidate(canonical_id, float(score)))
    return tuple(sorted(items, key=lambda item: (-item.score, item.canonical_id)))


_legacy_extract_constraints = extract_constraints


def _extract_dictionary_constraints(
    message: str,
    dictionary: _AttributeDictionary,
    *,
    semantic_matcher: SemanticMatcher | None = None,
    semantic_threshold: float = 0.70,
) -> CanonicalShoppingConstraints:
    text = message or ""
    values: dict[str, list[str]] = {name: [] for name in CATEGORICAL_FIELDS}
    evidence: list[ConstraintEvidence] = []
    unmapped: set[str] = set()
    claimed: list[tuple[int, int]] = []

    price_min, price_max = _extract_prices(text)
    if price_min is not None or price_max is not None:
        claimed.extend(
            (match.start(), match.end())
            for match in PRICE_EXPRESSION.finditer(text)
        )
        if price_min is not None:
            evidence.append(ConstraintEvidence("price_min", "price", "", "structured", 1.0))
        if price_max is not None:
            evidence.append(ConstraintEvidence("price_max", "price", "", "structured", 1.0))

    for match in SIZE_NUMERIC.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in claimed):
            continue
        claimed.append((match.start(), match.end()))
        value = match.group(1)
        if value not in values["size"]:
            values["size"].append(value)
            evidence.append(
                ConstraintEvidence(
                    f"size:{value}", "size", match.group(0), "structured", 1.0
                )
            )

    for match in _dictionary_surface_matches(dictionary, text):
        if any(match.start < end and match.end > start for start, end in claimed):
            continue
        claimed.append((match.start, match.end))
        entry = _resolve_dictionary_match(text, match, dictionary)
        if entry is None:
            unmapped.add(match.raw_text.strip().lower())
            continue
        if entry.value not in values[entry.attribute]:
            values[entry.attribute].append(entry.value)
            evidence.append(
                ConstraintEvidence(
                    entry.canonical_id, entry.attribute, match.raw_text, "exact", 1.0
                )
            )

    residual = _residual_phrase(text, claimed)
    if residual and semantic_matcher is not None:
        semantic_matches = _semantic_items(semantic_matcher, residual)
        if semantic_matches and semantic_matches[0].score >= semantic_threshold:
            best = semantic_matches[0]
            entry = dictionary.get(best.canonical_id)
            if entry is not None and entry.attribute in CATEGORICAL_FIELDS:
                if entry.value not in values[entry.attribute]:
                    values[entry.attribute].append(entry.value)
                    evidence.append(
                        ConstraintEvidence(
                            entry.canonical_id,
                            entry.attribute,
                            residual,
                            "semantic",
                            best.score,
                        )
                    )
            else:
                unmapped.add(residual)
        else:
            unmapped.add(residual)
    elif residual:
        unmapped.add(residual)

    for word in ATTRIBUTE_TOPIC.findall(text):
        field_name = _TOPIC_FIELD.get(word.lower(), "")
        if field_name in CATEGORICAL_FIELDS and not values[field_name]:
            unmapped.add(word.lower())

    return CanonicalShoppingConstraints(
        price_min=price_min,
        price_max=price_max,
        unmapped=tuple(sorted(unmapped)),
        evidence=tuple(evidence),
        **{name: tuple(found) for name, found in values.items()},
    )


def extract_constraints(
    message: str,
    *,
    dictionary: _AttributeDictionary | None = None,
    semantic_matcher: SemanticMatcher | None = None,
    semantic_threshold: float = 0.70,
) -> ShoppingConstraints:
    """Extract constraints using #8 exact lookup and optional semantic fallback.

    Structured price/size parsing runs first. Exact dictionary values are then
    matched longest-first, and only the unmatched meaningful phrase reaches the
    injected semantic matcher. Results below the threshold remain unresolved.
    Until a generated dictionary artifact exists, the legacy offline vocabulary
    is used so the starter remains runnable without Issue #5 data.
    """
    active_dictionary = dictionary if dictionary is not None else _load_default_dictionary()
    if active_dictionary is None:
        return _legacy_extract_constraints(message)
    return _extract_dictionary_constraints(
        message,
        active_dictionary,
        semantic_matcher=semantic_matcher,
        semantic_threshold=semantic_threshold,
    )
