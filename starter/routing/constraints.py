"""Extract canonical shopping constraints from one user utterance.

Produces the runtime constraint record shape:

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
nearest vocabulary item.

The generated attribute dictionary is the source of truth for categorical
values. Price and size remain structured runtime fields; size is intentionally
outside the semantic dictionary contract.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache


CATEGORICAL_FIELDS: tuple[str, ...] = (
    "category",
    "brand",
    "color",
    "material",
    "size",
    "style",
    "feature",
    "use_case",
)
STRUCTURED_CANONICAL_FIELDS: tuple[str, ...] = ("brand",)
STRUCTURED_RUNTIME_FIELDS: tuple[str, ...] = ("size",)

_PLAIN_NUMBER = r"\d[\d,]*(?:\.\d+)?"
# Do not let a failed match on a decimal or thousands-formatted value backtrack
# into a shorter number (for example, reading ``0.5"`` as the price ``0``).
_NUMBER_END = r"(?!\d|[.,]\d|\s*[:/]\s*\d)"
_CURRENCY_SYMBOL = r"[$\u00a3\u00a5\u20ac]"
_CURRENCY_WORD = (
    r"(?:dollars?|usd|bucks|euros?|eur|gbp|pounds?\s+sterling|cad|aud|sgd|"
    r"jpy|yen)"
)
_CURRENCY_TOKEN = rf"(?:{_CURRENCY_SYMBOL}|\b{_CURRENCY_WORD}\b)"
_NUMBER = rf"(?:{_CURRENCY_TOKEN}\s*)?({_PLAIN_NUMBER}){_NUMBER_END}"
_RANGE_NUMBER = (
    rf"(?:{_CURRENCY_TOKEN}\s*)?({_PLAIN_NUMBER}){_NUMBER_END}"
    rf"(?:\s*{_CURRENCY_TOKEN})?"
)
_NON_PRICE_UNIT = (
    r"(?:(?:years?|yrs?|year[-\s]?old|months?|weeks?|days?|hours?|hrs?|"
    r"minutes?|mins?|seconds?|secs?|inches?|inch|feet|foot|ft|yards?|yds?|"
    r"centimeters?|cm|millimeters?|mm|meters?|metres?|kilometers?|kilometres?|"
    r"km|kilograms?|kg|pounds?|lbs?|ounces?|oz|grams?|milligrams?|mg|liters?|"
    r"litres?|milliliters?|millilitres?|ml|gallons?|gal|quarts?|qt|pints?|"
    r"watts?|volts?|mah|kilohertz|megahertz|gigahertz|khz|mhz|ghz|bytes?|"
    r"kilobytes?|megabytes?|gigabytes?|terabytes?|kb|mb|gb|tb|pixels?|"
    r"megapixels?|mp|dpi|ppi|degrees?|percent|percentage|stars?|points?|"
    r"counts?|ct|packs?|pieces?|pcs?|pairs?|sets?|units?)\b|"
    r"[\x22\x25\x27\u00b0\u00d7\u201c\u201d\u2032\u2033])"
)
_NON_PRICE_TAIL = (
    rf"(?!\s*{_NON_PRICE_UNIT})"
    rf"(?!\s*(?:to|and|-)\s*{_PLAIN_NUMBER}\s*(?:years?|yrs?|year[-\s]?old)\b)"
    r"(?!\s+(?:\d+\s*/\s*\d+|[x\u00d7-]\s*\d))"
)

PRICE_MAX = re.compile(
    rf"(?:under|below|less than|no more than|cheaper than|up to|within|max(?:imum)?(?:\s+of)?|nothing over|not more than)\s*{_NUMBER}{_NON_PRICE_TAIL}",
    re.IGNORECASE,
)
PRICE_MIN = re.compile(
    rf"(?:over|above|more than|at least|min(?:imum)?(?:\s+of)?|starting (?:at|from)|from)\s*{_NUMBER}{_NON_PRICE_TAIL}",
    re.IGNORECASE,
)
PRICE_RANGE = re.compile(
    rf"(?<!size )(?<!sizes )(?<!sizing )(?:between\s+)?{_RANGE_NUMBER}\s*(?:-|–|to|and)\s*{_RANGE_NUMBER}{_NON_PRICE_TAIL}",
    re.IGNORECASE,
)
_APPROXIMATE_LEAD = (
    r"(?:around|about|roughly|approximately|approximate|approx\.?|"
    r"estimated(?:\s+at)?|close\s+to|~)"
)
_OPTIONAL_PRICE_LABEL = (
    r"(?:(?:price|budget|cost)\s*(?:is|was|of|at|:)?\s*)?"
)
PRICE_AROUND = re.compile(
    rf"{_APPROXIMATE_LEAD}\s*{_OPTIONAL_PRICE_LABEL}{_NUMBER}{_NON_PRICE_TAIL}",
    re.IGNORECASE,
)
PRICE_DIRECT = re.compile(
    rf"(?:{_CURRENCY_TOKEN}\s*({_PLAIN_NUMBER}){_NUMBER_END}|"
    rf"({_PLAIN_NUMBER}){_NUMBER_END}\s*{_CURRENCY_TOKEN})",
    re.IGNORECASE,
)
_CURRENCY_MARKER = re.compile(
    _CURRENCY_TOKEN,
    re.IGNORECASE,
)
_PRICE_CONTEXT_BEFORE = re.compile(
    r"\b(?:price|budget|cost|costs|costing)\b"
    r"(?:\s+(?:is|was|of|at|should\s+be|needs?\s+to\s+be|set\s+at))?\s*$",
    re.IGNORECASE,
)
_NON_PRICE_CONTEXT_BEFORE = re.compile(
    r"\b(?:age|aged|battery\s+life|bust|capacity|chest|circumference|count|"
    r"depth|diameter|dimensions?|display|duration|heel(?:\s+height)?|height|"
    r"hip|inseam|length|measurement|measures?|neck|platform(?:\s+height)?|"
    r"quantity|rated|rating|runtime|run\s+time|screen(?:\s+size)?|"
    r"shaft(?:\s+height)?|shoulder|size|sizes|sizing|sleeve|speed|temperature|"
    r"thickness|time|waist|weight|weighs?|width|contains?|includes?|holds?|"
    r"lasts?|comes?\s+with)\b"
    r"(?:\s+(?:is|are|was|were|of|at|from|runs?|measures?|weighs?|"
    r"measured|ranges?|var(?:y|ies|ied|ying)|between|should\s+be|"
    r"comes?\s+(?:to|in|with)))*\s*$",
    re.IGNORECASE,
)
_PRICE_CONTINUATION_WORDS = frozenset(
    {
        "after",
        "all",
        "and",
        "before",
        "but",
        "delivered",
        "depending",
        "each",
        "excluding",
        "for",
        "ideally",
        "if",
        "including",
        "is",
        "max",
        "maximum",
        "min",
        "minimum",
        "on",
        "or",
        "out",
        "per",
        "please",
        "preferably",
        "shipped",
        "tax",
        "total",
        "when",
        "with",
        "without",
    }
)

# Any expression that states a price at all. The ledger uses this as its
# budget signal; the extractor above uses the individual bounds to normalise.
PRICE_EXPRESSION = re.compile(
    "|".join(
        [
            PRICE_MAX.pattern,
            PRICE_MIN.pattern,
            PRICE_RANGE.pattern,
            PRICE_AROUND.pattern,
            PRICE_DIRECT.pattern,
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
    """Canonical constraints from one utterance in the runtime shape."""

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


def _number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _price_match_allowed(text: str, match: re.Match[str]) -> bool:
    """Reject numeric comparisons that are measurements, counts, or specs."""

    window_start = max(0, match.start() - 64)
    window_end = min(len(text), match.end() + 64)
    before = text[window_start:match.start()]
    after = text[match.end():window_end]
    expression = match.group(0)
    currency_after = _CURRENCY_MARKER.match(after.lstrip())
    explicit_price = bool(
        _CURRENCY_MARKER.search(expression)
        or currency_after
        or _PRICE_CONTEXT_BEFORE.search(before)
    )

    if explicit_price:
        return True
    if _NON_PRICE_CONTEXT_BEFORE.search(before):
        return False
    if re.match(rf"\s*{_NON_PRICE_UNIT}", after, re.IGNORECASE):
        return False
    if re.match(r"\s*out\s+of\s+\d", after, re.IGNORECASE):
        return False

    # A noun immediately after a bare number normally supplies its unit or
    # quantity ("about 4 stars", "at least 2 pockets"). Price expressions may
    # instead be followed by a small set of discourse/price continuations.
    following_word = re.match(
        r"\s*(?:[-,;]\s*)?([^\W\d_]+)", after, re.IGNORECASE
    )
    if (
        following_word is not None
        and following_word.group(1).casefold() not in _PRICE_CONTINUATION_WORDS
    ):
        return False

    raw_values = [group for group in match.groups() if group is not None]
    if any(1900 <= _number(raw_value) <= 2099 for raw_value in raw_values):
        return False
    return True


def iter_price_expression_matches(text: str):
    """Yield only price expressions that survive contextual validation."""

    for match in PRICE_EXPRESSION.finditer(text):
        if _price_match_allowed(text, match):
            yield match


def first_price_expression_match(text: str) -> re.Match[str] | None:
    """Return the first validated price expression for intent evidence."""

    return next(iter_price_expression_matches(text), None)


def _first_allowed_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return next(
        (match for match in pattern.finditer(text) if _price_match_allowed(text, match)),
        None,
    )


def _extract_prices(text: str) -> tuple[float | None, float | None]:

    match = _first_allowed_match(PRICE_RANGE, text)
    if match is not None:
        low, high = sorted((_number(match.group(1)), _number(match.group(2))))
        return low, high

    price_min = price_max = None
    match = _first_allowed_match(PRICE_MAX, text)
    if match is not None:
        price_max = _number(match.group(1))
    match = _first_allowed_match(PRICE_MIN, text)
    if match is not None:
        price_min = _number(match.group(1))

    if price_min is None and price_max is None:
        match = _first_allowed_match(PRICE_AROUND, text)
        if match is not None:
            # "around $60" is a soft bound in both directions.
            centre = _number(match.group(1))
            return centre * 0.8, centre * 1.2
        match = _first_allowed_match(PRICE_DIRECT, text)
        if match is not None:
            raw_value = next(group for group in match.groups() if group is not None)
            price_max = _number(raw_value)

    return price_min, price_max


_KNOWN_PHRASE_REWRITES: tuple[tuple[str, str], ...] = (
    # The catalog dictionary is intentionally broad and contains many noisy
    # surfaces. These user-facing phrases are stable concepts whose useful
    # canonical value is clearer than a literal substring match.
    (r"\bwould\s+work\s+best\b", "would be ideal"),
    (r"\bpractical\s+storage\s+pockets?\b", "feature pockets"),
    (r"\bstorage\s+pockets?\b", "feature pockets"),
    (r"\bsun\s+protection\b", "feature uv protection"),
    (r"\beasy\s+to\s+machine\s+wash\b", "feature machine washable"),
    (r"\bmachine\s+wash(?:ing)?\b", "feature machine washable"),
    (r"\bmachine\s+washability\b", "feature machine washable"),
    (r"\bgive\s+in\s+the\s+material\b", "feature stretch"),
    (r"\bzip\s+closure\b", "feature zipper closure"),
    (r"\bbuckle\s+fastening\b", "feature buckle closure"),
    (r"\bslip[- ]resistant\s+design\b", "feature non slip"),
    (r"\bgood\s+grip\b", "feature non slip"),
    (r"\bgood\s+protection\s+from\s+wind\b", "feature windproof"),
    (r"\bkeep\s+water\s+out\b", "feature waterproof"),
    (r"\bbreathability\b", "feature breathable"),
    (r"\bkeep\s+it\s+light\b", "feature lightweight"),
    (r"\bfit\s+i\s+can\s+adjust\b", "feature adjustable"),
    (r"\bparts\s+that\s+can\s+be\s+removed\s+when\s+needed\b", "feature removable"),
    (r"\bvisibility[- ]enhancing\s+reflective\s+details?\b", "feature reflective"),
    (r"\bmemory[- ]foam\s+cushioning\b", "feature memory foam"),
    (r"\bquick[- ]drying\b", "feature quick drying"),
    (r"\bdry\s+quickly\b", "feature quick drying"),
    (r"\b(?:something|a|an)\s+adjustable\b", "feature adjustable"),
    (r"\bfor\s+weddings?\b", "for wedding"),
)


def _normalise_known_phrases(message: str) -> str:
    text = message
    for pattern, replacement in _KNOWN_PHRASE_REWRITES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


_TOPIC_FIELD = {
    "color": "color", "colour": "color", "material": "material",
    "fabric": "material", "size": "size", "sizing": "size", "style": "style",
    "fit": "style", "brand": "brand", "feature": "feature",
    "budget": "price", "price": "price",
}

# Generated dictionary integration.
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path
from typing import Callable as _Callable, Iterable as _Iterable, Mapping as _Mapping

from dictionary.registry import AttributeDictionary as _AttributeDictionary
from dictionary.registry import ATTRIBUTE_FIELDS as _DICTIONARY_ATTRIBUTES
from dictionary.registry import CanonicalValue as _CanonicalValue
from dictionary.registry import DEFAULT_MIN_SIMILARITY as _DEFAULT_MIN_SIMILARITY
from dictionary.registry import LookupMatch as _LookupMatch
from dictionary.registry import SEMANTIC_ATTRIBUTES as _SEMANTIC_ATTRIBUTES
from dictionary.registry import normalize_text as _normalize_dictionary_text
from dictionary.registry import (
    SEMANTIC_QUERY_STOPWORDS as _SEMANTIC_QUERY_STOPWORDS,
)
from dictionary.registry import semantic_query_ngrams as _semantic_query_ngrams
from dictionary.registry import semantic_query_tokens as _semantic_query_tokens


@_dataclass(frozen=True)
class ConstraintEvidence:
    """Internal provenance for one resolved constraint."""

    canonical_id: str
    attribute: str
    raw_text: str
    match_method: str
    confidence: float
    layer: str = "layer1"


@_dataclass(frozen=True)
class SemanticShoppingConstraints:
    """Layer 2 canonical matches kept separate from exact constraints."""

    category: tuple[str, ...] = ()
    color: tuple[str, ...] = ()
    material: tuple[str, ...] = ()
    style: tuple[str, ...] = ()
    feature: tuple[str, ...] = ()
    use_case: tuple[str, ...] = ()
    evidence: tuple[ConstraintEvidence, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "category": list(self.category),
            "color": list(self.color),
            "material": list(self.material),
            "style": list(self.style),
            "feature": list(self.feature),
            "use_case": list(self.use_case),
        }

    def evidence_dict(self) -> list[dict[str, object]]:
        return [
            {
                "canonical_id": item.canonical_id,
                "attribute": item.attribute,
                "raw_text": item.raw_text,
                "match_method": item.match_method,
                "confidence": item.confidence,
                "layer": item.layer,
            }
            for item in self.evidence
        ]

    def populated_fields(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in _SEMANTIC_ATTRIBUTES
            if getattr(self, field_name)
        )


@_dataclass(frozen=True)
class CanonicalShoppingConstraints(ShoppingConstraints):
    """Runtime constraint output with optional resolver provenance."""

    evidence: tuple[ConstraintEvidence, ...] = _field(default=())
    semantic_constraints: SemanticShoppingConstraints = _field(
        default_factory=SemanticShoppingConstraints
    )

    def evidence_dict(self) -> list[dict[str, object]]:
        return [
            {
                "canonical_id": item.canonical_id,
                "attribute": item.attribute,
                "raw_text": item.raw_text,
                "match_method": item.match_method,
                "confidence": item.confidence,
                "layer": item.layer,
            }
            for item in self.evidence
        ]

    def structured_only(self) -> "CanonicalShoppingConstraints":
        """Return Layer 1's brand-only canonical view.

        Price and size remain structured runtime controls. The other canonical
        attributes are intentionally left to the independent Layer 2 semantic
        view so an exact dictionary match cannot create a second structured
        claim for the same user utterance.
        """

        semantic_ids = {
            item.canonical_id for item in self.semantic_constraints.evidence
        }
        structured_ids = {
            item.canonical_id
            for item in self.evidence
            if item.layer != "layer2"
            and item.attribute
            in (*STRUCTURED_CANONICAL_FIELDS, *STRUCTURED_RUNTIME_FIELDS, "price")
        }
        values: dict[str, tuple[str, ...]] = {}
        for field_name in CATEGORICAL_FIELDS:
            if field_name in STRUCTURED_RUNTIME_FIELDS:
                values[field_name] = tuple(getattr(self, field_name))
            elif field_name in STRUCTURED_CANONICAL_FIELDS and self.evidence:
                values[field_name] = tuple(
                    value
                    for value in getattr(self, field_name)
                    if f"{field_name}:{_normalize_dictionary_text(value).replace(' ', '_')}"
                    in structured_ids
                )
            elif field_name in STRUCTURED_CANONICAL_FIELDS:
                # Preserve compatibility for manually constructed constraint
                # objects that have no provenance attached.
                values[field_name] = tuple(
                    value
                    for value in getattr(self, field_name)
                    if f"{field_name}:{_normalize_dictionary_text(value).replace(' ', '_')}"
                    not in semantic_ids
                )
            else:
                values[field_name] = ()
        allowed_evidence_attributes = {
            *STRUCTURED_CANONICAL_FIELDS,
            *STRUCTURED_RUNTIME_FIELDS,
            "price",
        }
        return CanonicalShoppingConstraints(
            price_min=self.price_min,
            price_max=self.price_max,
            unmapped=self.unmapped,
            evidence=tuple(
                item
                for item in self.evidence
                if item.layer != "layer2"
                and item.attribute in allowed_evidence_attributes
            ),
            **values,
        )


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
    phrase: str


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
COMMON_BRAND_COLLISION_TERMS = frozenset(
    {"find", "it", "make", "on"}
)


# Backward-compatible local name used by the extraction path. The actual
# policy is owned by dictionary.registry so semantic filtering cannot diverge
# between query text construction and registry n-gram matching.
_RESIDUAL_STOPWORDS = _SEMANTIC_QUERY_STOPWORDS


@lru_cache(maxsize=1)
def _load_default_dictionary() -> _AttributeDictionary | None:
    directories = (
        _Path("data/derived/annotations/v5/dictionary"),
    )
    for directory in directories:
        if not (directory / "canonical_values.json").exists():
            continue
        if not (directory / "normalized_lookup.json").exists():
            continue
        try:
            dictionary = _AttributeDictionary.load(directory)
            if dictionary.has_semantic_embeddings:
                try:
                    from dictionary.semantic import load_bge_attribute_encoder

                    model_hint = dictionary.embedding_model
                    model_path = (
                        model_hint
                        if model_hint and _Path(model_hint).is_dir()
                        else None
                    )
                    dictionary.set_query_encoder(
                        load_bge_attribute_encoder(model_path)
                    )
                except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                    # Exact Layer 1 matching remains usable when the optional
                    # local BGE model is not installed.
                    pass
            return dictionary
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _dictionary_pattern_surface(value: str) -> str:
    """Build a separator-tolerant regex from a normalized dictionary value."""

    parts = value.split()
    return r"[\s_-]+".join(re.escape(part) for part in parts)


def alias_pattern(field: str, *extra: str) -> re.Pattern[str]:
    """Build an intent-signal pattern from the generated dictionary.

    ``size`` is a structured runtime field rather than a dictionary attribute,
    so its callers supply the numeric/topic patterns explicitly. All other
    attribute values come from the required generated registry. The optional
    expressions are retained for attribute-topic and contextual signal words.
    """

    if field == "size":
        return re.compile("|".join(extra), re.IGNORECASE)
    if field not in _DICTIONARY_ATTRIBUTES:
        raise ValueError(f"unknown canonical attribute: {field}")

    dictionary = _load_default_dictionary()
    if dictionary is None:
        raise RuntimeError(
            "generated attribute dictionary is required at "
            "data/derived/annotations/v5/dictionary"
        )

    alternatives = tuple(
        rf"(?<![A-Za-z0-9]){_dictionary_pattern_surface(value.normalized)}"
        rf"(?![A-Za-z0-9])"
        for value in dictionary.values
        if value.attribute == field and value.normalized
    )
    if not alternatives and not extra:
        raise RuntimeError(f"generated attribute dictionary has no values for {field}")
    return re.compile("|".join((*extra, *alternatives)), re.IGNORECASE)


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
    material_from_context = directly_before(("made", "from"))
    for attribute in _DICTIONARY_ATTRIBUTES:
        before_patterns = _EXPLICIT_CONTEXT_BEFORE.get(attribute, ())
        after_patterns = _EXPLICIT_CONTEXT_AFTER.get(attribute, ())

        # ``from`` can introduce a brand (``from Nike``), but in the longer
        # phrase ``made from polyester`` it is part of the material cue. The
        # specific two-token cue must win so the same value is not rejected as
        # having both brand and material context.
        if attribute == "brand" and material_from_context:
            before_patterns = tuple(
                pattern for pattern in before_patterns if pattern != ("from",)
            )

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
    allowed_attribute: str | None = None,
) -> _CanonicalValue | None:
    candidates = dictionary.get_candidates(match.raw_text)
    if allowed_attribute is not None:
        # A surface such as "amber" is a valid category, brand, colour and
        # material at once. When the shopper is answering a question about one
        # attribute, the other readings are not candidates at all, so the
        # ambiguity never has to be broken by frequency.
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.attribute == allowed_attribute
        )
        if not candidates:
            return None
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


def _semantic_text(text: str) -> str:
    """Build the independent Layer 2 input without Layer 1 span claims."""

    return " ".join(_semantic_query_tokens(text))


def _semantic_tokens(text: str) -> tuple[str, ...]:
    """Return contraction-aware tokens for semantic attribute matching."""

    return _semantic_query_tokens(text)


def _semantic_ngrams(phrase: str, *, max_ngram: int = 3) -> tuple[str, ...]:
    """Return deterministic stopword-filtered 1-, 2-, and 3-gram phrases."""

    return _semantic_query_ngrams(phrase, max_ngram=max_ngram)


def _semantic_items_from_result(
    result: object,
    phrase: str,
) -> tuple[_SemanticCandidate, ...]:
    if isinstance(result, (_LookupMatch, _Mapping)):
        result = (result,)
    items: list[_SemanticCandidate] = []
    for item in result or ():
        if isinstance(item, _LookupMatch):
            items.append(
                _SemanticCandidate(
                    item.canonical_id,
                    item.similarity,
                    item.raw_text or phrase,
                )
            )
        elif isinstance(item, _Mapping):
            canonical_id = str(item.get("canonical_id", "")).strip()
            if canonical_id:
                score = item.get("score", item.get("similarity", 0.0))
                items.append(_SemanticCandidate(canonical_id, float(score), phrase))
    return tuple(items)


def _semantic_items(
    matcher: SemanticMatcher,
    phrase: str,
) -> tuple[_SemanticCandidate, ...]:
    return _semantic_items_from_result(matcher(phrase), phrase)


def _dedupe_semantic_items(
    items: Iterable[_SemanticCandidate],
) -> tuple[_SemanticCandidate, ...]:
    best_by_id: dict[str, _SemanticCandidate] = {}
    for item in items:
        previous = best_by_id.get(item.canonical_id)
        if previous is None or item.score > previous.score:
            best_by_id[item.canonical_id] = item
    return tuple(
        sorted(best_by_id.values(), key=lambda item: (-item.score, item.canonical_id))
    )


def _extract_dictionary_constraints(
    message: str,
    dictionary: _AttributeDictionary,
    *,
    semantic_matcher: SemanticMatcher | None = None,
    semantic_threshold: float = _DEFAULT_MIN_SIMILARITY,
    asked_attribute: str | None = None,
) -> CanonicalShoppingConstraints:
    text = _normalise_known_phrases(message or "")
    # Only dictionary attributes can be narrowed. "budget", "size" and "other"
    # are structured or non-dictionary asks, and price/size parsing below stays
    # unscoped either way so a budget answer is still read.
    scoped_attribute = (
        asked_attribute if asked_attribute in _DICTIONARY_ATTRIBUTES else None
    )
    values: dict[str, list[str]] = {name: [] for name in CATEGORICAL_FIELDS}
    semantic_values: dict[str, list[str]] = {
        name: [] for name in _SEMANTIC_ATTRIBUTES
    }
    evidence: list[ConstraintEvidence] = []
    semantic_evidence: list[ConstraintEvidence] = []
    unmapped: set[str] = set()
    structured_claimed: list[tuple[int, int]] = []

    price_min, price_max = _extract_prices(text)
    if price_min is not None or price_max is not None:
        structured_claimed.extend(
            (match.start(), match.end())
            for match in iter_price_expression_matches(text)
        )
        if price_min is not None:
            evidence.append(ConstraintEvidence("price_min", "price", "", "structured", 1.0))
        if price_max is not None:
            evidence.append(ConstraintEvidence("price_max", "price", "", "structured", 1.0))

    for match in SIZE_NUMERIC.finditer(text):
        if any(
            match.start() < end and match.end() > start
            for start, end in structured_claimed
        ):
            continue
        structured_claimed.append((match.start(), match.end()))
        value = match.group(1)
        if value not in values["size"]:
            values["size"].append(value)
            evidence.append(
                ConstraintEvidence(
                    f"size:{value}", "size", match.group(0), "structured", 1.0
                )
            )

    for match in _dictionary_surface_matches(dictionary, text):
        if any(
            match.start < end and match.end > start
            for start, end in structured_claimed
        ):
            continue
        structured_claimed.append((match.start, match.end))
        entry = _resolve_dictionary_match(
            text, match, dictionary, allowed_attribute=scoped_attribute
        )
        if entry is None:
            unmapped.add(match.raw_text.strip().lower())
            continue
        if scoped_attribute is not None and entry.attribute != scoped_attribute:
            continue
        if entry.value not in values[entry.attribute]:
            values[entry.attribute].append(entry.value)
            evidence.append(
                ConstraintEvidence(
                    entry.canonical_id, entry.attribute, match.raw_text, "exact", 1.0
                )
            )

    semantic_text = _semantic_text(text)
    if semantic_text:
        semantic_matches: tuple[_SemanticCandidate, ...] = ()
        if semantic_matcher is None and dictionary.semantic_available:
            semantic_matches = _semantic_items_from_result(
                dictionary.semantic_match_ngrams(
                    semantic_text,
                    allowed_attribute=scoped_attribute,
                    stopwords=_RESIDUAL_STOPWORDS,
                    max_ngram=3,
                    min_similarity=semantic_threshold,
                ),
                semantic_text,
            )
        elif semantic_matcher is not None:
            semantic_items: list[_SemanticCandidate] = []
            for phrase in _semantic_ngrams(semantic_text, max_ngram=3):
                semantic_items.extend(_semantic_items(semantic_matcher, phrase))
            semantic_matches = _dedupe_semantic_items(semantic_items)

        accepted = [
            item for item in semantic_matches if item.score >= semantic_threshold
        ]
        accepted_count = 0
        for item in accepted:
            entry = dictionary.get(item.canonical_id)
            if entry is None or entry.attribute not in _SEMANTIC_ATTRIBUTES:
                continue
            if scoped_attribute is not None and entry.attribute != scoped_attribute:
                continue
            if entry.value not in values[entry.attribute]:
                values[entry.attribute].append(entry.value)
            if entry.value in semantic_values[entry.attribute]:
                continue
            semantic_values[entry.attribute].append(entry.value)
            item_evidence = ConstraintEvidence(
                entry.canonical_id,
                entry.attribute,
                item.phrase,
                f"semantic_{len(item.phrase.split())}gram",
                item.score,
                "layer2",
            )
            evidence.append(item_evidence)
            semantic_evidence.append(item_evidence)
            accepted_count += 1
        if accepted_count == 0:
            unmapped.add(semantic_text)

    for word in ATTRIBUTE_TOPIC.findall(text):
        field_name = _TOPIC_FIELD.get(word.lower(), "")
        if field_name in CATEGORICAL_FIELDS and not values[field_name]:
            unmapped.add(word.lower())

    return CanonicalShoppingConstraints(
        price_min=price_min,
        price_max=price_max,
        unmapped=tuple(sorted(unmapped)),
        evidence=tuple(evidence),
        semantic_constraints=SemanticShoppingConstraints(
            evidence=tuple(semantic_evidence),
            **{
                name: tuple(found)
                for name, found in semantic_values.items()
            },
        ),
        **{name: tuple(found) for name, found in values.items()},
    )


def extract_constraints(
    message: str,
    *,
    dictionary: _AttributeDictionary | None = None,
    semantic_matcher: SemanticMatcher | None = None,
    semantic_threshold: float = _DEFAULT_MIN_SIMILARITY,
    asked_attribute: str | None = None,
) -> ShoppingConstraints:
    """Extract constraints using the generated dictionary and exact lookup.

    Structured price/size parsing and exact dictionary matching run in their
    own path. Independently, the same utterance is stopword-filtered into
    deterministic 1/2/3-gram phrases for the Layer 2 semantic matcher; Layer 1
    exact-match claims do not remove text from that path.
    The generated dictionary is required for categorical extraction.

    ``asked_attribute`` narrows both the exact and the semantic pass to the
    attribute the shopper was asked about, so an answer is read as an answer
    to that question rather than as free text over all seven attributes.
    """
    text = _normalise_known_phrases(message or "")
    active_dictionary = dictionary if dictionary is not None else _load_default_dictionary()
    if active_dictionary is None:
        raise RuntimeError(
            "generated attribute dictionary is required at "
            "data/derived/annotations/v5/dictionary"
        )
    return _extract_dictionary_constraints(
        text,
        active_dictionary,
        semantic_matcher=semantic_matcher,
        semantic_threshold=semantic_threshold,
        asked_attribute=asked_attribute,
    )
