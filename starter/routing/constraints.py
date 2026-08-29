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

_PLAIN_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_NUMBER = rf"\$?\s?({_PLAIN_NUMBER})"
_NON_PRICE_TAIL = (
    r"(?!\s*(?:years?|yrs?|year[-\s]?old|inches?|inch|centimeters?|cm|"
    r"millimeters?|mm|kilograms?|kg|pounds?|lbs?)\b)"
    rf"(?!\s*(?:to|and|-)\s*{_PLAIN_NUMBER}\s*(?:years?|yrs?|year[-\s]?old)\b)"
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
    rf"(?<!size )(?<!sizes )(?<!sizing )(?:between\s+)?{_NUMBER}\s*(?:-|–|to|and)\s*{_NUMBER}\s*(?:dollars|usd|bucks)?{_NON_PRICE_TAIL}",
    re.IGNORECASE,
)
PRICE_AROUND = re.compile(
    rf"(?:around|about|approximately|roughly)\s*{_NUMBER}{_NON_PRICE_TAIL}",
    re.IGNORECASE,
)
PRICE_DIRECT = re.compile(
    rf"(?:\$\s?({_PLAIN_NUMBER})|({_PLAIN_NUMBER})\s*(?:dollars?|usd|bucks)\b)",
    re.IGNORECASE,
)
_EXPLICIT_PRICE_MARKER = re.compile(
    r"\$|\bdollars?\b|\busd\b|\bbucks\b|\bprice\b|\bbudget\b|\bcost\b",
    re.IGNORECASE,
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


def _extract_prices(text: str) -> tuple[float | None, float | None]:
    def number(raw: str) -> float:
        return float(raw.replace(",", ""))

    def allowed(match: re.Match[str]) -> bool:
        window_start = max(0, match.start() - 24)
        window_end = min(len(text), match.end() + 24)
        window = text[window_start:window_end]
        explicit_price = bool(_EXPLICIT_PRICE_MARKER.search(window))
        before = text[window_start:match.start()]
        after = text[match.end():window_end]
        if not explicit_price and re.search(
            r"\b(?:size|sizes|sizing)\s*(?:is|:)?\s*$|\b(?:waist|inseam|length|width|height|diameter)\s*$",
            before,
            re.IGNORECASE,
        ):
            return False
        if not explicit_price and re.match(
            r"\s*(?:years?|yrs?|year[-\s]?old|inches?|inch|centimeters?|cm|"
            r"millimeters?|mm|kilograms?|kg|pounds?|lbs?)\b",
            after,
            re.IGNORECASE,
        ):
            return False
        if not explicit_price:
            raw_values = [
                group
                for group in match.groups()
                if group is not None
            ]
            if any(1900 <= number(raw_value) <= 2099 for raw_value in raw_values):
                return False
        return True

    match = PRICE_RANGE.search(text)
    if match is not None and allowed(match):
        low, high = sorted((number(match.group(1)), number(match.group(2))))
        return low, high

    price_min = price_max = None
    match = PRICE_MAX.search(text)
    if match is not None and allowed(match):
        price_max = number(match.group(1))
    match = PRICE_MIN.search(text)
    if match is not None and allowed(match):
        price_min = number(match.group(1))

    if price_min is None and price_max is None:
        match = PRICE_AROUND.search(text)
        if match is not None and allowed(match):
            # "around $60" is a soft bound in both directions.
            centre = number(match.group(1))
            return centre * 0.8, centre * 1.2
        match = PRICE_DIRECT.search(text)
        if match is not None and allowed(match):
            raw_value = next(group for group in match.groups() if group is not None)
            price_max = number(raw_value)

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
        """Return only the Layer 1 view of this extraction."""

        semantic_ids = {
            item.canonical_id for item in self.semantic_constraints.evidence
        }
        values: dict[str, tuple[str, ...]] = {}
        for field_name in CATEGORICAL_FIELDS:
            values[field_name] = tuple(
                value
                for value in getattr(self, field_name)
                if f"{field_name}:{_normalize_dictionary_text(value).replace(' ', '_')}"
                not in semantic_ids
            )
        return CanonicalShoppingConstraints(
            price_min=self.price_min,
            price_max=self.price_max,
            unmapped=self.unmapped,
            evidence=tuple(
                item for item in self.evidence if item.layer != "layer2"
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
    directories = (
        _Path("data/derived/annotations/v5/dictionary"),
        _Path("data/derived/dictionary"),
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
            "data/derived/annotations/v5/dictionary or data/derived/dictionary"
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


def _semantic_ngrams(phrase: str, *, max_ngram: int = 3) -> tuple[str, ...]:
    """Return deterministic stopword-filtered 1-, 2-, and 3-gram phrases."""

    tokens = [
        token
        for token in _normalize_dictionary_text(phrase).split()
        if token not in _RESIDUAL_STOPWORDS
    ]
    phrases: list[str] = []
    for width in range(1, max_ngram + 1):
        for start in range(0, len(tokens) - width + 1):
            phrases.append(" ".join(tokens[start : start + width]))
    return tuple(dict.fromkeys(phrases))


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
) -> CanonicalShoppingConstraints:
    text = _normalise_known_phrases(message or "")
    values: dict[str, list[str]] = {name: [] for name in CATEGORICAL_FIELDS}
    semantic_values: dict[str, list[str]] = {
        name: [] for name in _SEMANTIC_ATTRIBUTES
    }
    evidence: list[ConstraintEvidence] = []
    semantic_evidence: list[ConstraintEvidence] = []
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
    if residual:
        semantic_matches: tuple[_SemanticCandidate, ...] = ()
        if semantic_matcher is None and dictionary.semantic_available:
            semantic_matches = _semantic_items_from_result(
                dictionary.semantic_match_ngrams(
                    residual,
                    stopwords=_RESIDUAL_STOPWORDS,
                    max_ngram=3,
                    min_similarity=semantic_threshold,
                ),
                residual,
            )
        elif semantic_matcher is not None:
            semantic_items: list[_SemanticCandidate] = []
            for phrase in _semantic_ngrams(residual, max_ngram=3):
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
            if entry.value in values[entry.attribute]:
                continue
            values[entry.attribute].append(entry.value)
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
) -> ShoppingConstraints:
    """Extract constraints using the generated dictionary and exact lookup.

    Structured price/size parsing runs first. Exact dictionary values are then
    matched longest-first. Remaining text is stopword-filtered into deterministic
    1/2/3-gram phrases for the Layer 2 semantic matcher; results below the
    threshold remain unresolved.
    The generated dictionary is required for categorical extraction.
    """
    text = _normalise_known_phrases(message or "")
    active_dictionary = dictionary if dictionary is not None else _load_default_dictionary()
    if active_dictionary is None:
        raise RuntimeError(
            "generated attribute dictionary is required at "
            "data/derived/annotations/v5/dictionary or data/derived/dictionary"
        )
    return _extract_dictionary_constraints(
        text,
        active_dictionary,
        semantic_matcher=semantic_matcher,
        semantic_threshold=semantic_threshold,
    )
