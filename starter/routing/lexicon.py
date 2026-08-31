"""Signal ledger for the Buying vs Browsing intent router (issue #6).

Every weight the rules tier uses lives in this file, so retuning the router
means editing one place. Nothing else in the package hard-codes a phrase.

The central rule is that a *category* keyword carries no weight. Naming a
product says which shelf the customer is standing at, not whether they have
decided. What separates the two intents is whether the message commits to
anything.

Vocabulary ownership: categorical attribute vocabularies come from the
generated dictionary loaded by `constraints.alias_pattern()`. Size remains a
structured runtime field and uses the compact size matcher in that helper;
price signals come from structured price expressions. Keeping values in the
generated dictionary ensures the extractor and intent ledger use one source.

What this file still owns is everything that is *not* a product attribute:
the hesitation signals, the request verbs, the weights, and the rules for how
attribute evidence is combined. It also adds two things the extractor
deliberately omits, because they are evidence of intent rather than
extractable constraints:

- **topic words.** "What material do you have?" names no material, so the
  extractor records nothing. The ledger still treats naming the subject as
  weak evidence of a shopper who is narrowing down.
- **qualitative price talk.** "nothing expensive" sets no numeric bound, so it
  is not a price constraint, but it is plainly budget-aware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from starter.routing.constraints import (
    ATTRIBUTE_TOPIC,
    PRICE_EXPRESSION,
    SIZE_EXPRESSION,
    alias_pattern,
    first_price_expression_match,
)


BUYING = "BUYING"
BROWSING = "BROWSING"


@dataclass(frozen=True)
class SignalSpec:
    """One piece of evidence the router knows how to look for."""

    name: str
    polarity: int  # +1 buying, -1 browsing
    weight: float
    pattern: re.Pattern[str]
    hard: bool = False
    requires_buying_evidence: bool = False
    suppressed_by_hard_evidence: bool = False


def _compile(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), re.IGNORECASE)


# --------------------------------------------------------------------------
# Evidence of commitment
# --------------------------------------------------------------------------

# Budget: any validated price expression from the extractor, plus qualitative
# price talk that sets no numeric bound. ``_BUDGET`` retains the complete
# inspectable vocabulary; ``match_signal`` applies contextual numeric checks.
_QUALITATIVE_BUDGET = _compile(
    r"\b(?:cheap|affordable|inexpensive|budget[-\s]friendly|price range)\b",
    r"\b(?:expensive|pricey|premium|high[-\s]end)\b",
    r"\bbudget\b",
)
_BUDGET = _compile(PRICE_EXPRESSION.pattern, _QUALITATIVE_BUDGET.pattern)

_BRAND = alias_pattern("brand", r"\bbrands?\b")

# Size is structured runtime evidence rather than a dictionary value. Numeric
# sizes, typed product measurements, and the topic word are matched explicitly;
# bare single letters are not.
# An unanchored \bm\b matches the "m" in "I'm" and invents a hard constraint
# out of a contraction, flipping the label confidently wrong.
_SIZE = alias_pattern("size", SIZE_EXPRESSION.pattern, r"\bsizes?\b")

_COLOR = alias_pattern("color", r"\bcolou?rs?\b")

_MATERIAL = alias_pattern("material", r"\b(?:material|fabric)\b")

_FEATURE = alias_pattern("feature")

# Soft evidence. A stated occasion is as consistent with someone describing a
# vague situation as with a decided buyer, so it must not suppress vagueness.
# Anchored on "for" and similar so that "hiking boots" -- a category -- is not
# read as a stated use case.
_USE_CASE = _compile(
    rf"\b(?:for|with|during|to)\s+(?:the\s+|a\s+|my\s+)?(?:{alias_pattern('use_case').pattern})",
    r"\b(?:mainly|primarily|mostly)\s+(?:use|wear|for)\b",
    r"\b(?:is the )?main use\b",
)

_STYLE = alias_pattern("style", r"\bstyle\b")

# "I need" / "show me" only count when the customer also said what they need.
# Without the guard the verb alone pushes every vague request toward BUYING.
_REQUEST_VERB = _compile(
    r"\bi\s+(?:need|want|require)\b",
    r"\b(?:show|find|get)\s+me\b",
    r"\bi'?m\s+(?:looking for|after|shopping for)\b",
)


# --------------------------------------------------------------------------
# Evidence of hesitation
# --------------------------------------------------------------------------

_UNDECIDED = _compile(
    r"\b(?:not|n'?t)\s+sure\b",
    r"\bhaven'?t\s+(?:settled|decided|figured|narrowed)\b",
    r"\bno idea\b",
    r"\bundecided\b",
    r"\b(?:don'?t|do not)\s+(?:really\s+)?(?:have|know)\s+(?:any\s+|a\s+|my\s+)?"
    r"(?:strong\s+|specific\s+|particular\s+)?(?:preferences?|requirements?|idea)\b",
    r"\bopen\s+(?:on|to|about)\b",
    r"\bflexible\s+(?:on|about)\b",
    r"\bstill\s+(?:deciding|figuring|working out)\b",
    r"\b(?:don'?t|do not|not)\s+(?:really\s+)?(?:know|sure)\s+"
    r"(?:what|where|which)\s+(?:i'?m|i am|i|to)\b",
    r"\bwhere (?:do i|should i) start\b",
)

_EXPLORE_VERB = _compile(
    r"\bjust\s+(?:browsing|looking|exploring|window shopping)\b",
    r"\b(?:browsing|exploring|window shopping)\b",
    r"\bshopping\s+around\b",
    r"\blooking\s+around\b",
    r"\b(?:considering|thinking about)\b",
    r"\bsee\s+what'?s?\s+(?:out there|available)\b",
)

_OPTION_SEEKING = _compile(
    r"\b(?:some\s+)?(?:options|ideas|suggestions|recommendations|inspiration)\b",
    r"\bwhat\s+(?:options|else|do you have|would you)\b",
    r"\bcompare\b",
    r"\bmakes?\s+sense\b",
    r"\bgifts?\b",
)

# "looking for something" is materially vaguer than a stray "something".
_VAGUE_HEAD = _compile(
    r"\b(?:looking for|shopping for|need|want|find|after|prefer)\s+"
    r"(?:some\s+)?(?:something|anything|some stuff|some things)\b",
)

# "good" is absent: it modifies real attributes ("good grip") far more often
# than it signals vagueness.
_VAGUE_QUALITY = _compile(
    r"\b(?:nice|cute|cool|pretty|comfortable|comfy|decent|suitable|"
    r"interesting|trendy|stylish)\b",
)

_HEDGED = _compile(r"\byet\b", r"\bfor now\b", r"\bat this point\b")


SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec("budget", +1, 1.50, _BUDGET, hard=True),
    SignalSpec("brand", +1, 1.20, _BRAND, hard=True),
    SignalSpec("size", +1, 1.00, _SIZE, hard=True),
    SignalSpec("color", +1, 1.00, _COLOR, hard=True),
    SignalSpec("material", +1, 1.00, _MATERIAL, hard=True),
    SignalSpec("feature", +1, 1.00, _FEATURE, hard=True),
    SignalSpec("use_case", +1, 0.60, _USE_CASE),
    SignalSpec("style", +1, 0.70, _STYLE),
    SignalSpec("request_verb", +1, 0.45, _REQUEST_VERB, requires_buying_evidence=True),
    SignalSpec("undecided", -1, 1.40, _UNDECIDED),
    SignalSpec("explore_verb", -1, 1.10, _EXPLORE_VERB),
    SignalSpec("option_seeking", -1, 0.90, _OPTION_SEEKING),
    SignalSpec("vague_head", -1, 1.00, _VAGUE_HEAD, suppressed_by_hard_evidence=True),
    SignalSpec("vague_quality", -1, 0.50, _VAGUE_QUALITY, suppressed_by_hard_evidence=True),
    SignalSpec("hedged", -1, 0.60, _HEDGED, suppressed_by_hard_evidence=True),
)

SIGNAL_NAMES: tuple[str, ...] = tuple(spec.name for spec in SIGNALS)


def match_signal(spec: SignalSpec, text: str) -> re.Match[str] | None:
    """Match a signal, sharing contextual price validation with extraction."""

    if spec.name != "budget":
        return spec.pattern.search(text)
    matches = tuple(
        match
        for match in (
            first_price_expression_match(text),
            _QUALITATIVE_BUDGET.search(text),
        )
        if match is not None
    )
    return min(matches, key=lambda match: match.start()) if matches else None


# Maps the contract's `ask_attribute` values onto the signal that fires if the
# customer answers that question. Used to tell an answer apart from an
# unprompted change of direction.
ATTRIBUTE_SIGNALS: dict[str, str] = {
    "budget": "budget",
    "brand": "brand",
    "size": "size",
    "color": "color",
    "material": "material",
    "feature": "feature",
    "style": "style",
    "use_case": "use_case",
}


# --------------------------------------------------------------------------
# Turn-level markers, used by the session tracker rather than by the scorer
# --------------------------------------------------------------------------

# These markers are shared by the session override detector and the routing
# tracker.  They indicate a possible change of direction; the session layer
# separately decides whether the change is a full goal replacement or only a
# preference correction.
OVERRIDE_MARKER_PATTERNS: tuple[str, ...] = (
    r"\bactually\b",
    r"\binstead\b",
    r"\bchanged my mind\b",
    r"\b(?:my )?priority changed\b",
    r"\bon second thought\b",
    r"\bscratch that\b",
    r"\bforget (?:that|it|the|what|about|my)\b",
    r"\b(?:ignore|disregard)\b(?:.{0,45}\b(?:earlier|previous|old|last|that)\b|\s+(?:that|it))",
    r"\bstart over\b",
    r"\bnew search\b",
)

OVERRIDE_MARKER = _compile(*OVERRIDE_MARKER_PATTERNS)

# Strong reset language is intentionally a separate view of the same shared
# vocabulary.  "actually black" and "my priority changed" are still
# overrides, but are preference-level changes unless the message also names a
# replacement product goal.
FULL_GOAL_OVERRIDE_MARKER = _compile(
    r"\bscratch that\b",
    r"\bforget (?:that|it|the|what|about|my)\b",
    r"\b(?:ignore|disregard)\b(?:.{0,45}\b(?:earlier|previous|old|last|that)\b|\s+(?:that|it))",
    r"\bstart over\b",
    r"\bnew search\b",
)

NO_PREFERENCE_MARKER = _compile(
    r"\b(?:don'?t|do not)\s+(?:really\s+)?have\s+(?:an?\s+|any\s+)?"
    r"(?:additional|specific|strong|particular)?\s*preference",
    r"\bno preference\b",
    r"\buse your judgment\b",
    r"^\s*(?:i\s+)?(?:don'?t|do not)\s+know(?:\s+(?:yet|what|which)\b[^.!?]*)?[.!?]*$",
    r"^\s*(?:i\s+)?(?:am|'m)\s+not\s+sure\s*[.!?]*$",
    r"^\s*(?:i\s+)?(?:don'?t|do not)\s+have\s*[.!?]*$",
    r"^\s*(?:anything|whatever)\s+(?:is|works?)\s+fine\s*[.!?]*$",
    r"^\s*(?:you\s+decide|it\s+doesn'?t\s+matter)\s*[.!?]*$",
)

FILLER_MARKER = _compile(
    r"\bask me about one specific attribute\b",
    r"\bnot quite right\b",
)

# The hard/local evaluators can supply this sentence when the Agent has no
# useful standard clarification question. It is conversational control text,
# not a user shopping request, so the Agent must not extract its words as
# product facts.
GENERIC_CLARIFICATION_REPLY = re.compile(
    r"^\s*those\s+options\s+are\s+not\s+quite\s+right\s+yet\.\s*"
    r"(?:you\s+can\s+)?ask\s+me\s+about\s+one\s+specific\s+attribute\.?\s*$",
    re.IGNORECASE,
)

# Messages made only of these carry no shopping content, so the BUYING prior
# must not apply to them.
CONVERSATIONAL_FILLER = frozenset(
    {
        "hi", "hello", "hey", "yo", "hiya", "help", "hmm", "hmmm", "um", "uh",
        "ok", "okay", "k", "sure", "yes", "yeah", "yep", "no", "nope", "thanks",
        "thank", "you", "please", "there", "again", "well", "so", "and", "a",
        "the", "im", "i", "m",
    }
)


# --------------------------------------------------------------------------
# Scoring constants
# --------------------------------------------------------------------------

# The bias makes BUYING the default reading of a bare category request
# ("I'm looking for cardigans"): a named product with no hedging is a shopper
# who knows what shelf they want.
BIAS = 0.35
BUYING_SCALE = 1.25
BROWSING_SCALE = 1.15
LOGISTIC_K = 1.60

# Reported when nothing at all was said. Inside the weak band on purpose: the
# honest answer is "explore, but I have no evidence".
NO_CONTENT_CONFIDENCE = 0.55

# Below this the rules tier does not trust itself. The `weak` flag keys off
# it, telling issue #9 to blend retrieval tracks rather than fork.
WEAK_CONFIDENCE = 0.70

# Phase 1: how many distinct canonical constraint fields a message must fill
# before it is routed BUYING without consulting the signal ledger. Two is the
# measured optimum on the labelled sets; see the ADR for the sweep.
BUYING_TAG_THRESHOLD = 2

# Category is excluded from the Phase 1 count on purpose. Naming a product
# says which shelf the customer is at, not that they have decided -- the same
# principle that gives category keywords zero weight in the ledger below.
TAG_COUNT_EXCLUDE: tuple[str, ...] = ("category",)

# Phase 1 asserts BUYING from the tag count alone, which is sound only while
# extraction is precise. When the ledger reads the same message as browsing at
# least this confidently, the tag count is treated as noise and the message
# falls through to Phase 2. Same asymmetry as DECISION_CONFIDENCE below: a
# wrongly confident BUYING has no natural way back.
BROWSING_VETO_CONFIDENCE = 0.70

# Terminal rule: a message the pipeline cannot decide with at least this much
# confidence is routed BROWSING. Browsing is the recoverable error -- broad
# retrieval plus a clarifying question still converges, whereas a wrongly
# confident BUYING narrows onto constraints the customer never gave.
DECISION_CONFIDENCE = 0.70

# A flip after the session has started needs more evidence than the first
# decision did, so ordinary answer turns cannot rock the router back and forth.
FLIP_MARGIN = 0.80
