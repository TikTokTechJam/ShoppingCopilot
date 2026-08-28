"""Signal ledger for the Buying vs Browsing intent router (issue #6).

Every weight the rules tier uses lives in this file, so retuning the router
means editing one place. Nothing else in the package hard-codes a phrase.

The central rule is that a *category* keyword carries no weight. Naming a
product says which shelf the customer is standing at, not whether they have
decided. What separates the two intents is whether the message commits to
anything.

Vocabulary ownership: the attribute vocabularies -- brand, budget, size,
colour, material, feature, use case and style -- are **not** defined here.
They come from `constraints.CANONICAL_VOCAB`, the canonical dictionary issue
#7 produces, via `constraints.alias_pattern()`. Keeping a second copy in this
file meant two lists that could disagree about what "navy" or "water
resistant" means; now there is one, and replacing it with the generated
dictionary from issues #5 and #8 updates both components at once.

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
    SIZE_NUMERIC,
    alias_pattern,
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
    sourced_from: str | None = None


def _compile(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), re.IGNORECASE)


# --------------------------------------------------------------------------
# Evidence of commitment
# --------------------------------------------------------------------------

# Budget: any written price expression from the extractor, plus qualitative
# price talk that sets no numeric bound.
_BUDGET = _compile(
    PRICE_EXPRESSION.pattern,
    r"\b(?:cheap|affordable|inexpensive|budget[-\s]friendly|price range)\b",
    r"\b(?:expensive|pricey|premium|high[-\s]end)\b",
    r"\bbudget\b",
)

_BRAND = alias_pattern("brand", r"\bbrands?\b")

# Sizes: the canonical vocabulary plus numeric sizes, which are values rather
# than dictionary entries. Note what is *not* here -- bare single letters. An
# unanchored \bm\b matches the "m" in "I'm" and invents a hard constraint out
# of a contraction, flipping the label confidently wrong. `SIZE_NUMERIC` and
# the canonical `size s` / `size m` aliases both require the word "size".
_SIZE = alias_pattern("size", SIZE_NUMERIC.pattern, r"\bsizes?\b")

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
    SignalSpec("brand", +1, 1.20, _BRAND, hard=True, sourced_from="#8"),
    SignalSpec("size", +1, 1.00, _SIZE, hard=True, sourced_from="#8"),
    SignalSpec("color", +1, 1.00, _COLOR, hard=True, sourced_from="#8"),
    SignalSpec("material", +1, 1.00, _MATERIAL, hard=True, sourced_from="#8"),
    SignalSpec("feature", +1, 1.00, _FEATURE, hard=True, sourced_from="#8"),
    SignalSpec("use_case", +1, 0.60, _USE_CASE, sourced_from="#8"),
    SignalSpec("style", +1, 0.70, _STYLE, sourced_from="#8"),
    SignalSpec("request_verb", +1, 0.45, _REQUEST_VERB, requires_buying_evidence=True),
    SignalSpec("undecided", -1, 1.40, _UNDECIDED),
    SignalSpec("explore_verb", -1, 1.10, _EXPLORE_VERB),
    SignalSpec("option_seeking", -1, 0.90, _OPTION_SEEKING),
    SignalSpec("vague_head", -1, 1.00, _VAGUE_HEAD, suppressed_by_hard_evidence=True),
    SignalSpec("vague_quality", -1, 0.50, _VAGUE_QUALITY, suppressed_by_hard_evidence=True),
    SignalSpec("hedged", -1, 0.60, _HEDGED, suppressed_by_hard_evidence=True),
)

SIGNAL_NAMES: tuple[str, ...] = tuple(spec.name for spec in SIGNALS)

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

OVERRIDE_MARKER = _compile(
    r"\bactually\b",
    r"\binstead\b",
    r"\bchanged my mind\b",
    r"\b(?:my )?priority changed\b",
    r"\bon second thought\b",
    r"\bscratch that\b",
    r"\bforget (?:the|what|about|my)\b",
    r"\b(?:ignore|disregard)\b.{0,45}\b(?:earlier|previous|old|last|that)\b",
)

NO_PREFERENCE_MARKER = _compile(
    r"\b(?:don'?t|do not)\s+(?:really\s+)?have\s+(?:an?\s+|any\s+)?"
    r"(?:additional|specific|strong|particular)?\s*preference",
    r"\bno preference\b",
    r"\buse your judgment\b",
)

FILLER_MARKER = _compile(
    r"\bask me about one specific attribute\b",
    r"\bnot quite right\b",
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

# Below this the rules tier does not trust itself. Two things key off it: the
# `weak` flag that tells issue #9 to blend retrieval tracks rather than fork,
# and escalation to the reranker.
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

# The reranker only overrides the rules tier when it is this sure. Below it,
# the rules answer stands: an unsure second opinion is not worth discarding a
# deterministic first one for.
RERANKER_ACCEPT_CONFIDENCE = 0.80

# A flip after the session has started needs more evidence than the first
# decision did, so ordinary answer turns cannot rock the router back and forth.
FLIP_MARGIN = 0.80
