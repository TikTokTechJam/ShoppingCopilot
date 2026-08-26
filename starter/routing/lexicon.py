"""Signal ledger for the Buying vs Browsing intent router (issue #6).

Every weight the rules tier uses lives in this file, so retuning the router
means editing one place. Nothing else in the package hard-codes a phrase.

The central rule is that a *category* keyword carries no weight. Naming a
product says which shelf the customer is standing at, not whether they have
decided. What separates the two intents is whether the message commits to
anything.

Vocabulary note: the value lists are a deliberately small starting point.
Issue #5 (canonical catalog facts) and issue #8 (canonical attribute
dictionary) are the intended owners of the `color`, `material`, `brand`,
`size` and `feature` vocabularies; `SignalSpec.sourced_from` records that so
the swap is mechanical rather than archaeological.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


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

_BUDGET = _compile(
    r"\b(?:under|below|less than|no more than|cheaper than|up to|within|max(?:imum)?|around|about)\s*\$\s?\d[\d,]*(?:\.\d+)?",
    r"\b(?:under|below|less than|no more than|cheaper than|up to)\s+\d[\d,]*(?:\.\d+)?\s*(?:dollars|usd|bucks)\b",
    r"\$\s?\d[\d,]*(?:\.\d+)?",
    r"\bbudget\b",
    r"\b(?:cheap|affordable|inexpensive|budget[-\s]friendly|price range)\b",
    r"\b(?:expensive|pricey|premium|high[-\s]end)\b",
)

# Placeholder vocabulary; issue #8 should supply the canonical brand list
# derived from the catalog `store` field.
_BRAND = _compile(
    r"\bbrands?\b",
    r"\b(?:nike|adidas|puma|reebok|new balance|under armour|columbia|carhartt|"
    r"levi'?s|wrangler|hanes|gildan|skechers|crocs|timberland|dr\.? martens|"
    r"clarks|vans|converse|asics|brooks|merrell|patagonia|north face|uniqlo)\b",
)

# Bare single-letter size tokens are deliberately absent. An unanchored \bm\b
# matches the "m" in "I'm" and invents a hard constraint out of a contraction,
# which flips the label *confidently* wrong. Single letters must follow "size".
_SIZE = _compile(
    r"\bsizes?\b",
    r"\bsize\s*[:\-]?\s*(?:\d+(?:\.\d+)?|x{0,3}[sml]\b)",
    r"\b(?:xxs|xs|xl|xxl|xxxl)\b",
    r"\b(?:petite|plus[-\s]size|big and tall)\b",
    r"\b(?:wide|narrow|tall|regular)\s+(?:fit|width|sizing)\b",
    r"\bin\s+an?\s+(?:small|medium|large)\b",
)

_COLOR = _compile(
    r"\bcolou?rs?\b",
    r"\b(?:black|white|blue|navy|red|pink|green|brown|gray|grey|purple|violet|"
    r"yellow|orange|beige|tan|cream|ivory|gold|silver|olive|burgundy|teal|"
    r"maroon|charcoal|khaki|turquoise)\b",
)

# "down" is absent on purpose: "down jacket" is a category, not a material.
# Same failure shape as the single-letter sizes above.
_MATERIAL = _compile(
    r"\b(?:material|fabric)\b",
    r"\b(?:cotton|leather|polyester|nylon|wool|silk|denim|rayon|spandex|linen|"
    r"suede|cashmere|fleece|satin|velvet|mesh|canvas|alloy|sterling silver|"
    r"stainless steel|faux leather|faux fur|microfiber|corduroy|flannel|rubber|"
    r"bamboo|acrylic|lycra|jersey|chiffon|lace|tweed|sherpa|gore[-\s]?tex|"
    r"neoprene|nubuck|shearling)\b",
    r"\b(?:goose|duck)\s+down\b",
    r"\bdown[-\s](?:filled|insulated|feather)\b",
)

_FEATURE = _compile(
    r"\b(?:waterproof|water[-\s]resistant|weatherproof|windproof|breathable|"
    r"insulated|thermal|reversible|adjustable|hypoallergenic|orthopedic|"
    r"wrinkle[-\s]free|moisture[-\s]wicking|non[-\s]slip|slip[-\s]resistant|"
    r"anti[-\s]slip|arch support|ankle support|memory foam|machine washable|"
    r"hand wash|quick[-\s]?dry(?:ing)?|dries quickly|dry quickly|uv protection|"
    r"lightweight|padded|hooded|sleeveless|pockets?|zipper|drawstring|"
    r"high[-\s]waisted|stretchy|elastic waist)\b",
    r"\b(?:long|short)\s+sleeves?\b",
    r"\b(?:grip|traction|cushioning|breathability|insulation|warmth|compression|"
    r"odou?r[-\s]resistant|stain[-\s]resistant|water[-\s]repellent|"
    r"sun protection|moisture management)\b",
    r"\bhandles?\s+sweat\b",
    r"\bprotection from\s+(?:wind|rain|sun|the cold)\b",
)

# Soft evidence. A stated occasion is as consistent with someone describing a
# vague situation as with a decided buyer, so it must not suppress vagueness.
_USE_CASE = _compile(
    r"\bfor\s+(?:the\s+|a\s+|my\s+)?(?:hiking|running|walking|jogging|the gym|gym|"
    r"swimming|tennis|golf|yoga|hunting|camping|skiing|snowboarding|climbing|"
    r"work|the office|office|school|weddings?|parties|a party|formal events?|"
    r"travel|travelling|traveling|the beach|beach|halloween|christmas|winter|"
    r"summer|everyday wear|everyday use|daily wear)\b",
    r"\b(?:mainly|primarily|mostly)\s+(?:use|wear|for)\b",
    r"\b(?:is the )?main use\b",
)

_STYLE = _compile(
    r"\bstyle\b",
    r"\b(?:casual|formal|classic|vintage|retro|boho|bohemian|preppy|sporty|"
    r"athletic|elegant|minimalist|slim fit|relaxed fit|oversized)\b",
)

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
