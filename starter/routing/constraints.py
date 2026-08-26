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
from dataclasses import dataclass, field


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

_COMPILED: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    name: tuple(
        (canonical, re.compile(rf"(?<![a-z]){alias}(?![a-z])", re.IGNORECASE))
        for canonical, alias in entries
    )
    for name, entries in CANONICAL_VOCAB.items()
}

_NUMBER = r"\$?\s?(\d[\d,]*(?:\.\d+)?)"

_PRICE_MAX = re.compile(
    rf"(?:under|below|less than|no more than|cheaper than|up to|within|max(?:imum)?(?:\s+of)?|nothing over|not more than)\s*{_NUMBER}",
    re.IGNORECASE,
)
_PRICE_MIN = re.compile(
    rf"(?:over|above|more than|at least|min(?:imum)?(?:\s+of)?|starting (?:at|from)|from)\s*{_NUMBER}",
    re.IGNORECASE,
)
_PRICE_RANGE = re.compile(
    rf"(?:between\s+)?{_NUMBER}\s*(?:-|–|to|and)\s*{_NUMBER}\s*(?:dollars|usd|bucks)?",
    re.IGNORECASE,
)
_PRICE_AROUND = re.compile(rf"(?:around|about|approximately|roughly)\s*{_NUMBER}", re.IGNORECASE)
# A price only counts as a price when it is written as one.
_CURRENCY = re.compile(r"\$|\bdollars?\b|\busd\b|\bbucks\b|\bprice\b|\bbudget\b|\bcost", re.IGNORECASE)

# Numeric sizes are values, not vocabulary entries, so they are matched
# separately. Anchored on the word "size" so a bare number is never mistaken
# for one.
_SIZE_NUMERIC = re.compile(r"\bsizes?\s*[:\-]?\s*(\d+(?:\.\d)?)\b", re.IGNORECASE)

# Words that look like a commitment but name no value. Recorded as unmapped so
# a later component can ask about them, never counted as a constraint.
_TOPIC_ONLY = re.compile(
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
    if not _CURRENCY.search(text):
        # "size 10" and "10 to 12 years" are not prices.
        return None, None

    def number(raw: str) -> float:
        return float(raw.replace(",", ""))

    match = _PRICE_RANGE.search(text)
    if match is not None:
        low, high = sorted((number(match.group(1)), number(match.group(2))))
        return low, high

    price_min = price_max = None
    match = _PRICE_MAX.search(text)
    if match is not None:
        price_max = number(match.group(1))
    match = _PRICE_MIN.search(text)
    if match is not None:
        price_min = number(match.group(1))

    if price_min is None and price_max is None:
        match = _PRICE_AROUND.search(text)
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

    numeric_sizes = tuple(dict.fromkeys(_SIZE_NUMERIC.findall(text)))
    if numeric_sizes:
        values["size"] = values["size"] + numeric_sizes

    price_min, price_max = _extract_prices(text)

    # An attribute named without a value is a topic, not a constraint. Keep it
    # so a later component can ask about it; never count it as commitment.
    unmapped = tuple(
        sorted(
            {
                word.lower()
                for word in _TOPIC_ONLY.findall(text)
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
