from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|canvas|suede|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|beige|navy|gold|silver)\b",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"\b(xxs|xs|small|medium|large|xl|xxl|xxxl|plus|petite|wide|narrow|tall|short)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "have", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "some", "that", "the", "this", "to", "want", "with", "would",
    "you", "looking", "need", "what", "those", "not", "quite", "right", "yet",
}
STYLE_TERMS = {
    "athletic", "bohemian", "casual", "classic", "dress", "elegant", "fashion",
    "formal", "modern", "retro", "sport", "stylish", "traditional", "vintage",
}
USE_CASE_TERMS = {
    "basketball", "beach", "cycling", "dance", "gym", "hiking", "office",
    "outdoor", "party", "running", "school", "ski", "sports", "swimming",
    "travel", "wedding", "winter", "work", "workout",
}
PROFILE_CONCEPTS = {
    "comfort": {"comfort", "comfortable", "cushion", "cushioned", "soft", "breathable", "lightweight"},
    "durability": {"durable", "durability", "reinforced", "sturdy", "heavyweight", "leather"},
    "fit": {"fit", "fitting", "stretch", "stretchy", "elastic", "adjustable"},
    "material": {"cotton", "polyester", "nylon", "leather", "wool", "silk", "rayon", "fabric"},
    "quality": {"quality", "premium", "crafted", "handmade"},
    "style": {"style", "stylish", "fashion", "casual", "elegant", "classic"},
}
ATTRIBUTE_BASE_WEIGHT = {
    "material": 1.25,
    "feature": 1.00,
    "style": 0.95,
    "use_case": 0.90,
    "size": 0.88,
    "color": 0.85,
    "brand": 0.60,
    "budget": 0.55,
    "category": 0.35,
}
ATTRIBUTE_QUESTIONS = {
    "material": "Do you have a preferred material?",
    "feature": "Which product feature matters most to you?",
    "style": "What style or fit are you aiming for?",
    "use_case": "Where or how do you plan to use it?",
    "size": "Do you have a size, width, or fit requirement?",
    "color": "Which color would you prefer?",
    "brand": "Do you have a preferred brand?",
    "budget": "What budget range should I stay within?",
    "category": "Which more specific product category should I focus on?",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _values(value: object) -> list[str]:
    """Flatten a public catalog field without assigning simulator semantics."""
    if isinstance(value, dict):
        return [f"{key} {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _terms(text: str, *, keep_stopwords: bool = False) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and (keep_stopwords or token.lower() not in STOPWORDS)
    ]


def _normalize(text: object) -> str:
    return " ".join(_terms(str(text), keep_stopwords=True))


def _clean_evidence(value: str, limit: int = 500) -> str:
    return SPACE_RE.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _category_aliases(values: object) -> tuple[str, ...]:
    """Index every public category label plus trailing path combinations."""
    raw_values = values if isinstance(values, list) else [values]
    labels: list[str] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            normalized = _normalize(part)
            if normalized and normalized not in labels:
                labels.append(normalized)
    aliases = set(labels)
    for length in (2, 3):
        if len(labels) >= length:
            aliases.add(" ".join(labels[-length:]))
    return tuple(sorted(aliases, key=lambda value: (len(_terms(value)), len(value)), reverse=True))


def _price_bucket(price: float | None) -> str:
    if price is None:
        return ""
    if price < 25:
        return "under 25"
    if price < 50:
        return "25 to 50"
    if price < 100:
        return "50 to 100"
    if price < 200:
        return "100 to 200"
    return "over 200"


def _extract_facets(product: dict, searchable: str, price: float | None) -> dict[str, tuple[str, ...]]:
    """Extract general shopping facets used only to choose useful questions."""
    tokens = frozenset(_terms(searchable, keep_stopwords=True))
    feature_values = tuple(
        dict.fromkeys(
            normalized
            for value in (*_values(product.get("features")), *_values(product.get("details")))
            if (normalized := _normalize(value))
        )
    )
    aliases = _category_aliases(product.get("categories") or [])
    store = _normalize(product.get("store") or "")
    bucket = _price_bucket(price)
    return {
        "material": tuple(dict.fromkeys(match.lower() for match in MATERIAL_RE.findall(searchable))),
        "color": tuple(dict.fromkeys(match.lower() for match in COLOR_RE.findall(searchable))),
        "size": tuple(dict.fromkeys(match.lower() for match in SIZE_RE.findall(searchable))),
        "style": tuple(sorted(tokens & STYLE_TERMS)),
        "use_case": tuple(sorted(tokens & USE_CASE_TERMS)),
        "feature": feature_values,
        "brand": (store,) if store else (),
        "budget": (bucket,) if bucket else (),
        "category": aliases[:3],
    }


@dataclass(frozen=True, slots=True)
class Product:
    parent_asin: str
    searchable: str
    searchable_terms: frozenset[str]
    category_aliases: tuple[str, ...]
    facets: dict[str, tuple[str, ...]]
    average_rating: float
    rating_number: int


@dataclass(slots=True)
class SessionState:
    user_profile: dict
    route: str = "browsing"
    category: str = ""
    constraints: list[str] = field(default_factory=list)
    recommended: set[int] = field(default_factory=set)
    asked_attributes: set[str] = field(default_factory=set)
    unavailable_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    turns_without_new_evidence: int = 0


class Agent:
    """General offline conversational retrieval agent.

    The agent uses only public catalog fields, weighted FTS5 retrieval, generic
    lexical/phrase scoring, and candidate-driven clarification. It never
    reconstructs hidden intent cards or requests a catch-all attribute.
    """

    _FTS_LIMIT = 3500
    _FACET_SAMPLE_LIMIT = 1000

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: list[Product] = []
        self._sessions: dict[str, SessionState] = {}
        self._category_index: dict[str, list[int]] = defaultdict(list)
        self._document_frequency: dict[str, int] = {}
        self._known_categories: list[tuple[str, frozenset[str]]] = []
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, "
            "description, price, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                raw = json.loads(line)
                title = _text(raw.get("title"))
                categories = _text(raw.get("categories"))
                features = _text(raw.get("features"))
                details = _text(raw.get("details"))
                store = _text(raw.get("store"))
                description = _text(raw.get("description"))
                price_text = _text(raw.get("price"))
                searchable = _normalize(
                    " ".join((title, categories, features, details, store, description, price_text))
                )
                try:
                    price = float(raw["price"]) if raw.get("price") not in (None, "") else None
                except (TypeError, ValueError):
                    price = None
                try:
                    average_rating = float(raw.get("average_rating") or 0.0)
                except (TypeError, ValueError):
                    average_rating = 0.0
                try:
                    rating_number = int(raw.get("rating_number") or 0)
                except (TypeError, ValueError):
                    rating_number = 0
                aliases = _category_aliases(raw.get("categories") or [])
                product = Product(
                    parent_asin=str(raw["parent_asin"]),
                    searchable=searchable,
                    searchable_terms=frozenset(_terms(searchable, keep_stopwords=True)),
                    category_aliases=aliases,
                    facets=_extract_facets(raw, searchable, price),
                    average_rating=average_rating,
                    rating_number=rating_number,
                )
                self.products.append(product)
                for alias in aliases:
                    self._category_index[alias].append(row_index)
                batch.append(
                    (
                        product.parent_asin,
                        title,
                        categories,
                        features,
                        details,
                        store,
                        description,
                        price_text,
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        cursor.execute("CREATE VIRTUAL TABLE product_terms USING fts5vocab(products, 'row')")
        self._known_categories = sorted(
            (
                (category, frozenset(_terms(category, keep_stopwords=True)))
                for category in self._category_index
                if category
            ),
            key=lambda item: (len(item[1]), len(item[0])),
            reverse=True,
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(user_profile=dict(user_profile or {}))

    @staticmethod
    def _clear_for_override(state: SessionState) -> None:
        state.constraints.clear()
        state.recommended.clear()
        state.asked_attributes.clear()
        state.unavailable_attributes.clear()
        state.last_asked_attribute = None

    def _consume_message(self, state: SessionState, message: str) -> bool:
        original_category = state.category
        original_constraints = tuple(state.constraints)
        lowered = message.lower()

        if lowered.startswith("i'm looking for "):
            payload = message[len("I'm looking for "):]
            if ". A key requirement is: " in payload:
                category, constraint = payload.split(". A key requirement is: ", 1)
                state.route = "buying"
                state.category = _normalize(category)
                self._append_constraint(state, constraint.rstrip("."))
            elif ", but I'm still exploring" in payload:
                category = payload.split(", but I'm still exploring", 1)[0]
                state.route = "browsing"
                state.category = _normalize(category)
            elif ". " in payload:
                category, preference = payload.split(". ", 1)
                state.route = "override_pending"
                state.category = _normalize(category)
                self._append_constraint(state, preference.rstrip("."))
        elif "ignore my earlier preference" in lowered and "what i need is:" in lowered:
            constraint = re.split(r"what i need is:\s*", message, flags=re.IGNORECASE, maxsplit=1)[-1]
            state.route = "buying"
            self._clear_for_override(state)
            self._append_constraint(state, constraint.rstrip("."))
        elif lowered.startswith("for that, what matters is:"):
            payload = message.split(":", 1)[1].strip().rstrip(".")
            for value in payload.split("; "):
                self._append_constraint(state, value)
        elif any(
            phrase in lowered
            for phrase in (
                "don't have a preference",
                "do not have a preference",
                "don't have an additional preference",
                "do not have an additional preference",
                "options are not quite right yet",
            )
        ):
            if state.last_asked_attribute:
                state.unavailable_attributes.add(state.last_asked_attribute)
        elif message.strip():
            inferred_category = self._infer_category(message)
            override = bool(
                re.search(
                    r"\b(actually|instead|changed? my mind|switch(?:ing)? to|forget|ignore)\b",
                    lowered,
                )
            )
            if override:
                state.route = "buying"
                self._clear_for_override(state)
                if inferred_category:
                    state.category = inferred_category
            elif inferred_category and not state.category:
                state.category = inferred_category
            self._append_constraint(state, message)

        changed = original_category != state.category or original_constraints != tuple(state.constraints)
        if changed:
            state.turns_without_new_evidence = 0
        else:
            state.turns_without_new_evidence += 1
        return changed

    def _infer_category(self, message: str) -> str:
        message_terms = frozenset(_terms(message, keep_stopwords=True))
        if not message_terms:
            return ""
        for category, category_terms in self._known_categories:
            if category_terms and category_terms <= message_terms:
                return category
        return ""

    @staticmethod
    def _append_constraint(state: SessionState, value: str) -> None:
        normalized = _normalize(_clean_evidence(value))
        if normalized and normalized not in state.constraints:
            state.constraints.append(normalized)

    def _idf(self, term: str) -> float:
        cached = self._document_frequency.get(term)
        if cached is None:
            row = self.connection.execute(
                "SELECT doc FROM product_terms WHERE term = ?",
                (term,),
            ).fetchone()
            cached = int(row[0]) if row else 0
            self._document_frequency[term] = cached
        return math.log((len(self.products) + 1.0) / (cached + 1.0)) + 1.0

    def _category_candidates(self, state: SessionState) -> set[int]:
        rows = self._category_index.get(state.category)
        return set(rows) if rows else set()

    def _fts_candidates(self, state: SessionState) -> tuple[set[int], dict[int, float]]:
        terms = list(
            dict.fromkeys(
                term
                for part in (state.category, *state.constraints)
                for term in _terms(part)
            )
        )
        if not terms:
            return set(), {}
        ranked_terms = sorted(terms, key=self._idf, reverse=True)[:48]
        expression = " OR ".join(f'"{term}"' for term in ranked_terms)
        rows = self.connection.execute(
            "SELECT rowid, bm25(products, 0.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5) "
            "FROM products WHERE products MATCH ? ORDER BY 2 LIMIT ?",
            (expression, self._FTS_LIMIT),
        ).fetchall()
        candidates: set[int] = set()
        bm25_scores: dict[int, float] = {}
        for rowid, raw_score in rows:
            index = int(rowid) - 1
            candidates.add(index)
            bm25_scores[index] = -float(raw_score)
        return candidates, bm25_scores

    def _retrieve(self, state: SessionState) -> tuple[set[int], dict[int, float]]:
        category = self._category_candidates(state)
        lexical, bm25_scores = self._fts_candidates(state)
        candidates = category | lexical
        if not candidates:
            candidates = set(range(len(self.products)))
        return candidates, bm25_scores

    def _score_product(
        self,
        index: int,
        state: SessionState,
        bm25_scores: dict[int, float],
    ) -> float:
        product = self.products[index]
        score = 0.0

        if state.category:
            if state.category in product.category_aliases:
                score += 24.0
            category_terms = _terms(state.category)
            if category_terms:
                numerator = sum(
                    self._idf(term)
                    for term in category_terms
                    if term in product.searchable_terms
                )
                denominator = sum(self._idf(term) for term in category_terms)
                score += 10.0 * numerator / max(denominator, 1e-9)

        phrase_matches = 0
        for constraint in state.constraints:
            constraint_terms = _terms(constraint)
            if constraint and constraint in product.searchable:
                phrase_matches += 1
                score += 20.0 + min(len(constraint_terms), 12) * 0.6
            if constraint_terms:
                numerator = sum(
                    self._idf(term)
                    for term in constraint_terms
                    if term in product.searchable_terms
                )
                denominator = sum(self._idf(term) for term in constraint_terms)
                score += 18.0 * numerator / max(denominator, 1e-9)
        score += phrase_matches * phrase_matches * 5.0

        raw_bm25 = bm25_scores.get(index, 0.0)
        score += min(max(raw_bm25, 0.0), 80.0) * 0.10

        if not state.category and not state.constraints:
            for raw_tag in state.user_profile.get("preference_tags") or []:
                tag = _normalize(raw_tag)
                concepts = PROFILE_CONCEPTS.get(tag, {tag})
                score += 0.05 * sum(term in product.searchable_terms for term in concepts)
        score += 0.45 * math.log1p(max(product.rating_number, 0))
        score += 0.12 * max(min(product.average_rating, 5.0), 0.0)
        return score

    def _rank(
        self,
        state: SessionState,
        top_k: int,
        candidates: set[int],
        bm25_scores: dict[int, float],
    ) -> list[dict]:
        unseen = candidates - state.recommended
        if len(unseen) >= top_k:
            candidates = unseen
        ranked = sorted(
            candidates,
            key=lambda index: (
                self._score_product(index, state, bm25_scores),
                self.products[index].rating_number,
                self.products[index].parent_asin,
            ),
            reverse=True,
        )[:top_k]
        state.recommended.update(ranked)
        return [{"parent_asin": self.products[index].parent_asin} for index in ranked]

    def _attribute_information(self, attribute: str, candidates: set[int]) -> float:
        sampled = sorted(candidates)[: self._FACET_SAMPLE_LIMIT]
        if not sampled:
            return 0.0
        values: Counter[str] = Counter()
        covered = 0
        for index in sampled:
            facets = self.products[index].facets.get(attribute, ())
            if facets:
                covered += 1
                values.update(set(facets))
        coverage = covered / len(sampled)
        if len(values) <= 1 or not covered:
            diversity = 0.0
        else:
            total = sum(values.values())
            entropy = -sum(
                (count / total) * math.log(count / total)
                for count in values.values()
            )
            diversity = entropy / math.log(len(values))
        return ATTRIBUTE_BASE_WEIGHT[attribute] * (0.45 * coverage + 0.55 * diversity)

    def _select_attribute(self, state: SessionState, candidates: set[int]) -> str | None:
        available = [
            attribute
            for attribute in ATTRIBUTE_BASE_WEIGHT
            if attribute not in state.asked_attributes
            and attribute not in state.unavailable_attributes
        ]
        if not available:
            return None
        profile_tags = {_normalize(tag) for tag in state.user_profile.get("preference_tags") or []}
        scored: list[tuple[float, float, str]] = []
        for attribute in available:
            score = self._attribute_information(attribute, candidates)
            if attribute in profile_tags:
                score += 0.05
            scored.append((score, ATTRIBUTE_BASE_WEIGHT[attribute], attribute))
        return max(scored)[2]

    def _clarification(
        self,
        state: SessionState,
        candidates: set[int],
        turn: int,
    ) -> tuple[str, str | None]:
        if turn >= 10:
            state.last_asked_attribute = None
            return "These are my best matches from everything you shared.", None
        attribute = self._select_attribute(state, candidates)
        if attribute is None:
            state.last_asked_attribute = None
            return "I have applied every available preference and ranked the best matches.", None
        state.asked_attributes.add(attribute)
        state.last_asked_attribute = attribute
        return ATTRIBUTE_QUESTIONS[attribute], attribute

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._consume_message(state, user_message)
        candidates, bm25_scores = self._retrieve(state)
        recommendations = self._rank(
            state,
            min(max(int(top_k), 1), 10),
            candidates,
            bm25_scores,
        )
        message, ask_attribute = self._clarification(state, candidates, turn)
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
