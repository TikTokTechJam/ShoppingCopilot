from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "exploring", "actually", "ignore", "earlier", "preference", "what",
    "matters", "options", "quite", "right", "yet", "ask", "one", "specific",
    "attribute", "additional", "have", "dont", "not", "use", "your", "judgment",
}

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|alloy)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"\b(xxs|xs|small|medium|large|xl|xxl|xxxl|wide|narrow|petite|tall|one size)\b",
    re.IGNORECASE,
)

FEATURE_MARKERS = (
    "water resistant", "waterproof", "breathable", "quick drying", "quick-drying",
    "moisture wicking", "moisture-wicking", "lightweight", "durable", "comfortable",
    "insulated", "hypoallergenic", "imported", "closure", "buckle", "zipper",
    "button", "pull on", "machine wash", "hand wash", "adjustable", "removable",
    "pocket", "battery", "indicator", "rubber sole", "sole", "heel", "sleeve",
    "fleece", "uv", "gift", "alloy", "resistant", "stretch", "soft", "warm",
)
STYLE_MARKERS = (
    "casual", "formal", "classic", "fashion", "athletic", "sporty", "slim",
    "regular fit", "relaxed", "loose", "vintage", "boho", "elegant", "short sleeve",
    "long sleeve", "crew", "v neck", "neckline", "fit",
)
USE_CASE_MARKERS = (
    "hiking", "running", "gym", "winter", "outdoor", "work", "trail", "travel",
    "everyday", "beach", "swim", "formal event",
)

QUESTION_ATTRIBUTES = (
    "material", "feature", "color", "style", "use_case", "size", "budget", "brand",
)
ATTRIBUTE_PRIOR = {
    "material": 0.62,
    "feature": 0.68,
    "color": 0.52,
    "style": 0.38,
    "use_case": 0.32,
    "size": 0.28,
    "budget": 0.20,
    "brand": 0.12,
}
ATTRIBUTE_QUESTIONS = {
    "material": "Which material should I prioritize?",
    "feature": "Which feature matters most for this item?",
    "color": "Do you have a preferred color?",
    "style": "Which style or fit should I prioritize?",
    "use_case": "What will you mainly use this for?",
    "size": "Do you have a preferred size or fit?",
    "budget": "What budget range should I stay within?",
    "brand": "Do you have a preferred brand or store?",
}

QUERY_POOL_SIZE = 800
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _is_override(message: str) -> bool:
    lowered = message.lower()
    if re.search(r"\b(ignore|disregard|forget)\b.{0,45}\b(earlier|previous|old)\b", lowered):
        return True
    return lowered.startswith("actually") and any(
        phrase in lowered for phrase in ("i need", "i want", "what i need", "prioritize")
    )


def _is_no_preference(message: str) -> bool:
    lowered = message.lower()
    return (
        "don't have a preference" in lowered
        or "do not have a preference" in lowered
        or "no preference" in lowered
        or "use your judgment" in lowered
        or "not quite right" in lowered
    )


def _is_noise(message: str) -> bool:
    lowered = message.lower()
    return _is_no_preference(message) or (
        "ask me about one specific attribute" in lowered
        or "i don't have an additional preference" in lowered
    )


def _category_context(message: str) -> str:
    first_sentence = re.split(r"[.!?]\s+", message, maxsplit=1)[0]
    return re.split(r",\s+but\b", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _price(value: object) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return None if match is None else float(match.group(0))


class Agent:
    """Adaptive lexical search with expected-utility attribute questions.

    The agent keeps a lightweight posterior approximation over the best lexical
    candidates. At every turn it compares the expected Top-10 concentration and
    information gain of each unused attribute query, while retaining the
    conversation constraints that produced the current candidate pool.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict[str, object]] = {}
        self._attribute_cache: dict[tuple[str, str], tuple[str, ...]] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[object, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                        product.get("price"),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "profile": user_profile,
            "initial_message": "",
            "messages": [],
            "asked": set(),
            "declined": set(),
            "last_asked": None,
        }

    def _ingest_message(self, state: dict[str, object], message: str) -> None:
        messages = state["messages"]
        assert isinstance(messages, list)
        if not messages:
            state["initial_message"] = message
            messages.append(message)
            return

        if _is_override(message):
            initial = str(state["initial_message"])
            state["messages"] = [_category_context(initial), message]
            state["declined"] = set()
            state["last_asked"] = None
            return

        last_asked = state.get("last_asked")
        if isinstance(last_asked, str) and _is_no_preference(message):
            declined = state["declined"]
            assert isinstance(declined, set)
            declined.add(last_asked)
        if not _is_noise(message):
            messages.append(message)

    def _search(self, query: str, limit: int) -> list[dict[str, object]]:
        terms = list(dict.fromkeys(_terms(query)))[:60]
        columns = "parent_asin, title, categories, features, details, store, description, price"
        if terms:
            expression = " OR ".join(f'"{term}"' for term in terms)
            rows = self.connection.execute(
                f"SELECT {columns} FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 7.0, 4.0, 2.0, 2.0, 1.5, 1.0, 0.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        else:
            rows = self.connection.execute(
                f"SELECT {columns} FROM products ORDER BY rowid LIMIT ?", (limit,)
            ).fetchall()

        candidates: list[dict[str, object]] = []
        for rank, row in enumerate(rows):
            product = dict(zip(columns.split(", "), row))
            product["_rank"] = rank
            product["_corpus"] = " ".join(str(product.get(field) or "") for field in SEARCH_FIELDS).lower()
            candidates.append(product)
        return candidates

    def _attribute_values(self, attribute: str, product: dict[str, object]) -> tuple[str, ...]:
        key = (str(product.get("parent_asin")), attribute)
        cached = self._attribute_cache.get(key)
        if cached is not None:
            return cached

        corpus = str(product.get("_corpus") or "")
        if attribute == "material":
            values = tuple(sorted(set(MATERIAL_RE.findall(corpus))))
        elif attribute == "color":
            values = tuple(sorted(set(COLOR_RE.findall(corpus))))
        elif attribute == "size":
            values = tuple(sorted(set(SIZE_RE.findall(corpus))))
        elif attribute == "use_case":
            values = tuple(sorted({marker for marker in USE_CASE_MARKERS if marker in corpus}))
        elif attribute == "style":
            values = tuple(sorted({marker for marker in STYLE_MARKERS if marker in corpus}))
        elif attribute == "feature":
            values = {marker for marker in FEATURE_MARKERS if marker in corpus}
            if not values and product.get("features"):
                values.add("__feature_present__")
            values = tuple(sorted(values))
        elif attribute == "budget":
            value = _price(product.get("price"))
            if value is None:
                values = ()
            elif value < 25:
                values = ("under_25",)
            elif value < 50:
                values = ("25_to_50",)
            elif value < 100:
                values = ("50_to_100",)
            elif value < 150:
                values = ("100_to_150",)
            else:
                values = ("over_150",)
        elif attribute == "brand":
            store = str(product.get("store") or "").strip().lower()
            values = (store,) if store else ()
        else:
            values = ()

        self._attribute_cache[key] = values
        return values

    def _question_utility(self, attribute: str, candidates: list[dict[str, object]]) -> float:
        if not candidates:
            return -math.inf

        weights = [1.0 / ((index + 1) ** 0.72) for index in range(len(candidates))]
        total_weight = sum(weights)
        baseline_top10 = sum(weights[:10]) / total_weight
        group_weights: defaultdict[str, float] = defaultdict(float)
        group_members: defaultdict[str, list[float]] = defaultdict(list)

        for product, weight in zip(candidates, weights):
            values = self._attribute_values(attribute, product) or ("__none__",)
            share = weight / len(values)
            for value in values:
                group_weights[value] += share
                group_members[value].append(share)

        expected_top10_mass = 0.0
        for value, group_weight in group_weights.items():
            if value == "__none__":
                outcome_top10_mass = baseline_top10
            else:
                outcome_top10_mass = sum(sorted(group_members[value], reverse=True)[:10]) / group_weight
            expected_top10_mass += (group_weight / total_weight) * outcome_top10_mass

        probabilities = [group_weight / total_weight for group_weight in group_weights.values()]
        gini_gain = 1.0 - sum(probability * probability for probability in probabilities)
        useful_answer = 1.0 - group_weights.get("__none__", 0.0) / total_weight
        top10_gain = max(0.0, expected_top10_mass - baseline_top10)
        return (
            2.2 * top10_gain
            + 0.35 * gini_gain
            + 0.55 * useful_answer
            + ATTRIBUTE_PRIOR[attribute]
        )

    def _choose_attribute(
        self,
        state: dict[str, object],
        candidates: list[dict[str, object]],
    ) -> str | None:
        asked = state["asked"]
        declined = state["declined"]
        assert isinstance(asked, set)
        assert isinstance(declined, set)
        available = [
            attribute
            for attribute in QUESTION_ATTRIBUTES
            if attribute not in asked and attribute not in declined
        ]
        if not available:
            return None
        return max(
            available,
            key=lambda attribute: (
                self._question_utility(attribute, candidates),
                -QUESTION_ATTRIBUTES.index(attribute),
            ),
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        state = self._sessions[session_id]
        self._ingest_message(state, user_message)
        messages = state["messages"]
        assert isinstance(messages, list)
        query = " ".join(str(message) for message in messages)
        candidates = self._search(query, max(QUERY_POOL_SIZE, top_k))
        recommendations = [
            {"parent_asin": str(product["parent_asin"])}
            for product in candidates[:top_k]
        ]

        attribute: str | None = None
        if turn < 10:
            attribute = self._choose_attribute(state, candidates)
        if attribute is not None:
            asked = state["asked"]
            assert isinstance(asked, set)
            asked.add(attribute)
        state["last_asked"] = attribute

        return {
            "message": ATTRIBUTE_QUESTIONS.get(attribute, "Here are the closest matches I found."),
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
