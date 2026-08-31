"""Small localhost-only web viewer for Manual400 or interactive sessions.

The session execution lives in :class:`evaluator.hard_evaluator.Manual400SessionRunner`;
this module also supports a developer-driven console session against a selected
catalog target, and serializes the existing evaluator-side diagnostics for a
browser.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from dictionary.registry import SEMANTIC_ATTRIBUTES
from evaluator.agent_factory import build_evaluator_agent
from evaluator.hard_evaluator import (
    DEBUG_FACT_FIELDS,
    MAX_TURNS,
    Manual400SessionRunner,
    add_score_fields,
    _debug_rank,
    _debug_ranking_snapshot,
    _debug_state_snapshot,
    _debug_target_facts,
    _debug_values,
    load_catalog_ids,
    load_jsonl,
    metric_summary,
    normalize_recommendations,
    TOP_K,
    validate_agent_response,
    validate_sessions,
)
from evaluator.local_evaluator import (
    catalog_index as local_catalog_index,
    coarse_category as local_coarse_category,
    customer_reply as local_customer_reply,
    initial_message as local_initial_message,
    materialize_hidden_fields as local_materialize_hidden_fields,
)
from starter import clarification
from starter.agent import CLARIFICATION_CANDIDATE_LIMIT
from starter.retrieval import MODE_SCORE_WEIGHTS
from starter.routing import constraints as constraint_module


DEFAULT_CATALOG = "data/catalog.jsonl"
DEFAULT_SESSIONS = "data/derived/gptannotation/sessions.jsonl"
DEFAULT_LOCAL_DATASET = "data/public_set.jsonl"
DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).with_name("debug_web_ui")


def _json_safe(value: Any) -> Any:
    """Convert small evaluator objects to values accepted by json.dumps."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _constraint_payload(constraints: Any) -> dict[str, Any]:
    """Serialize a ShoppingConstraints object, not a SessionState wrapper."""

    if constraints is None:
        return {}
    payload: dict[str, Any] = {}
    for field_name in DEBUG_FACT_FIELDS + ("size",):
        values = _debug_values(constraints, field_name)
        if values:
            payload[field_name] = values
    budget: dict[str, float] = {}
    price_min = getattr(constraints, "price_min", None)
    price_max = getattr(constraints, "price_max", None)
    if price_min is not None:
        budget["min"] = float(price_min)
    if price_max is not None:
        budget["max"] = float(price_max)
    if budget:
        payload["budget"] = budget
    return payload


def _changed_constraint_payload(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Show only state changes produced by the real Agent turn.

    Both sides are reported. Showing the after-value alone hid two cases that
    matter: an override replacing ``red`` with ``blue`` looked like a plain
    addition, and a removal looked like a no-op empty list.
    """

    field_order = (*DEBUG_FACT_FIELDS, "size", "budget")
    return {
        field_name: {
            "before": before.get(field_name, []),
            "after": after.get(field_name, []),
        }
        for field_name in field_order
        if before.get(field_name) != after.get(field_name)
    }


def _intent_payload(agent: Any, session_id: str, message: str) -> dict[str, Any]:
    """Re-run the router on this turn's message for its audit trail.

    ``SessionState`` keeps only the decided mode, so the tier/confidence/signal
    detail is recomputed here.  It is read-only and uses the Agent's own
    router, so it cannot change what the Agent decided.
    """

    router = getattr(agent, "router", None)
    classify = getattr(router, "classify", None)
    if not callable(classify):
        return {}
    try:
        result = classify(message or "")
    except Exception as exc:  # diagnostics must never break a turn
        return {"error": f"{type(exc).__name__}: {exc}"}
    as_dict = getattr(result, "as_dict", None)
    payload = dict(as_dict()) if callable(as_dict) else {}
    payload["signal_detail"] = [
        {
            "name": signal.name,
            "polarity": signal.polarity,
            "weight": signal.weight,
            "evidence": signal.evidence,
        }
        for signal in getattr(result, "signals", ())
    ]
    tier = str(payload.get("tier", ""))
    payload["tier_meaning"] = {
        "tags": "decided on constraint count alone; the ledger was not consulted",
        "rules": "decided by the signal ledger",
        "default": "undecided; the terminal rule routed BROWSING",
    }.get(tier, tier)
    return payload


def _profile_payload(agent: Any, session_id: str) -> dict[str, Any]:
    """Preference tags, their affinity factors, and which backend produced them."""

    affinity = getattr(agent, "_profile_affinity", {}).get(session_id)
    if affinity is None:
        return {"enabled": False, "reason": "user profile is disabled"}
    tags = list(getattr(affinity, "tags", ()) or ())
    # ``_similarity`` is None when there are no tags at all; otherwise the
    # backend is embedding only when an encoder was actually supplied.
    encoder = getattr(getattr(agent, "retriever", None), "query_encoder", None)
    backend = "embedding" if encoder is not None and tags else "lexical"
    factors = {}
    try:
        factors = {
            name: round(float(value), 4)
            for name, value in affinity.as_dict().items()
        }
    except Exception:
        factors = {}
    return {
        "enabled": True,
        "preference_tags": tags,
        "similarity_backend": backend if tags else "none",
        "backend_note": (
            "no encoder is configured, so tag/attribute similarity is Jaccard "
            "token overlap, not embeddings"
            if backend == "lexical" and tags
            else ""
        ),
        "affinity": {
            name: round(float(value), 4)
            for name, value in getattr(affinity, "affinity", {}).items()
        },
        "factors": factors,
        "refused": list(getattr(affinity, "_refused", ()) or ()),
    }


def _clarification_payload(
    agent: Any,
    session_id: str,
    before_state: Mapping[str, Any],
    after_state: Mapping[str, Any],
    candidates: Any,
) -> dict[str, Any]:
    """Per-attribute clarification utilities, plus why each was rejected."""

    base = {
        "previous_asked": before_state.get("last_asked"),
        "next_asked": after_state.get("last_asked"),
    }
    policy = getattr(agent, "clarification", None)
    if policy is None or candidates is None:
        return base

    try:
        stats = policy.analyze(candidates)
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    constraints_known = set(after_state.get("constraints_populated", ()))
    asked = set(after_state.get("asked_attributes", ()) or ())
    asked.update(after_state.get("no_preference_attributes", ()) or ())
    affinity = getattr(agent, "_profile_affinity", {}).get(session_id)
    profile_factor = affinity.factor if affinity is not None else None
    turn = after_state.get("turn")
    mode = after_state.get("mode") or "BROWSING"

    rows: list[dict[str, Any]] = []
    for attribute in clarification.NORMAL_CLARIFICATION_ATTRIBUTES:
        facet = stats.facets.get(attribute)
        row: dict[str, Any] = {"attribute": attribute}
        if attribute in asked:
            row["eligible"] = False
            row["reason"] = "already asked or declined"
        elif attribute in constraints_known:
            row["eligible"] = False
            row["reason"] = "already known from constraints"
        elif facet is None:
            row["eligible"] = False
            row["reason"] = "no facet in the candidate pool"
        else:
            row["coverage"] = round(facet.coverage, 4)
            row["values"] = len(facet.counts)
            row["gini"] = round(facet.expected_reduction, 4)
            row["entropy"] = round(facet.entropy, 4)
            row["top_values"] = list(facet.top_values(3))
            if facet.coverage < 0.20:
                row["eligible"] = False
                row["reason"] = f"coverage {facet.coverage:.2f} < 0.20"
            elif len(facet.counts) < 2:
                row["eligible"] = False
                row["reason"] = "fewer than 2 distinct values"
            elif facet.expected_reduction < 0.10:
                row["eligible"] = False
                row["reason"] = f"gini {facet.expected_reduction:.2f} < 0.10"
            else:
                row["eligible"] = True
                row["reason"] = None
        if row.get("eligible"):
            try:
                row["utility"] = round(
                    float(
                        policy._score(
                            attribute,
                            tuple(candidates),
                            mode,
                            profile_factor,
                            turn,
                            candidate_stats=stats,
                        )
                    ),
                    6,
                )
            except Exception:
                row["utility"] = None
        else:
            row["utility"] = None
        rows.append(row)

    rows.sort(key=lambda item: (item["utility"] is None, -(item["utility"] or 0.0)))
    base.update(
        {
            "candidate_count": stats.candidate_count,
            "pool_broad": bool(policy.is_broad_pool(stats)),
            "broad_threshold": clarification.BROAD_CANDIDATE_THRESHOLD,
            "floor": round(clarification.ASK_UTILITY_FLOOR, 6),
            "attributes": rows,
        }
    )
    return base


def _first_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if str(item).strip())
    return "" if value is None else str(value)


def _product_payload(agent: Any, asin: str) -> dict[str, Any]:
    product = getattr(agent.retriever, "product_by_asin", {}).get(asin)
    if product is None:
        return {"parent_asin": asin, "available": False}
    raw = getattr(product, "raw", {})
    taxonomy = product.facts.get("category", ())
    if not taxonomy:
        taxonomy = raw.get("categories", ())
    return {
        "parent_asin": asin,
        "available": True,
        "title": _first_text(raw.get("title", raw.get("name", ""))),
        "price": product.price,
        "taxonomy": list(taxonomy) if isinstance(taxonomy, (list, tuple)) else taxonomy,
        "facts": {
            field_name: list(product.facts.get(field_name, ()))
            for field_name in DEBUG_FACT_FIELDS
        },
    }


def _layer2_status(agent: Any) -> dict[str, Any]:
    """Report the BGE canonical-expansion path, not product-vector search."""

    del agent
    loader = getattr(constraint_module, "_load_default_dictionary", None)
    dictionary = loader() if callable(loader) else None
    available = bool(
        dictionary is not None
        and getattr(dictionary, "semantic_available", False)
    )
    if available:
        encoder = getattr(dictionary, "_query_encoder", None)
        return {
            "available": True,
            "model": getattr(dictionary, "embedding_model", None),
            "query_model": getattr(encoder, "model_id", None),
            "dimension": getattr(dictionary, "embedding_dimension", None),
            "attributes": list(SEMANTIC_ATTRIBUTES),
            "brand": "exact-only",
        }
    return {
        "available": False,
        "reason": "The local BGE canonical-expansion dictionary is unavailable.",
        "setup_hint": (
            "Install the local BGE model under models/bge-small-en-v1.5 or set "
            "SHOPPING_ATTRIBUTE_EMBEDDING_MODEL to its local path."
        ),
    }


def _product_dense_status(agent: Any) -> dict[str, Any]:
    """Report the V5 product-card vector path used by Browsing."""

    retriever = getattr(agent, "retriever", None)
    index = getattr(retriever, "product_embedding_index", None)
    encoder = getattr(retriever, "product_query_encoder", None)
    available = bool(getattr(retriever, "product_dense_available", False))
    manifest = getattr(index, "manifest", {}) if index is not None else {}
    retrieval_mode = getattr(retriever, "browsing_retrieval_mode", "hybrid")
    if not isinstance(manifest, Mapping):
        manifest = {}
    artifact_model = manifest.get("embedding_model", manifest.get("model"))
    dimension = getattr(index, "dimension", None)
    if available:
        return {
            "available": True,
            "artifact": "data/derived/product_embeddings_v5",
            "retrieval_mode": retrieval_mode,
            "model": artifact_model,
            "query_model": getattr(encoder, "model_id", None),
            "dimension": dimension,
            "products": int(manifest.get("product_count", len(getattr(index, "asins", ())))),
            "product_card_fields": list(
                manifest.get(
                    "product_card_fields",
                    ["category", "brand", "color", "material", "feature", "use_case"],
                )
            ),
            "hash_fallback": False,
        }
    if index is None:
        reason = "The V5 semantic product-card artifact is unavailable."
    else:
        reason = getattr(retriever, "product_embedding_compatibility_error", None) or (
            "The V5 product-card query encoder is unavailable or incompatible."
        )
    return {
        "available": False,
        "artifact_loaded": index is not None,
        "artifact": "data/derived/product_embeddings_v5",
        "retrieval_mode": retrieval_mode,
        "model": artifact_model,
        "query_model": getattr(encoder, "model_id", None),
        "dimension": dimension,
        "reason": reason,
        "setup_hint": (
            "Build the local Qwen product-card artifact, install its local "
            "query model, and set SHOPPING_PRODUCT_EMBEDDING_MODEL if needed."
        ),
        "hash_fallback": False,
    }


def _bm25_status(agent: Any) -> dict[str, Any]:
    retriever = getattr(agent, "retriever", None)
    available = bool(getattr(retriever, "bm25_available", False))
    reason = getattr(retriever, "bm25_error", None) or (
        "The local BM25 product-text index is unavailable."
    )
    index = getattr(retriever, "bm25_index", None)
    return {
        "available": available,
        "state": getattr(
            retriever,
            "bm25_state",
            "ready" if available else "unavailable",
        ),
        "reason": None if available else reason,
        "indexed_products": (
            int(getattr(index, "indexed_rows", 0)) if index is not None else 0
        ),
        "catalog_products": len(getattr(retriever, "_catalog_order", ()) or ()),
        "build_seconds": getattr(retriever, "bm25_build_seconds", None),
    }


def _state_payload(agent: Any, session_id: str) -> dict[str, Any]:
    state = agent.sessions.get(session_id)
    snapshot = _debug_state_snapshot(agent, session_id)
    snapshot.update(
        {
            "turn": int(getattr(state, "turn", 0)),
            "last_user_message": getattr(state, "last_user_message", None),
            "asked_attributes": sorted(
                str(value) for value in getattr(state, "asked_attributes", set())
            ),
            "last_asked": getattr(state, "last_asked", None),
            "retrieval_query_text": getattr(
                state,
                "retrieval_query_text",
                getattr(state, "query_text", ""),
            ),
        }
    )
    return snapshot


def _candidate_payload(
    agent: Any,
    candidate: Any,
    rank: int,
    target: str,
    dense_available: bool,
    canonical_available: bool,
    bm25_available: bool,
) -> dict[str, Any]:
    asin = str(candidate.parent_asin)
    product = _product_payload(agent, asin)
    return {
        "rank": rank,
        "parent_asin": asin,
        "title": product.get("title", ""),
        "price": product.get("price"),
        "dense_score": (
            float(candidate.dense_score) if dense_available else None
        ),
        "bm25_score": (
            float(getattr(candidate, "bm25_score", 0.0))
            if bm25_available
            else None
        ),
        "fusion_score": (
            float(getattr(candidate, "fusion_score", 0.0))
            if getattr(candidate, "fusion_score", None) is not None
            else None
        ),
        "mmr_score": (
            float(getattr(candidate, "mmr_score"))
            if getattr(candidate, "mmr_score", None) is not None
            else None
        ),
        "final_score": float(candidate.score),
        "base_score": float(candidate.score),
        "ranking_score": float(candidate.ranking_score),
        "matched_constraints": list(candidate.matched_constraints),
        "matched_semantic_constraints": list(
            getattr(candidate, "matched_semantic_constraints", ())
        ),
        "target": asin == target,
    }


def _ranking_payload(
    agent: Any,
    session_id: str,
    target: str,
    ranked: Iterable[str],
) -> dict[str, Any]:
    snapshot = _debug_ranking_snapshot(agent, session_id, target)
    canonical_status = _layer2_status(agent)
    canonical_available = bool(canonical_status.get("available", False))
    product_dense_status = _product_dense_status(agent)
    dense_available = bool(product_dense_status.get("available", False))
    bm25_status = _bm25_status(agent)
    bm25_available = bool(bm25_status.get("available", False))
    eligible = list(snapshot["eligible"])
    global_ranking = list(snapshot["global"])
    target_eligible = snapshot["target_eligible"]
    target_global = snapshot["target_global"]
    eligible_by_asin = {candidate.parent_asin: candidate for candidate in eligible}
    top10 = []
    for rank, asin in enumerate(list(ranked)[:TOP_K], 1):
        candidate = eligible_by_asin.get(asin)
        if candidate is not None:
            top10.append(
                _candidate_payload(
                    agent,
                    candidate,
                    rank,
                    target,
                    dense_available,
                    canonical_available,
                    bm25_available,
                )
            )
        else:
            top10.append(
                {
                    "rank": rank,
                    "parent_asin": asin,
                    "title": _product_payload(agent, asin).get("title", ""),
                    "price": _product_payload(agent, asin).get("price"),
                    "dense_score": None,
                    "bm25_score": None,
                    "fusion_score": None,
                    "mmr_score": None,
                    "final_score": None,
                    "base_score": None,
                    "ranking_score": None,
                    "matched_constraints": [],
                    "matched_semantic_constraints": [],
                    "target": asin == target,
                }
            )

    score_candidate = target_eligible or target_global

    target_in_eligible = target_eligible is not None
    retriever = getattr(agent, "retriever", None)
    product_by_asin = getattr(retriever, "product_by_asin", {})
    target_in_catalog = target in product_by_asin
    state = agent.sessions.get(session_id)
    excluded = {
        str(asin)
        for asin in getattr(state, "excluded_recommendations", set())
    }
    price_min = getattr(getattr(state, "constraints", None), "price_min", None)
    price_max = getattr(getattr(state, "constraints", None), "price_max", None)
    target_price = getattr(product_by_asin.get(target), "price", None)
    budget_active = price_min is not None or price_max is not None
    target_fails_budget = (
        budget_active
        and (
            target_price is None
            or (price_min is not None and target_price < price_min)
            or (price_max is not None and target_price > price_max)
        )
    )
    if target_in_eligible:
        target_status = "ELIGIBLE"
    elif target in excluded:
        target_status = "EXCLUDED_FROM_RECOMMENDATIONS"
    elif target_fails_budget:
        target_status = "BUDGET_INELIGIBLE"
    elif not target_in_catalog:
        target_status = "NOT_IN_CATALOG"
    elif target_global is not None:
        target_status = "OUTSIDE_ELIGIBLE_FILTER"
    else:
        target_status = "NOT_IN_RETRIEVAL_CANDIDATES"
    mode = str(snapshot.get("mode") or "BROWSING")
    weights = MODE_SCORE_WEIGHTS.get(mode, MODE_SCORE_WEIGHTS["BROWSING"])
    return {
        # Which signals actually rank this turn. The UI reads these so a score
        # that carries zero weight is not shown as though it decided anything.
        "mode": mode,
        "signal_roles": {
            "dense": "active" if weights["dense"] else "inactive",
            "bm25": "active" if weights.get("bm25", 0.0) else "inactive",
            "fusion": "active" if mode == "BROWSING" else "inactive",
            "mmr": "active" if mode == "BROWSING" else "inactive",
        },
        "mode_weights": dict(weights),
        "dense_rank": (
            _debug_rank(snapshot["dense"], target) if dense_available else None
        ),
        "bm25_rank": (
            _debug_rank(snapshot["bm25"], target) if bm25_available else None
        ),
        "hybrid_rank": _debug_rank(snapshot["hybrid"], target),
        "global_bm25_rank": (
            _debug_rank(snapshot["global_bm25"], target)
            if bm25_available
            else None
        ),
        "global_hybrid_rank": _debug_rank(snapshot["global_hybrid"], target),
        "eligible": target_in_eligible,
        "target_status": target_status,
        "target_in_catalog": target_in_catalog,
        "eligible_count": len(eligible),
        "global_count": len(global_ranking),
        "global_rank": _debug_rank(global_ranking, target),
        "global_rank_status": (
            "AVAILABLE" if target_global is not None else "NOT_FOUND"
        ),
        "dense_score": (
            float(score_candidate.dense_score)
            if score_candidate is not None and dense_available
            else None
        ),
        "bm25_score": (
            float(getattr(score_candidate, "bm25_score", 0.0))
            if score_candidate is not None and bm25_available
            else None
        ),
        "fusion_score": (
            float(getattr(score_candidate, "fusion_score", 0.0))
            if score_candidate is not None
            else None
        ),
        "mmr_score": (
            float(getattr(score_candidate, "mmr_score"))
            if score_candidate is not None
            and getattr(score_candidate, "mmr_score", None) is not None
            else None
        ),
        "final_score": (
            float(score_candidate.ranking_score)
            if score_candidate is not None
            else None
        ),
        "base_score": (
            float(score_candidate.score) if score_candidate is not None else None
        ),
        "ranking_score": (
            float(score_candidate.ranking_score)
            if score_candidate is not None
            else None
        ),
        "top10": top10,
        "bm25_debug": snapshot.get("bm25_debug", {}),
        "view_scores": None,
    }


class SessionPool:
    """Deterministic, no-repeat-until-exhausted session selection."""

    def __init__(self, sessions: Iterable[Mapping[str, Any]], seed: int | None = None) -> None:
        self.sessions = [dict(session) for session in sessions]
        if not self.sessions:
            raise ValueError("session dataset is empty")
        self._rng: random.Random | random.SystemRandom = (
            random.Random(seed) if seed is not None else random.SystemRandom()
        )
        self._orders: dict[str, list[dict[str, Any]]] = {}
        self._positions: dict[str, int] = {}

    def _matching(self, scenario: str) -> list[dict[str, Any]]:
        if scenario == "ANY":
            return list(self.sessions)
        return [
            session
            for session in self.sessions
            if str(session.get("scenario_type", "")).casefold() == scenario.casefold()
        ]

    def next(self, scenario: str = "ANY") -> dict[str, Any]:
        key = scenario.upper()
        if key not in self._orders or self._positions[key] >= len(self._orders[key]):
            order = self._matching(key)
            if not order:
                raise KeyError(f"no sessions found for scenario {scenario!r}")
            self._rng.shuffle(order)
            self._orders[key] = order
            self._positions[key] = 0
        position = self._positions[key]
        self._positions[key] += 1
        return dict(self._orders[key][position])

    def next_unseen(self, scenario: str = "ANY") -> dict[str, Any] | None:
        """Return the next session without restarting an exhausted pass."""

        key = scenario.upper()
        if key not in self._orders:
            order = self._matching(key)
            if not order:
                raise KeyError(f"no sessions found for scenario {scenario!r}")
            self._rng.shuffle(order)
            self._orders[key] = order
            self._positions[key] = 0

        position = self._positions[key]
        if position >= len(self._orders[key]):
            return None
        self._positions[key] += 1
        return dict(self._orders[key][position])

    def by_id(self, session_id: str) -> dict[str, Any] | None:
        wanted = str(session_id).strip()
        if not wanted:
            return None
        aliases = {wanted}
        wanted_folded = wanted.casefold()
        for prefix in ("manual400:", "public:"):
            if wanted_folded.startswith(prefix):
                aliases.add(wanted[len(prefix):])
                break
        alias_keys = {value.casefold() for value in aliases}
        for session in self.sessions:
            sample_id = str(session.get("sample_id", "")).strip()
            if sample_id.casefold() in alias_keys:
                return dict(session)
        return None


class InteractiveSessionRunner:
    """Run one developer-driven session with replies entered in the console.

    The selected catalog product is retained only for debug-side ranking
    diagnostics.  The Agent receives the operator's messages and an opaque
    session id, never the target ASIN or its product facts.
    """

    def __init__(
        self,
        agent: Any,
        target_asin: str,
        initial_message: str,
        catalog_ids: set[str],
        user_profile: Mapping[str, Any] | None = None,
    ) -> None:
        target = str(target_asin).strip()
        if target not in catalog_ids:
            raise ValueError(f"catalog product not found: {target!r}")
        message = str(initial_message or "").strip()
        if not message:
            raise ValueError("initial message must not be empty")

        self.agent = agent
        self.catalog_ids = catalog_ids
        self.target = target
        self.session_id = "interactive:debug"
        self.session = {
            "sample_id": "interactive",
            "scenario_type": "interactive",
            "target_asin": target,
            "initial_message": message,
        }
        self.agent.reset(self.session_id, dict(user_profile or {}))
        self.next_turn_number = 1
        self.events: list[dict[str, Any]] = []
        self.done = False

    def next_turn(self, user_message: str) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("the interactive session reached the 10-turn limit")
        message = str(user_message or "").strip()
        if not message:
            raise ValueError("reply must not be empty")

        turn = self.next_turn_number
        raw_response = self.agent.respond(
            self.session_id,
            message,
            turn,
            TOP_K,
        )
        response = validate_agent_response(raw_response)
        ranked = normalize_recommendations(
            response.get("recommendations"),
            self.catalog_ids,
        )
        target_in_top10 = self.target in ranked
        session_complete = turn == MAX_TURNS
        event = {
            "session_id": self.session_id,
            "sample_id": "interactive",
            "scenario_type": "interactive",
            "target_asin": self.target,
            "turn": turn,
            "user_message": message,
            "response": response,
            "ranked": ranked,
            "override_applied": False,
            "pre_override_hit": False,
            "scoreable_hit": target_in_top10,
            "session_complete": session_complete,
        }
        self.events.append(event)
        if session_complete:
            self.done = True
        else:
            self.next_turn_number += 1
        return event


class LocalEvaluatorSessionRunner:
    """Run one public-set session with the local evaluator's simulator.

    This intentionally mirrors ``evaluator.local_evaluator.evaluate`` one
    turn at a time so the browser is a diagnostic view of the same public-set
    flow, rather than a second evaluator implementation with different reply
    or override behavior.
    """

    def __init__(
        self,
        agent: Any,
        session: Mapping[str, Any],
        catalog_ids: set[str],
        categories: Mapping[str, list[str]],
        products: Mapping[str, dict[str, Any]],
    ) -> None:
        self.agent = agent
        self.session = dict(session)
        self.catalog_ids = catalog_ids
        self.categories = categories
        self.products = products
        self.sample_id = str(self.session["sample_id"])
        ground_truth = self.session.get("ground_truth")
        if not isinstance(ground_truth, Mapping):
            raise ValueError(f"{self.sample_id} is missing ground_truth")
        self.target = str(ground_truth.get("parent_asin", "")).strip()
        if self.target not in catalog_ids:
            raise ValueError(
                f"{self.sample_id} target is not in the catalog: {self.target!r}"
            )
        self.scenario = str(self.session["scenario_type"])
        self.session_id = f"public:{self.sample_id}"
        self.agent.reset(
            self.session_id,
            dict(self.session.get("user_profile") or {}),
        )

        card, behavior = local_materialize_hidden_fields(
            self.session,
            self.products,
        )
        self.effective_sample = {
            **self.session,
            "intent_card": card,
            "behavior": behavior,
        }
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = self.scenario != "intent_override"
        self.user_message = local_initial_message(
            self.effective_sample,
            local_coarse_category(self.categories.get(self.target, [])),
            self.disclosed,
        )
        self.next_turn_number = 1
        self.first_hit_turn: int | None = None
        self.best_rank: int | None = None
        self.events: list[dict[str, Any]] = []
        self.done = False

    def next_turn(self) -> dict[str, Any] | None:
        """Execute one public-set turn using local-evaluator semantics."""

        if self.done:
            return None

        turn = self.next_turn_number
        user_message = self.user_message
        try:
            raw_response = self.agent.respond(
                self.session_id,
                user_message,
                turn,
                TOP_K,
            )
        except Exception:
            raw_response = None
        if not isinstance(raw_response, dict) or not isinstance(
            raw_response.get("message"), str
        ):
            response = {
                "message": "",
                "ask_attribute": None,
                "recommendations": [],
            }
        else:
            response = raw_response

        ranked = normalize_recommendations(
            response.get("recommendations"),
            self.catalog_ids,
        )
        target_in_top10 = self.target in ranked
        scoreable_hit = bool(self.override_applied and target_in_top10)
        session_complete = scoreable_hit or turn == MAX_TURNS
        event = {
            "session_id": self.session_id,
            "sample_id": self.sample_id,
            "scenario_type": self.scenario,
            "target_asin": self.target,
            "turn": turn,
            "user_message": user_message,
            "response": response,
            "ranked": ranked,
            "override_applied": self.override_applied,
            "pre_override_hit": bool(target_in_top10 and not self.override_applied),
            "scoreable_hit": scoreable_hit,
            "session_complete": session_complete,
        }
        self.events.append(event)

        if scoreable_hit:
            self.best_rank = ranked.index(self.target) + 1
            self.first_hit_turn = turn
            self.done = True
        elif turn == MAX_TURNS:
            self.done = True
        else:
            override = self.effective_sample.get("behavior", {}).get("override") or {}
            if (
                not self.override_applied
                and turn + 1 == int(override.get("turn", 3))
            ):
                self.override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    self.disclosed.add(new_value)
                self.user_message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                self.user_message, self.boundary_used = local_customer_reply(
                    self.effective_sample,
                    response.get("ask_attribute"),
                    self.disclosed,
                    self.boundary_used,
                )
            self.next_turn_number += 1

        return event

    def result(self) -> dict[str, Any]:
        """Return the local evaluator's per-session score record."""

        return {
            "sample_id": self.sample_id,
            "scenario_type": self.scenario,
            "target_asin": self.target,
            "hit": self.first_hit_turn is not None,
            "first_hit_turn": self.first_hit_turn,
            "best_rank": self.best_rank,
            "reciprocal_rank": (
                0.0
                if self.best_rank is None
                else 1.0 / self.best_rank
            ),
        }


class DebugWebController:
    """Own one active evaluator runner and serialize its debug state."""

    def __init__(
        self,
        agent: Any,
        sessions: Iterable[Mapping[str, Any]],
        catalog_ids: set[str],
        *,
        evaluator_kind: str = "hard",
        categories: Mapping[str, list[str]] | None = None,
        products: Mapping[str, dict[str, Any]] | None = None,
        seed: int | None = None,
        interactive_mode: bool = False,
    ) -> None:
        self.agent = agent
        if evaluator_kind not in {"hard", "local"}:
            raise ValueError(f"unknown evaluator kind: {evaluator_kind!r}")
        self.evaluator_kind = evaluator_kind
        self.categories = dict(categories or {})
        self.products = dict(products or {})
        self.pool = (
            None
            if interactive_mode
            else SessionPool(sessions, seed=seed)
        )
        self.catalog_ids = catalog_ids
        self.interactive_mode = bool(interactive_mode)
        self.runner: Manual400SessionRunner | LocalEvaluatorSessionRunner | None = None
        self.interactive_runner: InteractiveSessionRunner | None = None
        self.turn_records: list[dict[str, Any]] = []
        self.before_state: dict[str, Any] = {}
        self.before_constraints: dict[str, Any] = {}
        self.before_semantic_constraints: dict[str, Any] = {}
        self.before_exclusions: list[str] = []
        self.miss_search: dict[str, Any] | None = None

    @property
    def session(self) -> Mapping[str, Any] | None:
        if self.runner is not None:
            return self.runner.session
        return None if self.interactive_runner is None else self.interactive_runner.session

    def _new_runner(self, session: Mapping[str, Any]) -> dict[str, Any]:
        self.interactive_runner = None
        if self.evaluator_kind == "local":
            self.runner = LocalEvaluatorSessionRunner(
                self.agent,
                session,
                self.catalog_ids,
                self.categories,
                self.products,
            )
        else:
            self.runner = Manual400SessionRunner(
                self.agent,
                session,
                self.catalog_ids,
            )
        self.turn_records = []
        self.before_state = {}
        self.before_constraints = {}
        self.before_semantic_constraints = {}
        self.before_exclusions = []
        self.miss_search = None
        return self.state_payload()

    def new_random(self, scenario: str = "ANY") -> dict[str, Any]:
        if self.pool is None:
            raise RuntimeError("random benchmark sessions are unavailable in interactive mode")
        return self._new_runner(self.pool.next(scenario))

    def load(self, session_id: str) -> dict[str, Any]:
        if self.pool is None:
            raise RuntimeError("benchmark sessions are unavailable in interactive mode")
        session = self.pool.by_id(session_id)
        if session is None:
            raise KeyError(f"session {session_id!r} was not found")
        return self._new_runner(session)

    def find_next_miss(self, scenario: str = "ANY") -> dict[str, Any]:
        """Run unseen sessions until the first non-hit session is found."""

        if self.pool is None:
            raise RuntimeError("miss search is unavailable in interactive mode")

        requested_scenario = str(scenario or "ANY").upper()
        searched = 0
        while True:
            session = self.pool.next_unseen(requested_scenario)
            if session is None:
                self.miss_search = {
                    "status": "exhausted",
                    "scenario": requested_scenario,
                    "searched": searched,
                    "message": "No non-hit session remains in this pass.",
                }
                return self.state_payload()

            self._new_runner(session)
            self.run_to_end()
            searched += 1
            result = self.runner.result() if self.runner is not None else {}
            if not bool(result.get("hit")):
                self.miss_search = {
                    "status": "found",
                    "scenario": requested_scenario,
                    "searched": searched,
                    "sample_id": str(session.get("sample_id", "")),
                    "result": result,
                    "message": "Stopped at the first non-hit session.",
                }
                return self.state_payload()

    def search_catalog(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Find catalog products for developer target selection only."""

        terms = tuple(
            part.casefold()
            for part in str(query).split()
            if part.strip()
        )
        if not terms:
            return []

        results: list[tuple[int, int, dict[str, Any]]] = []
        retriever = self.agent.retriever
        for asin in getattr(retriever, "_catalog_order", ()):
            product = retriever.product_by_asin.get(asin)
            if product is None:
                continue
            raw = getattr(product, "raw", {})
            title = _first_text(raw.get("title", raw.get("name", "")))
            haystack = title.casefold()
            matched = sum(term in haystack for term in terms)
            if matched == 0:
                continue
            exact_phrase = 1 if " ".join(terms) in haystack else 0
            results.append(
                (
                    exact_phrase,
                    matched,
                    {
                        "parent_asin": str(asin),
                        "title": title,
                        "price": product.price,
                    },
                )
            )
        results.sort(key=lambda item: (-item[0], -item[1], item[2]["parent_asin"]))
        return [item[2] for item in results[: max(1, int(limit))]]

    def start_interactive(
        self,
        target_asin: str,
        initial_message: str,
        user_profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.interactive_mode:
            raise RuntimeError("start the debug server with --interactive first")
        lookup = {asin.casefold(): asin for asin in self.catalog_ids}
        target = lookup.get(str(target_asin).strip().casefold())
        if target is None:
            raise ValueError(f"catalog product not found: {target_asin!r}")
        self.runner = None
        self.interactive_runner = InteractiveSessionRunner(
            self.agent,
            target,
            initial_message,
            self.catalog_ids,
            user_profile,
        )
        self.turn_records = []
        self.before_state = {}
        self.before_constraints = {}
        self.before_semantic_constraints = {}
        self.before_exclusions = []
        self.interactive_turn(initial_message)
        return self.state_payload()

    def interactive_turn(self, user_message: str) -> dict[str, Any]:
        runner = self.interactive_runner
        if runner is None:
            raise RuntimeError("start an interactive session before entering a reply")
        session_id = runner.session_id
        state = self.agent.sessions.get(session_id)
        self.before_state = _state_payload(self.agent, session_id)
        self.before_constraints = _constraint_payload(state.constraints)
        self.before_semantic_constraints = dict(
            self.before_state.get("semantic_constraints", {})
        )
        self.before_exclusions = sorted(
            str(value) for value in state.excluded_recommendations
        )
        event = runner.next_turn(user_message)
        record = self._record_event(
            event,
            runner.target,
            self.before_state,
            self.before_constraints,
            self.before_semantic_constraints,
            self.before_exclusions,
        )
        self.turn_records.append(record)
        return self.state_payload()

    def state_payload(self) -> dict[str, Any]:
        if self.interactive_runner is not None:
            runner = self.interactive_runner
            state = _state_payload(self.agent, runner.session_id)
            return {
                "interactive_mode": True,
                "evaluator": "interactive",
                "session": {
                    "session_id": "interactive",
                    "scenario": "interactive",
                    "turn": len(runner.events),
                    "total_turns": MAX_TURNS,
                    "initial_message": str(runner.session["initial_message"]),
                    "target": _product_payload(self.agent, runner.target),
                },
                "turn": len(runner.events),
                "total_turns": MAX_TURNS,
                "done": bool(runner.done),
                "state": state,
                "layer2": _layer2_status(self.agent),
                "product_dense": _product_dense_status(self.agent),
                "bm25": _bm25_status(self.agent),
                "benchmark": None,
                "miss_search": self.miss_search,
                "score_weights": MODE_SCORE_WEIGHTS,
                "turns": self.turn_records,
            }
        if self.runner is None:
            return {
                "interactive_mode": self.interactive_mode,
                "evaluator": self.evaluator_kind,
                "session": None,
                "turn": 0,
                "total_turns": MAX_TURNS,
                "done": True,
                "layer2": _layer2_status(self.agent),
                "product_dense": _product_dense_status(self.agent),
                "bm25": _bm25_status(self.agent),
                "score_weights": MODE_SCORE_WEIGHTS,
                "benchmark": None,
                "miss_search": self.miss_search,
            }
        session = self.runner.session
        state = _state_payload(self.agent, self.runner.session_id)
        target = str(self.runner.target)
        return {
            "interactive_mode": False,
            "evaluator": self.evaluator_kind,
            "session": {
                "session_id": str(session["sample_id"]),
                "scenario": str(session["scenario_type"]),
                "turn": int(getattr(self.agent.sessions.get(self.runner.session_id), "turn", 0)),
                "total_turns": MAX_TURNS,
                "initial_message": str(session.get("initial_message", "")),
                "target": _product_payload(self.agent, target),
            },
            "turn": int(getattr(self.agent.sessions.get(self.runner.session_id), "turn", 0)),
            "total_turns": MAX_TURNS,
            "done": bool(self.runner.done),
            "state": state,
            "layer2": _layer2_status(self.agent),
            "product_dense": _product_dense_status(self.agent),
            "bm25": _bm25_status(self.agent),
            "benchmark": self._benchmark_payload(),
            "miss_search": self.miss_search,
            "score_weights": MODE_SCORE_WEIGHTS,
            "turns": self.turn_records,
        }

    def _clarification_candidates(self, session_id: str) -> Any:
        """Rebuild the pool the clarification policy scored this turn.

        Mirrors ``Agent.respond``'s ``candidate_pool_only`` retrieval, which is
        deliberately broader than the recommendation pool. Diagnostics only:
        it never feeds the Agent.
        """

        try:
            state = self.agent.sessions.get(session_id)
            return self.agent.retriever.retrieve(
                getattr(state, "mode", None) or "BROWSING",
                getattr(state, "retrieval_query_text", ""),
                getattr(state, "constraints", None),
                semantic_constraints=getattr(state, "semantic_constraints", None),
                limit=CLARIFICATION_CANDIDATE_LIMIT,
                minimum_candidates=50,
                excluded_asins=getattr(state, "excluded_recommendations", None),
                candidate_pool_only=True,
            )
        except Exception:
            return None

    def _benchmark_payload(self) -> dict[str, Any] | None:
        if self.runner is None:
            return None
        result = self.runner.result()
        return {
            "complete": bool(self.runner.done),
            "result": result,
            "metrics": add_score_fields(metric_summary([result])),
        }

    def _record_event(
        self,
        event: Mapping[str, Any],
        target: str,
        before_state: Mapping[str, Any],
        before_constraints: Mapping[str, Any],
        before_semantic_constraints: Mapping[str, Any],
        before_exclusions: list[str],
    ) -> dict[str, Any]:
        """Build the same debug turn payload for benchmark and console runs."""

        session_id = str(event["session_id"])
        after_state = _state_payload(self.agent, session_id)
        extracted_this_turn = {
            "structured": _changed_constraint_payload(
                before_constraints,
                after_state.get("constraints", {}),
            ),
            "semantic": _changed_constraint_payload(
                before_semantic_constraints,
                after_state.get("semantic_constraints", {}),
            ),
        }
        ranking = _ranking_payload(
            self.agent,
            session_id,
            target,
            event["ranked"],
        )
        override_kind = after_state.get("override_kind")
        return {
            "turn": event["turn"],
            "user_message": event["user_message"],
            "agent": {
                "message": event["response"]["message"],
                "ask_attribute": event["response"]["ask_attribute"],
            },
            "state": {
                "mode": after_state.get("mode"),
                "constraints": after_state.get("constraints", {}),
                "semantic_constraints": after_state.get(
                    "semantic_constraints", {}
                ),
                "extracted_this_turn": extracted_this_turn,
                "query_text": after_state.get("query_text", ""),
                "retrieval_query_text": after_state.get(
                    "retrieval_query_text",
                    after_state.get("query_text", ""),
                ),
                "asked_attributes": after_state.get("asked_attributes", []),
                "last_asked": after_state.get("last_asked"),
                "exclusions": after_state.get("excluded", []),
            },
            "intent": _intent_payload(
                self.agent, session_id, event["user_message"]
            ),
            "profile": _profile_payload(self.agent, session_id),
            "clarification": _clarification_payload(
                self.agent,
                session_id,
                before_state,
                after_state,
                self._clarification_candidates(session_id),
            ),
            "target": _product_payload(self.agent, target),
            "target_facts": _debug_target_facts(self.agent, target),
            "ranking": ranking,
            "override": {
                "detected": override_kind in {"FULL_GOAL", "PREFERENCE"},
                "kind": override_kind,
                "old_mode": before_state.get("mode"),
                "new_mode": after_state.get("mode"),
                "constraints_before": before_constraints,
                "constraints_after": after_state.get("constraints", {}),
                "semantic_constraints_before": before_semantic_constraints,
                "semantic_constraints_after": after_state.get(
                    "semantic_constraints", {}
                ),
                "exclusions_before": before_exclusions,
                "exclusions_after": after_state.get("excluded", []),
            },
            "hit": bool(event["scoreable_hit"]),
            "pre_override_hit": bool(event["pre_override_hit"]),
            "scoreable": bool(event["override_applied"]),
            "done": bool(event["session_complete"]),
            "benchmark": self._benchmark_payload(),
        }

    def _run_one(self) -> dict[str, Any]:
        if self.runner is None:
            raise RuntimeError("load a session before running a turn")
        if self.runner.done:
            raise RuntimeError("the active session is complete; load another session")

        session_id = self.runner.session_id
        state = self.agent.sessions.get(session_id)
        self.before_state = _state_payload(self.agent, session_id)
        self.before_constraints = _constraint_payload(state.constraints)
        self.before_semantic_constraints = dict(
            self.before_state.get("semantic_constraints", {})
        )
        self.before_exclusions = sorted(
            str(value) for value in state.excluded_recommendations
        )

        event = self.runner.next_turn()
        if event is None:
            raise RuntimeError("the active session is complete")
        record = self._record_event(
            event,
            str(self.runner.target),
            self.before_state,
            self.before_constraints,
            self.before_semantic_constraints,
            self.before_exclusions,
        )
        self.turn_records.append(record)
        return record

    def next_turn(self) -> dict[str, Any]:
        self._run_one()
        return self.state_payload()

    def run_to_end(self) -> dict[str, Any]:
        start = len(self.turn_records)
        while self.runner is not None and not self.runner.done:
            self._run_one()
        return {
            **self.state_payload(),
            "new_turns": self.turn_records[start:],
        }


class DebugRequestHandler(BaseHTTPRequestHandler):
    """Minimal JSON API and static-file handler for the local dashboard."""

    app: DebugWebController

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal focused on startup/errors; this is a local tool.
        return None

    def _send_json(self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(self.app.state_payload())
            return
        if parsed.path.startswith("/debug_web_ui/"):
            name = parsed.path.removeprefix("/debug_web_ui/")
            if name not in {"app.js", "style.css"}:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            content_types = {
                "app.js": "text/javascript; charset=utf-8",
                "style.css": "text/css; charset=utf-8",
            }
            try:
                body = (STATIC_DIR / name).read_bytes()
            except OSError as exc:
                self._send_json({"error": f"UI asset unavailable: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_types[name])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path not in {"/", "/index.html"}:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            body = (STATIC_DIR / "index.html").read_bytes()
        except OSError as exc:
            self._send_json({"error": f"UI asset unavailable: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/session/random":
                result = self.app.new_random(str(body.get("scenario", "ANY")))
                self._send_json(result)
            elif parsed.path == "/api/session/load":
                result = self.app.load(str(body.get("session_id", "")))
                self._send_json(result)
            elif parsed.path == "/api/session/find-next-miss":
                result = self.app.find_next_miss(str(body.get("scenario", "ANY")))
                self._send_json(result)
            elif parsed.path == "/api/session/next":
                self._send_json(self.app.next_turn())
            elif parsed.path == "/api/session/run-to-end":
                self._send_json(self.app.run_to_end())
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive local-server guard
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def _print_interactive_turn(record: Mapping[str, Any]) -> None:
    """Print a compact console view while the browser shows full diagnostics."""

    agent = record.get("agent", {})
    ranking = record.get("ranking", {})
    print(f"\nTurn {record.get('turn')}")
    print(f"Agent: {agent.get('message', '')}")
    print(f"Asked: {agent.get('ask_attribute') or 'none'}")
    print(
        "Target ranks: "
        f"dense={ranking.get('dense_rank', 'N/A')} "
        f"bm25={ranking.get('bm25_rank', 'N/A')} "
        f"final={ranking.get('hybrid_rank', 'N/A')}"
    )
    print("Top 10:")
    for item in ranking.get("top10", ()):
        print(
            f"  {item.get('rank', '?'):>2} "
            f"{item.get('parent_asin', '')} "
            f"final={score if (score := item.get('final_score')) is not None else 'N/A'}"
        )
    if record.get("hit"):
        print("Target is currently in Top 10.")


def _run_interactive_session(
    app: DebugWebController,
    target_asin: str,
) -> tuple[str, str | None]:
    """Collect one initial message and manual replies for one target."""

    while True:
        try:
            initial_message = input("initial message> ").strip()
        except EOFError:
            return "quit", None
        if initial_message.casefold() == "q":
            return "quit", None
        if not initial_message:
            print("Please enter a message, or type q to quit.")
            continue

        try:
            app.start_interactive(target_asin, initial_message)
        except (RuntimeError, ValueError) as exc:
            print(f"Unable to start interactive session: {exc}")
            continue
        _print_interactive_turn(app.turn_records[-1])

        restart = False
        while app.interactive_runner is not None and not app.interactive_runner.done:
            asked = app.agent.sessions.get(
                app.interactive_runner.session_id
            ).last_asked
            try:
                reply = input(
                    f"reply (asked={asked or 'none'}, q/restart/target <ASIN>)> "
                ).strip()
            except EOFError:
                return "quit", None
            lowered = reply.casefold()
            if lowered == "q":
                return "quit", None
            if lowered == "restart":
                restart = True
                break
            if lowered.startswith("target "):
                return "target", reply.split(None, 1)[1].strip()
            if not reply:
                print("Please enter a reply, or use q/restart.")
                continue
            try:
                app.interactive_turn(reply)
            except (RuntimeError, ValueError) as exc:
                print(f"Unable to process reply: {exc}")
                continue
            _print_interactive_turn(app.turn_records[-1])

        if restart:
            print("Restarting the same target.")
            continue
        print("Interactive session reached the 10-turn limit.")
        return "choose", None


def run_interactive_console(app: DebugWebController) -> None:
    """Run the target picker and stdin-driven Agent conversation."""

    print("Interactive debug mode")
    print("Choose a catalog ASIN, or use search <words>. Type q to quit.")
    pending_target: str | None = None
    choices: dict[str, str] = {}
    while True:
        if pending_target is not None:
            command = pending_target
            pending_target = None
        else:
            try:
                command = input("target> ").strip()
            except EOFError:
                return
        if command.casefold() == "q":
            return
        if command.casefold().startswith("search "):
            matches = app.search_catalog(command.split(None, 1)[1])
            choices = {str(index): item["parent_asin"] for index, item in enumerate(matches, 1)}
            if not matches:
                print("No title matches found.")
            else:
                for index, item in enumerate(matches, 1):
                    price = item.get("price")
                    price_text = "price N/A" if price is None else f"${float(price):.2f}"
                    print(
                        f"{index}. {item['parent_asin']} · {price_text} · "
                        f"{item.get('title', '')}"
                    )
            continue
        if command.casefold().startswith("target "):
            command = command.split(None, 1)[1].strip()
        target = choices.get(command, command)
        resolved = {asin.casefold(): asin for asin in app.catalog_ids}.get(
            target.casefold()
        )
        if resolved is None:
            print(f"Catalog product not found: {target}")
            continue
        result, next_target = _run_interactive_session(app, resolved)
        if result == "quit":
            return
        if result == "target":
            pending_target = next_target


def validate_local_sessions(
    sessions: Iterable[Mapping[str, Any]],
    catalog_ids: set[str],
) -> list[dict[str, Any]]:
    """Validate the public-set shape without changing its records."""

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, session in enumerate(sessions, 1):
        if not isinstance(session, Mapping):
            raise ValueError(f"public session {index} must be an object")
        sample_id = str(session.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"public session {index} is missing sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate public session sample_id: {sample_id}")
        seen_ids.add(sample_id)
        scenario = str(session.get("scenario_type", "")).strip()
        if not scenario:
            raise ValueError(f"public session {sample_id} is missing scenario_type")
        ground_truth = session.get("ground_truth")
        target = (
            str(ground_truth.get("parent_asin", "")).strip()
            if isinstance(ground_truth, Mapping)
            else ""
        )
        if not target or target not in catalog_ids:
            raise ValueError(
                f"public session {sample_id} has an invalid catalog target: {target!r}"
            )
        validated.append(dict(session))
    if not validated:
        raise ValueError("public session dataset is empty")
    return validated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local evaluator and interactive debug UI"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument(
        "--evaluator",
        choices=("hard", "local"),
        default="hard",
        help="benchmark simulator to show (default: hard Manual400)",
    )
    parser.add_argument(
        "--sessions",
        "--dataset",
        dest="sessions",
        default=None,
        help="benchmark session JSONL file",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="select a catalog target and enter shopper replies in the console",
    )
    parser.add_argument(
        "--browsing-retrieval",
        choices=("hybrid", "qwen_dense"),
        default=None,
        help=(
            "Browsing retrieval arm: hybrid (default) or qwen_dense for the "
            "raw Qwen product-embedding experiment"
        ),
    )
    return parser


def create_application(args: argparse.Namespace) -> DebugWebController:
    categories: dict[str, list[str]] = {}
    products: dict[str, dict[str, Any]] = {}
    if args.evaluator == "local" and not args.interactive:
        catalog_ids, categories, products = local_catalog_index(args.catalog)
    else:
        catalog_ids = load_catalog_ids(args.catalog)
    sessions: list[dict[str, Any]] = []
    if not args.interactive:
        session_path = args.sessions or (
            DEFAULT_LOCAL_DATASET
            if args.evaluator == "local"
            else DEFAULT_SESSIONS
        )
        sessions = load_jsonl(session_path)
        if args.evaluator == "local":
            sessions = validate_local_sessions(sessions, catalog_ids)
        else:
            sessions = validate_sessions(sessions, catalog_ids)
    agent = build_evaluator_agent(
        args.catalog,
        browsing_retrieval_mode=args.browsing_retrieval,
    )
    return DebugWebController(
        agent,
        sessions,
        catalog_ids,
        evaluator_kind=args.evaluator,
        categories=categories,
        products=products,
        seed=args.seed,
        interactive_mode=args.interactive,
    )


def main() -> None:
    args = _build_parser().parse_args()
    try:
        app = create_application(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Unable to start debug UI: {exc}\n"
            "Install the embedding dependencies and place the local BGE model "
            "under models/bge-small-en-v1.5, or set "
            "SHOPPING_ATTRIBUTE_EMBEDDING_MODEL."
        ) from exc

    handler = type("ShoppingCopilotDebugHandler", (DebugRequestHandler,), {})
    handler.app = app
    server_type = ThreadingHTTPServer if args.interactive else HTTPServer
    server = server_type((args.host, args.port), handler)
    print(f"Debug UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    if args.interactive:
        server_thread = threading.Thread(
            target=server.serve_forever,
            name="shopping-copilot-debug-web",
            daemon=True,
        )
        server_thread.start()
        try:
            run_interactive_console(app)
        except KeyboardInterrupt:
            print("\nInteractive debug stopped.")
        finally:
            server.shutdown()
            server_thread.join(timeout=2.0)
            server.server_close()
        return
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDebug UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
