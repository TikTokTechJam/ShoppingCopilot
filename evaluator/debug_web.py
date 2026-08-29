"""Small localhost-only web viewer for one Manual400 session at a time.

The session execution lives in :class:`evaluator.hard_evaluator.Manual400SessionRunner`;
this module only selects sessions, calls one runner turn, and serializes the
existing evaluator-side diagnostics for a browser.
"""

from __future__ import annotations

import argparse
import json
import random
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from evaluator.agent_factory import build_evaluator_agent
from evaluator.hard_evaluator import (
    DEBUG_FACT_FIELDS,
    MAX_TURNS,
    Manual400SessionRunner,
    _debug_rank,
    _debug_ranking_snapshot,
    _debug_state_snapshot,
    _debug_target_facts,
    _debug_values,
    load_catalog_ids,
    load_jsonl,
    TOP_K,
)
from starter.retrieval import MODE_SCORE_WEIGHTS


DEFAULT_CATALOG = "data/catalog.jsonl"
DEFAULT_SESSIONS = "data/derived/gptannotation/sessions.jsonl"
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
    retriever = agent.retriever
    available = bool(getattr(retriever, "dense_available", False))
    index = getattr(retriever, "layer2_index", None)
    encoder = getattr(retriever, "query_encoder", None)
    manifest = getattr(index, "manifest", {}) if index is not None else {}
    if available:
        return {
            "available": True,
            "artifact_model": manifest.get("embedding_model", manifest.get("model")),
            "query_model": getattr(encoder, "model_id", None),
            "dimension": getattr(index, "dimension", None),
        }
    reason = getattr(retriever, "layer2_compatibility_error", None)
    if not reason:
        if index is None:
            reason = "No compatible Layer 2 artifact was loaded."
        else:
            reason = "Layer 2 query encoder is not configured."
    return {
        "available": False,
        "reason": reason,
        "setup_hint": (
            "Install the local Jina model under model/jina-embeddings-v5-text-nano "
            "or set SHOPPING_EMBEDDING_MODEL to its local path."
        ),
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
        }
    )
    return snapshot


def _candidate_payload(
    agent: Any,
    candidate: Any,
    rank: int,
    target: str,
    dense_available: bool,
) -> dict[str, Any]:
    asin = str(candidate.parent_asin)
    product = _product_payload(agent, asin)
    return {
        "rank": rank,
        "parent_asin": asin,
        "title": product.get("title", ""),
        "price": product.get("price"),
        "structured_score": float(candidate.constraint_score),
        "dense_score": (
            float(candidate.dense_score) if dense_available else None
        ),
        "final_score": float(candidate.score),
        "matched_constraints": list(candidate.matched_constraints),
        "target": asin == target,
    }


def _ranking_payload(
    agent: Any,
    session_id: str,
    target: str,
    ranked: Iterable[str],
) -> dict[str, Any]:
    snapshot = _debug_ranking_snapshot(agent, session_id, target)
    dense_available = bool(getattr(agent.retriever, "dense_available", False))
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
                    agent, candidate, rank, target, dense_available
                )
            )
        else:
            top10.append(
                {
                    "rank": rank,
                    "parent_asin": asin,
                    "title": _product_payload(agent, asin).get("title", ""),
                    "price": _product_payload(agent, asin).get("price"),
                    "structured_score": None,
                    "dense_score": None,
                    "final_score": None,
                    "matched_constraints": [],
                    "target": asin == target,
                }
            )

    score_candidate = target_eligible or target_global

    target_in_eligible = target_eligible is not None
    return {
        "structured_rank": _debug_rank(snapshot["structured"], target),
        "dense_rank": (
            _debug_rank(snapshot["dense"], target) if dense_available else None
        ),
        "hybrid_rank": _debug_rank(snapshot["hybrid"], target),
        "eligible": target_in_eligible,
        "eligible_count": len(eligible),
        "global_count": len(global_ranking),
        "global_rank": _debug_rank(global_ranking, target),
        "global_rank_status": "AVAILABLE" if target_in_eligible else "INELIGIBLE",
        "structured_score": (
            float(score_candidate.constraint_score)
            if score_candidate is not None
            else None
        ),
        "dense_score": (
            float(score_candidate.dense_score)
            if score_candidate is not None and dense_available
            else None
        ),
        "final_score": (
            float(score_candidate.score) if score_candidate is not None else None
        ),
        "top10": top10,
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

    def by_id(self, session_id: str) -> dict[str, Any] | None:
        wanted = str(session_id).strip()
        aliases = {wanted}
        if wanted.startswith("manual400:"):
            aliases.add(wanted.split(":", 1)[1])
        if wanted.startswith("manual400_"):
            aliases.add(wanted)
        for session in self.sessions:
            if str(session.get("sample_id", "")).strip() in aliases:
                return dict(session)
        return None


class DebugWebController:
    """Own one active evaluator runner and serialize its debug state."""

    def __init__(
        self,
        agent: Any,
        sessions: Iterable[Mapping[str, Any]],
        catalog_ids: set[str],
        *,
        seed: int | None = None,
    ) -> None:
        self.agent = agent
        self.pool = SessionPool(sessions, seed=seed)
        self.catalog_ids = catalog_ids
        self.runner: Manual400SessionRunner | None = None
        self.turn_records: list[dict[str, Any]] = []
        self.before_state: dict[str, Any] = {}
        self.before_constraints: dict[str, Any] = {}
        self.before_exclusions: list[str] = []

    @property
    def session(self) -> Mapping[str, Any] | None:
        return None if self.runner is None else self.runner.session

    def _new_runner(self, session: Mapping[str, Any]) -> dict[str, Any]:
        self.runner = Manual400SessionRunner(
            self.agent,
            session,
            self.catalog_ids,
        )
        self.turn_records = []
        self.before_state = {}
        self.before_constraints = {}
        self.before_exclusions = []
        return self.state_payload()

    def new_random(self, scenario: str = "ANY") -> dict[str, Any]:
        return self._new_runner(self.pool.next(scenario))

    def load(self, session_id: str) -> dict[str, Any]:
        session = self.pool.by_id(session_id)
        if session is None:
            raise KeyError(f"session {session_id!r} was not found")
        return self._new_runner(session)

    def state_payload(self) -> dict[str, Any]:
        if self.runner is None:
            return {
                "session": None,
                "turn": 0,
                "total_turns": MAX_TURNS,
                "done": True,
                "layer2": _layer2_status(self.agent),
                "score_weights": MODE_SCORE_WEIGHTS,
            }
        session = self.runner.session
        state = _state_payload(self.agent, self.runner.session_id)
        target = str(session["target_asin"])
        return {
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
            "score_weights": MODE_SCORE_WEIGHTS,
            "turns": self.turn_records,
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
        self.before_exclusions = sorted(
            str(value) for value in state.excluded_recommendations
        )
        user_message = self.runner.user_message

        try:
            delta = self.agent._extract(user_message)
        except Exception:
            delta = None
        event = self.runner.next_turn()
        if event is None:
            raise RuntimeError("the active session is complete")
        after_state = _state_payload(self.agent, session_id)
        ranking = _ranking_payload(
            self.agent,
            session_id,
            str(self.runner.target),
            event["ranked"],
        )
        override_kind = after_state.get("override_kind")
        record = {
            "turn": event["turn"],
            "user_message": event["user_message"],
            "agent": {
                "message": event["response"]["message"],
                "ask_attribute": event["response"]["ask_attribute"],
            },
            "state": {
                "mode": after_state.get("mode"),
                "constraints": after_state.get("constraints", {}),
                "extracted_this_turn": _constraint_payload(delta),
                "query_text": after_state.get("query_text", ""),
                "asked_attributes": after_state.get("asked_attributes", []),
                "last_asked": after_state.get("last_asked"),
                "exclusions": after_state.get("excluded", []),
            },
            "clarification": {
                "previous_asked": self.before_state.get("last_asked"),
                "next_asked": after_state.get("last_asked"),
            },
            "target": _product_payload(self.agent, str(self.runner.target)),
            "target_facts": _debug_target_facts(
                self.agent, str(self.runner.target)
            ),
            "ranking": ranking,
            "override": {
                "detected": override_kind in {"FULL_GOAL", "PREFERENCE"},
                "kind": override_kind,
                "old_mode": self.before_state.get("mode"),
                "new_mode": after_state.get("mode"),
                "constraints_before": self.before_constraints,
                "constraints_after": after_state.get("constraints", {}),
                "exclusions_before": self.before_exclusions,
                "exclusions_after": after_state.get("excluded", []),
            },
            "hit": bool(event["scoreable_hit"]),
            "pre_override_hit": bool(event["pre_override_hit"]),
            "scoreable": bool(event["override_applied"]),
            "done": bool(event["session_complete"]),
        }
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Manual400 session debug UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--sessions", default=DEFAULT_SESSIONS)
    parser.add_argument("--layer2-artifact-dir")
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    parser.add_argument("--half-precision", action="store_true")
    return parser


def create_application(args: argparse.Namespace) -> DebugWebController:
    sessions = load_jsonl(args.sessions)
    catalog_ids = load_catalog_ids(args.catalog)
    agent = build_evaluator_agent(
        args.catalog,
        layer2_artifact_dir=args.layer2_artifact_dir,
        embedding_model=args.embedding_model,
        half_precision=args.half_precision,
        device=args.device,
    )
    return DebugWebController(agent, sessions, catalog_ids, seed=args.seed)


def main() -> None:
    args = _build_parser().parse_args()
    try:
        app = create_application(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Unable to start debug UI: {exc}\n"
            "Install the embedding dependencies and place the local Jina model "
            "under model/jina-embeddings-v5-text-nano, or pass "
            "--embedding-model PATH."
        ) from exc

    handler = type("ShoppingCopilotDebugHandler", (DebugRequestHandler,), {})
    handler.app = app
    server = HTTPServer((args.host, args.port), handler)
    print(f"Debug UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDebug UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
