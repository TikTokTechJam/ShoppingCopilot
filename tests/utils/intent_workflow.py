"""Adapter exposing our workflow's intent decision as one three-way call.

The specification asks for a classifier over ``BUYING``, ``BROWSING`` and
``INTENT OVERRIDE``. Our workflow has no single module that returns all three:

* ``starter.routing.intent_router.TwoPhaseIntentRouter`` decides BUYING vs
  BROWSING and knows nothing about overrides;
* ``starter.session.detect_override_kind`` decides whether an utterance
  replaces earlier state, and knows nothing about buying vs browsing.

``Agent.respond`` composes them in a fixed order (agent.py:168-210): override
detection runs first, and only against a session that already has state --
``if state.mode is not None`` -- so the opening utterance of a session can
never be an override. ``predict_intent`` reproduces exactly that order, so the
suite measures the shipped decision rather than a reimplementation of it.
"""

from __future__ import annotations

from typing import Sequence

from starter.routing.constraints import extract_constraints
from starter.routing.intent_router import TwoPhaseIntentRouter
from starter.session import OverrideKind, detect_override_kind, merge_constraints
from starter.routing.constraints import ShoppingConstraints


BUYING = "BUYING"
BROWSING = "BROWSING"
INTENT_OVERRIDE = "INTENT OVERRIDE"
INTENTS = (BUYING, BROWSING, INTENT_OVERRIDE)

_router = TwoPhaseIntentRouter()


def normalize_intent(value: object) -> str:
    """Accept INTENT_OVERRIDE, intent-override, etc. as one canonical label."""

    text = " ".join(str(value).replace("_", " ").replace("-", " ").split()).upper()
    return text if text in INTENTS else text


def _accumulate(prior_turns: Sequence[str]) -> ShoppingConstraints:
    """Replay earlier turns into session constraints, as the Agent would."""

    state = ShoppingConstraints()
    for turn in prior_turns:
        if str(turn).strip():
            state = merge_constraints(state, extract_constraints(str(turn)))
    return state


def predict_intent(
    utterance: str,
    prior_context: str | Sequence[str] | None = None,
) -> str:
    """Return the workflow's intent for ``utterance``.

    ``prior_context`` is the conversation before this turn. Without it the
    utterance is a session opener, which the Agent never tests for an
    override, so only BUYING/BROWSING are reachable.
    """

    if prior_context is None:
        prior_turns: list[str] = []
    elif isinstance(prior_context, str):
        prior_turns = [line for line in prior_context.splitlines() if line.strip()]
        if not prior_turns and prior_context.strip():
            prior_turns = [prior_context.strip()]
    else:
        prior_turns = [str(turn) for turn in prior_context if str(turn).strip()]

    if prior_turns:
        current = _accumulate(prior_turns)
        delta = extract_constraints(utterance or "")
        if detect_override_kind(utterance or "", current, delta) is not OverrideKind.NONE:
            return INTENT_OVERRIDE

    result = _router.classify(utterance or "")
    return BUYING if str(result.intent).upper() == BUYING else BROWSING


__all__ = [
    "BROWSING",
    "BUYING",
    "INTENTS",
    "INTENT_OVERRIDE",
    "normalize_intent",
    "predict_intent",
]
