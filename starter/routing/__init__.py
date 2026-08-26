"""Reusable routing components for the shopping agent."""

from starter.routing.constraints import (
    ShoppingConstraints,
    extract_constraints,
)
from starter.routing.intent_router import (
    CascadingIntentRouter,
    Intent,
    IntentResult,
    IntentRouter,
    LexicalIntentRouter,
    SessionIntentTracker,
    Signal,
    Tier,
    TwoPhaseIntentRouter,
    classify,
)
from starter.routing.lexicon import BROWSING, BUYING

__all__ = [
    "BROWSING",
    "BUYING",
    "CascadingIntentRouter",
    "Intent",
    "IntentResult",
    "IntentRouter",
    "LexicalIntentRouter",
    "SessionIntentTracker",
    "ShoppingConstraints",
    "Signal",
    "Tier",
    "TwoPhaseIntentRouter",
    "classify",
    "extract_constraints",
]


def build_default_router(*, use_model: bool = True) -> IntentRouter:
    """The two-phase pipeline, with the reranker when it is installed.

    Importing the model backend is deferred so that a repo without
    `onnxruntime` never pays for it, and never fails because of it.
    """
    if not use_model:
        return TwoPhaseIntentRouter()
    try:
        from starter.routing.local_model import build_backend

        return TwoPhaseIntentRouter(backend=build_backend())
    except Exception:
        return TwoPhaseIntentRouter()
