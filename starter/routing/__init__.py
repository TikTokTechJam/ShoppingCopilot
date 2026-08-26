"""Reusable routing components for the shopping agent."""

from starter.routing.intent_router import (
    CascadingIntentRouter,
    Intent,
    IntentResult,
    IntentRouter,
    LexicalIntentRouter,
    SessionIntentTracker,
    Signal,
    Tier,
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
    "Signal",
    "Tier",
    "classify",
]


def build_default_router(*, use_model: bool = True) -> IntentRouter:
    """Rules plus the reranker when it is installed, rules alone otherwise.

    Importing the model backend is deferred so that a repo without
    `onnxruntime` never pays for it, and never fails because of it.
    """
    if not use_model:
        return LexicalIntentRouter()
    try:
        from starter.routing.local_model import build_backend

        return CascadingIntentRouter(backend=build_backend())
    except Exception:
        return LexicalIntentRouter()
