"""Regression gate over the saved intent dataset.

    pytest tests/test_intent_dataset.py -v

This is the deterministic counterpart to ``test_intent_classifier.py``. That
suite asks a model for fresh cases every run, which is what makes it good at
finding new failures and useless as a gate: it can go red because the
generator got creative. This one replays the fixed corpus written by
``tests.generate_intent_dataset``, calls no model, and fails only when the
classifier itself changes.

The thresholds below are a **ratchet, not a target**. They record what the
classifier scores today, with a little headroom because the corpus can be
regenerated. Raising them is the point: when a lexicon or router change
improves an intent, raise its floor in the same commit so the gain cannot be
silently lost later.

Measured on the 400-case corpus at the time of writing:

    BUYING           100/100   100.0%
    BROWSING          29/100    29.0%
    INTENT OVERRIDE  148/200    74.0%
    overall          277/400    69.2%

BROWSING is the outlier, and it is a missing-signal problem rather than a
tuning one: 71 exploratory queries read as BUYING while no BUYING case ever
reads as BROWSING. See the ``option_seeking`` pattern in
``starter/routing/lexicon.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from tests.score_intent_dataset import (
    DEFAULT_DATA_DIR,
    DEFAULT_PRIOR_TURN,
    load_cases,
    score,
)
from tests.utils.intent_workflow import BROWSING, BUYING, INTENT_OVERRIDE, INTENTS


DATA_DIR = os.environ.get("INTENT_DATA_DIR", DEFAULT_DATA_DIR)

# Floors, not goals. Each sits a few points under the measured value so a
# regenerated corpus does not turn a healthy classifier red.
MIN_ACCURACY = {
    BUYING: float(os.environ.get("INTENT_MIN_BUYING", "95.0")),
    BROWSING: float(os.environ.get("INTENT_MIN_BROWSING", "25.0")),
    INTENT_OVERRIDE: float(os.environ.get("INTENT_MIN_OVERRIDE", "70.0")),
}
MIN_OVERALL = float(os.environ.get("INTENT_MIN_OVERALL", "65.0"))

# How many misclassified utterances a failure message lists.
FAILURE_SAMPLE = 15


@pytest.fixture(scope="session")
def scored() -> List[Dict[str, Any]]:
    """Classify the whole corpus once; no model is involved."""

    cases = load_cases(DATA_DIR)
    if not cases:
        pytest.skip(
            f"no dataset in {DATA_DIR}; generate one with "
            "python -m tests.generate_intent_dataset",
            allow_module_level=True,
        )
    return score(cases, prior_turn=DEFAULT_PRIOR_TURN)


def _subset(scored: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
    return [item for item in scored if item["expected"] == intent]


def _report(intent: str, subset: List[Dict[str, Any]], floor: float) -> str:
    """Accuracy, where the errors went, and the utterances that missed."""

    hits = sum(1 for item in subset if item["correct"])
    accuracy = 100.0 * hits / len(subset)
    misses = [item for item in subset if not item["correct"]]

    went: Dict[str, int] = {}
    for item in misses:
        went[item["predicted"]] = went.get(item["predicted"], 0) + 1
    spread = ", ".join(f"{count} -> {label}" for label, count in sorted(went.items()))

    lines = [
        f"{intent}: {hits}/{len(subset)} = {accuracy:.1f}%, below the {floor:.1f}% floor",
        f"misclassified as: {spread}",
        "",
    ]
    for item in misses[:FAILURE_SAMPLE]:
        lines.append(f"  {item['id'] or '-'}  -> {item['predicted']}")
        lines.append(f"     {item['utterance']}")
    if len(misses) > FAILURE_SAMPLE:
        lines.append(f"  ... and {len(misses) - FAILURE_SAMPLE} more")
    return "\n".join(lines)


def _assert_intent(scored: List[Dict[str, Any]], intent: str) -> None:
    subset = _subset(scored, intent)
    if not subset:
        pytest.skip(f"no {intent} cases in {DATA_DIR}")
    floor = MIN_ACCURACY[intent]
    hits = sum(1 for item in subset if item["correct"])
    accuracy = 100.0 * hits / len(subset)
    print(f"\n{intent}: {hits}/{len(subset)} = {accuracy:.1f}% (floor {floor:.1f}%)")
    if accuracy < floor:
        pytest.fail(_report(intent, subset, floor), pytrace=False)


def test_dataset_is_well_formed(scored: List[Dict[str, Any]]) -> None:
    """The corpus itself: labelled, unique, and non-trivial."""

    assert len(scored) >= 3, "the dataset is too small to gate on"

    unlabelled = [item for item in scored if item["expected"] not in INTENTS]
    assert not unlabelled, (
        f"{len(unlabelled)} case(s) carry a label outside {INTENTS}: "
        + ", ".join(sorted({item["expected"] for item in unlabelled}))
    )

    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for item in scored:
        key = item["utterance"].casefold()
        if key in seen:
            duplicates.append(f"{item['id']} repeats {seen[key]}: {item['utterance']!r}")
        else:
            seen[key] = item["id"]
    assert not duplicates, "duplicate utterances:\n" + "\n".join(duplicates[:10])

    counts = {intent: len(_subset(scored, intent)) for intent in INTENTS}
    print(f"\ncorpus: {counts}, {len(scored)} cases total")


def test_buying_accuracy(scored: List[Dict[str, Any]]) -> None:
    """BUYING must not regress: it is the intent the router handles best."""

    _assert_intent(scored, BUYING)


def test_browsing_accuracy(scored: List[Dict[str, Any]]) -> None:
    """BROWSING floor. Currently the weakest intent -- raise this as it improves."""

    _assert_intent(scored, BROWSING)


def test_override_accuracy(scored: List[Dict[str, Any]]) -> None:
    """INTENT OVERRIDE floor, scored with a generic prior turn.

    The saved records hold the override utterance alone, but the workflow only
    tests a turn for an override once the session has state, so the scorer
    replays one neutral opening turn first.
    """

    _assert_intent(scored, INTENT_OVERRIDE)


def test_overall_accuracy(scored: List[Dict[str, Any]]) -> None:
    hits = sum(1 for item in scored if item["correct"])
    accuracy = 100.0 * hits / len(scored)
    print(f"\noverall: {hits}/{len(scored)} = {accuracy:.1f}% (floor {MIN_OVERALL:.1f}%)")
    if accuracy < MIN_OVERALL:
        pytest.fail(
            f"overall accuracy {accuracy:.1f}% is below the {MIN_OVERALL:.1f}% floor "
            f"({hits}/{len(scored)} correct)",
            pytrace=False,
        )


def test_no_intent_collapses(scored: List[Dict[str, Any]]) -> None:
    """No intent may become unreachable.

    An accuracy floor per intent does not catch a classifier that stops
    emitting a label entirely, because the other intents can absorb the loss
    while each stays above its own floor. This is the cheapest guard against
    that: every label the corpus contains must be predicted at least once.
    """

    predicted = {item["predicted"] for item in scored}
    expected = {item["expected"] for item in scored}
    missing = sorted(expected - predicted)
    assert not missing, (
        f"the classifier never predicted {missing}; "
        f"it only ever returned {sorted(predicted)}"
    )
