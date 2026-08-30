"""LLM-as-a-Judge for intent classification (Task 4).

The judge is deliberately not a second opinion on the label. Ground truth is
already fixed by the generator, so string equality answers "did the workflow
match". What equality cannot answer is whether the *case* was fair: a
generator can emit an utterance that reads as BROWSING however it was
labelled, and failing the workflow for that would measure the generator.

So the judge is asked to rule on the utterance itself, and a disagreement with
ground truth is reported as a defective case rather than as a workflow error.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tests.utils.catalog_loader import summarize_products
from tests.utils.intent_workflow import INTENTS, normalize_intent
from tests.utils.llm_client import parse_json


def build_prompt(
    utterance: str,
    expected_intent: str,
    actual_intent: str,
    catalog_context: List[Dict],
    prior_context: str | None = None,
) -> str:
    prior = (
        f'Earlier turns in this session:\n"""{prior_context}"""'
        if prior_context
        else "This is the opening utterance of the session; there are no earlier turns."
    )
    return f"""
You are judging an e-commerce intent classifier. The three possible intents are:

- BUYING: high-intent, specific purchase query with explicit constraints
  (budget, size, specifications), urgency, or a direct purchase signal.
- BROWSING: low-intent, exploratory, open-ended query with no hard purchasing
  commitment.
- INTENT OVERRIDE: a mid-session pivot that shifts category, changes a
  constraint the shopper already gave, or cancels a previous preference.
  This is only possible when earlier turns exist.

{prior}

Shopper utterance under test:
\"\"\"{utterance}\"\"\"

Catalog context the utterance was written from:
{json.dumps(summarize_products(list(catalog_context or [])), indent=2, ensure_ascii=False)}

The intended label was {expected_intent}. The classifier predicted {actual_intent}.

Decide independently which single intent the utterance actually expresses,
then answer:
- "judged_intent": your own label, one of BUYING, BROWSING, INTENT OVERRIDE.
- "is_correct": true if the classifier's prediction matches YOUR label.
- "case_is_valid": false if the intended label is not defensible for this
  utterance, i.e. the test case itself is bad.
- "reasoning": one or two sentences citing the words that decided it.

Return ONLY this JSON object:
{{"judged_intent": "...", "is_correct": true, "case_is_valid": true, "reasoning": "..."}}
""".strip()


def evaluate_intent_classification(
    utterance: str,
    expected_intent: str,
    actual_intent: str,
    catalog_context: List[Dict],
    prior_context: str = None,
    *,
    client: Any = None,
) -> Dict:
    """Uses LLM to judge whether the workflow's predicted intent is valid.

    Returns: {"is_correct": bool, "reasoning": str}
    """

    expected = normalize_intent(expected_intent)
    actual = normalize_intent(actual_intent)

    if client is None:
        from tests.utils.llm_client import build_client

        client = build_client()

    try:
        payload = parse_json(
            client.annotate(
                build_prompt(
                    utterance, expected, actual, catalog_context, prior_context
                )
            )
        )
    except Exception as exc:  # noqa: BLE001 - never let the judge mask a result
        return {
            "is_correct": actual == expected,
            "judged_intent": expected,
            "case_is_valid": True,
            "judge_available": False,
            "reasoning": (
                f"judge unavailable ({type(exc).__name__}: {exc}); "
                "fell back to exact match against the intended label"
            ),
        }

    judged = normalize_intent(payload.get("judged_intent", expected))
    if judged not in INTENTS:
        judged = expected
    return {
        "is_correct": bool(payload.get("is_correct", actual == judged)),
        "judged_intent": judged,
        "case_is_valid": bool(payload.get("case_is_valid", True)),
        "judge_available": True,
        "reasoning": str(payload.get("reasoning", ""))[:400],
    }


__all__ = ["build_prompt", "evaluate_intent_classification"]
