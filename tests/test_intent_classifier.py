"""LLM-generated intent classification suite (Task 3).

    pytest tests/test_intent_classifier.py -v

Three isolated tests, one per intent. Each samples real products from
``data/catalog.jsonl``, has the annotation model generate utterances for that
intent alone, feeds them to the shipped workflow classifier, and has the same
model judge the outcome.

Every test needs the annotation endpoint, so all three **skip** rather than
fail when it is unreachable: a red test would report a model outage as a
classifier regression. Configure it with ``ANNOTATION_BASE_URL`` /
``ANNOTATION_MODEL`` / ``ANNOTATION_API_KEY`` (``.env`` is read automatically).

Knobs, all environment variables:
    INTENT_TEST_CASES     cases generated per intent   (default 10)
    INTENT_CATALOG_SAMPLE products sampled for context (default 5)
    INTENT_CATALOG        catalog path                 (default data/catalog.jsonl)
    INTENT_REPORT_DIR     where per-run reports land   (default test_reports)

Every generated case is printed with its prediction and the judge's verdict, so
a run can be checked by hand rather than trusted on its score. pytest captures
stdout for passing tests, so pass ``-s`` to watch it live::

    pytest tests/test_intent_classifier.py -v -s

The same per-case detail is always written to ``INTENT_REPORT_DIR`` as Markdown
and JSON, which is unaffected by capture.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.utils.catalog_loader import (
    DEFAULT_CATALOG,
    sample_catalog_context,
    summarize_products,
)
from tests.utils.intent_generator import generate_synthetic_intent_cases
from tests.utils.intent_workflow import (
    BROWSING,
    BUYING,
    INTENT_OVERRIDE,
    normalize_intent,
    predict_intent,
)
from tests.utils.llm_client import EndpointUnavailable, build_client, preflight
from tests.utils.llm_judge import evaluate_intent_classification


CASE_COUNT = int(os.environ.get("INTENT_TEST_CASES", "10"))
CATALOG_SAMPLE_SIZE = int(os.environ.get("INTENT_CATALOG_SAMPLE", "5"))
CATALOG_PATH = os.environ.get("INTENT_CATALOG", DEFAULT_CATALOG)
REPORT_DIR = os.environ.get("INTENT_REPORT_DIR", "test_reports")


@pytest.fixture(scope="session")
def llm_client() -> Any:
    """The annotation model, or skip the whole suite with the reason why."""

    try:
        client = build_client()
    except EndpointUnavailable as exc:
        pytest.skip(f"annotation endpoint unavailable: {exc}", allow_module_level=True)
    failure = preflight(client)
    if failure:
        pytest.skip(f"annotation endpoint unreachable: {failure}", allow_module_level=True)
    return client


@pytest.fixture(scope="session")
def catalog_context() -> List[Dict]:
    if not os.path.exists(CATALOG_PATH):
        pytest.skip(f"catalog not found at {CATALOG_PATH}", allow_module_level=True)
    return sample_catalog_context(CATALOG_PATH, CATALOG_SAMPLE_SIZE)


def _describe_failure(
    case: Dict[str, Any],
    expected: str,
    actual: str,
    verdict: Dict[str, Any],
    catalog_context: List[Dict],
) -> str:
    """Actionable failure text: utterance, context, both labels, the reason."""

    lines = [
        f"utterance      : {case['utterance']!r}",
        f"expected       : {expected}",
        f"predicted      : {actual}",
        f"judge label    : {verdict.get('judged_intent')}",
        f"judge reasoning: {verdict.get('reasoning')}",
    ]
    if case.get("prior_context"):
        lines.insert(1, f"prior context  : {case['prior_context']!r}")
    titles = [item.get("title", "") for item in summarize_products(catalog_context)]
    lines.append(f"catalog seed   : {titles}")
    return "\n".join(lines)


def _render_case(index: int, record: Dict[str, Any]) -> str:
    """One case, in full, for manual verification."""

    status = record["status"]
    lines = [
        f"[{index:>2}] {status:<8} expected={record['expected']}  "
        f"predicted={record['predicted']}  judge={record['judged'] or '-'}",
        f"     utterance: {record['utterance']}",
        f"     tags     : {len(record['constraint_fields'])} "
        f"{record['constraint_fields']}",
    ]
    if record["prior_context"]:
        prior = record["prior_context"].replace("\n", " | ")
        lines.append(f"     prior    : {prior}")
    if record.get("band_warning"):
        lines.append(f"     WARNING  : {record['band_warning']}")
    if record["reasoning"]:
        lines.append(f"     judge    : {record['reasoning']}")
    return "\n".join(lines)


def _render_report(
    intent: str,
    records: List[Dict[str, Any]],
    catalog_context: List[Dict],
) -> str:
    correct = sum(1 for r in records if r["status"] == "PASS")
    wrong = sum(1 for r in records if r["status"] == "FAIL")
    rejected = sum(1 for r in records if r["status"] == "REJECTED")
    scored = correct + wrong

    header = [
        "=" * 78,
        f"{intent}: {len(records)} generated | {correct}/{scored} correct"
        + (f" | {rejected} rejected by judge" if rejected else ""),
        "=" * 78,
    ]
    body = [_render_case(i, r) for i, r in enumerate(records, start=1)]
    seeds = [item.get("title", "") for item in summarize_products(catalog_context)]
    footer = ["-" * 78, "catalog seed products:"]
    footer += [f"  - {title}" for title in seeds]
    return "\n".join([*header, *body, *footer])


def _write_report(
    intent: str,
    records: List[Dict[str, Any]],
    catalog_context: List[Dict],
) -> Path | None:
    """Persist the per-case detail, immune to pytest's stdout capture."""

    try:
        directory = Path(REPORT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        slug = intent.lower().replace(" ", "_")
        payload = {
            "intent": intent,
            "catalog_seed": summarize_products(catalog_context),
            "cases": records,
        }
        (directory / f"intent_cases_{slug}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report = directory / f"intent_cases_{slug}.md"
        report.write_text(
            "# " + intent + " cases\n\n```\n"
            + _render_report(intent, records, catalog_context)
            + "\n```\n",
            encoding="utf-8",
        )
        return report
    except OSError:
        return None


def _run_intent_suite(
    intent: str,
    client: Any,
    catalog_context: List[Dict],
) -> None:
    """Generate, classify, judge, and report -- shared by the three tests."""

    cases = generate_synthetic_intent_cases(
        intent_type=intent,
        catalog_sample=catalog_context,
        count=CASE_COUNT,
        client=client,
    )
    assert cases, f"the generator returned no usable {intent} cases"

    records: List[Dict[str, Any]] = []
    failures: List[str] = []

    for case in cases:
        prior = case.get("prior_context") or None
        actual_intent = predict_intent(case["utterance"], prior)
        verdict = evaluate_intent_classification(
            utterance=case["utterance"],
            expected_intent=intent,
            actual_intent=actual_intent,
            catalog_context=catalog_context,
            prior_context=prior,
            client=client,
        )
        # A case the judge rejects measures the generator, not the classifier,
        # so it is reported and excluded rather than counted as a failure.
        if not verdict.get("case_is_valid", True):
            status = "REJECTED"
        elif verdict["is_correct"]:
            status = "PASS"
        else:
            status = "FAIL"

        records.append(
            {
                "status": status,
                "utterance": case["utterance"],
                "prior_context": case.get("prior_context", ""),
                "expected": intent,
                "predicted": actual_intent,
                "constraint_fields": case.get("constraint_fields", []),
                "band_warning": case.get("band_warning", ""),
                "judged": verdict.get("judged_intent"),
                "reasoning": verdict.get("reasoning", ""),
                "judge_available": verdict.get("judge_available", True),
            }
        )
        if status == "FAIL":
            failures.append(
                _describe_failure(case, intent, actual_intent, verdict, catalog_context)
            )

    # Printed for every run, not only failing ones, so a pass can be checked
    # by hand. Use -s to see it live; pytest shows it automatically on failure.
    print("\n" + _render_report(intent, records, catalog_context))
    written = _write_report(intent, records, catalog_context)
    if written is not None:
        print(f"report written to {written}")

    scored = sum(1 for record in records if record["status"] != "REJECTED")
    if scored == 0:
        pytest.skip(f"every generated {intent} case was rejected by the judge")

    assert not failures, (
        f"{len(failures)}/{scored} {intent} case(s) misclassified\n\n"
        + "\n\n".join(failures)
    )


@pytest.mark.parametrize("run_id", range(1))  # Configurable test runs
def test_intent_buying(run_id, llm_client, catalog_context):
    """Test Suite 1: Evaluates classification accuracy for BUYING intent."""

    _run_intent_suite(BUYING, llm_client, catalog_context)


def test_intent_browsing(llm_client, catalog_context):
    """Test Suite 2: Evaluates classification accuracy for BROWSING intent."""

    _run_intent_suite(BROWSING, llm_client, catalog_context)


def test_intent_override(llm_client, catalog_context):
    """Test Suite 3: Evaluates classification accuracy for INTENT OVERRIDE intent.

    The generator supplies prior turns with every case, and ``predict_intent``
    replays them into session constraints before classifying the override, so
    the workflow's state machine is exercised rather than a single utterance.
    """

    _run_intent_suite(INTENT_OVERRIDE, llm_client, catalog_context)
