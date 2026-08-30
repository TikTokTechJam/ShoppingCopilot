"""Score the workflow's intent classifier against the saved dataset.

    python -m tests.score_intent_dataset

Reads the ground-truth files written by ``tests.generate_intent_dataset`` and
runs every utterance through the shipped classifier. No model is called: the
labels are already in the files, so this is deterministic, offline, and fast
enough to run on every change to the router or the lexicon.

**Prior context.** The saved override records are the override utterance
alone, but ``Agent.respond`` only tests a turn for an override when the
session already has state (``if state.mode is not None``), so an utterance
classified in isolation can never come back INTENT OVERRIDE. Scoring the
override set therefore replays a generic opening turn first. It supplies the
session state, not the content: the override utterances name what they are
abandoning themselves, which is why the context could be dropped at save time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tests.utils.intent_workflow import (
    BROWSING,
    BUYING,
    INTENT_OVERRIDE,
    INTENTS,
    normalize_intent,
    predict_intent,
)


DEFAULT_DATA_DIR = "data/derived/intent_cases"

# A neutral opening turn, used only to give the session the state an override
# needs to be detectable. It names no attribute the override could collide
# with beyond the category, which every override replaces anyway.
DEFAULT_PRIOR_TURN = "I am shopping for something from your catalog."


def _slug(intent: str) -> str:
    return intent.lower().replace(" ", "_")


def load_cases(data_dir: str) -> List[Dict[str, Any]]:
    """Read every intent file present, newest schema only."""

    cases: List[Dict[str, Any]] = []
    directory = Path(data_dir)
    for intent in INTENTS:
        path = directory / f"{_slug(intent)}.json"
        if not path.exists():
            print(f"[score] missing {path}, skipped", file=sys.stderr)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            utterance = str(case.get("utterance", "")).strip()
            if not utterance:
                continue
            cases.append(
                {
                    "id": case.get("id", ""),
                    "utterance": utterance,
                    "expected": normalize_intent(
                        case.get("expected_intent", payload.get("intent", intent))
                    ),
                }
            )
    return cases


def score(
    cases: List[Dict[str, Any]],
    *,
    prior_turn: str = DEFAULT_PRIOR_TURN,
) -> List[Dict[str, Any]]:
    """Classify every case, recording what the workflow said."""

    results: List[Dict[str, Any]] = []
    for case in cases:
        # Only the override set needs session state; giving the single-turn
        # intents a prior turn would change what they are testing.
        prior = prior_turn if case["expected"] == INTENT_OVERRIDE else None
        predicted = predict_intent(case["utterance"], prior)
        results.append({**case, "predicted": predicted, "correct": predicted == case["expected"]})
    return results


def render(results: List[Dict[str, Any]], *, show_failures: int = 10) -> str:
    lines: List[str] = []
    total = len(results)
    correct = sum(1 for item in results if item["correct"])

    lines.append("=" * 74)
    lines.append(
        f"Intent classification on the saved dataset: "
        f"{correct}/{total} correct ({100.0 * correct / total if total else 0.0:.1f}%)"
    )
    lines.append("=" * 74)

    lines.append("")
    lines.append("Per intent:")
    lines.append("")
    lines.append(f"  {'expected':<18}{'n':>5}{'correct':>9}{'accuracy':>10}")
    for intent in INTENTS:
        subset = [item for item in results if item["expected"] == intent]
        if not subset:
            continue
        hits = sum(1 for item in subset if item["correct"])
        lines.append(
            f"  {intent:<18}{len(subset):>5}{hits:>9}"
            f"{100.0 * hits / len(subset):>9.1f}%"
        )

    lines.append("")
    lines.append("Confusion (rows expected, columns predicted):")
    lines.append("")
    header = "  " + " " * 18 + "".join(f"{intent:>18}" for intent in INTENTS)
    lines.append(header)
    for expected in INTENTS:
        row = Counter(
            item["predicted"] for item in results if item["expected"] == expected
        )
        if not sum(row.values()):
            continue
        lines.append(
            f"  {expected:<18}" + "".join(f"{row.get(p, 0):>18}" for p in INTENTS)
        )

    failures = [item for item in results if not item["correct"]]
    if failures:
        lines.append("")
        lines.append(f"Misclassified ({len(failures)}), first {min(show_failures, len(failures))}:")
        lines.append("")
        for item in failures[:show_failures]:
            lines.append(
                f"  {item['id'] or '-'}  {item['expected']} -> {item['predicted']}"
            )
            lines.append(f"     {item['utterance']}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the intent classifier against the saved dataset",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--prior-turn", default=DEFAULT_PRIOR_TURN)
    parser.add_argument("--show-failures", type=int, default=10)
    parser.add_argument("--json", dest="json_path", help="write full results as JSON")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="exit non-zero below this overall accuracy, for CI use",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cases = load_cases(args.data_dir)
    if not cases:
        print(f"no cases found in {args.data_dir}", file=sys.stderr)
        return 2

    results = score(cases, prior_turn=args.prior_turn)
    print(render(results, show_failures=args.show_failures))

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nfull results written to {args.json_path}", file=sys.stderr)

    accuracy = 100.0 * sum(1 for item in results if item["correct"]) / len(results)
    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(
            f"\naccuracy {accuracy:.1f}% is below the required {args.min_accuracy:.1f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
