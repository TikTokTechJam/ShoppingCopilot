"""Score the live intent workflow against ``data/derived/intent_cases``.

    python -m tools.eval_intent_cases                 # all three case files
    python -m tools.eval_intent_cases --set buying    # one file
    python -m tools.eval_intent_cases --misses        # print every miss

The benchmark spans two different components, so this tool runs each case
against the one that actually decides it:

``buying.json`` / ``browsing.json``
    ``TwoPhaseIntentRouter.classify`` -- the live runtime router, the same
    object ``Agent`` builds.  Reported with a per-tier breakdown, because a
    miss at the ``default`` tier (the router could not decide) is a different
    failure from a confident miss at ``rules``.

``intent_override.json``
    ``session.detect_override_kind`` -- ``INTENT OVERRIDE`` is not a label the
    router can emit.  The router answers BUYING vs BROWSING; override
    detection is a separate marker-plus-delta check.  A case counts as correct
    when the kind is ``FULL_GOAL`` or ``PREFERENCE``.

Each case is a single utterance with no conversation history, so overrides are
scored against empty current constraints -- the first-turn reading.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from starter.routing import BROWSING, BUYING, TwoPhaseIntentRouter
from starter.routing.constraints import ShoppingConstraints, extract_constraints
from starter.session import OverrideKind, detect_override_kind


ROOT = Path(__file__).parents[1]
CASE_DIR = ROOT / "data" / "derived" / "intent_cases"
ROUTER_SETS = ("buying", "browsing")
OVERRIDE_SETS = ("intent_override",)
ALL_SETS = (*ROUTER_SETS, *OVERRIDE_SETS)


def load_cases(name: str) -> list[dict]:
    path = CASE_DIR / f"{name}.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases", [])
    return [case for case in cases if isinstance(case, dict)]


def score_router(name: str, cases: list[dict], show_misses: bool) -> tuple[int, int]:
    router = TwoPhaseIntentRouter()
    by_tier: dict[str, list[bool]] = defaultdict(list)
    misses: list[tuple[str, str, str, float, str]] = []
    correct = 0

    for case in cases:
        utterance = str(case.get("utterance", ""))
        expected = str(case.get("expected_intent", "")).strip().upper()
        result = router.classify(utterance)
        hit = result.intent == expected
        correct += hit
        by_tier[result.tier].append(hit)
        if not hit:
            misses.append(
                (
                    str(case.get("id", "")),
                    expected,
                    result.intent,
                    result.confidence,
                    utterance,
                )
            )

    print(f"\n{name}  ({len(cases)} cases, expects {cases[0].get('expected_intent') if cases else '?'})")
    print(f"  accuracy   {correct}/{len(cases)} = {correct / max(len(cases), 1):.4f}")
    print("  by tier:")
    for tier, hits in sorted(by_tier.items()):
        print(f"    {tier:16} {sum(hits):4d}/{len(hits):<4}  = {sum(hits) / len(hits):.4f}")
    print(f"  phase 1 (tags) {router.phase1_decisions:4d}   vetoed {router.phase1_vetoed:4d}"
          f"   defaulted BROWSING {router.defaulted:4d}")
    if misses and show_misses:
        print("  misses:")
        for case_id, expected, got, confidence, utterance in misses:
            print(f"    {case_id}  want {expected:8} got {got:8} @{confidence:.2f}  {utterance}")
    return correct, len(cases)


def score_override(name: str, cases: list[dict], show_misses: bool) -> tuple[int, int]:
    empty = ShoppingConstraints()
    kinds: Counter[str] = Counter()
    misses: list[tuple[str, str]] = []
    correct = 0

    for case in cases:
        utterance = str(case.get("utterance", ""))
        delta = extract_constraints(utterance)
        kind = detect_override_kind(utterance, empty, delta)
        kinds[kind.value] += 1
        if kind in (OverrideKind.FULL_GOAL, OverrideKind.PREFERENCE):
            correct += 1
        else:
            misses.append((str(case.get("id", "")), utterance))

    print(f"\n{name}  ({len(cases)} cases, expects an override)")
    print(f"  detected   {correct}/{len(cases)} = {correct / max(len(cases), 1):.4f}")
    print("  by kind:")
    for kind, count in sorted(kinds.items()):
        print(f"    {kind:16} {count:4d}")
    if misses and show_misses:
        print("  missed (read as NONE):")
        for case_id, utterance in misses:
            print(f"    {case_id}  {utterance}")
    return correct, len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=ALL_SETS, action="append", dest="sets",
                        help="score one case file; repeatable (default: all)")
    parser.add_argument("--misses", action="store_true",
                        help="print every miss, not just the totals")
    args = parser.parse_args()
    selected = tuple(args.sets) if args.sets else ALL_SETS

    print("=" * 72)
    print(f"Intent case benchmark - {CASE_DIR}")
    print("=" * 72)

    total_correct = 0
    total_cases = 0
    for name in selected:
        cases = load_cases(name)
        if not cases:
            print(f"\n{name}: missing or empty, skipped")
            continue
        scorer = score_override if name in OVERRIDE_SETS else score_router
        correct, count = scorer(name, cases, args.misses)
        total_correct += correct
        total_cases += count

    if total_cases:
        print(f"\nTOTAL  {total_correct}/{total_cases} = {total_correct / total_cases:.4f}")
        print("\nNote: buying/browsing score TwoPhaseIntentRouter; intent_override")
        print("scores session.detect_override_kind. They are different components.")


if __name__ == "__main__":
    main()
