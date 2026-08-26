"""Score the intent router against the labelled sets.

    python -m tools.eval_intent_router              # rules tier only
    python -m tools.eval_intent_router --model      # rules + reranker cascade

Reports accuracy, macro-F1 and a confidence-band calibration table, plus a
per-tier breakdown when the reranker is enabled.

This tool deliberately does not read the generated benchmark sessions. Any
labelled set built from templated sessions teaches a classifier the template;
such a set belongs in CI as a floor, never as the number the work optimises.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from starter.routing import (
    BROWSING,
    BUYING,
    CascadingIntentRouter,
    LexicalIntentRouter,
    TwoPhaseIntentRouter,
)


ROOT = Path(__file__).parents[1]
GOLDEN = ROOT / "tests" / "data" / "intent_golden.jsonl"
DEV = ROOT / "data" / "derived" / "intent" / "dev_set.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    scores = []
    for label in (BUYING, BROWSING):
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


def report(name: str, rows: list[dict], router) -> list[tuple[str, str, float]]:
    if not rows:
        print(f"\n{name}: missing, skipped")
        return []

    pairs: list[tuple[str, str]] = []
    scored: list[tuple[str, str, float]] = []
    by_tier: dict[str, list[bool]] = defaultdict(list)
    misses = []

    started = time.perf_counter()
    for row in rows:
        result = router.classify(row["message"])
        gold = row["intent"]
        pairs.append((gold, result.intent))
        scored.append((gold, result.intent, result.confidence))
        by_tier[result.tier].append(gold == result.intent)
        if gold != result.intent:
            misses.append((gold, result.intent, result.confidence, result.tier, row["message"]))
    elapsed = time.perf_counter() - started

    correct = sum(1 for gold, pred in pairs if gold == pred)
    print(f"\n{name}")
    print(f"  accuracy   {correct}/{len(pairs)} = {correct / len(pairs):.4f}")
    print(f"  macro-F1   {macro_f1(pairs):.4f}")
    print(f"  wall clock {elapsed * 1000:.1f} ms total, {elapsed / len(rows) * 1000:.2f} ms/message")
    if len(by_tier) > 1 or "rules" not in by_tier:
        print("  by tier:")
        for tier, hits in sorted(by_tier.items()):
            print(f"    {tier:28} {sum(hits)}/{len(hits)}")
    if misses:
        print("  misses:")
        for gold, pred, confidence, tier, message in misses:
            print(f"    want {gold:8} got {pred:8} @{confidence:.2f} [{tier}]  {message}")
    return scored


def calibration(rows: list[tuple[str, str, float]]) -> None:
    print("\nCALIBRATION")
    for low, high in [(0.50, 0.60), (0.60, 0.70), (0.70, 0.85), (0.85, 1.01)]:
        bucket = [(g, p) for g, p, c in rows if low <= c < high]
        if not bucket:
            print(f"  {low:.2f}-{high:.2f}  n=   0")
            continue
        correct = sum(1 for g, p in bucket if g == p)
        print(f"  {low:.2f}-{high:.2f}  n={len(bucket):4d}  acc={correct / len(bucket):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="enable the reranker tier")
    parser.add_argument("--ledger-only", action="store_true",
                        help="skip Phase 1 and the BROWSING default")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep the Phase 1 tag threshold")
    args = parser.parse_args()

    if args.sweep:
        sweep(args.model)
        return

    if args.model:
        from starter.routing.local_model import QwenRerankerBackend

        backend = QwenRerankerBackend()
        missing = backend.missing_requirements()
        if missing:
            print("reranker unavailable:")
            for item in missing:
                print("   ", item)
            print("  see requirements-reranker.txt and tools/fetch_reranker.py")
            return
        backend_obj = backend
    else:
        backend_obj = None

    if args.ledger_only:
        router = CascadingIntentRouter(backend=backend_obj)
        label = "signal ledger only" + (" + reranker" if backend_obj else "")
    else:
        router = TwoPhaseIntentRouter(backend=backend_obj)
        label = (
            f"two-phase (tags >= {router.tag_threshold}, "
            f"default BROWSING below {router.decision_confidence})"
            + (" + reranker" if backend_obj else "")
        )

    print("=" * 72)
    print(f"Intent router evaluation - {label}")
    print("=" * 72)

    scored = report("GOLDEN (issue #6 spec, never tuned against)", load(GOLDEN), router)
    scored += report("DEV SET (hand-written)", load(DEV), router)
    if scored:
        calibration(scored)

    total = max(1, len(scored))
    print(f"\nROUTING  {total} messages")
    if isinstance(router, TwoPhaseIntentRouter):
        print(f"  phase 1 (tags)      {router.phase1_decisions:4d}  ({router.phase1_decisions / total:.1%})")
        print(f"  defaulted BROWSING  {router.defaulted:4d}  ({router.defaulted / total:.1%})")
    print(f"  escalated to model  {router.escalations:4d}  ({router.escalations / total:.1%})"
          f"   failures: {router.backend_failures}")


def sweep(use_model: bool) -> None:
    """Tag threshold against accuracy, so the choice of 2 is a measured one."""
    backend = None
    if use_model:
        from starter.routing.local_model import QwenRerankerBackend

        backend = QwenRerankerBackend()
        if backend.missing_requirements():
            print("reranker unavailable; sweeping without it")
            backend = None

    rows = load(GOLDEN) + load(DEV)
    print(f"{'config':44} {'correct':>9}  {'macro-F1':>8}  {'phase1':>7}  {'default':>8}")
    print("-" * 84)

    for threshold in (1, 2, 3, 4):
        for decision in (0.50, 0.70):
            router = TwoPhaseIntentRouter(
                backend=backend, tag_threshold=threshold, decision_confidence=decision
            )
            pairs = [(r["intent"], router.classify(r["message"]).intent) for r in rows]
            correct = sum(1 for g, p in pairs if g == p)
            name = f"tags>={threshold}, default below {decision:.2f}"
            print(
                f"{name:44} {correct:4d}/{len(rows):<4} {macro_f1(pairs):8.4f}"
                f"  {router.phase1_decisions:7d}  {router.defaulted:8d}"
            )
    print("\ndefault below 0.50 disables the terminal BROWSING rule")


if __name__ == "__main__":
    main()
