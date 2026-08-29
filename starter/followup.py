"""Follow-up strategy: whether a turn returns a new list or holds.

Implements Section 12b of ``Architecture.md``.

The scored objective decomposes exactly into a per-session utility, so the
turn policy optimizes ``utility()`` and nothing else.  The wait branch stays
disabled until ``gamma`` is measured (Section 12b.6); the invariants that
constrain it are enforced here regardless, so enabling it later cannot bypass
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Iterable, Sequence

# Competition constants.  A miss is scored as though it converted one turn
# after the last, which is what makes the efficiency term collapse to a clean
# per-session value.
MAX_TURNS = 10
MISS_TURN = MAX_TURNS + 1
HIT_WEIGHT = 0.50
RANK_WEIGHT = 0.30
EFFICIENCY_WEIGHT = 0.20


def utility(turn: int, rank: int | None, *, max_turns: int = MAX_TURNS) -> float:
    """Per-session score contribution of a hit at ``turn`` and ``rank``.

    Section 12b.1.  ``rank`` of ``None`` is a miss and scores zero.  Because
    every per-session efficiency term already lies in [0, 1], the corpus-level
    clip never binds and ``TechnicalScore`` is the mean of this function.
    """
    if rank is None:
        return 0.0
    if rank < 1 or turn < 1 or turn > max_turns:
        raise ValueError("turn must be within the session and rank must be >= 1")
    efficiency = (max_turns + 1 - turn) / max_turns
    return HIT_WEIGHT + RANK_WEIGHT / rank + EFFICIENCY_WEIGHT * efficiency


def promotion_threshold(turn: int, rank: int, *, max_turns: int = MAX_TURNS) -> float:
    """Minimum ``gamma`` that would justify withholding ``rank`` at ``turn``.

    Section 12b.3.  Withholding is worthwhile only when the probability of
    re-retrieving the item next turn *and* ranking it first exceeds this.  A
    value above 1.0 means withholding can never pay off.
    """
    if turn >= max_turns:
        return float("inf")  # nothing left to wait for; invariant 3
    return utility(turn, rank, max_turns=max_turns) / utility(
        turn + 1, 1, max_turns=max_turns
    )


def fill_to_top_k(
    primary: Iterable[str],
    backfill: Iterable[str],
    top_k: int,
    *,
    valid_asins: Collection[str] | None = None,
) -> list[str]:
    """Return exactly ``top_k`` unique valid IDs when that many are available.

    Invariant 2 of Section 12b.4.  Appending candidates *below* the ranked ones
    cannot change the rank of anything above them, so padding is weakly
    dominant and a short list is a pure expected-value loss.
    """
    try:
        limit = max(0, int(top_k))
    except (TypeError, ValueError):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for source in (primary, backfill):
        for value in source:
            if len(result) >= limit:
                return result
            asin = str(value).strip()
            if not asin or asin in seen:
                continue
            if valid_asins is not None and asin not in valid_asins:
                continue
            seen.add(asin)
            result.append(asin)
    return result


@dataclass(frozen=True)
class FollowUpDecision:
    """Outcome of one turn-level choice."""

    recommend: bool
    reason: str

    def __bool__(self) -> bool:
        return self.recommend


@dataclass
class FollowUpPolicy:
    """Decide whether to return a newly ranked list this turn.

    ``allow_wait`` stays false until ``gamma`` is measured with
    ``evaluator/followup_probe.py``.  Section 12b.6 requires that measurement
    before any wait branch ships.
    """

    allow_wait: bool = False
    max_turns: int = MAX_TURNS
    _waits: list[int] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        self._waits.clear()

    def decide(
        self,
        turn: int,
        *,
        ranked_ranks: Sequence[int] = (),
        promotion_probabilities: Sequence[float] = (),
        ask_attribute: str | None = None,
        made_progress: bool = True,
    ) -> FollowUpDecision:
        """Apply Section 12b.3 subject to the invariants of Section 12b.4."""
        turn = int(turn)
        if not self.allow_wait:
            return FollowUpDecision(True, "wait branch disabled pending gamma measurement")
        # Invariant 3: nothing left to wait for on the final turn.
        if turn >= self.max_turns:
            return FollowUpDecision(True, "final turn")
        # Invariant 6: waiting without asking cannot gather information.
        if not ask_attribute:
            return FollowUpDecision(True, "no clarification asked")
        # Invariant 4: waiting requires the previous turn to have moved state.
        if not made_progress:
            return FollowUpDecision(True, "no new information last turn")
        # Invariant 5: never wait twice in a row.
        if self._waits and self._waits[-1] == turn - 1:
            return FollowUpDecision(True, "consecutive wait cap")
        if not ranked_ranks or len(promotion_probabilities) != len(ranked_ranks):
            return FollowUpDecision(True, "no calibrated promotion estimate")
        # Section 12b.3: hold only if every ranked item clears its threshold.
        for rank, gamma in zip(ranked_ranks, promotion_probabilities):
            if float(gamma) <= promotion_threshold(turn, int(rank), max_turns=self.max_turns):
                return FollowUpDecision(True, f"rank {rank} below promotion threshold")
        self._waits.append(turn)
        return FollowUpDecision(False, "all ranked items clear the promotion threshold")


__all__ = [
    "EFFICIENCY_WEIGHT",
    "FollowUpDecision",
    "FollowUpPolicy",
    "HIT_WEIGHT",
    "MAX_TURNS",
    "MISS_TURN",
    "RANK_WEIGHT",
    "fill_to_top_k",
    "promotion_threshold",
    "utility",
]
