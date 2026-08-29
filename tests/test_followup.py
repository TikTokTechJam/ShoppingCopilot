from __future__ import annotations

import unittest

from starter.clarification import _utility
from starter.followup import (
    FollowUpPolicy,
    MAX_TURNS,
    fill_to_top_k,
    promotion_threshold,
    utility,
)


def _pool(*values: str) -> tuple[object, ...]:
    """A candidate pool that splits perfectly evenly on ``color``."""

    class Candidate:
        def __init__(self, value: str) -> None:
            self.attributes = {"color": [value]}

    return tuple(Candidate(value) for value in values)


def _two_field_pool(left: str, right: str) -> tuple[object, ...]:
    """A pool that splits identically on two fields, isolating the priors."""

    class Candidate:
        def __init__(self, value: str) -> None:
            self.attributes = {left: [value], right: [value]}

    return tuple(Candidate(v) for v in ("a", "b", "c", "d"))


def _factor(overrides: dict[str, float]):
    """A profile whose likelihood ratio is 1.0 except where stated."""
    return lambda attribute: overrides.get(attribute, 1.0)


def _choose(pool, mode: str, profile_factor=None, turn: int = 1):
    from starter.clarification import ClarificationPolicy
    from starter.routing.constraints import ShoppingConstraints

    return ClarificationPolicy().choose(
        pool,
        ShoppingConstraints(),
        (),
        mode=mode,
        profile_factor=profile_factor,
        turn=turn,
    )


class UtilityTest(unittest.TestCase):
    def test_matches_the_documented_grid(self) -> None:
        self.assertAlmostEqual(utility(1, 1), 1.000)
        self.assertAlmostEqual(utility(1, 10), 0.730)
        self.assertAlmostEqual(utility(10, 1), 0.820)
        self.assertAlmostEqual(utility(10, 10), 0.550)

    def test_miss_scores_zero(self) -> None:
        self.assertEqual(utility(1, None), 0.0)

    def test_decomposition_reproduces_the_evaluator(self) -> None:
        """TechnicalScore is the mean of utility() over sessions."""
        import statistics

        sessions = [(1, 1), (4, 3), (None, None)]
        hit_rate = sum(1 for turn, _ in sessions if turn is not None) / len(sessions)
        mrr = statistics.fmean(
            0.0 if rank is None else 1.0 / rank for _, rank in sessions
        )
        mttc = statistics.fmean(
            MAX_TURNS + 1 if turn is None else turn for turn, _ in sessions
        )
        efficiency = max(0.0, min(1.0, (MAX_TURNS + 1 - mttc) / MAX_TURNS))
        expected = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
        actual = statistics.fmean(
            utility(turn, rank) if turn is not None else 0.0 for turn, rank in sessions
        )
        self.assertAlmostEqual(expected, actual, places=12)

    def test_delay_costs_less_than_rank(self) -> None:
        self.assertAlmostEqual(utility(1, 5) - utility(2, 5), 0.02)
        self.assertGreater(utility(1, 1) - utility(1, 10), 0.25)

    def test_rejects_out_of_range(self) -> None:
        for turn, rank in ((0, 1), (11, 1), (1, 0)):
            with self.assertRaises(ValueError):
                utility(turn, rank)


class PromotionThresholdTest(unittest.TestCase):
    def test_withholding_rank_one_is_never_optimal(self) -> None:
        for turn in range(1, MAX_TURNS):
            self.assertGreater(promotion_threshold(turn, 1), 1.0)

    def test_final_turn_can_never_wait(self) -> None:
        self.assertEqual(promotion_threshold(MAX_TURNS, 5), float("inf"))

    def test_lower_ranks_are_cheaper_to_withhold(self) -> None:
        thresholds = [promotion_threshold(1, rank) for rank in range(1, 11)]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))


class FillToTopKTest(unittest.TestCase):
    def test_pads_from_backfill(self) -> None:
        self.assertEqual(fill_to_top_k(["a", "b"], ["c", "d"], 4), ["a", "b", "c", "d"])

    def test_preserves_primary_order_and_dedupes(self) -> None:
        self.assertEqual(fill_to_top_k(["a", "b", "a"], ["b", "c"], 3), ["a", "b", "c"])

    def test_drops_invalid_ids(self) -> None:
        self.assertEqual(
            fill_to_top_k(["a", "x"], ["b"], 3, valid_asins={"a", "b"}), ["a", "b"]
        )

    def test_never_exceeds_top_k(self) -> None:
        self.assertEqual(fill_to_top_k(["a", "b", "c"], ["d"], 2), ["a", "b"])

    def test_degrades_on_bad_top_k(self) -> None:
        self.assertEqual(fill_to_top_k(["a"], [], "nope"), [])
        self.assertEqual(fill_to_top_k(["a"], [], -1), [])


class FollowUpPolicyTest(unittest.TestCase):
    def test_wait_branch_is_disabled_by_default(self) -> None:
        policy = FollowUpPolicy()
        decision = policy.decide(
            1, ranked_ranks=(10,), promotion_probabilities=(1.0,), ask_attribute="color"
        )
        self.assertTrue(decision.recommend)
        self.assertIn("gamma", decision.reason)

    def test_final_turn_always_recommends(self) -> None:
        policy = FollowUpPolicy(allow_wait=True)
        decision = policy.decide(
            MAX_TURNS,
            ranked_ranks=(10,),
            promotion_probabilities=(1.0,),
            ask_attribute="color",
        )
        self.assertTrue(decision.recommend)

    def test_wait_requires_a_question(self) -> None:
        policy = FollowUpPolicy(allow_wait=True)
        decision = policy.decide(
            1, ranked_ranks=(10,), promotion_probabilities=(1.0,), ask_attribute=None
        )
        self.assertTrue(decision.recommend)

    def test_wait_requires_progress(self) -> None:
        policy = FollowUpPolicy(allow_wait=True)
        decision = policy.decide(
            1,
            ranked_ranks=(10,),
            promotion_probabilities=(1.0,),
            ask_attribute="color",
            made_progress=False,
        )
        self.assertTrue(decision.recommend)

    def test_consecutive_wait_cap(self) -> None:
        policy = FollowUpPolicy(allow_wait=True)
        first = policy.decide(
            1, ranked_ranks=(10,), promotion_probabilities=(1.0,), ask_attribute="color"
        )
        self.assertFalse(first.recommend)
        second = policy.decide(
            2, ranked_ranks=(10,), promotion_probabilities=(1.0,), ask_attribute="color"
        )
        self.assertTrue(second.recommend)
        self.assertIn("consecutive", second.reason)

    def test_low_gamma_recommends(self) -> None:
        policy = FollowUpPolicy(allow_wait=True)
        decision = policy.decide(
            1, ranked_ranks=(10,), promotion_probabilities=(0.1,), ask_attribute="color"
        )
        self.assertTrue(decision.recommend)

    def test_rank_one_never_waits_even_with_certain_promotion(self) -> None:
        policy = FollowUpPolicy(allow_wait=True)
        decision = policy.decide(
            1, ranked_ranks=(1,), promotion_probabilities=(1.0,), ask_attribute="color"
        )
        self.assertTrue(decision.recommend)


class ProfileFlagTest(unittest.TestCase):
    """The --disable-user-profile switch must reach the clarification prior."""

    def test_factor_is_identity_without_tags(self) -> None:
        from starter.profile_affinity import ProfileAffinity

        for profile in (None, {}, {"preference_tags": []}, {"preference_tags": "x"}):
            affinity = ProfileAffinity(profile)
            self.assertTrue(
                all(affinity.factor(a) == 1.0 for a in affinity.attributes),
                profile,
            )

    def test_factor_stays_bounded(self) -> None:
        from starter.profile_affinity import PROFILE_WEIGHT, ProfileAffinity

        affinity = ProfileAffinity({"preference_tags": ["fit", "material", "comfort"]})
        for attribute in affinity.attributes:
            self.assertGreaterEqual(affinity.factor(attribute), 1.0 - PROFILE_WEIGHT)
            self.assertLessEqual(affinity.factor(attribute), 1.0 + PROFILE_WEIGHT)

    def test_reset_honours_the_switch(self) -> None:
        from starter.agent import Agent

        class Sessions:
            def reset(self, session_id, user_profile):
                return None

        def build(enabled: bool):
            agent = Agent.__new__(Agent)
            agent.use_user_profile = enabled
            agent._profile_affinity = {}
            agent.sessions = Sessions()
            agent.retriever = None
            Agent.reset(agent, "s", {"preference_tags": ["fit"]})
            return agent._profile_affinity

        self.assertEqual(build(False), {})
        self.assertIn("s", build(True))

    def test_neutral_profile_factor_changes_nothing(self) -> None:
        base = _utility("color", _pool("black", "white", "blue", "red"), "BUYING")
        neutral = _utility(
            "color", _pool("black", "white", "blue", "red"), "BUYING", lambda _: 1.0
        )
        self.assertAlmostEqual(neutral, base)

    def test_profile_factor_moves_the_utility_monotonically(self) -> None:
        pool = _pool("black", "white", "blue", "red")
        base = _utility("color", pool, "BUYING")
        self.assertGreater(_utility("color", pool, "BUYING", lambda _: 1.25), base)
        self.assertLess(_utility("color", pool, "BUYING", lambda _: 0.75), base)

    def test_bad_profile_factor_is_ignored(self) -> None:
        pool = _pool("black", "white", "blue")
        base = _utility("color", pool, "BUYING")
        for bad in ("nope", float("nan"), float("inf"), -1.0):
            self.assertAlmostEqual(_utility("color", pool, "BUYING", lambda _: bad), base)

    def test_refusal_factor_vetoes_the_attribute(self) -> None:
        pool = _pool("black", "white", "blue")
        self.assertEqual(_utility("color", pool, "BUYING", lambda _: 0.0), 0.0)


class CombinedUtilityTest(unittest.TestCase):
    """Section 12's split term combined with Section 12b's session utility."""

    def test_answer_probability_stays_a_probability(self) -> None:
        from starter.clarification import MODE_PRIORS, _answer_probability

        for attribute in MODE_PRIORS["BUYING"]:
            for ratio in (0.5, 0.75, 1.0, 1.25, 4.0, 1000.0):
                probability = _answer_probability(
                    attribute, "BUYING", lambda _, r=ratio: r
                )
                self.assertGreater(probability, 0.0)
                self.assertLess(probability, 1.0)

    def test_odds_update_never_truncates_a_strong_prior(self) -> None:
        """A multiplier on the probability would clip material at 1.0."""
        from starter.clarification import MODE_PRIORS, _answer_probability

        self.assertEqual(MODE_PRIORS["BUYING"]["material"], 1.00)
        favoured = _answer_probability("material", "BUYING", lambda _: 1.25)
        neutral = _answer_probability("material", "BUYING", lambda _: 1.0)
        disfavoured = _answer_probability("material", "BUYING", lambda _: 0.75)
        self.assertGreater(favoured, neutral)
        self.assertGreater(neutral, disfavoured)

    def test_utility_is_in_score_units_when_a_turn_is_given(self) -> None:
        pool = _pool("black", "white", "blue", "red")
        unscaled = _utility("color", pool, "BUYING")
        self.assertAlmostEqual(
            _utility("color", pool, "BUYING", None, 1), unscaled * utility(2, 1)
        )

    def test_late_questions_are_worth_less(self) -> None:
        pool = _pool("black", "white", "blue", "red")
        values = [_utility("color", pool, "BUYING", None, t) for t in range(1, 10)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], values[-1])

    def test_final_turn_has_no_horizon(self) -> None:
        pool = _pool("black", "white", "blue", "red")
        self.assertEqual(_utility("color", pool, "BUYING", None, MAX_TURNS), 0.0)

    def test_choose_abstains_on_the_final_turn(self) -> None:
        """The old `turn < 10` literal now falls out of the utility."""
        from starter.clarification import ClarificationPolicy
        from starter.routing.constraints import ShoppingConstraints

        pool = _pool("black", "white", "blue", "red")
        policy = ClarificationPolicy()
        constraints = ShoppingConstraints()
        self.assertEqual(
            policy.choose(pool, constraints, (), mode="BUYING", turn=1), "color"
        )
        self.assertIsNone(
            policy.choose(pool, constraints, (), mode="BUYING", turn=MAX_TURNS)
        )

    def test_floors_preserve_the_historical_bar(self) -> None:
        """Re-expressing the bet in score units must not move the bar.

        A question whose pool term sits exactly on the legacy 0.035 threshold
        must still sit exactly on the floor after the prior is read as a
        probability and discounted by the horizon.
        """
        from starter.clarification import (
            ASK_SPLIT_FLOOR,
            ASK_UTILITY_FLOOR,
            PRIOR_CEILING,
        )

        split, weight = ASK_SPLIT_FLOOR, 1.00  # a top-priority attribute
        self.assertAlmostEqual(
            split * PRIOR_CEILING * weight * utility(2, 1), ASK_UTILITY_FLOOR
        )

    def test_profile_reorders_a_near_tie(self) -> None:
        """BUYING ranks color 0.92 and size 0.90: close enough for evidence."""
        from starter.profile_affinity import PROFILE_WEIGHT

        pool = _two_field_pool("color", "size")
        self.assertEqual(_choose(pool, "BUYING"), "color")
        favours_size = _factor({"size": 1.0 + PROFILE_WEIGHT})
        self.assertEqual(_choose(pool, "BUYING", favours_size), "size")

    def test_profile_cannot_overturn_a_decisive_prior(self) -> None:
        """material 1.00 against budget 0.66 is not a near-tie."""
        from starter.profile_affinity import PROFILE_WEIGHT

        pool = _two_field_pool("material", "budget")
        self.assertEqual(_choose(pool, "BUYING"), "material")
        favours_budget = _factor(
            {"budget": 1.0 + PROFILE_WEIGHT, "material": 1.0 - PROFILE_WEIGHT}
        )
        self.assertEqual(_choose(pool, "BUYING", favours_budget), "material")

    def test_strong_enough_evidence_still_moves_a_decisive_prior(self) -> None:
        """The top weight must not read as certainty, or nothing can move it."""
        pool = _two_field_pool("material", "budget")
        self.assertEqual(_choose(pool, "BUYING", _factor({"budget": 12.0})), "budget")


if __name__ == "__main__":
    unittest.main()
