# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the reward computation system."""

import pytest
from content_moderation_env.server.reward import (
    compute_duplicate_action_penalty,
    compute_reward,
    compute_efficiency_bonus,
    compute_investigation_reward,
    compute_invalid_action_penalty,
    compute_calibration_bonus,
    compute_no_decision_penalty,
)
from content_moderation_env.server.grader import (
    PUBLIC_TASK_GRADE_EPSILON,
    compute_calibration_grade,
    clamp_public_task_grade,
    compute_task_grade,
    grade_decision,
    grade_factors,
    compute_factor_jaccard,
)


class TestDecisionGrading:
    """Tests for decision grading logic."""

    def test_perfect_match(self):
        """Test perfect decision match gives 0.55."""
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
        }
        score, details = grade_decision(
            agent_decision="remove",
            agent_violation="5.1",
            agent_severity="high",
            ground_truth=ground_truth,
        )
        assert score == 0.55
        assert details["grade"] == "perfect"

    def test_correct_decision_and_violation(self):
        """Test correct decision + violation but wrong severity gives 0.50."""
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
        }
        score, details = grade_decision(
            agent_decision="remove",
            agent_violation="5.1",
            agent_severity="medium",  # Wrong severity
            ground_truth=ground_truth,
        )
        assert score == 0.50
        assert details["grade"] == "decision_and_violation_correct"

    def test_correct_decision_only(self):
        """Test correct decision but wrong violation gives 0.40."""
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
        }
        score, details = grade_decision(
            agent_decision="remove",
            agent_violation="4.1",  # Wrong violation
            agent_severity="high",
            ground_truth=ground_truth,
        )
        assert score == 0.40
        assert details["grade"] == "decision_correct_only"

    def test_adjacent_decision(self):
        """Test adjacent decisions get partial credit."""
        ground_truth = {
            "correct_decision": "escalate",
            "primary_violation": "6.1",
            "severity": "medium",
        }
        # warn is adjacent to escalate
        score, details = grade_decision(
            agent_decision="warn",
            agent_violation="6.1",
            agent_severity="medium",
            ground_truth=ground_truth,
        )
        assert score == 0.08
        assert details["grade"] == "adjacent_decision"

    def test_dangerous_false_negative(self):
        """Test approving dangerous content gives negative score."""
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "4.1",
            "severity": "critical",
        }
        score, details = grade_decision(
            agent_decision="approve",
            agent_violation="none",
            agent_severity="none",
            ground_truth=ground_truth,
        )
        assert score == -0.18
        assert details["grade"] == "dangerous_false_negative"

    def test_wrong_decision_penalty(self):
        """Test completely wrong decision gives negative score."""
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
        }
        # warn is not adjacent to remove, so this is wrong_decision
        score, details = grade_decision(
            agent_decision="approve",
            agent_violation="none",
            agent_severity="none",
            ground_truth=ground_truth,
        )
        # This is false_negative (approve vs remove with high severity)
        # Let's test a true wrong_decision case instead
        ground_truth2 = {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
        }
        score2, details2 = grade_decision(
            agent_decision="warn",  # Not adjacent to approve
            agent_violation="3.1",
            agent_severity="medium",
            ground_truth=ground_truth2,
        )
        assert score2 == -0.10
        assert details2["grade"] == "wrong_decision"


class TestFactorGrading:
    """Tests for factor grading logic."""

    def test_perfect_factor_match(self):
        """Test perfect factor match gives 0.15."""
        ground_truth = {
            "key_factors": ["spam_commercial", "clear_violation_no_exception"]
        }
        score, details = grade_factors(
            agent_factors=["spam_commercial", "clear_violation_no_exception"],
            ground_truth=ground_truth,
        )
        assert score == 0.15
        assert details["jaccard"] == 1.0

    def test_partial_factor_match(self):
        """Test partial factor match gives proportional score."""
        ground_truth = {
            "key_factors": ["spam_commercial", "repeat_offender"]
        }
        # Agent got one right, one wrong
        score, details = grade_factors(
            agent_factors=["spam_commercial", "new_account"],
            ground_truth=ground_truth,
        )
        # Intersection: spam_commercial (1)
        # Union: spam_commercial, repeat_offender, new_account (3)
        # Jaccard: 1/3 = 0.333
        assert 0.04 < score < 0.06  # 0.333 * 0.15 = 0.05

    def test_no_factor_overlap(self):
        """Test no overlap gives 0.0."""
        ground_truth = {
            "key_factors": ["spam_commercial", "repeat_offender"]
        }
        score, details = grade_factors(
            agent_factors=["new_account", "trusted_contributor"],
            ground_truth=ground_truth,
        )
        assert score == 0.0


class TestEfficiencyBonus:
    """Tests for efficiency bonus."""

    def test_zero_actions_correct(self):
        """Test 0 actions with correct decision gives max bonus."""
        score, _ = compute_efficiency_bonus(actions_taken=0, eligible_for_efficiency=True)
        assert score == 0.10

    def test_one_action_correct(self):
        """Test 1 action with correct decision gives 0.08."""
        score, _ = compute_efficiency_bonus(actions_taken=1, eligible_for_efficiency=True)
        assert score == 0.08

    def test_many_actions_correct(self):
        """Test 6+ actions gives no bonus."""
        score, _ = compute_efficiency_bonus(actions_taken=6, eligible_for_efficiency=True)
        assert score == 0.0

    def test_wrong_decision_no_bonus(self):
        """Test wrong decision gets no efficiency bonus."""
        score, _ = compute_efficiency_bonus(actions_taken=0, eligible_for_efficiency=False)
        assert score == 0.0


class TestTrajectoryRewards:
    """Tests for non-terminal trajectory shaping."""

    def test_needed_context_reward(self):
        """Requesting needed context should produce positive reward."""
        score, details = compute_investigation_reward(
            context_field="thread_context",
            ground_truth={"context_needed": ["thread_context"]},
            retrieved=True,
        )
        assert score > 0
        assert details["is_needed"] is True
        assert details["retrieved"] is True

    def test_needed_but_unavailable_context_is_penalized(self):
        """Unavailable context should never receive positive shaping reward."""
        score, details = compute_investigation_reward(
            context_field="thread_context",
            ground_truth={"context_needed": ["thread_context"]},
            retrieved=False,
        )
        assert score < 0
        assert details["is_needed"] is True
        assert details["retrieved"] is False
        assert details["reason"] == "needed_context_unavailable"

    def test_irrelevant_context_penalty(self):
        """Unneeded context should produce a small penalty."""
        score, details = compute_investigation_reward(
            context_field="author_profile",
            ground_truth={"context_needed": ["thread_context"]},
            retrieved=True,
        )
        assert score < 0
        assert details["is_needed"] is False

    def test_duplicate_action_penalty(self):
        """Duplicate requests should be penalized more strongly."""
        score, details = compute_duplicate_action_penalty("thread_context")
        assert score < 0
        assert details["reason"] == "duplicate_context_request"

    def test_invalid_action_penalty(self):
        """Invalid actions should carry a penalty."""
        score, details = compute_invalid_action_penalty(
            reason="invalid_action_type",
            action_type="invalid_action",
        )
        assert score < 0
        assert details["action_type"] == "invalid_action"

    def test_no_decision_penalty(self):
        """Budget exhaustion without a decision should be negative."""
        score, details = compute_no_decision_penalty(actions_taken=8)
        assert score < 0
        assert details["reason"] == "no_decision_made"


class TestCalibrationBonus:
    """Tests for calibration bonus."""

    def test_high_confidence_correct_easy(self):
        """Test high confidence + correct on easy gives 0.10."""
        score, _ = compute_calibration_bonus(
            confidence=0.9,
            agent_decision="remove",
            correct_decision="remove",
            difficulty="easy",
        )
        assert score == 0.10

    def test_high_confidence_correct_hard(self):
        """Test high confidence + correct on hard gives 0.10."""
        score, _ = compute_calibration_bonus(
            confidence=0.9,
            agent_decision="remove",
            correct_decision="remove",
            difficulty="hard",
        )
        assert score == 0.10

    def test_overconfident_wrong(self):
        """Test high confidence + wrong gives -0.10 penalty."""
        score, _ = compute_calibration_bonus(
            confidence=0.9,
            agent_decision="approve",
            correct_decision="remove",
            difficulty="easy",
        )
        assert score == -0.10

    def test_low_confidence_escalate_hard(self):
        """Test low confidence escalation on hard case gives 0.08."""
        score, _ = compute_calibration_bonus(
            confidence=0.3,
            agent_decision="escalate",
            correct_decision="escalate",
            difficulty="hard",
        )
        assert score == 0.08

    def test_medium_confidence_correct(self):
        """Test medium confidence + correct gives 0.04."""
        score, _ = compute_calibration_bonus(
            confidence=0.6,
            agent_decision="remove",
            correct_decision="remove",
            difficulty="medium",
        )
        assert score == 0.04


class TestCalibrationGrade:
    """Tests for normalized calibration grading."""

    def test_overconfident_wrong_gets_zero_grade(self):
        """Wrong answers at very high confidence should receive no calibration credit."""
        score, details = compute_calibration_grade(
            confidence=0.9,
            agent_decision="approve",
            correct_decision="remove",
            difficulty="medium",
        )
        assert score == 0.0
        assert details["calibration_type"] == "overconfident_wrong"

    def test_confident_wrong_gets_small_credit_only(self):
        """Moderately confident wrong answers should receive only minimal calibration credit."""
        score, details = compute_calibration_grade(
            confidence=0.7,
            agent_decision="approve",
            correct_decision="remove",
            difficulty="medium",
        )
        assert score == 0.05
        assert details["calibration_type"] == "confident_wrong"

    def test_cautious_wrong_is_not_rewarded_like_correct_uncertainty(self):
        """Low-confidence wrong answers should not receive a large calibration score."""
        score, details = compute_calibration_grade(
            confidence=0.35,
            agent_decision="approve",
            correct_decision="remove",
            difficulty="hard",
        )
        assert score == 0.02
        assert details["calibration_type"] == "cautious_wrong"


class TestPublicTaskGrade:
    """Tests for public task-grade interval handling."""

    def test_clamp_public_task_grade_preserves_interior_values(self):
        """Interior task grades should pass through unchanged."""
        assert clamp_public_task_grade(0.42) == pytest.approx(0.42)

    def test_clamp_public_task_grade_moves_boundaries_inward(self):
        """Exact endpoints should be nudged into the open interval."""
        assert clamp_public_task_grade(0.0) == pytest.approx(PUBLIC_TASK_GRADE_EPSILON)
        assert clamp_public_task_grade(1.0) == pytest.approx(1.0 - PUBLIC_TASK_GRADE_EPSILON)

    def test_compute_task_grade_keeps_perfect_episode_below_one(self):
        """Perfect episodes should remain validator-safe without changing rank order."""
        score, breakdown = compute_task_grade(
            agent_decision={
                "decision": "remove",
                "primary_violation": "5.1",
                "severity": "high",
                "confidence": 0.95,
                "key_factors": ["spam_commercial", "clear_violation_no_exception"],
            },
            ground_truth={
                "correct_decision": "remove",
                "primary_violation": "5.1",
                "severity": "high",
                "key_factors": ["spam_commercial", "clear_violation_no_exception"],
            },
            actions_taken=0,
            difficulty="easy",
        )

        assert 0.0 < score < 1.0
        assert score == pytest.approx(1.0 - PUBLIC_TASK_GRADE_EPSILON)
        assert breakdown["raw_total"] == pytest.approx(1.0)

    def test_compute_task_grade_keeps_worst_case_above_zero(self):
        """Worst-case public grades should still be strictly positive."""
        score, breakdown = compute_task_grade(
            agent_decision={
                "decision": "approve",
                "primary_violation": "none",
                "severity": "none",
                "confidence": 0.95,
                "key_factors": ["no_violation_found"],
            },
            ground_truth={
                "correct_decision": "remove",
                "primary_violation": "4.1",
                "severity": "critical",
                "key_factors": ["explicit_threat"],
            },
            actions_taken=0,
            difficulty="easy",
        )

        assert 0.0 < score < 1.0
        assert score == pytest.approx(PUBLIC_TASK_GRADE_EPSILON)
        assert breakdown["raw_total"] == pytest.approx(0.0)


class TestFullReward:
    """Tests for full reward computation."""

    def test_perfect_easy_case(self):
        """Test perfect easy case gives 0.90."""
        agent_decision = {
            "decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
            "confidence": 0.95,
            "key_factors": ["spam_commercial", "clear_violation_no_exception"],
        }
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
            "key_factors": ["spam_commercial", "clear_violation_no_exception"],
        }
        total, breakdown = compute_reward(
            agent_decision=agent_decision,
            ground_truth=ground_truth,
            actions_taken=0,
            difficulty="easy",
        )
        assert total == 0.90
        assert breakdown["decision"]["score"] == 0.55
        assert breakdown["factor"]["score"] == 0.15
        assert breakdown["efficiency"]["score"] == 0.10
        assert breakdown["calibration"]["score"] == 0.10

    def test_reward_clamped_to_zero(self):
        """Test reward never goes below 0."""
        agent_decision = {
            "decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "confidence": 0.95,  # Overconfident wrong
            "key_factors": ["no_violation_found"],
        }
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "4.1",
            "severity": "critical",
            "key_factors": ["explicit_threat"],
        }
        total, _ = compute_reward(
            agent_decision=agent_decision,
            ground_truth=ground_truth,
            actions_taken=0,
            difficulty="easy",
        )
        # -0.18 (dangerous) + 0 (factors) + 0 (efficiency) - 0.10 (overconf) = -0.28
        # But clamped to MIN_EPISODE_REWARD = -0.25
        assert total == -0.25

    def test_reward_clamped_to_one(self):
        """Test reward never exceeds 1.0."""
        # Even with theoretical maximum components, should cap at 1.0
        agent_decision = {
            "decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
            "confidence": 0.95,
            "key_factors": ["spam_commercial", "clear_violation_no_exception"],
        }
        ground_truth = {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
            "key_factors": ["spam_commercial", "clear_violation_no_exception"],
        }
        total, _ = compute_reward(
            agent_decision=agent_decision,
            ground_truth=ground_truth,
            actions_taken=0,
            difficulty="easy",
        )
        assert total <= 1.0
