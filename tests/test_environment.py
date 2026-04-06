# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the SafeSpace environment."""

import pytest
from pydantic import ValidationError

from content_moderation_env.server.environment import SafeSpaceEnvironment
from content_moderation_env.server.reward import normalize_public_reward
from content_moderation_env.models import ModerationAction


class TestEnvironmentReset:
    """Tests for environment reset."""

    def test_reset_returns_observation(self):
        """Test reset returns valid observation."""
        env = SafeSpaceEnvironment()
        obs = env.reset()

        assert obs.content_item is not None
        assert obs.trigger_info is not None
        assert obs.platform_policy != ""
        assert len(obs.available_factors) > 0
        assert obs.actions_taken == 0
        assert obs.max_actions == 8

    def test_reset_specific_scenario(self):
        """Test reset with specific scenario ID."""
        env = SafeSpaceEnvironment()
        obs = env.reset("easy_001")

        assert env.state.scenario_id == "easy_001"
        assert env.state.task_id == "clear_violations"
        assert env.state.difficulty == "easy"

    def test_reset_clears_context(self):
        """Test reset clears gathered context."""
        env = SafeSpaceEnvironment()

        # First episode - gather some context
        env.reset("easy_001")
        env.step(ModerationAction(action_type="request_author_profile"))

        # Reset should clear context
        obs = env.reset("easy_002")
        assert obs.gathered_context.author_profile is None

    def test_reset_seed_is_deterministic(self):
        """Test seeded reset picks the same scenario repeatedly."""
        env = SafeSpaceEnvironment()

        first = env.reset(seed=7)
        first_scenario_id = env.state.scenario_id
        second = env.reset(seed=7)

        assert first.content_item.post_id == second.content_item.post_id
        assert first_scenario_id == env.state.scenario_id


class TestEnvironmentStep:
    """Tests for environment step."""

    def test_investigation_action(self):
        """Test investigation action returns context."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        obs = env.step(ModerationAction(action_type="request_author_profile"))

        assert obs.gathered_context.author_profile is not None
        assert obs.actions_taken == 1
        assert "request_author_profile" in obs.action_history
        assert obs.done is False
        assert obs.reward == pytest.approx(normalize_public_reward(-0.03))
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.raw_applied_score == pytest.approx(-0.03)

    def test_duplicate_investigation_warning(self):
        """Test duplicate investigation gives warning."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        # First request
        env.step(ModerationAction(action_type="request_author_profile"))

        # Duplicate request
        obs = env.step(ModerationAction(action_type="request_author_profile"))

        assert "already retrieved" in obs.feedback.lower() or "wasted" in obs.feedback.lower()
        assert obs.reward == pytest.approx(normalize_public_reward(-0.05))
        assert obs.actions_taken == 2
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.raw_applied_score == pytest.approx(-0.05)

    def test_decide_action_ends_episode(self):
        """Test decide action ends episode."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        obs = env.step(ModerationAction(
            action_type="decide",
            decision="remove",
            primary_violation="5.1",
            severity="high",
            confidence=0.95,
            key_factors=["spam_commercial"],
        ))

        assert obs.done is True
        assert 0.0 <= obs.reward <= 1.0
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.raw_applied_score > 0

    def test_decide_requires_fields(self):
        """Test decide action requires all fields."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        # Missing required fields
        obs = env.step(ModerationAction(
            action_type="decide",
            decision="remove",
            # Missing other fields
        ))

        assert obs.done is False
        assert "missing required fields" in obs.feedback.lower()
        assert obs.reward == pytest.approx(normalize_public_reward(-0.06))
        assert obs.actions_taken == 1
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.raw_applied_score == pytest.approx(-0.06)

    def test_invalid_action_type(self):
        """Test invalid action types are rejected by the model."""
        with pytest.raises(ValidationError):
            ModerationAction(action_type="invalid_action")

    def test_invalid_action_branch_penalizes(self):
        """Test bypassing validation still triggers environment penalty."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        obs = env.step(ModerationAction.model_construct(action_type="invalid_action"))

        assert obs.done is False
        assert "invalid" in obs.feedback.lower()
        assert obs.reward == pytest.approx(normalize_public_reward(-0.06))
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.raw_applied_score == pytest.approx(-0.06)


class TestEnvironmentState:
    """Tests for environment state."""

    def test_state_tracking(self):
        """Test state tracks episode correctly."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        assert env.state.scenario_id == "easy_001"
        assert env.state.step_count == 0
        assert env.state.actions_taken == 0

        env.step(ModerationAction(action_type="request_author_profile"))

        assert env.state.step_count == 1
        assert env.state.actions_taken == 1
        assert "author_profile" in env.state.context_requested

    def test_decision_updates_state(self):
        """Test decision updates state correctly."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        env.step(ModerationAction(
            action_type="decide",
            decision="remove",
            primary_violation="5.1",
            severity="high",
            confidence=0.95,
            key_factors=["spam_commercial"],
        ))

        assert env.state.decision_made is True
        assert env.state.episode_reward > 0
        assert env.state.raw_episode_reward > 0

    def test_hard_case_rewards_needed_context(self):
        """Test hard scenarios reward useful context gathering."""
        env = SafeSpaceEnvironment()
        env.reset("hard_001")

        obs = env.step(ModerationAction(action_type="request_author_violations"))

        assert obs.reward == pytest.approx(normalize_public_reward(0.05))
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.is_needed is True
        assert obs.reward_breakdown.raw_applied_score == pytest.approx(0.05)

    def test_budget_exhaustion_without_decision_is_terminal(self):
        """Test repeated wasted actions end the episode with a penalty."""
        env = SafeSpaceEnvironment()
        env.reset("easy_001")

        obs = None
        for _ in range(8):
            obs = env.step(ModerationAction.model_construct(action_type="invalid_action"))

        assert obs is not None
        assert obs.done is True
        assert 0.0 <= obs.reward <= 1.0
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.no_decision["reason"] == "no_decision_made"
        assert obs.reward_breakdown.no_decision["raw_penalty"] == pytest.approx(-0.15)

    def test_three_useful_context_requests_reach_new_trajectory_cap(self):
        """Three needed context requests should fully benefit from the higher cap."""
        env = SafeSpaceEnvironment()
        env.reset("hard_plus_002")

        env.step(ModerationAction(action_type="request_thread_context"))
        env.step(ModerationAction(action_type="request_author_violations"))
        obs = env.step(ModerationAction(action_type="request_similar_precedents"))

        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.applied_score == pytest.approx(
            normalize_public_reward(0.05)
        )
        assert obs.reward_breakdown.raw_applied_score == pytest.approx(0.05)
        assert obs.reward_breakdown.trajectory_total == pytest.approx(
            normalize_public_reward(0.15)
        )
        assert obs.reward_breakdown.raw_trajectory_total == pytest.approx(0.15)
        assert env.state.episode_reward == pytest.approx(normalize_public_reward(0.15))
        assert env.state.raw_episode_reward == pytest.approx(0.15)


class TestScenarioDiversity:
    """Tests for scenario loading and diversity."""

    def test_all_difficulties_loadable(self):
        """Test scenarios from all difficulties load."""
        env = SafeSpaceEnvironment()

        for scenario_id in ["easy_001", "med_001", "hard_001"]:
            obs = env.reset(scenario_id)
            assert obs.content_item is not None

    def test_trigger_types(self):
        """Test different trigger types are handled."""
        env = SafeSpaceEnvironment()

        # User report
        obs = env.reset("easy_002")
        assert obs.trigger_info.trigger_type == "user_report"
        assert obs.trigger_info.report_count > 0

        # Auto flag
        obs = env.reset("easy_001")
        assert obs.trigger_info.trigger_type == "auto_flag"
        assert obs.trigger_info.auto_flag_reason is not None

        # Appeal
        obs = env.reset("med_005")
        assert obs.trigger_info.trigger_type == "appeal"
        assert obs.trigger_info.original_decision is not None

        # Proactive audit
        obs = env.reset("hard_002")
        assert obs.trigger_info.trigger_type == "proactive_audit"
        assert obs.trigger_info.audit_reason is not None
