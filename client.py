# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SafeSpace Content Moderation Environment Client."""

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        ContentItem,
        GatheredContext,
        ModerationAction,
        ModerationObservation,
        ModerationState,
        RewardBreakdown,
        TaskGradeBreakdown,
        TriggerInfo,
    )
except ImportError:  # pragma: no cover
    from models import (
        ContentItem,
        GatheredContext,
        ModerationAction,
        ModerationObservation,
        ModerationState,
        RewardBreakdown,
        TaskGradeBreakdown,
        TriggerInfo,
    )


class SafeSpaceEnv(
    EnvClient[ModerationAction, ModerationObservation, ModerationState]
):
    """
    Client for the SafeSpace Content Moderation Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SafeSpaceEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     print(result.observation.content_item.text)
        ...
        ...     # Investigate
        ...     result = client.step(ModerationAction(action_type="request_thread_context"))
        ...     print(result.observation.gathered_context.thread_context)
        ...
        ...     # Make decision
        ...     result = client.step(ModerationAction(
        ...         action_type="decide",
        ...         decision="approve",
        ...         primary_violation="none",
        ...         severity="none",
        ...         confidence=0.9,
        ...         key_factors=["gaming_or_competition_context"]
        ...     ))
        ...     print(f"Reward: {result.reward}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SafeSpaceEnv.from_docker_image("safespace-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ModerationAction(
        ...         action_type="decide",
        ...         decision="remove",
        ...         primary_violation="5.1",
        ...         severity="high",
        ...         confidence=0.95,
        ...         key_factors=["spam_commercial"]
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ModerationAction) -> Dict[str, Any]:
        """
        Convert ModerationAction to JSON payload for step message.

        Args:
            action: ModerationAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
        }

        # Include decision fields only for decide action
        if action.action_type == "decide":
            payload["decision"] = action.decision
            payload["primary_violation"] = action.primary_violation
            payload["severity"] = action.severity
            payload["confidence"] = action.confidence
            payload["key_factors"] = action.key_factors

        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ModerationObservation]:
        """
        Parse server response into StepResult[ModerationObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ModerationObservation
        """
        obs_data = payload.get("observation", {})

        # Parse nested objects
        content_item = None
        if obs_data.get("content_item"):
            content_item = ContentItem(**obs_data["content_item"])

        trigger_info = None
        if obs_data.get("trigger_info"):
            trigger_info = TriggerInfo(**obs_data["trigger_info"])

        gathered_context = GatheredContext()
        if obs_data.get("gathered_context"):
            gathered_context = GatheredContext(**obs_data["gathered_context"])

        reward_breakdown = None
        if obs_data.get("reward_breakdown") is not None:
            reward_breakdown = RewardBreakdown.model_validate(
                obs_data["reward_breakdown"]
            )

        grade_breakdown = None
        if obs_data.get("grade_breakdown") is not None:
            grade_breakdown = TaskGradeBreakdown.model_validate(
                obs_data["grade_breakdown"]
            )

        observation = ModerationObservation(
            content_item=content_item,
            trigger_info=trigger_info,
            gathered_context=gathered_context,
            platform_policy=obs_data.get("platform_policy", ""),
            available_factors=obs_data.get("available_factors", []),
            actions_taken=obs_data.get("actions_taken", 0),
            max_actions=obs_data.get("max_actions", 8),
            action_history=obs_data.get("action_history", []),
            feedback=obs_data.get("feedback", ""),
            error_code=obs_data.get("error_code"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            reward_breakdown=reward_breakdown,
            task_grade=obs_data.get("task_grade"),
            grade_breakdown=grade_breakdown,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ModerationState:
        """
        Parse server response into ModerationState object.

        Args:
            payload: JSON response from state request

        Returns:
            ModerationState object
        """
        return ModerationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario_id=payload.get("scenario_id"),
            task_id=payload.get("task_id"),
            difficulty=payload.get("difficulty"),
            trigger_type=payload.get("trigger_type"),
            actions_taken=payload.get("actions_taken", 0),
            max_actions=payload.get("max_actions", 8),
            context_requested=payload.get("context_requested", []),
            decision_made=payload.get("decision_made", False),
            episode_reward=payload.get("episode_reward", 0.0),
            done=payload.get("done", False),
            last_error_code=payload.get("last_error_code"),
        )


# Alias for backward compatibility
ContentModerationEnv = SafeSpaceEnv
