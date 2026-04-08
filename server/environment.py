"""SafeSpace environment implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        ContentItem,
        GatheredContext,
        ModerationAction,
        ModerationObservation,
        ModerationState,
        RewardBreakdown,
        TaskGradeBreakdown,
        TriggerInfo,
    )
except (ModuleNotFoundError, ImportError):  # pragma: no cover
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

from .grader import clamp_public_task_grade, compute_task_grade
from .policy import FACTOR_LIST, PLATFORM_POLICY
from .reward import (
    MAX_EPISODE_REWARD,
    MIN_EPISODE_REWARD,
    TRAJECTORY_REWARD_CAP,
    compute_duplicate_action_penalty,
    compute_investigation_reward,
    compute_invalid_action_penalty,
    compute_no_decision_penalty,
    compute_reward,
    normalize_public_reward,
    normalize_reward_breakdown,
)
from .scenarios import get_task_id_for_scenario, load_scenario


class SafeSpaceEnvironment(Environment):
    """Content-moderation environment with trajectory shaping and strict episode handling."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    INVESTIGATION_ACTIONS: Dict[str, str] = {
        "request_author_profile": "author_profile",
        "request_author_violations": "author_violations",
        "request_thread_context": "thread_context",
        "request_community_rules": "community_rules",
        "request_linked_content": "linked_content_summary",
        "request_similar_precedents": "similar_precedents",
        "request_reporter_credibility": "reporter_credibility",
    }

    def __init__(self) -> None:
        self._state = ModerationState(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Dict[str, Any]] = None
        self._gathered_context = GatheredContext()
        self._action_history: list[str] = []
        self._trajectory_reward_total = 0.0
        self._raw_episode_reward_total = 0.0
        self._episode_done = False
        self._last_reward_breakdown: Optional[Dict[str, Any]] = None
        self._last_grade_breakdown: Optional[Dict[str, Any]] = None
        self._last_task_grade: Optional[float] = None
        self._last_error_code: Optional[str] = None

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        """Reset the environment and return the initial observation."""
        requested_task = kwargs.get("task_id") or kwargs.get("scenario_id") or task_id
        self._scenario = load_scenario(requested_task, seed=seed)
        resolved_task_id = get_task_id_for_scenario(self._scenario)

        self._state = ModerationState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scenario_id=self._scenario.get("scenario_id"),
            task_id=resolved_task_id,
            difficulty=self._scenario.get("difficulty"),
            trigger_type=self._scenario.get("trigger_info", {}).get("trigger_type"),
            actions_taken=0,
            max_actions=8,
            context_requested=[],
            decision_made=False,
            episode_reward=0.0,
            raw_episode_reward=0.0,
            done=False,
            last_error_code=None,
        )
        self._gathered_context = GatheredContext()
        self._action_history = []
        self._trajectory_reward_total = 0.0
        self._raw_episode_reward_total = 0.0
        self._episode_done = False
        self._last_reward_breakdown = None
        self._last_grade_breakdown = None
        self._last_task_grade = None
        self._last_error_code = None

        return self._build_observation(
            feedback="Episode started. Review the content and make a moderation decision.",
            done=False,
            reward=None,
            error_code=None,
            reward_breakdown={"reward_type": "reset", "total": 0.0},
            task_grade=None,
            grade_breakdown=None,
        )

    def step(
        self,
        action: ModerationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        """Execute one step in the environment."""
        del timeout_s, kwargs

        if self._scenario is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self._episode_done:
            return self._build_observation(
                feedback="Episode already completed. Call reset() before taking another action.",
                done=True,
                reward=0.0,
                error_code="episode_already_done",
                reward_breakdown={
                    "reward_type": "terminal_guard",
                    "reason": "episode_already_done",
                    "score": 0.0,
                    "total": 0.0,
                    "last_terminal_breakdown": self._last_reward_breakdown,
                },
                task_grade=self._last_task_grade,
                grade_breakdown=self._last_grade_breakdown,
            )

        self._state.step_count += 1

        if action.action_type in self.INVESTIGATION_ACTIONS:
            return self._handle_investigation(action)
        if action.action_type == "decide":
            return self._handle_decision(action)
        return self._handle_invalid_action(action)

    @property
    def state(self) -> ModerationState:
        """Return the current public environment state."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return metadata for validators, clients, and the web UI."""
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        readme_content = readme_path.read_text() if readme_path.exists() else None
        return EnvironmentMetadata(
            name="safespace",
            description=(
                "Content moderation environment with multi-step investigation, "
                "deterministic grading, and trajectory-level reward shaping."
            ),
            version="0.2.1",
            author="SafeSpace Hackathon Team",
            readme_content=readme_content,
        )

    def _build_observation(
        self,
        *,
        feedback: str,
        done: bool,
        reward: Optional[float],
        error_code: Optional[str],
        reward_breakdown: Optional[Dict[str, Any]],
        task_grade: Optional[float],
        grade_breakdown: Optional[Dict[str, Any]],
    ) -> ModerationObservation:
        """Build an observation from the current internal state."""
        typed_reward_breakdown = None
        if reward_breakdown is not None:
            typed_reward_breakdown = RewardBreakdown.model_validate(
                normalize_reward_breakdown(reward_breakdown)
            )

        typed_grade_breakdown = None
        if grade_breakdown is not None:
            typed_grade_breakdown = TaskGradeBreakdown.model_validate(grade_breakdown)

        return ModerationObservation(
            content_item=ContentItem(**self._scenario["content_item"]) if self._scenario else None,
            trigger_info=TriggerInfo(**self._scenario["trigger_info"]) if self._scenario else None,
            gathered_context=self._gathered_context,
            platform_policy=PLATFORM_POLICY,
            available_factors=FACTOR_LIST,
            actions_taken=self._state.actions_taken,
            max_actions=self._state.max_actions,
            action_history=self._action_history.copy(),
            feedback=feedback,
            error_code=error_code,
            done=done,
            reward=self._normalize_step_reward(reward),
            reward_breakdown=typed_reward_breakdown,
            task_grade=task_grade,
            grade_breakdown=typed_grade_breakdown,
            metadata={
                "episode_reward": self._state.episode_reward,
                "raw_episode_reward": self._raw_episode_reward_total,
                "trajectory_reward_total": self._trajectory_reward_total,
                "decision_made": self._state.decision_made,
                "raw_reward": reward,
            },
        )

    def _sync_public_reward_state(self) -> None:
        """Expose normalized reward fields on the public state model."""
        self._state.raw_episode_reward = self._raw_episode_reward_total
        if self._state.step_count == 0 and self._raw_episode_reward_total == 0.0:
            self._state.episode_reward = 0.0
            return
        normalized_total = normalize_public_reward(self._raw_episode_reward_total)
        self._state.episode_reward = 0.0 if normalized_total is None else normalized_total

    def _normalize_step_reward(self, reward: Optional[float]) -> Optional[float]:
        """Normalize a per-step reward for the public observation surface."""
        return normalize_public_reward(reward)

    def _consume_budget(self, action_label: str) -> None:
        """Consume one non-terminal action from the budget."""
        self._state.actions_taken += 1
        self._action_history.append(action_label)

    def _apply_trajectory_delta(
        self,
        raw_delta: float,
        breakdown: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply the capped trajectory reward delta for an investigation action."""
        previous_total = self._trajectory_reward_total
        next_total = max(
            -TRAJECTORY_REWARD_CAP,
            min(TRAJECTORY_REWARD_CAP, previous_total + raw_delta),
        )
        applied_delta = next_total - previous_total
        self._trajectory_reward_total = next_total

        self._raw_episode_reward_total = max(
            MIN_EPISODE_REWARD,
            min(MAX_EPISODE_REWARD, next_total),
        )
        self._sync_public_reward_state()
        enriched = {
            **breakdown,
            "requested_score": raw_delta,
            "applied_score": applied_delta,
            "trajectory_total": self._trajectory_reward_total,
            "episode_total": self._raw_episode_reward_total,
        }
        return applied_delta, enriched

    def _apply_episode_delta(self, delta: float) -> Tuple[float, float]:
        """Apply a bounded delta to the cumulative episode reward."""
        current_total = self._raw_episode_reward_total
        next_total = max(
            MIN_EPISODE_REWARD,
            min(MAX_EPISODE_REWARD, current_total + delta),
        )
        applied_delta = next_total - current_total
        self._raw_episode_reward_total = next_total
        self._sync_public_reward_state()
        return applied_delta, next_total

    def _finalize_if_budget_exhausted(
        self,
        *,
        feedback: str,
        step_reward: float,
        step_breakdown: Dict[str, Any],
        error_code: Optional[str],
    ) -> ModerationObservation:
        """Return a non-terminal observation or end the episode on budget exhaustion."""
        self._last_error_code = error_code
        self._state.last_error_code = error_code

        if self._state.actions_taken < self._state.max_actions:
            self._last_reward_breakdown = step_breakdown
            self._state.done = False
            return self._build_observation(
                feedback=feedback,
                done=False,
                reward=step_reward,
                error_code=error_code,
                reward_breakdown=step_breakdown,
                task_grade=None,
                grade_breakdown=None,
            )

        no_decision_penalty, terminal_breakdown = compute_no_decision_penalty(
            self._state.actions_taken
        )
        applied_penalty, cumulative_total = self._apply_episode_delta(no_decision_penalty)
        combined_reward = step_reward + applied_penalty
        self._state.decision_made = True
        self._state.done = True
        self._episode_done = True
        self._last_task_grade = clamp_public_task_grade(0.0)
        self._last_grade_breakdown = {
            "decision": {"weight": 0.70, "score": 0.0, "details": {"reason": "no_decision"}},
            "factor_overlap": {"weight": 0.15, "score": 0.0, "details": {}},
            "efficiency": {"weight": 0.05, "score": 0.0, "details": {}},
            "calibration": {"weight": 0.10, "score": 0.0, "details": {}},
            "raw_total": 0.0,
            "public_total_adjustment": self._last_task_grade,
            "total": self._last_task_grade,
        }
        combined_breakdown = {
            "reward_type": "trajectory_terminal",
            "trajectory": step_breakdown,
            "no_decision": {
                **terminal_breakdown,
                "applied_score": applied_penalty,
            },
            "step_total": combined_reward,
            "cumulative_total": cumulative_total,
        }
        self._last_reward_breakdown = combined_breakdown
        self._last_error_code = "no_decision_made"
        self._state.last_error_code = "no_decision_made"

        return self._build_observation(
            feedback=f"{feedback} Action budget exhausted before a final decision.",
            done=True,
            reward=combined_reward,
            error_code="no_decision_made",
            reward_breakdown=combined_breakdown,
            task_grade=self._last_task_grade,
            grade_breakdown=self._last_grade_breakdown,
        )

    def _handle_investigation(self, action: ModerationAction) -> ModerationObservation:
        """Handle a context-gathering action."""
        context_field = self.INVESTIGATION_ACTIONS[action.action_type]

        if self._state.actions_taken >= self._state.max_actions:
            self._episode_done = True
            self._state.done = True
            return self._build_observation(
                feedback="Action budget exhausted. Episode ended with no decision.",
                done=True,
                reward=0.0,
                error_code="action_budget_exhausted",
                reward_breakdown=self._last_reward_breakdown,
                task_grade=self._last_task_grade,
                grade_breakdown=self._last_grade_breakdown,
            )

        if context_field in self._state.context_requested:
            raw_reward, breakdown = compute_duplicate_action_penalty(context_field)
            self._consume_budget(f"{action.action_type} (duplicate)")
            step_reward, step_breakdown = self._apply_trajectory_delta(raw_reward, breakdown)
            return self._finalize_if_budget_exhausted(
                feedback=f"Warning: {context_field} was already retrieved. Action wasted.",
                step_reward=step_reward,
                step_breakdown=step_breakdown,
                error_code="duplicate_context_request",
            )

        available_context = self._scenario.get("available_context", {})
        context_value = available_context.get(context_field)
        raw_reward, raw_breakdown = compute_investigation_reward(
            context_field=context_field,
            ground_truth=self._scenario["ground_truth"],
            retrieved=context_value is not None,
        )

        if context_value is not None:
            setattr(self._gathered_context, context_field, context_value)
            feedback = f"Retrieved {context_field}."
        else:
            feedback = f"No data available for {context_field}."

        self._consume_budget(action.action_type)
        self._state.context_requested.append(context_field)
        raw_breakdown["retrieved"] = context_value is not None
        step_reward, step_breakdown = self._apply_trajectory_delta(raw_reward, raw_breakdown)

        return self._finalize_if_budget_exhausted(
            feedback=feedback,
            step_reward=step_reward,
            step_breakdown=step_breakdown,
            error_code=None,
        )

    def _handle_decision(self, action: ModerationAction) -> ModerationObservation:
        """Handle the terminal moderation decision."""
        if not all(
            [
                action.decision,
                action.primary_violation,
                action.severity,
                action.confidence is not None,
                action.key_factors is not None,
            ]
        ):
            raw_reward, raw_breakdown = compute_invalid_action_penalty(
                reason="missing_decision_fields",
                action_type=action.action_type,
            )
            self._consume_budget("decide (invalid)")
            step_reward, step_breakdown = self._apply_trajectory_delta(raw_reward, raw_breakdown)
            return self._finalize_if_budget_exhausted(
                feedback=(
                    "Invalid decision: missing required fields (decision, "
                    "primary_violation, severity, confidence, key_factors)."
                ),
                step_reward=step_reward,
                step_breakdown=step_breakdown,
                error_code="missing_decision_fields",
            )

        agent_decision = {
            "decision": action.decision,
            "primary_violation": action.primary_violation,
            "severity": action.severity,
            "confidence": action.confidence,
            "key_factors": action.key_factors,
        }
        ground_truth = self._scenario["ground_truth"]

        terminal_reward, reward_breakdown = compute_reward(
            agent_decision=agent_decision,
            ground_truth=ground_truth,
            actions_taken=self._state.actions_taken,
            difficulty=self._scenario["difficulty"],
        )
        applied_terminal_reward, cumulative_total = self._apply_episode_delta(terminal_reward)
        task_grade, grade_breakdown = compute_task_grade(
            agent_decision=agent_decision,
            ground_truth=ground_truth,
            actions_taken=self._state.actions_taken,
            difficulty=self._scenario["difficulty"],
        )

        self._state.decision_made = True
        self._state.done = True
        self._episode_done = True
        self._action_history.append(f"decide: {action.decision}")
        self._last_error_code = None
        self._state.last_error_code = None
        self._last_task_grade = task_grade
        self._last_grade_breakdown = grade_breakdown
        reward_breakdown["applied_score"] = applied_terminal_reward
        reward_breakdown["trajectory_total"] = self._trajectory_reward_total
        reward_breakdown["cumulative_total"] = cumulative_total
        self._last_reward_breakdown = reward_breakdown

        decision_grade = reward_breakdown.get("decision", {}).get("details", {}).get("grade", "unknown")
        feedback = (
            f"Decision recorded: {action.decision}. "
            f"Grade: {decision_grade}. Step reward: {applied_terminal_reward:.3f}. "
            f"Task grade: {task_grade:.3f}."
        )
        return self._build_observation(
            feedback=feedback,
            done=True,
            reward=applied_terminal_reward,
            error_code=None,
            reward_breakdown=reward_breakdown,
            task_grade=task_grade,
            grade_breakdown=grade_breakdown,
        )

    def _handle_invalid_action(self, action: ModerationAction) -> ModerationObservation:
        """Handle an invalid action type."""
        valid_actions = list(self.INVESTIGATION_ACTIONS) + ["decide"]
        raw_reward, raw_breakdown = compute_invalid_action_penalty(
            reason="invalid_action_type",
            action_type=action.action_type,
        )
        self._consume_budget(f"invalid:{action.action_type}")
        step_reward, step_breakdown = self._apply_trajectory_delta(raw_reward, raw_breakdown)
        return self._finalize_if_budget_exhausted(
            feedback=(
                f"Invalid action_type: '{action.action_type}'. "
                f"Valid types: {valid_actions}"
            ),
            step_reward=step_reward,
            step_breakdown=step_breakdown,
            error_code="invalid_action_type",
        )


ContentModerationEnvironment = SafeSpaceEnvironment
