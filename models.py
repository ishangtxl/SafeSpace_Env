# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SafeSpace Content Moderation Environment.

SafeSpace is an RL environment where an AI agent acts as a content moderator,
investigating reported posts and making structured moderation decisions.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

ActionType = Literal[
    "request_author_profile",
    "request_author_violations",
    "request_thread_context",
    "request_community_rules",
    "request_linked_content",
    "request_similar_precedents",
    "request_reporter_credibility",
    "decide",
]
DecisionType = Literal["approve", "remove", "escalate", "warn"]
SeverityType = Literal["none", "low", "medium", "high", "critical"]
TriggerType = Literal["user_report", "auto_flag", "appeal", "proactive_audit"]
MediaType = Literal["text", "text+image", "text+link"]
DifficultyType = Literal["easy", "medium", "hard"]


# ============================================================================
# Supporting Models (nested in Observation)
# ============================================================================


class ContentItem(BaseModel):
    """A content item (post) that needs moderation review."""

    post_id: str = Field(..., description="Unique identifier for the post")
    text: str = Field(..., description="The text content of the post")
    author_id: str = Field(..., description="Unique identifier of the author")
    community: str = Field(
        ..., description="Community where the post was made (e.g., 'gaming', 'health')"
    )
    timestamp: str = Field(..., description="ISO timestamp when the post was created")
    media_type: MediaType = Field(
        ..., description="Type of media: 'text', 'text+image', or 'text+link'"
    )
    media_description: Optional[str] = Field(
        default=None, description="Text description of image/link if present"
    )


class TriggerInfo(BaseModel):
    """How this content entered the moderation queue."""

    trigger_type: TriggerType = Field(
        ...,
        description="One of: 'user_report', 'auto_flag', 'appeal', 'proactive_audit'",
    )
    # For user_report
    report_count: int = Field(default=0, description="Number of reports received")
    report_categories: List[str] = Field(
        default_factory=list, description="Categories selected by reporters"
    )
    sample_report_reason: Optional[str] = Field(
        default=None, description="Example report reason from a user"
    )
    # For auto_flag
    auto_flag_reason: Optional[str] = Field(
        default=None, description="Why automated system flagged this content"
    )
    # For appeal
    original_decision: Optional[str] = Field(
        default=None, description="The original moderation decision being appealed"
    )
    appeal_text: Optional[str] = Field(
        default=None, description="User's appeal message"
    )
    # For proactive_audit
    audit_reason: Optional[str] = Field(
        default=None, description="Why this content was selected for audit"
    )


class GatheredContext(BaseModel):
    """Context gathered through investigation actions. Starts empty."""

    author_profile: Optional[Dict[str, Any]] = Field(
        default=None, description="Author's bio, account age, follower count"
    )
    author_violations: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Author's past moderation violations"
    )
    thread_context: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Full conversation thread"
    )
    community_rules: Optional[str] = Field(
        default=None, description="Community-specific moderation guidelines"
    )
    linked_content_summary: Optional[str] = Field(
        default=None, description="What the linked content contains"
    )
    similar_precedents: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="How similar posts were moderated before"
    )
    reporter_credibility: Optional[Dict[str, Any]] = Field(
        default=None, description="Reporter's history of accurate vs false reports"
    )


class BreakdownComponent(BaseModel):
    """Typed reward or grading component with room for structured details."""

    model_config = ConfigDict(extra="allow")

    score: Optional[float] = Field(default=None, description="Component score")
    max: Optional[float] = Field(default=None, description="Maximum component score")
    min: Optional[float] = Field(default=None, description="Minimum component score")
    raw_score: Optional[float] = Field(
        default=None, description="Raw component score before normalization"
    )
    raw_max: Optional[float] = Field(
        default=None, description="Raw maximum component score before normalization"
    )
    raw_min: Optional[float] = Field(
        default=None, description="Raw minimum component score before normalization"
    )
    weight: Optional[float] = Field(
        default=None, description="Normalized weighting used by the task grade"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details for the component calculation",
    )


class RewardBreakdown(BaseModel):
    """Typed reward breakdown returned on reset, intermediate, and terminal steps."""

    model_config = ConfigDict(extra="allow")

    reward_type: str = Field(default="unknown", description="Reward breakdown category")
    total: float = Field(default=0.0, description="Total reward for this step")
    raw_total: Optional[float] = Field(
        default=None, description="Raw total reward for this step before normalization"
    )
    score: Optional[float] = Field(
        default=None, description="Normalized score for simple cases"
    )
    raw_score: Optional[float] = Field(
        default=None, description="Raw score for simple cases before normalization"
    )
    requested_score: Optional[float] = Field(
        default=None, description="Normalized uncapped score requested by the reward rule"
    )
    raw_requested_score: Optional[float] = Field(
        default=None,
        description="Raw uncapped score requested by the reward rule before normalization",
    )
    applied_score: Optional[float] = Field(
        default=None, description="Normalized score applied after caps or bounds"
    )
    raw_applied_score: Optional[float] = Field(
        default=None,
        description="Raw score applied after caps or bounds before normalization",
    )
    step_total: Optional[float] = Field(
        default=None, description="Normalized combined step reward in multi-part terminal cases"
    )
    raw_step_total: Optional[float] = Field(
        default=None,
        description="Raw combined step reward in multi-part terminal cases before normalization",
    )
    trajectory_total: Optional[float] = Field(
        default=None, description="Normalized cumulative trajectory shaping reward"
    )
    raw_trajectory_total: Optional[float] = Field(
        default=None,
        description="Raw cumulative trajectory shaping reward before normalization",
    )
    episode_total: Optional[float] = Field(
        default=None, description="Normalized running episode reward after this step"
    )
    raw_episode_total: Optional[float] = Field(
        default=None,
        description="Raw running episode reward after this step before normalization",
    )
    cumulative_total: Optional[float] = Field(
        default=None,
        description="Normalized episode reward total after terminal application",
    )
    raw_cumulative_total: Optional[float] = Field(
        default=None,
        description="Raw episode reward total after terminal application before normalization",
    )
    theoretical_terminal_max: Optional[float] = Field(
        default=None, description="Normalized maximum possible terminal reward"
    )
    theoretical_terminal_min: Optional[float] = Field(
        default=None, description="Normalized minimum possible terminal reward"
    )
    raw_theoretical_terminal_max: Optional[float] = Field(
        default=None,
        description="Raw maximum possible terminal reward before normalization",
    )
    raw_theoretical_terminal_min: Optional[float] = Field(
        default=None,
        description="Raw minimum possible terminal reward before normalization",
    )
    context_field: Optional[str] = Field(
        default=None, description="Context source involved in the reward"
    )
    context_needed: List[str] = Field(
        default_factory=list, description="Ground-truth context sources needed"
    )
    is_needed: Optional[bool] = Field(
        default=None, description="Whether the requested context was useful"
    )
    retrieved: Optional[bool] = Field(
        default=None, description="Whether the context source had retrievable data"
    )
    reason: Optional[str] = Field(default=None, description="Machine-readable reason")
    action_type: Optional[str] = Field(
        default=None, description="Action type involved in the reward"
    )
    trajectory_cap: Optional[float] = Field(
        default=None, description="Trajectory reward cap in effect"
    )
    decision: Optional[BreakdownComponent] = Field(
        default=None, description="Decision-scoring component"
    )
    factor: Optional[BreakdownComponent] = Field(
        default=None, description="Factor overlap component"
    )
    efficiency: Optional[BreakdownComponent] = Field(
        default=None, description="Efficiency component"
    )
    calibration: Optional[BreakdownComponent] = Field(
        default=None, description="Calibration component"
    )
    trajectory: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Nested trajectory reward payload for no-decision terminal cases",
    )
    no_decision: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Nested no-decision penalty payload when the budget is exhausted",
    )
    last_terminal_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Previous terminal reward payload when guarding completed episodes",
    )


class TaskGradeBreakdown(BaseModel):
    """Typed normalized grader breakdown returned on terminal steps."""

    model_config = ConfigDict(extra="allow")

    decision: Optional[BreakdownComponent] = Field(
        default=None, description="Decision grading component"
    )
    factor_overlap: Optional[BreakdownComponent] = Field(
        default=None, description="Factor-overlap grading component"
    )
    efficiency: Optional[BreakdownComponent] = Field(
        default=None, description="Efficiency grading component"
    )
    calibration: Optional[BreakdownComponent] = Field(
        default=None, description="Calibration grading component"
    )
    total: float = Field(default=0.0, description="Normalized task grade in [0, 1]")


# ============================================================================
# Core OpenEnv Models
# ============================================================================


class ModerationAction(Action):
    """
    Action to be executed in the SafeSpace environment.

    Investigation actions (cost 1 action each):
    - request_author_profile
    - request_author_violations
    - request_thread_context
    - request_community_rules
    - request_linked_content
    - request_similar_precedents
    - request_reporter_credibility

    Terminal action:
    - decide (requires decision fields)
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "One of: 'request_author_profile', 'request_author_violations', "
            "'request_thread_context', 'request_community_rules', "
            "'request_linked_content', 'request_similar_precedents', "
            "'request_reporter_credibility', 'decide'"
        ),
    )

    # === Decision fields (required ONLY when action_type == "decide") ===

    decision: Optional[DecisionType] = Field(
        default=None,
        description="One of: 'approve', 'remove', 'escalate', 'warn'",
    )
    primary_violation: Optional[str] = Field(
        default=None,
        description="Policy section ID (e.g., '1.0', '2.1', '3.1') or 'none'",
    )
    severity: Optional[SeverityType] = Field(
        default=None,
        description="One of: 'none', 'low', 'medium', 'high', 'critical'",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the decision (0.0 to 1.0)",
    )
    key_factors: Optional[List[str]] = Field(
        default=None,
        description="Selected factors from the FACTOR_LIST that influenced the decision",
    )


class ModerationObservation(Observation):
    """
    Observation returned from the SafeSpace environment.

    Contains the content to moderate, trigger information, gathered context,
    platform policy, and episode progress.
    """

    # Content and trigger info
    content_item: Optional[ContentItem] = Field(
        default=None, description="The content item being moderated"
    )
    trigger_info: Optional[TriggerInfo] = Field(
        default=None, description="How this content entered the moderation queue"
    )

    # Investigation results (populated as agent gathers context)
    gathered_context: GatheredContext = Field(
        default_factory=GatheredContext,
        description="Context gathered through investigation actions",
    )

    # Policy and factors
    platform_policy: str = Field(
        default="", description="The platform's content moderation policy document"
    )
    available_factors: List[str] = Field(
        default_factory=list,
        description="List of factors the agent can cite in its decision",
    )

    # Episode progress
    actions_taken: int = Field(
        default=0, description="Number of actions taken this episode"
    )
    max_actions: int = Field(
        default=8, description="Maximum actions allowed per episode"
    )
    action_history: List[str] = Field(
        default_factory=list, description="List of actions taken so far"
    )
    feedback: str = Field(
        default="", description="Feedback message from the last action"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Structured error code for invalid or rejected actions",
    )

    # Reward breakdown (populated after terminal decision)
    reward_breakdown: Optional[RewardBreakdown] = Field(
        default=None,
        description="Breakdown of reward components for the last step",
    )
    task_grade: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deterministic normalized task grade for the current episode",
    )
    grade_breakdown: Optional[TaskGradeBreakdown] = Field(
        default=None,
        description="Breakdown of normalized task-grade components",
    )


class ModerationState(State):
    """
    State of the SafeSpace environment.

    Tracks episode metadata and progress.
    """

    # Override base State fields
    episode_id: Optional[str] = Field(
        default=None, description="Unique identifier for this episode"
    )
    step_count: int = Field(default=0, description="Number of steps taken")

    # Episode identification
    scenario_id: Optional[str] = Field(
        default=None, description="Current scenario ID"
    )
    task_id: Optional[str] = Field(
        default=None, description="Task ID used to load this scenario"
    )
    difficulty: Optional[DifficultyType] = Field(
        default=None, description="Scenario difficulty: easy, medium, or hard"
    )
    trigger_type: Optional[TriggerType] = Field(
        default=None, description="How this content entered the moderation queue"
    )

    # SafeSpace-specific public progress fields
    actions_taken: int = Field(
        default=0, description="Number of investigation actions taken"
    )
    max_actions: int = Field(
        default=8, description="Maximum actions allowed per episode"
    )
    context_requested: List[str] = Field(
        default_factory=list, description="List of context types requested"
    )
    decision_made: bool = Field(
        default=False, description="Whether a terminal decision has been made"
    )
    episode_reward: float = Field(
        default=0.0, description="Normalized total reward for episode"
    )
    raw_episode_reward: float = Field(
        default=0.0, description="Raw total reward for episode before normalization"
    )
    done: bool = Field(default=False, description="Whether the episode is terminal")
    last_error_code: Optional[str] = Field(
        default=None,
        description="Structured error code from the most recent rejected action",
    )
