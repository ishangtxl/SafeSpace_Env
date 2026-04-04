# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SafeSpace - Content Moderation RL Environment for OpenEnv.

SafeSpace is an RL environment where an AI agent acts as a content moderator,
investigating reported posts and making structured moderation decisions.

Example:
    >>> from content_moderation_env import SafeSpaceEnv, ModerationAction
    >>>
    >>> with SafeSpaceEnv(base_url="http://localhost:8000").sync() as client:
    ...     result = client.reset()
    ...     print(result.observation.content_item.text)
    ...
    ...     result = client.step(ModerationAction(
    ...         action_type="decide",
    ...         decision="remove",
    ...         primary_violation="5.1",
    ...         severity="high",
    ...         confidence=0.95,
    ...         key_factors=["spam_commercial"]
    ...     ))
    ...     print(f"Reward: {result.reward}")
"""

from .client import SafeSpaceEnv, ContentModerationEnv
from .models import (
    BreakdownComponent,
    ContentItem,
    GatheredContext,
    ModerationAction,
    ModerationObservation,
    ModerationState,
    RewardBreakdown,
    TaskGradeBreakdown,
    TriggerInfo,
)

__all__ = [
    "SafeSpaceEnv",
    "ContentModerationEnv",
    "BreakdownComponent",
    "ModerationAction",
    "ModerationObservation",
    "ModerationState",
    "RewardBreakdown",
    "TaskGradeBreakdown",
    "ContentItem",
    "TriggerInfo",
    "GatheredContext",
]
