"""SafeSpace Content Moderation environment server components."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SafeSpaceEnvironment",
    "ContentModerationEnvironment",
    "FACTOR_LIST",
    "PLATFORM_POLICY",
    "POLICY_SECTIONS",
    "grade_decision",
    "grade_factors",
    "compute_reward",
    "load_scenario",
    "get_all_scenarios",
]


def __getattr__(name: str) -> Any:
    """Lazily expose server helpers without importing the full runtime eagerly."""
    if name in {"SafeSpaceEnvironment", "ContentModerationEnvironment"}:
        from .environment import ContentModerationEnvironment, SafeSpaceEnvironment

        return {
            "SafeSpaceEnvironment": SafeSpaceEnvironment,
            "ContentModerationEnvironment": ContentModerationEnvironment,
        }[name]

    if name in {"FACTOR_LIST", "PLATFORM_POLICY", "POLICY_SECTIONS"}:
        from .policy import FACTOR_LIST, PLATFORM_POLICY, POLICY_SECTIONS

        return {
            "FACTOR_LIST": FACTOR_LIST,
            "PLATFORM_POLICY": PLATFORM_POLICY,
            "POLICY_SECTIONS": POLICY_SECTIONS,
        }[name]

    if name in {"grade_decision", "grade_factors"}:
        from .grader import grade_decision, grade_factors

        return {
            "grade_decision": grade_decision,
            "grade_factors": grade_factors,
        }[name]

    if name == "compute_reward":
        from .reward import compute_reward

        return compute_reward

    if name in {"load_scenario", "get_all_scenarios"}:
        from .scenarios import get_all_scenarios, load_scenario

        return {
            "load_scenario": load_scenario,
            "get_all_scenarios": get_all_scenarios,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
