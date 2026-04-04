"""Reward system for SafeSpace."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .grader import grade_decision, grade_factors


INVESTIGATION_REWARD = 0.05
IRRELEVANT_CONTEXT_PENALTY = -0.03
DUPLICATE_ACTION_PENALTY = -0.05
INVALID_ACTION_PENALTY = -0.06
NO_DECISION_PENALTY = -0.15
TRAJECTORY_REWARD_CAP = 0.15
MIN_EPISODE_REWARD = -0.25
MAX_EPISODE_REWARD = 1.00

EFFICIENCY_BONUS_TABLE = {
    0: 0.10,
    1: 0.08,
    2: 0.06,
    3: 0.04,
    4: 0.02,
    5: 0.01,
}


def compute_investigation_reward(
    context_field: str,
    ground_truth: Dict[str, Any],
    *,
    retrieved: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Score a context-gathering action."""
    context_needed = ground_truth.get("context_needed", [])
    is_needed = context_field in context_needed
    score = (
        INVESTIGATION_REWARD
        if is_needed and retrieved
        else IRRELEVANT_CONTEXT_PENALTY
    )
    if is_needed and retrieved:
        reason = "needed_context_retrieved"
    elif is_needed:
        reason = "needed_context_unavailable"
    else:
        reason = "irrelevant_context"

    return score, {
        "reward_type": "trajectory",
        "context_field": context_field,
        "context_needed": context_needed,
        "is_needed": is_needed,
        "retrieved": retrieved,
        "reason": reason,
        "score": score,
        "trajectory_cap": TRAJECTORY_REWARD_CAP,
    }


def compute_duplicate_action_penalty(context_field: str) -> Tuple[float, Dict[str, Any]]:
    """Penalty for repeating an already-seen investigation action."""
    return DUPLICATE_ACTION_PENALTY, {
        "reward_type": "trajectory",
        "reason": "duplicate_context_request",
        "context_field": context_field,
        "score": DUPLICATE_ACTION_PENALTY,
        "trajectory_cap": TRAJECTORY_REWARD_CAP,
    }


def compute_invalid_action_penalty(
    reason: str,
    action_type: str,
) -> Tuple[float, Dict[str, Any]]:
    """Penalty for malformed or unsupported actions."""
    return INVALID_ACTION_PENALTY, {
        "reward_type": "trajectory",
        "reason": reason,
        "action_type": action_type,
        "score": INVALID_ACTION_PENALTY,
        "trajectory_cap": TRAJECTORY_REWARD_CAP,
    }


def compute_efficiency_bonus(
    actions_taken: int,
    eligible_for_efficiency: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Bonus for correct, efficient investigation."""
    details = {
        "actions_taken": actions_taken,
        "eligible_for_efficiency": eligible_for_efficiency,
    }
    if not eligible_for_efficiency:
        details["reason"] = "requires_correct_decision_and_violation"
        return 0.0, details

    bonus = EFFICIENCY_BONUS_TABLE.get(actions_taken, 0.0)
    details["efficiency_bonus"] = bonus
    return bonus, details


def compute_calibration_bonus(
    confidence: float,
    agent_decision: str,
    correct_decision: str,
    difficulty: str,
) -> Tuple[float, Dict[str, Any]]:
    """Bonus or penalty for well-calibrated confidence."""
    is_correct = agent_decision == correct_decision
    details = {
        "confidence": confidence,
        "is_correct": is_correct,
        "difficulty": difficulty,
        "agent_decision": agent_decision,
        "correct_decision": correct_decision,
    }

    if not is_correct:
        if confidence >= 0.85:
            details["calibration_type"] = "overconfident_wrong"
            return -0.10, details
        if confidence >= 0.70:
            details["calibration_type"] = "confident_wrong"
            return -0.05, details
        details["calibration_type"] = "wrong"
        return 0.0, details

    if agent_decision in {"warn", "escalate"} and difficulty == "hard":
        if confidence <= 0.55:
            details["calibration_type"] = "appropriate_uncertainty"
            return 0.08, details
        details["calibration_type"] = "too_confident_escalation"
        return 0.02, details

    if confidence >= 0.85:
        details["calibration_type"] = "high_confidence_correct"
        return 0.10, details
    if confidence >= 0.60:
        details["calibration_type"] = "medium_confidence_correct"
        return 0.04, details
    if difficulty == "easy":
        details["calibration_type"] = "underconfident_easy"
        return 0.0, details

    details["calibration_type"] = "cautious_correct"
    return 0.02, details


def compute_reward(
    agent_decision: Dict[str, Any],
    ground_truth: Dict[str, Any],
    actions_taken: int,
    difficulty: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute the terminal reward components for a final moderation decision."""
    breakdown: Dict[str, Any] = {}

    decision_score, decision_details = grade_decision(
        agent_decision=agent_decision.get("decision", ""),
        agent_violation=agent_decision.get("primary_violation", ""),
        agent_severity=agent_decision.get("severity", ""),
        ground_truth=ground_truth,
    )
    breakdown["decision"] = {
        "score": decision_score,
        "max": 0.55,
        "min": -0.18,
        "details": decision_details,
    }

    factor_score, factor_details = grade_factors(
        agent_factors=agent_decision.get("key_factors", []),
        ground_truth=ground_truth,
    )
    breakdown["factor"] = {
        "score": factor_score,
        "max": 0.15,
        "details": factor_details,
    }

    efficiency_score, efficiency_details = compute_efficiency_bonus(
        actions_taken=actions_taken,
        eligible_for_efficiency=decision_details["eligible_for_efficiency"],
    )
    breakdown["efficiency"] = {
        "score": efficiency_score,
        "max": 0.10,
        "details": efficiency_details,
    }

    calibration_score, calibration_details = compute_calibration_bonus(
        confidence=float(agent_decision.get("confidence", 0.0)),
        agent_decision=agent_decision.get("decision", ""),
        correct_decision=ground_truth.get("correct_decision", ""),
        difficulty=difficulty,
    )
    breakdown["calibration"] = {
        "score": calibration_score,
        "max": 0.10,
        "min": -0.10,
        "details": calibration_details,
    }

    total = decision_score + factor_score + efficiency_score + calibration_score
    total = max(MIN_EPISODE_REWARD, min(0.90, total))

    breakdown["reward_type"] = "terminal"
    breakdown["total"] = total
    breakdown["theoretical_terminal_max"] = 0.90
    breakdown["theoretical_terminal_min"] = MIN_EPISODE_REWARD
    return total, breakdown


def compute_no_decision_penalty(actions_taken: int) -> Tuple[float, Dict[str, Any]]:
    """Reward payload when the action budget expires without a final decision."""
    return NO_DECISION_PENALTY, {
        "reward_type": "trajectory_terminal",
        "reason": "no_decision_made",
        "actions_taken": actions_taken,
        "decision": {"score": 0.0, "max": 0.55},
        "factor": {"score": 0.0, "max": 0.15},
        "efficiency": {"score": 0.0, "max": 0.10},
        "calibration": {"score": 0.0, "max": 0.10},
        "penalty": NO_DECISION_PENALTY,
        "total": NO_DECISION_PENALTY,
    }
