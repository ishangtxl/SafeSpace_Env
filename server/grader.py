"""Deterministic grading logic for SafeSpace."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


PUBLIC_TASK_GRADE_EPSILON = 0.001

ADJACENT_DECISIONS: List[Tuple[str, str]] = [
    ("escalate", "warn"),
    ("warn", "escalate"),
    ("remove", "escalate"),
    ("escalate", "remove"),
]

DECISION_REWARD_SCORES = {
    "perfect": 0.55,
    "decision_and_violation_correct": 0.50,
    "decision_correct_only": 0.40,
    "adjacent_decision": 0.08,
    "dangerous_false_negative": -0.18,
    "false_negative": -0.05,
    "false_positive": -0.05,
    "wrong_decision": -0.10,
}

DECISION_GRADE_SCORES = {
    "perfect": 1.00,
    "decision_and_violation_correct": 0.91,
    "decision_correct_only": 0.73,
    "adjacent_decision": 0.15,
    "dangerous_false_negative": 0.00,
    "false_negative": 0.05,
    "false_positive": 0.08,
    "wrong_decision": 0.00,
}

EFFICIENCY_GRADE_TABLE = {
    0: 1.00,
    1: 0.85,
    2: 0.70,
    3: 0.55,
    4: 0.40,
    5: 0.20,
}


def clamp_public_task_grade(score: float | None) -> float:
    """Clamp a public task grade into the validator-safe open interval (0, 1)."""
    if score is None:
        return PUBLIC_TASK_GRADE_EPSILON

    bounded = max(0.0, min(1.0, float(score)))
    if bounded <= 0.0:
        return PUBLIC_TASK_GRADE_EPSILON
    if bounded >= 1.0:
        return 1.0 - PUBLIC_TASK_GRADE_EPSILON
    return bounded


def is_adjacent_decision(agent_decision: str, correct_decision: str) -> bool:
    """Check if two decisions are considered adjacent."""
    return (agent_decision, correct_decision) in ADJACENT_DECISIONS


def evaluate_decision(
    agent_decision: str,
    agent_violation: str,
    agent_severity: str,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a deterministic decision evaluation payload."""
    correct_decision = ground_truth["correct_decision"]
    correct_violation = ground_truth["primary_violation"]
    correct_severity = ground_truth["severity"]

    details = {
        "agent_decision": agent_decision,
        "correct_decision": correct_decision,
        "decision_match": agent_decision == correct_decision,
        "agent_violation": agent_violation,
        "correct_violation": correct_violation,
        "violation_match": agent_violation == correct_violation,
        "agent_severity": agent_severity,
        "correct_severity": correct_severity,
        "severity_match": agent_severity == correct_severity,
    }

    if (
        agent_decision == correct_decision
        and agent_violation == correct_violation
        and agent_severity == correct_severity
    ):
        details["grade"] = "perfect"
    elif agent_decision == correct_decision and agent_violation == correct_violation:
        details["grade"] = "decision_and_violation_correct"
    elif agent_decision == correct_decision:
        details["grade"] = "decision_correct_only"
    elif is_adjacent_decision(agent_decision, correct_decision):
        details["grade"] = "adjacent_decision"
    elif agent_decision == "approve" and correct_decision == "remove":
        if correct_severity in ["high", "critical"]:
            details["grade"] = "dangerous_false_negative"
        else:
            details["grade"] = "false_negative"
    elif agent_decision == "remove" and correct_decision == "approve":
        details["grade"] = "false_positive"
    else:
        details["grade"] = "wrong_decision"

    details["reward_score"] = DECISION_REWARD_SCORES[details["grade"]]
    details["normalized_score"] = DECISION_GRADE_SCORES[details["grade"]]
    details["eligible_for_efficiency"] = (
        agent_decision == correct_decision and agent_violation == correct_violation
    )
    details["decision_correct"] = agent_decision == correct_decision
    return details


def grade_decision(
    agent_decision: str,
    agent_violation: str,
    agent_severity: str,
    ground_truth: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Score the decision component for the reward function."""
    details = evaluate_decision(
        agent_decision=agent_decision,
        agent_violation=agent_violation,
        agent_severity=agent_severity,
        ground_truth=ground_truth,
    )
    return details["reward_score"], details


def compute_factor_jaccard(
    agent_factors: List[str],
    ground_truth_factors: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """Compute Jaccard similarity between agent and ground-truth factors."""
    agent_set: Set[str] = set(agent_factors) if agent_factors else set()
    truth_set: Set[str] = set(ground_truth_factors) if ground_truth_factors else set()

    if not agent_set and not truth_set:
        return 1.0, {
            "intersection": [],
            "union": [],
            "agent_only": [],
            "truth_only": [],
            "jaccard": 1.0,
        }

    intersection = agent_set & truth_set
    union = agent_set | truth_set
    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard, {
        "intersection": sorted(intersection),
        "union": sorted(union),
        "agent_only": sorted(agent_set - truth_set),
        "truth_only": sorted(truth_set - agent_set),
        "jaccard": jaccard,
    }


def grade_factors(
    agent_factors: List[str],
    ground_truth: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Return the reward component for factor overlap."""
    truth_factors = ground_truth.get("key_factors", [])
    jaccard, details = compute_factor_jaccard(agent_factors, truth_factors)
    score = jaccard * 0.15
    details["factor_score"] = score
    return score, details


def compute_efficiency_grade(actions_taken: int, eligible_for_efficiency: bool) -> float:
    """Normalized efficiency grade used by the task grader."""
    if not eligible_for_efficiency:
        return 0.0
    return EFFICIENCY_GRADE_TABLE.get(actions_taken, 0.0)


def compute_calibration_grade(
    confidence: float,
    agent_decision: str,
    correct_decision: str,
    difficulty: str,
) -> Tuple[float, Dict[str, Any]]:
    """Deterministic normalized calibration grade in [0, 1]."""
    is_correct = agent_decision == correct_decision
    details = {
        "confidence": confidence,
        "agent_decision": agent_decision,
        "correct_decision": correct_decision,
        "difficulty": difficulty,
        "is_correct": is_correct,
    }

    if not is_correct:
        if confidence >= 0.85:
            details["calibration_type"] = "overconfident_wrong"
            return 0.0, details
        if confidence >= 0.60:
            details["calibration_type"] = "confident_wrong"
            return 0.05, details
        details["calibration_type"] = "cautious_wrong"
        return 0.02, details

    if agent_decision in {"warn", "escalate"} and difficulty == "hard":
        if confidence <= 0.55:
            details["calibration_type"] = "appropriate_uncertainty"
            return 1.0, details
        details["calibration_type"] = "too_confident_escalation"
        return 0.55, details

    if confidence >= 0.85:
        details["calibration_type"] = "high_confidence_correct"
        return 1.0, details
    if confidence >= 0.6:
        details["calibration_type"] = "medium_confidence_correct"
        return 0.7, details
    if difficulty == "easy":
        details["calibration_type"] = "underconfident_easy"
        return 0.35, details

    details["calibration_type"] = "cautious_correct"
    return 0.55, details


def compute_task_grade(
    agent_decision: Dict[str, Any],
    ground_truth: Dict[str, Any],
    actions_taken: int,
    difficulty: str,
) -> Tuple[float, Dict[str, Any]]:
    """Return the public normalized episode grade in the open interval (0, 1)."""
    decision_details = evaluate_decision(
        agent_decision=agent_decision.get("decision", ""),
        agent_violation=agent_decision.get("primary_violation", ""),
        agent_severity=agent_decision.get("severity", ""),
        ground_truth=ground_truth,
    )
    factor_jaccard, factor_details = compute_factor_jaccard(
        agent_decision.get("key_factors", []),
        ground_truth.get("key_factors", []),
    )
    efficiency_grade = compute_efficiency_grade(
        actions_taken,
        decision_details["eligible_for_efficiency"],
    )
    calibration_grade, calibration_details = compute_calibration_grade(
        confidence=float(agent_decision.get("confidence", 0.0)),
        agent_decision=agent_decision.get("decision", ""),
        correct_decision=ground_truth.get("correct_decision", ""),
        difficulty=difficulty,
    )

    raw_total = (
        0.70 * decision_details["normalized_score"]
        + 0.15 * factor_jaccard
        + 0.05 * efficiency_grade
        + 0.10 * calibration_grade
    )
    total = clamp_public_task_grade(raw_total)

    breakdown = {
        "decision": {
            "weight": 0.70,
            "score": decision_details["normalized_score"],
            "details": decision_details,
        },
        "factor_overlap": {
            "weight": 0.15,
            "score": factor_jaccard,
            "details": factor_details,
        },
        "efficiency": {
            "weight": 0.05,
            "score": efficiency_grade,
            "details": {
                "actions_taken": actions_taken,
                "eligible_for_efficiency": decision_details["eligible_for_efficiency"],
            },
        },
        "calibration": {
            "weight": 0.10,
            "score": calibration_grade,
            "details": calibration_details,
        },
        "raw_total": raw_total,
        "total": total,
    }
    if total != raw_total:
        breakdown["public_total_adjustment"] = total - raw_total
    return total, breakdown
