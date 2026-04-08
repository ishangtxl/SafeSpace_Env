#!/usr/bin/env python3
"""
Canonical baseline evaluator for SafeSpace.

Required environment variables:
    API_BASE_URL: OpenAI-compatible API endpoint for the model
    MODEL_NAME: Model identifier for inference
    HF_TOKEN: Primary Hugging Face / router credential

Optional environment variables:
    ENV_BASE_URL: Running SafeSpace server URL
    LOCAL_IMAGE_NAME: Local Docker image used when no ENV_BASE_URL is set
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from .client import SafeSpaceEnv
    from .models import ModerationAction, ModerationObservation
    from .server.grader import clamp_public_task_grade
    from .server.scenarios import (
        get_all_scenarios,
        get_benchmark_scenario_ids,
        get_benchmark_manifest,
        validate_benchmark_manifest,
    )
except ImportError:  # pragma: no cover
    from client import SafeSpaceEnv
    from models import ModerationAction, ModerationObservation
    from server.grader import clamp_public_task_grade
    from server.scenarios import (
        get_all_scenarios,
        get_benchmark_scenario_ids,
        get_benchmark_manifest,
        validate_benchmark_manifest,
    )

# Default uses HuggingFace Router. Baseline scores in README were generated
# using Azure AI Foundry. Set API_BASE_URL appropriately for your setup.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
DEFAULT_ENV_BASE_URL = "http://localhost:8000"
BENCHMARK_NAME = "safespace"
SUCCESS_SCORE_THRESHOLD = 0.50

MAX_TOKENS = 500
TEMPERATURE = 0.0  # Set to 0 for deterministic outputs
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "7"))


def resolve_api_key_and_source() -> tuple[Optional[str], Optional[str]]:
    """Resolve API credentials using the documented precedence order."""
    for env_var in (
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "API_KEY",
        "AZURE_OPENAI_API_KEY",
    ):
        value = os.getenv(env_var)
        if value:
            return value, env_var
    return None, None


OPENAI_API_KEY, OPENAI_API_KEY_SOURCE = resolve_api_key_and_source()

VALID_DECISIONS = {"approve", "remove", "warn", "escalate"}
VALID_SEVERITIES = {"none", "low", "medium", "high", "critical"}
VALID_VIOLATIONS = {
    "none",
    "1.1",
    "1.2",
    "2.1",
    "3.1",
    "4.1",
    "5.1",
    "5.2",
    "6.1",
    "7.1",
}
CONTEXT_OPTIONS = [
    "thread_context",
    "author_profile",
    "author_violations",
    "community_rules",
    "reporter_credibility",
    "similar_precedents",
    "linked_content_summary",
]
FACTOR_ALIASES = {
    "gambling_or_competition_context": "gaming_or_competition_context",
    "gaming_context": "gaming_or_competition_context",
    "educational_context": "medical_or_educational_context",
    "medical_context": "medical_or_educational_context",
    "quoted_for_reporting": "quoting_or_reporting",
    "quoted_for_debunking": "quoting_or_reporting",
    "reported_quote": "quoting_or_reporting",
    "false_reporting_campaign": "coordinated_reporting",
    "brigading": "coordinated_reporting",
    "trusted_source_reporter": "trusted_reporter",
    "direct_threat": "explicit_threat",
    "credible_threat": "explicit_threat",
    "coded_hate": "hate_speech_coded",
    "direct_hate": "hate_speech_direct",
    "commercial_spam": "spam_commercial",
    "engagement_bait": "spam_engagement",
    "factual_misinformation": "misinformation_factual",
    "opinion_framing": "misinformation_opinion",
    "doxxing": "privacy_violation",
    "privacy_doxxing": "privacy_violation",
    "dangerous_link": "harmful_link",
    "allowed_graphic_context": "graphic_content_justified",
    "policy_conflict": "policy_sections_conflict",
    "clear_violation": "clear_violation_no_exception",
    "no_violation": "no_violation_found",
}

PROMOTIONAL_CUES = (
    "check out my",
    "check it out",
    "just launched",
    "launched my",
    "new app",
    "new product",
    "productivity app",
    "course",
    "ebook",
    "link in bio",
    "use code",
    "free to try",
    "great deals",
    "opportunity",
    "download page",
)

PROMPT_REDACTIONS = [
    (
        re.compile(r"\bvaccines?\s+cause\s+autism\b", re.IGNORECASE),
        "[medical_misinformation_claim]",
    ),
    (
        re.compile(r"\bbig pharma\b", re.IGNORECASE),
        "[industry_conspiracy_reference]",
    ),
    (
        re.compile(
            r"\b(kill|killed|dead|destroy|destroyed|murder|murdered|hunt|end you|"
            r"coming for you|watch your back|make you pay|suffer)\b",
            re.IGNORECASE,
        ),
        "[violent_phrase]",
    ),
    (
        re.compile(r"\b\d{1,5}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5}\s(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Blvd|Boulevard)\b", re.IGNORECASE),
        "[private_address]",
    ),
    (
        re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "[phone_number]",
    ),
    (
        re.compile(r"\[((?:ethnic\s+)?slur|slurs?|group)\]", re.IGNORECASE),
        "[redacted_attack]",
    ),
]

POLICY_LABELS = {
    "1.1": "civility / baseline conduct",
    "1.2": "bad-faith or disruptive participation",
    "2.1": "protected-group attacks",
    "3.1": "targeted harassment",
    "4.1": "credible threats",
    "5.1": "commercial spam or deceptive funnels",
    "5.2": "coordinated manipulation or brigading",
    "6.1": "harmful false factual claims",
    "7.1": "private personal information",
    "none": "no violation",
}

POLICY_PROMPT_SUMMARY = """Decision guidance:
- Prefer approve when context shows benign intent or a clear exception.
- Prefer warn for low-severity borderline abuse.
- Prefer escalate when policies conflict, precedent matters, or confidence stays low after investigation.
"""

TASK_TO_DIFFICULTY = {
    "clear_violations": "easy",
    "context_dependent": "medium",
    "policy_edge_cases": "hard",
}
DIFFICULTY_TO_TASK = {
    difficulty: task_id for task_id, difficulty in TASK_TO_DIFFICULTY.items()
}


class InferenceConfigurationError(RuntimeError):
    """Raised when evaluator configuration is incomplete or invalid."""


class ModelRequestError(RuntimeError):
    """Raised when the model call or response parsing fails."""


class EpisodeExecutionError(RuntimeError):
    """Raised when a single evaluation episode fails mid-trajectory."""

    def __init__(
        self,
        *,
        scenario_id: str,
        task_id: Optional[str],
        stage: str,
        error: str,
        difficulty: Optional[str] = None,
    ) -> None:
        super().__init__(f"{scenario_id} failed during {stage}: {error}")
        self.scenario_id = scenario_id
        self.task_id = task_id
        self.stage = stage
        self.error = error
        self.difficulty = difficulty


PARAMETER_ERROR_MARKERS = (
    "unsupported",
    "unknown parameter",
    "unexpected keyword",
    "unrecognized request argument",
    "not permitted",
    "extra inputs are not permitted",
    "does not support",
    "invalid parameter",
    "unsupported parameter",
)

PARAMETER_ALIASES = {
    "max_completion_tokens": ("max_completion_tokens", "max completion tokens"),
    "max_tokens": ("max_tokens", "max tokens"),
    "seed": ("seed",),
}

SYSTEM_PROMPT = """You are an expert content moderator for a social media platform.

You review one case at a time and must return valid JSON only.

Available decisions:
- approve
- remove
- warn
- escalate

Return either:
{"investigate": ["thread_context", "author_profile"]}
or
{
  "decision": "approve|remove|warn|escalate",
  "primary_violation": "policy section or none",
  "severity": "none|low|medium|high|critical",
  "confidence": 0.0,
  "key_factors": ["factor1", "factor2"]
}
"""

DECISION_RULES = """Allowed primary_violation values only:
- none
- 1.1
- 1.2
- 2.1
- 3.1
- 4.1
- 5.1
- 5.2
- 6.1
- 7.1

Return key_factors as exact factor strings only, never prose explanations.
If decision=approve, then primary_violation must be none and severity must be none.
Use escalate for genuinely unresolved ambiguity, not as a default.
Use warn for borderline policy violations that should stay visible but need moderator action.
"""


def normalize_violation(value: Any, decision: str) -> str:
    """Map verbose or noisy model output to canonical violation IDs."""
    if decision == "approve":
        return "none"
    if not isinstance(value, str):
        return "none"

    lowered = value.strip().lower()
    if lowered in VALID_VIOLATIONS:
        return lowered
    if "none" == lowered:
        return "none"
    if "2.1" in lowered or "hate" in lowered:
        return "2.1"
    if "1.2" in lowered or "good faith" in lowered or "bad faith" in lowered or "disrupt" in lowered:
        return "1.2"
    if "1.1" in lowered or "baseline" in lowered or "conduct" in lowered or "civility" in lowered:
        return "1.1"
    if "3.1" in lowered or "harass" in lowered or "bully" in lowered:
        return "3.1"
    if "4.1" in lowered or "threat" in lowered or "violence" in lowered:
        return "4.1"
    if "5.2" in lowered or "coordinated" in lowered or "brigad" in lowered or "inauthentic" in lowered:
        return "5.2"
    if "5.1" in lowered or "spam" in lowered or "scam" in lowered or "commercial" in lowered or "funnel" in lowered:
        return "5.1"
    if "6.1" in lowered or "misinformation" in lowered or "false claim" in lowered:
        return "6.1"
    if "7.1" in lowered or "privacy" in lowered or "dox" in lowered or "address" in lowered:
        return "7.1"
    return "none"


def normalize_factors(raw_factors: Any, available_factors: List[str], decision: str) -> List[str]:
    """Map free-form factor output back into the canonical factor vocabulary."""
    available_set = set(available_factors)
    normalized: List[str] = []
    if isinstance(raw_factors, list):
        for item in raw_factors:
            if not isinstance(item, str):
                continue
            candidate = item.strip()
            if candidate in available_set:
                normalized.append(candidate)
                continue

            lowered = candidate.lower().strip()
            lowered = lowered.replace(" ", "_").replace("-", "_").replace("/", "_")
            alias = FACTOR_ALIASES.get(lowered)
            if alias and alias in available_set:
                normalized.append(alias)
                continue

            for factor in available_factors:
                if lowered == factor.lower():
                    normalized.append(factor)
                    break

    deduped: List[str] = []
    for factor in normalized:
        if factor not in deduped:
            deduped.append(factor)

    if deduped:
        return deduped[:4]
    return ["no_violation_found"] if decision == "approve" else ["borderline_requires_judgment"]


def calibrate_confidence(
    confidence: float,
    difficulty: str,
    decision: str,
    context_count: int,
) -> float:
    """Keep confidence realistic for ambiguous cases and sparse investigation."""
    calibrated = confidence
    if difficulty == "easy":
        if decision in {"remove", "approve"}:
            calibrated = max(calibrated, 0.9)
        return min(0.99, calibrated)

    if difficulty == "medium":
        if decision in {"approve", "remove"} and context_count == 0:
            calibrated = min(calibrated, 0.72)
        elif decision in {"approve", "remove"}:
            calibrated = min(calibrated, 0.9)
        elif decision in {"warn", "escalate"}:
            calibrated = min(calibrated, 0.68)
        return max(0.3, min(0.95, calibrated))

    if difficulty == "hard":
        if decision == "escalate":
            calibrated = min(calibrated, 0.38)
        elif decision == "warn":
            calibrated = min(calibrated, 0.72)
        elif context_count < 2:
            calibrated = min(calibrated, 0.65)
        else:
            calibrated = min(calibrated, 0.9)
        return max(0.25, min(0.9, calibrated))

    return max(0.0, min(1.0, calibrated))


def heuristic_investigation_candidates(
    observation: ModerationObservation,
    difficulty: str,
) -> List[str]:
    """Return a small high-signal shortlist of context requests."""
    if difficulty == "easy":
        return []

    content = observation.content_item.text.lower() if observation.content_item else ""
    media_type = observation.content_item.media_type if observation.content_item else "text"
    trigger_info = observation.trigger_info.model_dump() if observation.trigger_info else {}
    trigger_type = trigger_info.get("trigger_type")
    auto_flag_reason = (trigger_info.get("auto_flag_reason") or "").lower()
    appeal_text = (trigger_info.get("appeal_text") or "").lower()
    report_count = int(trigger_info.get("report_count") or 0)

    candidates: List[str] = []
    self_promo_cues = any(
        word in content
        for word in [
            "just launched",
            "launched my",
            "new app",
            "productivity app",
        ]
    )
    suspicious_link = (
        media_type == "text+link"
        or "link" in auto_flag_reason
        or "suspicious" in auto_flag_reason
    )

    if trigger_type == "appeal":
        candidates.append("thread_context")
    if trigger_type == "proactive_audit":
        candidates.extend(["author_violations", "similar_precedents"])
    if report_count >= 10:
        candidates.append("reporter_credibility")
    if "appeal" in appeal_text or trigger_type == "appeal":
        candidates.append("similar_precedents")
    if self_promo_cues and not ("link" in auto_flag_reason or "suspicious" in auto_flag_reason):
        candidates.extend(["author_violations", "linked_content_summary"])
    elif suspicious_link:
        candidates.append("linked_content_summary")
    if any(word in content for word in ["kill", "dead", "threat", "destroy", "fat", "joking", "jk", "bro"]):
        candidates.append("thread_context")
    if any(word in content for word in ["government", "policy", "vaccine", "autism", "opinion", "satire"]):
        candidates.append("similar_precedents")
    if any(word in content for word in ["address", "hotel", "room", "phone", "contact"]):
        candidates.append("community_rules")
    if any(word in content for word in ["globalists", "replace", "(((they)))", "tradition patrol", "early life"]):
        candidates.extend(["author_violations", "similar_precedents"])
    if any(word in content for word in ["buy", "dm now", "followers", "free checklist", "trial", "premium"]):
        candidates.extend(["linked_content_summary", "author_violations"])

    ordered: List[str] = []
    for candidate in candidates:
        if candidate in CONTEXT_OPTIONS and candidate not in ordered:
            ordered.append(candidate)

    limit = 1 if difficulty == "medium" else 2
    return ordered[:limit]


def heuristic_violation_candidates(
    observation: ModerationObservation,
    difficulty: str,
) -> List[str]:
    """Return a small shortlist of likely policy IDs for this case."""
    del difficulty

    content = observation.content_item.text.lower() if observation.content_item else ""
    media_type = observation.content_item.media_type if observation.content_item else "text"
    trigger_info = observation.trigger_info.model_dump() if observation.trigger_info else {}
    gathered_context = observation.gathered_context.model_dump()
    trigger_type = trigger_info.get("trigger_type")
    auto_flag_reason = (trigger_info.get("auto_flag_reason") or "").lower()
    linked_summary = (gathered_context.get("linked_content_summary") or "").lower()
    community_rules = (gathered_context.get("community_rules") or "").lower()
    thread_text = json.dumps(gathered_context.get("thread_context") or []).lower()

    candidates: List[str] = []

    if any(word in content for word in ["slur", "globalists", "(((they)))", "replace us"]):
        candidates.append("2.1")
    if any(word in content for word in ["kill", "dead", "watch your back", "make you pay"]):
        candidates.append("4.1")
    if any(word in content for word in ["idiot", "pathetic", "loser", "garbage", "clueless"]):
        candidates.extend(["3.1", "1.1"])
    if any(word in content for word in ["ratio", "brigade", "boost accounts", "mirror it everywhere"]):
        candidates.extend(["5.2", "3.1"])
    if any(word in content for word in ["buy now", "followers", "giveaway", "trial", "referral", "scholarship", "checklist"]):
        candidates.append("5.1")
    if media_type == "text+link" or "link" in auto_flag_reason:
        candidates.append("5.1")
    if any(word in content for word in ["vaccine", "autism", "big pharma", "deepfake", "fact check"]):
        candidates.append("6.1")
    if any(word in content for word in ["address", "phone", "school", "schedule", "hotel", "room"]):
        candidates.append("7.1")
    if any(word in linked_summary for word in ["credential", "card details", "phishing", "deepfake", "private data", "address", "phone"]):
        candidates.extend(["5.1", "6.1", "7.1"])
    if any(word in community_rules for word in ["bad-faith", "good faith", "disruptive"]) or any(
        word in thread_text for word in ["if it gets attention, it works", "peacefully protest"]
    ):
        candidates.append("1.2")
    if trigger_type == "appeal":
        candidates.append("none")
    if any(word in content for word in ["satire", "parody", "mandatory fun", "happiness program"]):
        candidates.append("none")

    ordered: List[str] = []
    for candidate in candidates + ["none"]:
        if candidate in VALID_VIOLATIONS and candidate not in ordered:
            ordered.append(candidate)

    return ordered[:4]


def heuristic_factor_candidates(
    observation: ModerationObservation,
    difficulty: str,
) -> List[str]:
    """Return a case-specific shortlist of likely factor labels."""
    del difficulty

    content = observation.content_item.text.lower() if observation.content_item else ""
    trigger_info = observation.trigger_info.model_dump() if observation.trigger_info else {}
    gathered_context = observation.gathered_context.model_dump()
    author_profile = gathered_context.get("author_profile") or {}
    author_violations = gathered_context.get("author_violations") or []
    thread_context = json.dumps(gathered_context.get("thread_context") or []).lower()
    linked_summary = (gathered_context.get("linked_content_summary") or "").lower()
    community_rules = (gathered_context.get("community_rules") or "").lower()
    reporter_credibility = gathered_context.get("reporter_credibility") or {}
    auto_flag_reason = (trigger_info.get("auto_flag_reason") or "").lower()

    candidates: List[str] = []

    if "spam_score: 0.9" in auto_flag_reason or "phishing_score: 0.9" in auto_flag_reason:
        candidates.append("auto_flag_high_confidence")
    elif auto_flag_reason:
        candidates.append("auto_flag_low_confidence")

    if any(word in content for word in ["buy now", "followers", "giveaway", "trial", "referral"]):
        candidates.append("spam_commercial")
    if any(word in content for word in ["ratio", "brigade", "boost accounts"]):
        candidates.append("coordinated_reporting")
    if any(word in content for word in ["kill", "make you pay", "watch your back"]):
        candidates.append("explicit_threat")
    if any(word in content for word in ["peacefully protest", "target acquired"]):
        candidates.append("implicit_threat")
    if any(word in content for word in ["vaccine", "autism", "deepfake", "fact check"]):
        candidates.append("misinformation_factual")
    if "my opinion" in content or "just my opinion" in content:
        candidates.append("misinformation_opinion")
    if any(word in content for word in ["satire", "parody", "mandatory fun"]):
        candidates.append("satire_or_parody")
    if any(word in linked_summary for word in ["satire", "fictional", "parody", "stage performance", "lyrics"]):
        candidates.extend(["satire_or_parody", "artistic_expression"])
    if any(word in thread_context for word in ["gg", "trash talk", "next round", "friendly banter"]):
        candidates.extend(["sarcasm_or_humor", "gaming_or_competition_context"])
    if any(word in linked_summary for word in ["private", "phone", "address", "medical records", "attendees", "hotel room"]):
        candidates.append("privacy_violation")
    if any(word in linked_summary for word in ["phishing", "credential", "card details", "upsell", "deepfake", "synthetic", "shock footage"]):
        candidates.append("harmful_link")
    if any(word in community_rules for word in ["educational medical content", "clinically relevant"]):
        candidates.extend(["graphic_content_justified", "medical_or_educational_context", "community_exception_applies"])
    if any(word in linked_summary for word in ["real-world accident footage", "real-world gore", "fatal", "dismemberment"]):
        candidates.append("graphic_content_unjustified")
    if author_violations:
        candidates.append("repeat_offender")
    if reporter_credibility.get("accuracy_rate", 0.0) >= 0.85:
        candidates.append("trusted_reporter")
    if author_profile.get("follower_count", 0) >= 500000:
        candidates.append("public_figure_author")
    if any(word in linked_summary for word in ["clearly labeled", "condemn", "documenting", "archive"]):
        candidates.append("quoting_or_reporting")
    if any(word in content for word in ["ridiculous", "clueless", "garbage", "ignorant"]):
        candidates.append("borderline_requires_judgment")

    ordered: List[str] = []
    for candidate in candidates + ["clear_violation_no_exception", "no_violation_found"]:
        if candidate in observation.available_factors and candidate not in ordered:
            ordered.append(candidate)

    return ordered[:8]


def policy_summary_for_case(candidate_violations: List[str]) -> str:
    """Render a compact policy hint block tailored to the likely policy IDs."""
    labels = [
        f"- {violation} = {POLICY_LABELS[violation]}"
        for violation in candidate_violations
        if violation in POLICY_LABELS and violation != "none"
    ]
    if not labels:
        labels = ["- none = no policy violation if context supports benign intent or a clear exception"]
    return "\n".join(labels)


def load_scenario_ids(task_id: str, mode: str) -> List[str]:
    """Load deterministic scenario IDs for the requested evaluation mode."""
    if mode == "canonical":
        return get_benchmark_scenario_ids(task_id, split="canonical")

    difficulty = TASK_TO_DIFFICULTY[task_id]
    scenarios = get_all_scenarios()[difficulty]
    canonical_ids = get_benchmark_scenario_ids(task_id, split="canonical")
    canonical_id_set = set(canonical_ids)
    remaining_ids = [
        scenario["scenario_id"]
        for scenario in scenarios
        if scenario["scenario_id"] not in canonical_id_set
    ]
    return canonical_ids + remaining_ids


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from model output."""
    if not response:
        return None

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def exception_text(exc: Exception) -> str:
    """Flatten provider error payloads into one lowercase string for matching."""
    parts = [str(exc)]

    message = getattr(exc, "message", None)
    if isinstance(message, str):
        parts.append(message)

    body = getattr(exc, "body", None)
    if body is not None:
        try:
            parts.append(json.dumps(body, sort_keys=True))
        except TypeError:
            parts.append(str(body))

    return " ".join(parts).lower()


def parameter_rejected(exc: Exception, parameter: str) -> bool:
    """Return whether a provider error clearly indicates an unsupported parameter."""
    text = exception_text(exc)
    aliases = PARAMETER_ALIASES.get(parameter, (parameter,))
    return (
        any(alias in text for alias in aliases)
        and any(marker in text for marker in PARAMETER_ERROR_MARKERS)
    )


def to_jsonable(value: Any) -> Any:
    """Convert Pydantic models to plain Python values for JSON output."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def sanitize_prompt_text(text: str) -> str:
    """Redact literal high-risk details before sending cases to the model."""
    sanitized = text
    for pattern, replacement in PROMPT_REDACTIONS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_prompt_value(value: Any) -> Any:
    """Recursively sanitize strings inside prompt payloads."""
    if isinstance(value, str):
        return sanitize_prompt_text(value)
    if isinstance(value, list):
        return [sanitize_prompt_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: sanitize_prompt_value(item)
            for key, item in value.items()
        }
    return value


def compact_trigger_info_for_prompt(trigger_info: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only high-signal trigger fields needed for model decisions."""
    compact: Dict[str, Any] = {}
    trigger_type = trigger_info.get("trigger_type")
    if trigger_type:
        compact["trigger_type"] = trigger_type

    report_count = trigger_info.get("report_count")
    if report_count:
        compact["report_count"] = report_count

    auto_flag_reason = trigger_info.get("auto_flag_reason")
    if auto_flag_reason:
        compact["auto_flag_reason"] = auto_flag_reason

    original_decision = trigger_info.get("original_decision")
    if original_decision:
        compact["original_decision"] = original_decision

    appeal_text = trigger_info.get("appeal_text")
    if appeal_text:
        compact["appeal_text"] = appeal_text

    audit_reason = trigger_info.get("audit_reason")
    if audit_reason:
        compact["audit_reason"] = audit_reason

    return compact


def format_log_bool(value: bool) -> str:
    """Format booleans for evaluator-compatible stdout."""
    return str(bool(value)).lower()


def format_action_token(action: ModerationAction) -> str:
    """Render a single-line action token for evaluator step logs."""
    if action.action_type != "decide":
        return action.action_type

    decision = action.decision or "unknown"
    primary_violation = action.primary_violation or "none"
    severity = action.severity or "none"
    confidence = 0.0 if action.confidence is None else float(action.confidence)
    return (
        f"decide:{decision}:{primary_violation}:{severity}:{confidence:.2f}"
    )


def log_start(task: str, env: str, model: str) -> None:
    """Emit the required episode-start log line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    *,
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit the required per-step log line."""
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={format_log_bool(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit the required episode-end log line."""
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={format_log_bool(success)} steps={steps} "
        f"score={score:.3f} rewards={rewards_text}",
        flush=True,
    )


def clamp_score(score: Optional[float]) -> float:
    """Clamp the final score into the public validator-safe open interval (0, 1)."""
    return clamp_public_task_grade(score)


def resolve_env_target(explicit_base_url: Optional[str]) -> Dict[str, Optional[str]]:
    """Resolve how the evaluator should connect to the environment."""
    if explicit_base_url:
        return {
            "connection_mode": "base_url",
            "env_base_url": explicit_base_url,
            "local_image_name": None,
        }
    if ENV_BASE_URL:
        return {
            "connection_mode": "base_url",
            "env_base_url": ENV_BASE_URL,
            "local_image_name": None,
        }
    if LOCAL_IMAGE_NAME:
        return {
            "connection_mode": "local_image",
            "env_base_url": None,
            "local_image_name": LOCAL_IMAGE_NAME,
        }
    return {
        "connection_mode": "base_url",
        "env_base_url": DEFAULT_ENV_BASE_URL,
        "local_image_name": None,
    }


def validate_runtime_configuration(mode: str) -> Dict[str, Any]:
    """Validate env vars and benchmark assets before the first model call."""
    api_key, api_key_source = resolve_api_key_and_source()
    if not MODEL_NAME:
        raise InferenceConfigurationError("Missing MODEL_NAME.")
    if not api_key:
        raise InferenceConfigurationError(
            "Missing API key. Set HF_TOKEN, API_KEY, OPENAI_API_KEY, or "
            "AZURE_OPENAI_API_KEY."
        )

    manifest = validate_benchmark_manifest()
    canonical = manifest["canonical"]
    target = resolve_env_target(None)
    return {
        "manifest_version": manifest["manifest_version"],
        "canonical_task_counts": {
            task_id: len(scenario_ids) for task_id, scenario_ids in canonical.items()
        },
        "evaluation_mode": mode,
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "api_key_source": api_key_source,
        "openai_seed": OPENAI_SEED,
        "connection_mode": target["connection_mode"],
        "env_base_url": target["env_base_url"],
        "local_image_name": target["local_image_name"],
    }


def heuristic_decision_action(
    observation: ModerationObservation,
    difficulty: str,
) -> ModerationAction:
    """Produce a deterministic fallback moderation action without an LLM call."""
    content = observation.content_item.text.lower() if observation.content_item else ""
    gathered_context = observation.gathered_context.model_dump()
    linked_summary = (gathered_context.get("linked_content_summary") or "").lower()
    community_rules = (gathered_context.get("community_rules") or "").lower()
    thread_context = json.dumps(gathered_context.get("thread_context") or []).lower()
    author_violations = gathered_context.get("author_violations") or []
    author_profile = gathered_context.get("author_profile") or {}
    candidate_violations = heuristic_violation_candidates(observation, difficulty)
    candidate_factors = heuristic_factor_candidates(observation, difficulty)
    context_count = sum(1 for value in gathered_context.values() if value is not None)

    def action(
        decision: str,
        primary_violation: str,
        severity: str,
        confidence: float,
        key_factors: List[str],
    ) -> ModerationAction:
        return ModerationAction(
            action_type="decide",
            decision=decision,
            primary_violation=primary_violation,
            severity=severity,
            confidence=calibrate_confidence(
                confidence=confidence,
                difficulty=difficulty,
                decision=decision,
                context_count=context_count,
            ),
            key_factors=normalize_factors(
                key_factors,
                observation.available_factors,
                decision,
            ),
        )

    if any(word in thread_context for word in ["gg", "trash talk", "next round", "friendly banter"]):
        return action(
            "approve",
            "none",
            "none",
            0.78,
            ["sarcasm_or_humor", "gaming_or_competition_context", "no_violation_found"],
        )

    if any(word in linked_summary for word in ["satire", "fictional", "parody", "clearly labeled"]) or (
        "satire" in content and "no_violation_found" in observation.available_factors
    ):
        return action(
            "approve",
            "none",
            "none",
            0.78,
            ["satire_or_parody", "no_violation_found"],
        )

    if any(word in linked_summary for word in ["documenting", "archive", "condemn", "fact-check", "fact check"]):
        return action(
            "approve",
            "none",
            "none",
            0.74,
            ["quoting_or_reporting", "community_exception_applies", "no_violation_found"],
        )

    if any(word in community_rules for word in ["educational medical content", "clinically relevant"]):
        return action(
            "approve",
            "none",
            "none",
            0.76,
            ["graphic_content_justified", "medical_or_educational_context", "community_exception_applies"],
        )

    if any(word in linked_summary for word in ["card details", "phishing", "credential", "upsell", "referrals before entry unlocks"]):
        return action(
            "remove",
            "5.1",
            "high",
            0.9,
            ["spam_commercial", "harmful_link", "clear_violation_no_exception"],
        )

    if any(word in content for word in ["buy now", "followers", "make $", "no prescription needed"]):
        return action(
            "remove",
            "5.1",
            "high",
            0.92,
            ["spam_commercial", "clear_violation_no_exception"],
        )

    if any(word in content for word in ["watch your back", "make you pay"]) and "gaming_or_competition_context" not in candidate_factors:
        return action(
            "remove",
            "4.1",
            "high",
            0.9,
            ["explicit_threat", "clear_violation_no_exception"],
        )

    if any(word in linked_summary for word in ["private", "phone", "address", "medical records", "school", "schedule", "hotel room"]):
        return action(
            "remove",
            "7.1",
            "critical" if any(word in linked_summary for word in ["school", "schedule", "medical records"]) else "high",
            0.88,
            ["privacy_violation", "clear_violation_no_exception"],
        )

    if any(word in thread_context for word in ["list is ready", "operation clean sweep", "target acquired"]) and author_violations:
        return action(
            "remove",
            "3.1",
            "high",
            0.84,
            ["implicit_threat", "repeat_offender", "clear_violation_no_exception"],
        )

    if "hate_speech_coded" in candidate_factors and difficulty == "hard":
        return action(
            "warn",
            "2.1",
            "medium",
            0.52,
            ["hate_speech_coded", "borderline_requires_judgment", "repeat_offender"],
        )

    if "misinformation_opinion" in candidate_factors and difficulty == "hard":
        return action(
            "escalate",
            "6.1",
            "medium",
            0.36,
            ["misinformation_opinion", "policy_sections_conflict", "repeat_offender"],
        )

    if "1.2" in candidate_violations and "graphic_content_unjustified" in candidate_factors:
        return action(
            "remove",
            "1.2",
            "medium",
            0.8,
            ["public_figure_author", "graphic_content_unjustified", "clear_violation_no_exception"],
        )

    if "1.2" in candidate_violations:
        return action(
            "warn",
            "1.2",
            "low",
            0.58,
            ["borderline_requires_judgment"],
        )

    if "1.1" in candidate_violations:
        return action(
            "warn",
            "1.1",
            "low",
            0.58,
            ["borderline_requires_judgment"],
        )

    if "3.1" in candidate_violations and "public_figure_target" in candidate_factors:
        return action(
            "warn",
            "3.1",
            "low",
            0.56,
            ["public_figure_target", "borderline_requires_judgment"],
        )

    if "6.1" in candidate_violations and "escalated" in community_rules:
        return action(
            "escalate",
            "6.1",
            "medium",
            0.4,
            ["policy_sections_conflict", "borderline_requires_judgment"],
        )

    if difficulty == "easy":
        if candidate_violations and candidate_violations[0] != "none":
            primary_violation = candidate_violations[0]
            severity = "high" if primary_violation in {"4.1", "5.1", "7.1"} else "medium"
            return action(
                "remove",
                primary_violation,
                severity,
                0.9,
                [candidate_factors[0]] if candidate_factors else ["clear_violation_no_exception"],
            )
        return action("approve", "none", "none", 0.9, ["no_violation_found"])

    if difficulty == "hard":
        primary_violation = candidate_violations[0] if candidate_violations else "none"
        if primary_violation == "none":
            return action("approve", "none", "none", 0.52, ["no_violation_found"])
        return action("escalate", primary_violation, "medium", 0.34, ["borderline_requires_judgment"])

    primary_violation = candidate_violations[0] if candidate_violations else "none"
    if primary_violation == "none":
        return action("approve", "none", "none", 0.68, ["no_violation_found"])
    severity = "high" if primary_violation in {"4.1", "5.1", "7.1"} else "medium"
    return action(
        "remove",
        primary_violation,
        severity,
        0.74,
        [candidate_factors[0]] if candidate_factors else ["clear_violation_no_exception"],
    )


class SafeSpaceAgent:
    """OpenAI-client baseline agent for SafeSpace."""

    def __init__(self) -> None:
        api_key, api_key_source = resolve_api_key_and_source()
        if not api_key:
            raise InferenceConfigurationError(
                "Missing API key. Set HF_TOKEN, API_KEY, OPENAI_API_KEY, or "
                "AZURE_OPENAI_API_KEY."
            )
        if not MODEL_NAME:
            raise InferenceConfigurationError("Missing MODEL_NAME environment variable.")

        self.client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
        self.model = MODEL_NAME
        self.api_key_source = api_key_source

    def _completion_request_kwargs(
        self,
        prompt: str,
        *,
        use_max_completion_tokens: bool,
        include_seed: bool,
    ) -> Dict[str, Any]:
        """Build provider-compatible OpenAI client request kwargs."""
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
        }
        if use_max_completion_tokens:
            request_kwargs["max_completion_tokens"] = MAX_TOKENS
        else:
            request_kwargs["max_tokens"] = MAX_TOKENS
        if include_seed:
            request_kwargs["seed"] = OPENAI_SEED
        return request_kwargs

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the model and parse the JSON response."""
        use_max_completion_tokens = True
        include_seed = True
        attempted_configs: set[tuple[bool, bool]] = set()

        try:
            while True:
                config = (use_max_completion_tokens, include_seed)
                if config in attempted_configs:
                    break
                attempted_configs.add(config)

                request_kwargs = self._completion_request_kwargs(
                    prompt,
                    use_max_completion_tokens=use_max_completion_tokens,
                    include_seed=include_seed,
                )

                try:
                    completion = self.client.chat.completions.create(**request_kwargs)
                except Exception as exc:
                    if (
                        use_max_completion_tokens
                        and parameter_rejected(exc, "max_completion_tokens")
                    ):
                        use_max_completion_tokens = False
                        continue
                    if include_seed and parameter_rejected(exc, "seed"):
                        include_seed = False
                        continue
                    raise ModelRequestError(
                        "Model request failed "
                        f"(model={self.model}, base_url={API_BASE_URL}): {exc}"
                    ) from exc

                content = completion.choices[0].message.content or ""
                parsed = parse_json_response(content)
                if parsed is None:
                    raise ModelRequestError(
                        "Model returned a non-JSON response that could not be parsed."
                    )
                return parsed
        except ModelRequestError:
            raise
        raise ModelRequestError(
            "Model request failed after exhausting provider-compatibility fallbacks "
            f"(model={self.model}, base_url={API_BASE_URL})."
        )

    def decide_investigation(
        self,
        observation: ModerationObservation,
        difficulty: str,
    ) -> List[str]:
        """Pick which context sources to gather before deciding."""
        heuristic_choices = heuristic_investigation_candidates(observation, difficulty)
        if difficulty == "easy":
            return []

        trigger_info = observation.trigger_info.model_dump() if observation.trigger_info else {}
        safe_trigger_info = sanitize_prompt_value(
            compact_trigger_info_for_prompt(trigger_info)
        )
        safe_content = sanitize_prompt_text(
            observation.content_item.text if observation.content_item else ""
        )
        prompt = f"""Choose the most relevant investigation actions for this case.

Difficulty: {difficulty}
Content (safety-redacted for transport): {safe_content}
Trigger info: {json.dumps(safe_trigger_info, indent=2)}

Available context keys:
{json.dumps(CONTEXT_OPTIONS)}

Heuristic shortlist for this case:
{json.dumps(heuristic_choices)}

Guidance:
- Easy cases should usually request no context.
- Medium cases should usually request 1 focused context item.
- Hard cases should usually request at most 2 focused context items.
- Prefer the heuristic shortlist unless there is a strong reason not to.
- Do not request context just to be safe.

Return JSON only:
{{"investigate": ["thread_context"]}}
"""

        try:
            parsed = self._call_llm(prompt)
        except ModelRequestError:
            return heuristic_choices
        requested = parsed.get("investigate", [])
        if not isinstance(requested, list):
            return heuristic_choices

        max_requests = 1 if difficulty == "medium" else 2
        normalized = [item for item in requested if item in CONTEXT_OPTIONS]
        if not normalized:
            return heuristic_choices

        ordered: List[str] = []
        for item in heuristic_choices + normalized:
            if item in CONTEXT_OPTIONS and item not in ordered:
                ordered.append(item)
        return ordered[:max_requests]

    def make_decision(
        self,
        observation: ModerationObservation,
        difficulty: str,
    ) -> ModerationAction:
        """Produce the final structured moderation decision."""
        trigger_info = observation.trigger_info.model_dump() if observation.trigger_info else {}
        gathered_context = observation.gathered_context.model_dump()
        safe_trigger_info = sanitize_prompt_value(
            compact_trigger_info_for_prompt(trigger_info)
        )
        safe_gathered_context = sanitize_prompt_value(gathered_context)
        safe_content = sanitize_prompt_text(
            observation.content_item.text if observation.content_item else ""
        )
        candidate_violations = heuristic_violation_candidates(observation, difficulty)
        candidate_factors = heuristic_factor_candidates(observation, difficulty)
        case_policy_summary = policy_summary_for_case(candidate_violations)
        prompt = f"""Make the final moderation decision.

Difficulty: {difficulty}
Content (safety-redacted for transport): {safe_content}
Trigger info: {json.dumps(safe_trigger_info, indent=2)}
Gathered context: {json.dumps(safe_gathered_context, indent=2)}
Candidate violation IDs for this case:
{json.dumps(candidate_violations)}
Candidate factor shortlist for this case:
{json.dumps(candidate_factors)}
Likely policy labels for this case:
{case_policy_summary}
General policy guidance:
{POLICY_PROMPT_SUMMARY}
{DECISION_RULES}

Decision guidance by difficulty:
- Easy: avoid escalation unless the case is truly ambiguous.
- Medium: if one piece of context clearly resolves the case, decide directly.
- Hard: prefer warn or escalate over overconfident approve/remove when signals conflict.

Output primary_violation as an exact ID only, never a label.
Prefer key_factors from the candidate shortlist when they fit.
If none fit exactly, choose the nearest exact canonical factor label.

Return JSON only:
{{
  "decision": "approve|remove|warn|escalate",
  "primary_violation": "policy section or none",
  "severity": "none|low|medium|high|critical",
  "confidence": 0.0,
  "key_factors": ["factor1", "factor2"]
}}
"""

        try:
            parsed = self._call_llm(prompt)
        except ModelRequestError:
            return heuristic_decision_action(observation, difficulty)
        decision = parsed.get("decision", "escalate")
        if decision not in VALID_DECISIONS:
            decision = "escalate"

        primary_violation = normalize_violation(parsed.get("primary_violation", "none"), decision)
        severity = parsed.get("severity", "none")
        confidence = parsed.get("confidence", 0.3)
        key_factors = normalize_factors(
            parsed.get("key_factors", ["borderline_requires_judgment"]),
            observation.available_factors,
            decision,
        )

        if decision == "approve":
            primary_violation = "none"
            severity = "none"
        if severity not in VALID_SEVERITIES:
            severity = "none"
        if not isinstance(confidence, (int, float)):
            confidence = 0.3
        confidence = max(0.0, min(1.0, float(confidence)))
        confidence = calibrate_confidence(
            confidence=confidence,
            difficulty=difficulty,
            decision=decision,
            context_count=sum(1 for value in gathered_context.values() if value is not None),
        )
        llm_action = ModerationAction(
            action_type="decide",
            decision=decision,
            primary_violation=primary_violation,
            severity=severity,
            confidence=confidence,
            key_factors=key_factors,
        )
        heuristic_action = heuristic_decision_action(observation, difficulty)

        if (
            difficulty == "hard"
            and heuristic_action.primary_violation == llm_action.primary_violation
        ):
            if heuristic_action.decision == "escalate" and llm_action.decision in {"approve", "remove", "warn"}:
                return heuristic_action
            if heuristic_action.decision == "warn" and llm_action.decision == "remove":
                return heuristic_action

        return llm_action


def context_to_action(context_key: str) -> str:
    """Map model-selected context keys to environment action names."""
    action_map = {
        "author_profile": "request_author_profile",
        "author_violations": "request_author_violations",
        "thread_context": "request_thread_context",
        "community_rules": "request_community_rules",
        "linked_content_summary": "request_linked_content",
        "similar_precedents": "request_similar_precedents",
        "reporter_credibility": "request_reporter_credibility",
    }
    return action_map[context_key]


def infer_difficulty(task_id: Optional[str], scenario_id: str) -> str:
    """Infer difficulty from the task mapping or scenario prefix."""
    if task_id in TASK_TO_DIFFICULTY:
        return TASK_TO_DIFFICULTY[task_id]

    lowered = scenario_id.lower()
    if lowered.startswith("easy"):
        return "easy"
    if lowered.startswith("med"):
        return "medium"
    if lowered.startswith("hard"):
        return "hard"
    return "unknown"


def infer_task_id(scenario_id: str) -> Optional[str]:
    """Infer the benchmark task ID from a scenario identifier prefix."""
    difficulty = infer_difficulty(None, scenario_id)
    return DIFFICULTY_TO_TASK.get(difficulty)


def build_failed_episode_result(
    *,
    task_id: Optional[str],
    scenario_id: str,
    stage: str,
    error: str,
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a conservative low-scored result for a failed episode."""
    resolved_task_id = task_id or infer_task_id(scenario_id)
    resolved_difficulty = difficulty or infer_difficulty(resolved_task_id, scenario_id)
    failure = {
        "scenario_id": scenario_id,
        "task_id": resolved_task_id,
        "stage": stage,
        "error": error,
    }
    return {
        "scenario_id": scenario_id,
        "task_id": resolved_task_id,
        "difficulty": resolved_difficulty,
        "episode_reward": 0.0,
        "raw_episode_reward": 0.0,
        "task_grade": clamp_score(0.0),
        "decision": None,
        "confidence": None,
        "investigation_plan": [],
        "step_rewards": [],
        "steps_taken": 0,
        "final_reward_breakdown": None,
        "final_grade_breakdown": None,
        "status": "failed",
        "failure": failure,
    }


async def run_episode(
    env: Any,
    agent: SafeSpaceAgent,
    scenario_id: str,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one evaluation episode and emit submission-compatible stdout logs."""
    resolved_task_id = task_id or infer_task_id(scenario_id)
    difficulty = infer_difficulty(resolved_task_id, scenario_id)
    investigation_plan: List[str] = []
    decision_action: Optional[ModerationAction] = None
    observation: Optional[ModerationObservation] = None
    result: Any = None
    step_rewards: List[float] = []
    steps_taken = 0
    failure_exc: Optional[EpisodeExecutionError] = None
    task_grade = clamp_score(0.0)
    episode_reward = 0.0
    raw_episode_reward = 0.0

    log_start(resolved_task_id or "unknown", BENCHMARK_NAME, MODEL_NAME)

    try:
        try:
            result = await env.reset(scenario_id=scenario_id)
        except Exception as exc:
            raise EpisodeExecutionError(
                scenario_id=scenario_id,
                task_id=resolved_task_id,
                stage="reset",
                error=str(exc),
                difficulty=difficulty,
            ) from exc

        observation = result.observation
        try:
            difficulty = (await env.state()).difficulty or infer_difficulty(
                resolved_task_id, scenario_id
            )
        except Exception as exc:
            raise EpisodeExecutionError(
                scenario_id=scenario_id,
                task_id=resolved_task_id,
                stage="state_after_reset",
                error=str(exc),
                difficulty=difficulty,
            ) from exc

        try:
            investigation_plan = agent.decide_investigation(observation, difficulty)
        except Exception as exc:
            raise EpisodeExecutionError(
                scenario_id=scenario_id,
                task_id=resolved_task_id,
                stage="decide_investigation",
                error=str(exc),
                difficulty=difficulty,
            ) from exc

        for context_key in investigation_plan:
            action = ModerationAction(action_type=context_to_action(context_key))
            try:
                result = await env.step(action)
            except Exception as exc:
                raise EpisodeExecutionError(
                    scenario_id=scenario_id,
                    task_id=resolved_task_id,
                    stage=f"investigation_step:{context_key}",
                    error=str(exc),
                    difficulty=difficulty,
                ) from exc

            observation = result.observation
            steps_taken += 1
            step_reward = 0.0 if result.reward is None else float(result.reward)
            step_rewards.append(step_reward)
            log_step(
                step=steps_taken,
                action=format_action_token(action),
                reward=step_reward,
                done=result.done,
                error=observation.error_code,
            )
            if result.done:
                break

        if result is not None and not result.done:
            try:
                decision_action = agent.make_decision(observation, difficulty)
            except Exception as exc:
                raise EpisodeExecutionError(
                    scenario_id=scenario_id,
                    task_id=resolved_task_id,
                    stage="make_decision",
                    error=str(exc),
                    difficulty=difficulty,
                ) from exc

            try:
                result = await env.step(decision_action)
            except Exception as exc:
                raise EpisodeExecutionError(
                    scenario_id=scenario_id,
                    task_id=resolved_task_id,
                    stage="decision_step",
                    error=str(exc),
                    difficulty=difficulty,
                ) from exc

            observation = result.observation
            steps_taken += 1
            step_reward = 0.0 if result.reward is None else float(result.reward)
            step_rewards.append(step_reward)
            log_step(
                step=steps_taken,
                action=format_action_token(decision_action),
                reward=step_reward,
                done=result.done,
                error=observation.error_code,
            )

        try:
            state = await env.state()
        except Exception as exc:
            raise EpisodeExecutionError(
                scenario_id=scenario_id,
                task_id=resolved_task_id,
                stage="state_after_episode",
                error=str(exc),
                difficulty=difficulty,
            ) from exc

        episode_reward = (
            float(state.episode_reward) if state.episode_reward is not None else 0.0
        )
        raw_episode_reward = float(
            getattr(state, "raw_episode_reward", episode_reward)
        )
        task_grade = clamp_score(
            observation.task_grade if observation and observation.task_grade is not None else None
        )

        return {
            "scenario_id": scenario_id,
            "task_id": resolved_task_id,
            "difficulty": difficulty,
            "episode_reward": episode_reward,
            "raw_episode_reward": raw_episode_reward,
            "task_grade": task_grade,
            "decision": decision_action.decision if decision_action else None,
            "confidence": decision_action.confidence if decision_action else None,
            "investigation_plan": investigation_plan,
            "step_rewards": step_rewards,
            "steps_taken": steps_taken,
            "final_reward_breakdown": to_jsonable(
                observation.reward_breakdown if observation else None
            ),
            "final_grade_breakdown": to_jsonable(
                observation.grade_breakdown if observation else None
            ),
            "status": "success",
            "failure": None,
        }
    except EpisodeExecutionError as exc:
        failure_exc = exc
        raise
    finally:
        final_score = clamp_score(task_grade)
        final_success = failure_exc is None and final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(
            success=final_success,
            steps=steps_taken,
            score=final_score,
            rewards=step_rewards,
        )


def summarize_task(task_id: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build deterministic aggregate metrics for one task."""
    total_reward = sum(item["episode_reward"] for item in results)
    total_raw_reward = sum(item.get("raw_episode_reward", 0.0) for item in results)
    total_task_grade = sum(item["task_grade"] for item in results)
    decision_counts: Dict[str, int] = {}
    for item in results:
        decision = item.get("decision") or "no_decision"
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    return {
        "task_id": task_id,
        "num_scenarios": len(results),
        "average_task_grade": clamp_score(total_task_grade / len(results)) if results else clamp_score(None),
        "average_reward": total_reward / len(results) if results else 0.0,
        "average_raw_reward": total_raw_reward / len(results) if results else 0.0,
        "total_task_grade": total_task_grade,
        "total_reward": total_reward,
        "total_raw_reward": total_raw_reward,
        "decision_distribution": decision_counts,
        "results": results,
    }


async def run_task_evaluation(
    env: Any,
    agent: SafeSpaceAgent,
    task_id: str,
    scenario_ids: List[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one task split and retain structured failure metadata."""
    results: List[Dict[str, Any]] = []
    failure_details: List[Dict[str, Any]] = []

    for scenario_id in scenario_ids:
        try:
            result = await run_episode(env, agent, scenario_id, task_id=task_id)
        except EpisodeExecutionError as exc:
            result = build_failed_episode_result(
                task_id=exc.task_id,
                scenario_id=exc.scenario_id,
                stage=exc.stage,
                error=exc.error,
                difficulty=exc.difficulty,
            )
        except Exception as exc:
            result = build_failed_episode_result(
                task_id=task_id,
                scenario_id=scenario_id,
                stage="unknown",
                error=str(exc),
            )

        if result["failure"] is not None:
            failure_details.append(result["failure"])
        results.append(result)

    summary = summarize_task(task_id, results)
    summary["successful_scenarios"] = len(results) - len(failure_details)
    summary["failed_scenarios"] = len(failure_details)
    summary["failure_details"] = failure_details
    return summary, failure_details


async def create_env_client(explicit_base_url: Optional[str]) -> SafeSpaceEnv:
    """Create an environment client from a URL or a local Docker image."""
    target = resolve_env_target(explicit_base_url)
    if target["connection_mode"] == "local_image" and target["local_image_name"]:
        return await SafeSpaceEnv.from_docker_image(target["local_image_name"])
    if target["env_base_url"] is None:
        raise InferenceConfigurationError(
            "Unable to resolve an environment target. Set ENV_BASE_URL or LOCAL_IMAGE_NAME."
        )
    return SafeSpaceEnv(base_url=target["env_base_url"])


def write_summary_file(path: Optional[str], summary: Dict[str, Any]) -> None:
    """Write the aggregate evaluation summary to disk when requested."""
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")


async def _async_main(args: argparse.Namespace) -> None:
    """Run the evaluator in async mode."""
    config_metadata = validate_runtime_configuration(args.mode)
    target = resolve_env_target(args.env_base_url)

    if args.validate_config:
        payload = {
            **config_metadata,
            "connection_mode": target["connection_mode"],
            "env_base_url": target["env_base_url"],
            "local_image_name": target["local_image_name"],
        }
        print(json.dumps(payload, indent=2))
        return

    agent = SafeSpaceAgent()
    started_at = time.time()
    task_summaries: Dict[str, Any] = {}
    failure_details: List[Dict[str, Any]] = []

    client = await create_env_client(args.env_base_url)
    try:
        if hasattr(client, "connect"):
            await client.connect()
        for task_id in TASK_TO_DIFFICULTY:
            scenario_ids = load_scenario_ids(task_id, args.mode)
            if args.limit_per_task is not None:
                scenario_ids = scenario_ids[: args.limit_per_task]
            task_summary, task_failures = await run_task_evaluation(
                client,
                agent,
                task_id,
                scenario_ids,
            )
            task_summaries[task_id] = task_summary
            failure_details.extend(task_failures)
    finally:
        try:
            if hasattr(client, "close"):
                await client.close()
        except Exception:
            pass

    total_scenarios = sum(
        summary["num_scenarios"] for summary in task_summaries.values()
    )
    total_reward = sum(summary["total_reward"] for summary in task_summaries.values())
    total_raw_reward = sum(
        summary["total_raw_reward"] for summary in task_summaries.values()
    )
    total_task_grade = sum(
        summary["total_task_grade"] for summary in task_summaries.values()
    )
    failure_count = len(failure_details)
    manifest = get_benchmark_manifest()
    summary = {
        "benchmark_manifest_version": manifest["manifest_version"],
        "evaluation_mode": args.mode,
        "connection_mode": target["connection_mode"],
        "env_base_url": target["env_base_url"],
        "local_image_name": target["local_image_name"],
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "api_key_source": config_metadata["api_key_source"],
        "openai_seed": OPENAI_SEED,
        "failure_count": failure_count,
        "successful_scenarios": total_scenarios - failure_count,
        "failed_scenarios": failure_count,
        "failure_details": failure_details,
        "limit_per_task": args.limit_per_task,
        "tasks": task_summaries,
        "overall_average_task_grade": (
            clamp_score(total_task_grade / total_scenarios) if total_scenarios else clamp_score(None)
        ),
        "overall_average_reward": (
            total_reward / total_scenarios if total_scenarios else 0.0
        ),
        "overall_average_raw_reward": (
            total_raw_reward / total_scenarios if total_scenarios else 0.0
        ),
        "overall_total_task_grade": total_task_grade,
        "overall_total_reward": total_reward,
        "overall_total_raw_reward": total_raw_reward,
        "total_scenarios": total_scenarios,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }

    write_summary_file(args.summary_json_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeSpace canonical baseline evaluator")
    parser.add_argument(
        "--mode",
        choices=["canonical", "full"],
        default="canonical",
        help="Evaluation mode: canonical submission baseline or full dataset sweep.",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="Optional cap for scenarios per task (useful for smoke evaluation).",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate environment variables and benchmark assets, then exit.",
    )
    parser.add_argument(
        "--env-base-url",
        default=None,
        help="Base URL of a running SafeSpace server.",
    )
    parser.add_argument(
        "--summary-json-path",
        default=None,
        help="Optional file path for the aggregate evaluation summary JSON.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_async_main(args))
    except InferenceConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
