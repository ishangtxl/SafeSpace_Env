#!/usr/bin/env python3
"""
Scenario generation utilities for SafeSpace.

This script generates scenarios by:
1. Creating procedural stress-test variations of gold scenarios
2. Generating new scenarios from category templates
3. Validating all scenarios against the schema
"""

import copy
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.policy import FACTOR_LIST, POLICY_SECTIONS
from server.scenarios import validate_scenario_record

# Paths
DATA_DIR = Path(__file__).parent.parent / "server" / "data"

# Random name pools
USERNAMES = [
    "user_alpha", "user_beta", "user_gamma", "user_delta", "user_epsilon",
    "user_zeta", "user_eta", "user_theta", "user_iota", "user_kappa",
    "user_lambda", "user_mu", "user_nu", "user_xi", "user_omicron",
    "player_one", "player_two", "gamer_pro", "tech_guru", "code_ninja",
    "data_wizard", "ai_master", "ml_expert", "dev_ops", "full_stack",
    "casual_user", "power_user", "new_member", "veteran", "moderator",
]

COMMUNITIES = [
    "general", "tech", "gaming", "news", "politics", "health", "science",
    "sports", "music", "movies", "books", "food", "travel", "photography",
    "art", "fashion", "fitness", "finance", "crypto", "programming",
    "memes", "wholesome", "debate", "askexperts", "todayilearned",
]

BIOS = [
    "Just here to chat", "Tech enthusiast", "Coffee addict",
    "Learning every day", "Building cool stuff", "Gaming is life",
    "Music lover", "Foodie", "Travel junkie", "Photography hobbyist",
    "Coder by day", "Student", "Professional lurker", "Part-time troll",
    "Full-time dreamer", "Open minded", "Critical thinker", "Free spirit",
]


def random_timestamp(seed: int) -> str:
    """Generate a random timestamp within the last 30 days."""
    rng = random.Random(seed)
    base = datetime(2026, 3, 15)
    offset = timedelta(
        days=rng.randint(-30, 0),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
    )
    return (base + offset).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_procedural_variation(
    base_scenario: Dict[str, Any],
    seed: int,
    prefix: str = "var",
) -> Dict[str, Any]:
    """
    Create a procedural variation of a base scenario.

    Varies:
    - scenario_id, post_id, author_id
    - timestamp
    - report_count (slightly)
    - follower/post counts
    - account age

    Preserves:
    - text content
    - ground truth
    - trigger type
    - category

    These variants are intended for stress-testing and broader training
    coverage. They are not canonical benchmark cases unless explicitly
    selected in the benchmark manifest.
    """
    rng = random.Random(seed)
    variant = copy.deepcopy(base_scenario)

    base_id = base_scenario["scenario_id"]
    variant["scenario_id"] = f"{base_id}_{prefix}{seed:03d}"

    # Vary content item metadata
    ci = variant["content_item"]
    ci["post_id"] = f"p_{rng.randint(10000, 99999)}"
    ci["author_id"] = rng.choice(USERNAMES) + f"_{rng.randint(100, 999)}"
    ci["timestamp"] = random_timestamp(seed)
    ci["community"] = rng.choice(COMMUNITIES) if rng.random() < 0.3 else ci["community"]

    # Vary trigger info
    ti = variant["trigger_info"]
    if ti.get("report_count", 0) > 0:
        ti["report_count"] = max(1, ti.get("report_count", 0) + rng.randint(-3, 5))

    # Vary available context
    ctx = variant.get("available_context", {})
    if ctx.get("author_profile"):
        profile = ctx["author_profile"]
        profile["account_age_days"] = max(1, profile.get("account_age_days", 100) + rng.randint(-50, 100))
        profile["follower_count"] = max(0, profile.get("follower_count", 50) + rng.randint(-20, 50))
        profile["post_count"] = max(1, profile.get("post_count", 100) + rng.randint(-30, 100))
        if rng.random() < 0.3:
            profile["bio"] = rng.choice(BIOS)

    if ctx.get("reporter_credibility"):
        rc = ctx["reporter_credibility"]
        rc["total_reports"] = max(1, rc.get("total_reports", 10) + rng.randint(-3, 5))
        accurate = int(rc["total_reports"] * rc.get("accuracy_rate", 0.5))
        rc["accurate_reports"] = accurate
        rc["false_reports"] = rc["total_reports"] - accurate

    return variant


def validate_scenario(scenario: Dict[str, Any]) -> List[str]:
    """
    Validate a scenario against the schema.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Required top-level fields
    required = ["scenario_id", "difficulty", "category", "content_item", "trigger_info", "ground_truth"]
    for field in required:
        if field not in scenario:
            errors.append(f"Missing required field: {field}")

    if errors:
        return errors

    # Validate difficulty
    if scenario["difficulty"] not in ["easy", "medium", "hard"]:
        errors.append(f"Invalid difficulty: {scenario['difficulty']}")

    # Validate content_item
    ci = scenario["content_item"]
    ci_required = ["post_id", "text", "author_id", "community", "timestamp", "media_type"]
    for field in ci_required:
        if field not in ci:
            errors.append(f"content_item missing: {field}")

    # Validate trigger_info
    ti = scenario["trigger_info"]
    if ti.get("trigger_type") not in ["user_report", "auto_flag", "appeal", "proactive_audit"]:
        errors.append(f"Invalid trigger_type: {ti.get('trigger_type')}")

    # Validate ground_truth
    gt = scenario["ground_truth"]
    gt_required = ["correct_decision", "primary_violation", "severity", "key_factors"]
    for field in gt_required:
        if field not in gt:
            errors.append(f"ground_truth missing: {field}")

    if "correct_decision" in gt:
        if gt["correct_decision"] not in ["approve", "remove", "escalate", "warn"]:
            errors.append(f"Invalid decision: {gt['correct_decision']}")

    if "primary_violation" in gt:
        if gt["primary_violation"] not in POLICY_SECTIONS:
            errors.append(f"Invalid violation: {gt['primary_violation']}")

    if "severity" in gt:
        if gt["severity"] not in ["none", "low", "medium", "high", "critical"]:
            errors.append(f"Invalid severity: {gt['severity']}")

    if "key_factors" in gt:
        for factor in gt["key_factors"]:
            if factor not in FACTOR_LIST:
                errors.append(f"Invalid factor: {factor}")

    errors.extend(validate_scenario_record(scenario))
    return errors


def load_scenarios(filename: str) -> List[Dict[str, Any]]:
    """Load scenarios from a JSON file."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return []
    with open(filepath) as f:
        return json.load(f)


def save_scenarios(scenarios: List[Dict[str, Any]], filename: str):
    """Save scenarios to a JSON file."""
    filepath = DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"Saved {len(scenarios)} scenarios to {filepath}")


def generate_procedural_variations(
    base_scenarios: List[Dict[str, Any]],
    variations_per_scenario: int = 6,
    seed_offset: int = 1000,
) -> List[Dict[str, Any]]:
    """Generate procedural variations of all base scenarios."""
    variations = []
    for i, base in enumerate(base_scenarios):
        for j in range(variations_per_scenario):
            seed = seed_offset + i * 100 + j
            variant = create_procedural_variation(base, seed)
            errors = validate_scenario(variant)
            if errors:
                print(f"Warning: Invalid variant {variant['scenario_id']}: {errors}")
            else:
                variations.append(variant)
    return variations


# ============================================================================
# Category-based scenario templates
# ============================================================================

EASY_TEMPLATES = [
    # Obvious spam
    {
        "category": "obvious_spam",
        "texts": [
            "FREE FOLLOWERS! Get 10K real followers for just $5! DM now! Limited offer!",
            "MAKE $5000 DAILY working from home! Click here for secret method!",
            "Hot singles in your area want to meet YOU! Click the link below!",
            "Congratulations! You've won $1,000,000! Click to claim your prize!",
            "Buy cheap medications online! No prescription needed! Best prices!",
        ],
        "trigger": {"trigger_type": "auto_flag", "auto_flag_reason": "spam_score: 0.95"},
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "high",
            "key_factors": ["spam_commercial", "clear_violation_no_exception"],
        },
    },
    # Clear threats
    {
        "category": "explicit_threat",
        "texts": [
            "I know where you live and I'm coming for you tonight",
            "You better watch your back, I'll make you regret this",
            "I will hunt you down and destroy everything you love",
            "Your days are numbered, I'm going to end you",
            "Sleep with one eye open, I know your address",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 5, "report_categories": ["threats"]},
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "4.1",
            "severity": "critical",
            "key_factors": ["explicit_threat", "clear_violation_no_exception"],
        },
    },
    # Obvious false positives (metaphorical language)
    {
        "category": "false_positive_metaphor",
        "texts": [
            "This new album absolutely KILLS IT! Best music I've heard all year!",
            "My presentation totally bombed today, so embarrassing",
            "I'm dying to try that new restaurant everyone's talking about",
            "This code is murder to debug, been at it for hours",
            "The competition was brutal but we crushed them in the finals!",
        ],
        "trigger": {"trigger_type": "auto_flag", "auto_flag_reason": "keyword_match: violent terms"},
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["metaphorical_language", "no_violation_found", "auto_flag_low_confidence"],
        },
    },
    # Clear hate speech
    {
        "category": "clear_hate",
        "texts": [
            "All [group] are subhuman and should be removed from society",
            "[Slur] don't belong in this country, send them back",
            "The world would be better without [group], they're all criminals",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 10, "report_categories": ["hate_speech"]},
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "2.1",
            "severity": "high",
            "key_factors": ["hate_speech_direct", "clear_violation_no_exception"],
        },
    },
    # Doxxing
    {
        "category": "doxxing",
        "texts": [
            "Here's @user's real info: John Smith, 123 Main St, Phone: 555-1234",
            "Found their address: 456 Oak Ave Apt 7, let's pay them a visit",
            "Exposing @target: real name is Sarah, works at XYZ Corp",
            "@victim lives at 789 Pine St, here's their license plate too",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 8, "report_categories": ["privacy", "harassment"]},
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "7.1",
            "severity": "critical",
            "key_factors": ["privacy_violation", "clear_violation_no_exception"],
        },
    },
    # Scam links
    {
        "category": "scam_link",
        "texts": [
            "URGENT: Your account will be deleted! Verify now: bit.ly/scam123",
            "Bank alert: Suspicious activity detected! Login here to secure: fake-bank.com/verify",
            "Your package is waiting! Pay $2 shipping: tracking-scam.com/pay",
            "IRS NOTICE: You owe back taxes! Pay immediately or face arrest: irs-scam.net",
            "Apple Support: Your iCloud was hacked! Reset password: apple-phish.com",
        ],
        "trigger": {"trigger_type": "auto_flag", "auto_flag_reason": "phishing_score: 0.92"},
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "critical",
            "key_factors": ["spam_commercial", "harmful_link", "clear_violation_no_exception"],
        },
    },
    # Legitimate questions
    {
        "category": "legitimate_question",
        "texts": [
            "Can anyone recommend a good Python IDE? VS Code vs PyCharm?",
            "What's the best way to learn machine learning as a beginner?",
            "Has anyone tried the new iPhone? Is it worth upgrading?",
            "Looking for book recommendations on personal finance",
            "What's your favorite recipe for chocolate chip cookies?",
            "How do I fix a leaky faucet? Any DIY tips?",
        ],
        "trigger": {"trigger_type": "auto_flag", "auto_flag_reason": "new_account_post"},
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["no_violation_found", "auto_flag_low_confidence"],
        },
    },
    # Bot spam
    {
        "category": "bot_spam",
        "texts": [
            "Great post! Check out my profile for more amazing content! #follow4follow",
            "Wow this is so true! Visit my bio link for similar content!!!",
            "I agree 100%! BTW I make $500/day from home, DM for info",
            "This! So much this! Also check my page for exclusive deals",
        ],
        "trigger": {"trigger_type": "auto_flag", "auto_flag_reason": "bot_score: 0.88"},
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "5.2",
            "severity": "medium",
            "key_factors": ["spam_engagement", "clear_violation_no_exception"],
        },
    },
    # Normal disagreement
    {
        "category": "normal_disagreement",
        "texts": [
            "I completely disagree with your opinion on this topic",
            "That's a terrible take and here's why I think you're wrong",
            "Your argument doesn't make any sense, have you even researched this?",
            "I can't believe anyone actually thinks this way",
            "This is factually incorrect on multiple levels",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 1, "report_categories": ["harassment"]},
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["no_violation_found"],
        },
    },
]

MEDIUM_TEMPLATES = [
    # Gaming banter (needs thread context)
    {
        "category": "gaming_banter",
        "texts": [
            "I'm going to destroy you next match, prepare to die noob",
            "You're so dead when I respawn, camping trash",
            "I'll murder your whole team in the next round",
            "Watch your back in ranked, I'm coming for that elo",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 2, "report_categories": ["threats"]},
        "context_needed": ["thread_context"],
        "thread_context": [
            {"author": "opponent_user", "text": "gg lol you only won because I lagged"},
            {"author": "current_author", "text": "queue again then, I'm going to destroy you next match"},
            {"author": "opponent_user", "text": "fine, first to ten, no excuses this time"},
        ],
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["gaming_or_competition_context", "sarcasm_or_humor", "no_violation_found"],
        },
    },
    # Repeat offender borderline content (needs violation history)
    {
        "category": "repeat_offender",
        "texts": [
            "Check out my new product! Use code SALE20 for 20% off",
            "Just launched my course, link in bio!",
            "My new ebook is out, grab it while it's hot",
        ],
        "trigger": {"trigger_type": "auto_flag", "auto_flag_reason": "spam_score: 0.6"},
        "context_needed": ["author_violations"],
        "author_violations": [
            {"type": "spam", "date": "2026-03-01", "severity": "medium"},
            {"type": "spam", "date": "2026-02-15", "severity": "low"},
            {"type": "spam", "date": "2026-02-01", "severity": "low"},
        ],
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "5.1",
            "severity": "medium",
            "key_factors": ["spam_commercial", "repeat_offender"],
        },
    },
    # Public figure commentary (needs thread context)
    {
        "category": "public_figure",
        "texts": [
            "This politician is completely incompetent and should resign immediately",
            "The CEO's decisions are destroying the company, fire them now",
            "This celebrity is a terrible role model and shouldn't have a platform",
            "The coach made terrible calls all game, they need to be replaced",
            "This influencer is toxic and their fans should know the truth",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 5, "report_categories": ["harassment"]},
        "context_needed": ["thread_context"],
        "thread_context": [
            {
                "author": "policywatch",
                "text": "We are discussing the politician's public comments and voting record from yesterday's press conference.",
            },
            {
                "author": "community_mod",
                "text": "Keep criticism focused on public actions and decisions, not family members or private life.",
            },
        ],
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["public_figure_target", "no_violation_found"],
        },
    },
    # Medical discussion (needs community rules)
    {
        "category": "medical_content",
        "texts": [
            "Here's my experience with medication side effects [detailed medical info]",
            "Sharing my surgery recovery journey with photos",
            "Discussing dosage adjustments I made to my prescription",
            "My symptoms and how I've been self-managing them",
        ],
        "trigger": {"trigger_type": "auto_flag", "auto_flag_reason": "medical_content_detected"},
        "context_needed": ["community_rules"],
        "community_rules": (
            "Health communities allow first-person recovery stories and educational medical content, "
            "but prohibit personalized diagnosis, direct dosage instructions, or advice that overrides a clinician."
        ),
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["medical_or_educational_context", "community_exception_applies", "no_violation_found"],
        },
    },
    # Coordinated false reports (needs reporter credibility)
    {
        "category": "brigading_target",
        "texts": [
            "I think the latest movie was actually pretty good",
            "Unpopular opinion but I enjoyed the game ending",
            "Hot take: the old version was better",
            "Controversial take: this team deserved to win",
            "I know this is unpopular but I support the decision",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 35, "report_categories": ["spam", "trolling"]},
        "context_needed": ["reporter_credibility"],
        "reporter_credibility": {
            "total_reports": 48,
            "accurate_reports": 4,
            "false_reports": 44,
            "accuracy_rate": 0.08,
            "note": "Most reports came from accounts linked to an off-platform dogpile thread targeting the poster.",
        },
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["coordinated_reporting", "no_violation_found"],
        },
    },
    # Appeal with merit
    {
        "category": "appeal_merit",
        "texts": [
            "I hate the government's new policy, it's destroying our economy",
            "The administration is completely corrupt and everyone knows it",
            "These politicians are ruining everything with their terrible decisions",
        ],
        "trigger": {
            "trigger_type": "appeal",
            "original_decision": "remove",
            "appeal_text": "This was removed for hate speech but it's political criticism",
        },
        "context_needed": ["thread_context"],
        "thread_context": [
            {
                "author": "current_author",
                "text": "I am talking about elected officials and their public decisions, not race or religion.",
            },
            {
                "author": "appeal_reviewer",
                "text": "The surrounding replies are political argument about a tax bill, not attacks on a protected group.",
            },
        ],
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["appeal_has_merit", "no_violation_found"],
        },
    },
    # Trusted contributor borderline (needs author profile)
    {
        "category": "trusted_borderline",
        "texts": [
            "This is ridiculous, I can't believe the mods allow this garbage",
            "Some people in this community are completely clueless",
            "I'm so tired of seeing the same stupid questions every day",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 3, "report_categories": ["toxicity"]},
        "context_needed": ["author_profile"],
        "ground_truth": {
            "correct_decision": "warn",
            "primary_violation": "1.1",
            "severity": "low",
            "key_factors": ["trusted_contributor", "borderline_requires_judgment"],
        },
    },
]

HARD_TEMPLATES = [
    # Policy conflict (misinfo vs opinion)
    {
        "category": "misinfo_opinion",
        "texts": [
            "In my opinion, the official statistics are wrong and here's why...",
            "I personally believe the mainstream narrative is misleading",
            "Based on my research, the experts are ignoring key evidence",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 15, "report_categories": ["misinformation"]},
        "context_needed": ["author_violations", "similar_precedents"],
        "author_violations": [
            {"type": "misleading_health_claim", "date": "2026-02-11", "severity": "low"},
        ],
        "similar_precedents": [
            {
                "post": "A user framed unverified health statistics as personal skepticism.",
                "decision": "escalate",
                "reason": "Opinion framing and factual public-health claims were mixed together.",
            },
            {
                "post": "A user challenged official numbers without linking direct harm.",
                "decision": "escalate",
                "reason": "Borderline misinformation with precedent-setting implications.",
            },
        ],
        "ground_truth": {
            "correct_decision": "escalate",
            "primary_violation": "6.1",
            "severity": "medium",
            "key_factors": ["misinformation_opinion", "policy_sections_conflict", "borderline_requires_judgment"],
        },
    },
    # Cross-cultural expression
    {
        "category": "cultural_context",
        "texts": [
            "You look so fat and healthy! You must be eating well!",
            "Your baby is so chubby! The parents are doing great!",
            "You've gained weight! You look prosperous now!",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 6, "report_categories": ["body_shaming"]},
        "context_needed": ["author_profile", "thread_context"],
        "thread_context": [
            {
                "author": "family_friend",
                "text": "In our family this is a compliment about being cared for, not an insult about appearance.",
            },
            {
                "author": "current_author",
                "text": "My aunt says this every holiday when she thinks someone looks well-fed and healthy.",
            },
        ],
        "ground_truth": {
            "correct_decision": "escalate",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["cultural_context", "borderline_requires_judgment", "no_violation_found"],
        },
    },
    # Coded hate speech
    {
        "category": "coded_hate",
        "texts": [
            "Check the early life section, it tells you everything you need to know",
            "The usual suspects are behind this, just follow the money",
            "Certain people always seem to be in charge, coincidence?",
        ],
        "trigger": {"trigger_type": "proactive_audit", "audit_reason": "high_risk_user_sample"},
        "context_needed": ["author_violations", "similar_precedents"],
        "similar_precedents": [
            {
                "post": "The phrase 'early life section' was used to imply ethnic conspiracy in a politics thread.",
                "decision": "warn",
                "reason": "Coded hate indicator with enough ambiguity to avoid immediate removal.",
            },
            {
                "post": "A finance thread used 'usual suspects' to insinuate antisemitic stereotypes.",
                "decision": "warn",
                "reason": "Dogwhistle phrasing toward a protected group.",
            },
        ],
        "ground_truth": {
            "correct_decision": "warn",
            "primary_violation": "2.1",
            "severity": "medium",
            "key_factors": ["hate_speech_coded", "borderline_requires_judgment"],
        },
    },
    # Satire detection
    {
        "category": "satire",
        "texts": [
            "BREAKING: Government Announces Mandatory Happiness Program",
            "Scientists Discover That Water Is Indeed Wet, More at 11",
            "Local Man Convinced His Opinion Is The Only Valid One",
            "Study Shows 100% of People Who Drink Water Eventually Die",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 20, "report_categories": ["misinformation"]},
        "context_needed": ["linked_content_summary", "author_profile"],
        "linked_content_summary": (
            "Linked article is from The Irony Ledger, a satire outlet. The page footer and about box "
            "explicitly label the publication as parody, and the article exaggerates events for comedic effect."
        ),
        "ground_truth": {
            "correct_decision": "approve",
            "primary_violation": "none",
            "severity": "none",
            "key_factors": ["satire_or_parody", "no_violation_found"],
        },
    },
    # Conflicting signals (violation + brigading)
    {
        "category": "conflicting_signals",
        "texts": [
            "You people are all pathetic, but jk love you guys",
            "This community is full of idiots, except the cool ones",
            "Everyone here is wrong but in a friendly way",
        ],
        "trigger": {"trigger_type": "user_report", "report_count": 15, "report_categories": ["harassment"]},
        "context_needed": ["author_violations", "reporter_credibility", "thread_context"],
        "author_violations": [
            {"type": "targeted_harassment", "date": "2026-02-17", "severity": "low"},
            {"type": "targeted_harassment", "date": "2026-01-28", "severity": "low"},
        ],
        "reporter_credibility": {
            "total_reports": 19,
            "accurate_reports": 15,
            "false_reports": 4,
            "accuracy_rate": 0.79,
            "note": "Reporters usually flag genuine harassment, but this thread is between long-time regulars.",
        },
        "thread_context": [
            {"author": "current_author", "text": "you are all pathetic, but jk love you guys"},
            {"author": "friend_user", "text": "lol we know, now stop dodging the queue and run it back"},
            {"author": "another_regular", "text": "same energy as always, keep it playful and stop pinging the mods"},
        ],
        "ground_truth": {
            "correct_decision": "warn",
            "primary_violation": "3.1",
            "severity": "low",
            "key_factors": ["sarcasm_or_humor", "repeat_offender", "borderline_requires_judgment"],
        },
    },
    # Proactive audit new account
    {
        "category": "audit_new_account",
        "texts": [
            "Just joined! Here's a link to my website with great deals",
            "New here but wanted to share this amazing opportunity",
            "First post! Check out this investment I found",
        ],
        "trigger": {"trigger_type": "proactive_audit", "audit_reason": "new_account_high_activity"},
        "context_needed": ["author_profile", "author_violations"],
        "ground_truth": {
            "correct_decision": "escalate",
            "primary_violation": "5.1",
            "severity": "medium",
            "key_factors": ["new_account", "spam_commercial", "borderline_requires_judgment"],
        },
    },
    # Appeal lacking merit
    {
        "category": "appeal_no_merit",
        "texts": [
            "I'll find you and make you pay for banning me",
            "The mods are dead men walking and I hope they all suffer for this",
            "You're all going to regret this, I know people",
        ],
        "trigger": {
            "trigger_type": "appeal",
            "original_decision": "remove",
            "appeal_text": "This was unfair, I was just expressing my frustration",
        },
        "context_needed": ["thread_context", "author_violations"],
        "author_violations": [
            {"type": "threats", "date": "2026-03-03", "severity": "high"},
            {"type": "abusive_appeal", "date": "2026-02-22", "severity": "medium"},
        ],
        "thread_context": [
            {
                "author": "moderator_note",
                "text": "Original removal followed direct threats sent after a temporary suspension notice.",
            },
            {
                "author": "current_author",
                "text": "I know where the mod meetup is and someone should make them pay for this ban.",
            },
        ],
        "ground_truth": {
            "correct_decision": "remove",
            "primary_violation": "4.1",
            "severity": "high",
            "key_factors": ["explicit_threat", "appeal_lacks_merit", "clear_violation_no_exception"],
        },
    },
]


def generate_from_template(
    template: Dict[str, Any],
    text_index: int,
    scenario_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate a scenario from a template."""
    rng = random.Random(seed)

    text = template["texts"][text_index % len(template["texts"])]

    scenario = {
        "scenario_id": scenario_id,
        "difficulty": "easy" if template in EASY_TEMPLATES else ("medium" if template in MEDIUM_TEMPLATES else "hard"),
        "category": template["category"],
        "content_item": {
            "post_id": f"p_{rng.randint(30000, 99999)}",
            "text": text,
            "author_id": f"{rng.choice(USERNAMES)}_{rng.randint(100, 999)}",
            "community": rng.choice(COMMUNITIES),
            "timestamp": random_timestamp(seed),
            "media_type": "text",
            "media_description": None,
        },
        "trigger_info": {
            "report_categories": template["trigger"].get("report_categories", []),
            "sample_report_reason": None,
            "original_decision": None,
            "appeal_text": None,
            "audit_reason": None,
            **template["trigger"],
        },
        "available_context": {
            "author_profile": {
                "bio": rng.choice(BIOS),
                "account_age_days": rng.randint(30, 1000),
                "follower_count": rng.randint(10, 5000),
                "post_count": rng.randint(10, 2000),
                "communities": rng.sample(COMMUNITIES, k=min(4, rng.randint(1, 5))),
            },
            "author_violations": template.get("author_violations", []),
            "thread_context": None,
            "community_rules": "Standard community guidelines apply.",
            "linked_content_summary": None,
            "similar_precedents": None,
            "reporter_credibility": None,
        },
        "ground_truth": copy.deepcopy(template["ground_truth"]),
    }

    for context_key in [
        "author_profile",
        "author_violations",
        "thread_context",
        "community_rules",
        "linked_content_summary",
        "similar_precedents",
        "reporter_credibility",
    ]:
        if context_key in template:
            scenario["available_context"][context_key] = copy.deepcopy(template[context_key])

    # Add context_needed if specified
    if "context_needed" in template:
        scenario["ground_truth"]["context_needed"] = template["context_needed"]
    else:
        scenario["ground_truth"]["context_needed"] = []

    # Add explanation
    scenario["ground_truth"]["explanation"] = f"Generated scenario for category: {template['category']}"

    return scenario


def generate_all_scenarios():
    """Generate all scenarios and save to files."""
    print("Loading existing gold scenarios...")

    easy_gold = load_scenarios("scenarios_easy.json")
    medium_gold = load_scenarios("scenarios_medium.json")
    hard_gold = load_scenarios("scenarios_hard.json")

    print(f"Found {len(easy_gold)} easy, {len(medium_gold)} medium, {len(hard_gold)} hard gold scenarios")

    # Generate procedural variations
    print("\nGenerating procedural variations...")
    easy_vars = generate_procedural_variations(easy_gold, variations_per_scenario=14, seed_offset=1000)
    medium_vars = generate_procedural_variations(medium_gold, variations_per_scenario=16, seed_offset=2000)
    hard_vars = generate_procedural_variations(hard_gold, variations_per_scenario=12, seed_offset=3000)

    print(f"Generated {len(easy_vars)} easy, {len(medium_vars)} medium, {len(hard_vars)} hard variations")

    # Generate from templates
    print("\nGenerating from templates...")
    easy_template_scenarios = []
    for i, template in enumerate(EASY_TEMPLATES):
        for j in range(len(template["texts"])):
            scenario = generate_from_template(
                template, j, f"easy_gen_{i:02d}_{j:02d}", seed=4000 + i * 100 + j
            )
            errors = validate_scenario(scenario)
            if not errors:
                easy_template_scenarios.append(scenario)

    medium_template_scenarios = []
    for i, template in enumerate(MEDIUM_TEMPLATES):
        for j in range(len(template["texts"])):
            scenario = generate_from_template(
                template, j, f"med_gen_{i:02d}_{j:02d}", seed=5000 + i * 100 + j
            )
            scenario["difficulty"] = "medium"
            errors = validate_scenario(scenario)
            if not errors:
                medium_template_scenarios.append(scenario)

    hard_template_scenarios = []
    for i, template in enumerate(HARD_TEMPLATES):
        for j in range(len(template["texts"])):
            scenario = generate_from_template(
                template, j, f"hard_gen_{i:02d}_{j:02d}", seed=6000 + i * 100 + j
            )
            scenario["difficulty"] = "hard"
            errors = validate_scenario(scenario)
            if not errors:
                hard_template_scenarios.append(scenario)

    print(f"Generated {len(easy_template_scenarios)} easy, {len(medium_template_scenarios)} medium, {len(hard_template_scenarios)} hard from templates")

    # Combine all scenarios
    all_easy = easy_gold + easy_vars + easy_template_scenarios
    all_medium = medium_gold + medium_vars + medium_template_scenarios
    all_hard = hard_gold + hard_vars + hard_template_scenarios

    # Save
    save_scenarios(all_easy, "scenarios_easy.json")
    save_scenarios(all_medium, "scenarios_medium.json")
    save_scenarios(all_hard, "scenarios_hard.json")

    # Print summary
    total = len(all_easy) + len(all_medium) + len(all_hard)
    print(f"\n{'='*50}")
    print(f"TOTAL SCENARIOS: {total}")
    print(f"  Easy: {len(all_easy)}")
    print(f"  Medium: {len(all_medium)}")
    print(f"  Hard: {len(all_hard)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    generate_all_scenarios()
