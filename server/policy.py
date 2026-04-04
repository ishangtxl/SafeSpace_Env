# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Platform policy and factor definitions for SafeSpace.

Contains:
- FACTOR_LIST: Predefined factors the agent can cite in decisions
- POLICY_SECTIONS: Valid policy section IDs
- PLATFORM_POLICY: Full policy document text
"""

from typing import Dict, List

# ============================================================================
# Factor List - All factors an agent can cite in their decision
# ============================================================================

FACTOR_LIST: List[str] = [
    # Context factors
    "sarcasm_or_humor",
    "metaphorical_language",
    "cultural_context",
    "gaming_or_competition_context",
    "medical_or_educational_context",
    "artistic_expression",
    "satire_or_parody",
    "quoting_or_reporting",
    # User/history factors
    "repeat_offender",
    "new_account",
    "trusted_contributor",
    "public_figure_target",
    "public_figure_author",
    # Report/trigger factors
    "coordinated_reporting",
    "trusted_reporter",
    "auto_flag_high_confidence",
    "auto_flag_low_confidence",
    "appeal_has_merit",
    "appeal_lacks_merit",
    # Content factors
    "explicit_threat",
    "implicit_threat",
    "hate_speech_direct",
    "hate_speech_coded",
    "spam_commercial",
    "spam_engagement",
    "misinformation_factual",
    "misinformation_opinion",
    "privacy_violation",
    "harmful_link",
    "graphic_content_justified",
    "graphic_content_unjustified",
    # Policy factors
    "community_exception_applies",
    "policy_sections_conflict",
    "borderline_requires_judgment",
    "clear_violation_no_exception",
    "no_violation_found",
]

# ============================================================================
# Policy Sections - Valid policy section IDs
# ============================================================================

POLICY_SECTIONS: Dict[str, str] = {
    "1.0": "General Conduct",
    "1.1": "Baseline Expectations",
    "1.2": "Good Faith Participation",
    "2.0": "Hate Speech & Discrimination",
    "2.1": "Attacks on Protected Groups",
    "2.2": "Hate Speech Exceptions",
    "3.0": "Harassment & Bullying",
    "3.1": "Targeted Harassment",
    "3.2": "Harassment Exceptions",
    "4.0": "Violence & Threats",
    "4.1": "Credible Threats",
    "4.2": "Violence Exceptions",
    "5.0": "Spam & Manipulation",
    "5.1": "Commercial Spam",
    "5.2": "Coordinated Inauthentic Behavior",
    "6.0": "Misinformation",
    "6.1": "Factual Misinformation",
    "6.2": "Opinion vs Fact",
    "7.0": "Privacy",
    "7.1": "Doxxing",
    "7.2": "Privacy Exceptions",
    "8.0": "Repeat Offenders & Escalation",
    "8.1": "Escalation Thresholds",
    "8.2": "Repeat Offender Policy",
    "none": "No Violation",
}

# Valid decisions
VALID_DECISIONS = ["approve", "remove", "escalate", "warn"]

# Valid severities
VALID_SEVERITIES = ["none", "low", "medium", "high", "critical"]

# ============================================================================
# Platform Policy Document
# ============================================================================

PLATFORM_POLICY = """
# SafeSpace Platform Content Moderation Policy

## 1. General Conduct

### 1.1 Baseline Expectations
All users are expected to engage respectfully with others. Content should contribute constructively to discussions and communities.

### 1.2 Good Faith Participation
Users should participate in good faith, avoiding intentional disruption or manipulation of platform features.

---

## 2. Hate Speech & Discrimination

### 2.1 Attacks on Protected Groups
**Prohibited:** Content that attacks, demeans, or incites hatred against individuals or groups based on protected characteristics including race, ethnicity, national origin, religion, gender, gender identity, sexual orientation, disability, or serious medical condition.

**Severity Levels:**
- **Critical:** Calls for violence against protected groups, dehumanizing language
- **High:** Direct slurs, explicit hate speech targeting groups
- **Medium:** Coded hate speech, dogwhistles, stereotyping
- **Low:** Insensitive generalizations without malicious intent

### 2.2 Hate Speech Exceptions
Content may be permitted when:
- Quoting hate speech to condemn or discuss it
- Educational or documentary context
- Satire or parody with clear social commentary
- Reclaimed language used within the affected community
- Academic or journalistic reporting

---

## 3. Harassment & Bullying

### 3.1 Targeted Harassment
**Prohibited:** Repeated, targeted attacks against specific individuals intended to humiliate, intimidate, or silence them.

**Severity Levels:**
- **Critical:** Threats of physical harm, coordinated harassment campaigns
- **High:** Sustained personal attacks, revealing private information
- **Medium:** Repeated negative attention, mocking
- **Low:** Isolated negative comments without pattern

### 3.2 Harassment Exceptions
Content may be permitted when:
- Criticizing public figures in their public capacity
- Mutual heated debate where both parties are engaged
- Competitive banter in gaming or sports contexts
- Fair criticism of public statements or actions

---

## 4. Violence & Threats

### 4.1 Credible Threats
**Prohibited:** Content that threatens violence against individuals or groups, particularly when specific, credible, and imminent.

**Severity Levels:**
- **Critical:** Specific threats with means, target, and timeframe
- **High:** Direct threats of violence without specificity
- **Medium:** Implied threats, intimidation
- **Low:** Aggressive language without clear violent intent

### 4.2 Violence Exceptions
Content may be permitted when:
- Metaphorical language clearly not literal ("I killed that presentation")
- Gaming context ("I'm going to destroy you next round")
- Fiction or creative writing
- Self-defense discussion or advocacy
- Historical documentation or news reporting

---

## 5. Spam & Manipulation

### 5.1 Commercial Spam
**Prohibited:** Unsolicited commercial content, get-rich-quick schemes, fraudulent offers, excessive self-promotion.

**Severity Levels:**
- **Critical:** Scam links, financial fraud, phishing
- **High:** Aggressive commercial spam, MLM promotion
- **Medium:** Excessive self-promotion, repetitive posting
- **Low:** Borderline promotional content in context

### 5.2 Coordinated Inauthentic Behavior
**Prohibited:** Bot networks, coordinated brigading, engagement manipulation, astroturfing.

**Severity Levels:**
- **Critical:** Large-scale bot operations, election manipulation
- **High:** Coordinated harassment campaigns, vote manipulation
- **Medium:** Small-scale brigading, artificial engagement
- **Low:** Suspicious but unconfirmed coordination

---

## 6. Misinformation

### 6.1 Factual Misinformation
**Prohibited:** Demonstrably false claims that pose risk to public health, safety, or democratic processes.

**Severity Levels:**
- **Critical:** False health claims causing harm, election misinformation
- **High:** Dangerous conspiracy theories, false emergency information
- **Medium:** Misleading claims with some factual basis
- **Low:** Inaccuracies without harmful intent

### 6.2 Opinion vs Fact
Content expressing opinions, even unpopular ones, is generally protected. The line is crossed when:
- Opinions are presented as established facts
- False factual claims are used to support opinions
- Content targets specific individuals with false claims

---

## 7. Privacy

### 7.1 Doxxing
**Prohibited:** Sharing private information without consent, including home addresses, phone numbers, financial information, or private medical information.

**Severity Levels:**
- **Critical:** Full doxxing with address and identity, endangering safety
- **High:** Partial private information with clear malicious intent
- **Medium:** Private information shared without clear malice
- **Low:** Semi-public information in inappropriate context

### 7.2 Privacy Exceptions
Content may be permitted when:
- Information is already public record
- Public figures in their public capacity
- Newsworthy events with clear public interest
- Information shared with explicit consent

---

## 8. Repeat Offenders & Escalation

### 8.1 Escalation Thresholds
Moderators should escalate decisions to senior review when:
- The case involves significant ambiguity
- Multiple policy sections apply with conflicting guidance
- The decision could set important precedent
- High-profile accounts or sensitive topics are involved
- Confidence is low despite investigation

### 8.2 Repeat Offender Policy
Users with prior violations receive heightened scrutiny:
- **3+ violations in 90 days:** Borderline content leans toward removal
- **5+ violations in 90 days:** Consider account-level action
- **Pattern of similar violations:** Treat as intentional
- **Long history with recent good behavior:** Consider rehabilitation

De-escalation: Users who maintain good standing for 6+ months after violations may return to standard review thresholds.

---

## Moderation Decision Guidelines

### Available Decisions
1. **Approve:** Content does not violate policy, or exceptions apply
2. **Remove:** Content clearly violates policy
3. **Warn:** Borderline content; keep visible but notify author
4. **Escalate:** Genuinely ambiguous; send to senior moderator

### Decision Factors
Consider these factors when making decisions:
- Literal vs. contextual meaning of content
- Author's history and intent signals
- Community norms and expectations
- Precedent from similar cases
- Potential harm vs. free expression value
- Reporter credibility and possible brigading

### Confidence & Calibration
- High confidence decisions (>80%) should be made when evidence is clear
- Medium confidence (40-80%) is appropriate for most context-dependent cases
- Low confidence (<40%) should trigger escalation consideration
- Overconfident wrong decisions are worse than humble escalations
"""


def get_policy_section_name(section_id: str) -> str:
    """Get the human-readable name for a policy section ID."""
    return POLICY_SECTIONS.get(section_id, "Unknown Section")


def is_valid_factor(factor: str) -> bool:
    """Check if a factor is in the valid FACTOR_LIST."""
    return factor in FACTOR_LIST


def is_valid_policy_section(section_id: str) -> bool:
    """Check if a policy section ID is valid."""
    return section_id in POLICY_SECTIONS


def is_valid_decision(decision: str) -> bool:
    """Check if a decision is valid."""
    return decision in VALID_DECISIONS


def is_valid_severity(severity: str) -> bool:
    """Check if a severity level is valid."""
    return severity in VALID_SEVERITIES
