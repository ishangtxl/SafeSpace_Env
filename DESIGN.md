# SafeSpace Design Document

This document explains the design decisions behind SafeSpace, a content moderation RL environment. It serves as both internal reference and an artifact demonstrating thoughtful, principled design.

---

## 1. Reward System Design

### Why 4-Component Terminal Reward

The terminal reward combines four orthogonal signals:

| Component | Weight | Range | Purpose |
|-----------|--------|-------|---------|
| **Decision** | ~61% | -0.18 to 0.55 | Core moderation accuracy |
| **Factor Overlap** | ~17% | 0.00 to 0.15 | Interpretability / reasoning quality |
| **Efficiency** | ~11% | 0.00 to 0.10 | Investigation parsimony |
| **Calibration** | ~11% | -0.10 to 0.10 | Confidence accuracy |

**Rationale:**
- **Decision dominates** because correct moderation is the primary goal
- **Factor overlap** rewards agents that identify the right reasoning, not just the right answer
- **Efficiency** incentivizes direct decisions when context isn't needed
- **Calibration** penalizes overconfident wrong answers and rewards appropriate uncertainty

### Trajectory Shaping

Investigation actions receive immediate feedback:
- `+0.05` for requesting context that's actually needed and retrievable
- `-0.03` for requesting irrelevant context (wastes budget)
- `-0.05` for duplicate requests (penalizes stateless behavior)
- `-0.06` for invalid actions (penalizes malformed inputs)

**Design choice:** Trajectory rewards are capped at `±0.15` total to prevent them from overwhelming terminal reward. The trajectory signal teaches search strategy without dominating the moderation quality signal.

### Score Table Alignment

We maintain **consistent ordering** between reward scores (for training) and grade scores (for evaluation):

```
perfect > decision_and_violation_correct > decision_correct_only > adjacent > wrong
```

This alignment prevents policy drift where agents optimize for reward but are evaluated on grades with different relative values.

---

## 2. Grading Philosophy

### Decision Grade Tiers

| Grade | Reward | Condition |
|-------|--------|-----------|
| Perfect | 0.55 | Decision + violation + severity all correct |
| Decision+Violation | 0.50 | Decision + violation correct, severity wrong |
| Decision Only | 0.40 | Only decision correct |
| Adjacent | 0.08 | Decision is adjacent (escalate↔warn, remove↔escalate) |
| Wrong | -0.10 | Completely wrong decision |

**Why adjacent partial credit?**

In real moderation, the difference between "warn" and "escalate" is often judgment-dependent. Adjacent decisions show the agent understood the case was borderline, even if it picked the wrong side. However, we keep this bonus small (0.08) to prevent gaming.

### Dangerous False Negative Penalty

Approving content that should be removed with high/critical severity receives the harshest penalty (-0.18). This reflects real-world asymmetric costs: missing a genuine threat is worse than over-moderating borderline content.

### Wrong Decision Penalty

We explicitly penalize wrong decisions with -0.10 (not 0.0). This provides a learning signal to avoid random guessing and creates clear separation from adjacent decisions.

---

## 3. Scenario Design

### Workflow Framing

We model three concrete moderation workflows rather than three abstract buckets:

- **Direct triage** (`clear_violations`): obvious removals and obvious false positives where deciding immediately is correct
- **Context-dependent investigation** (`context_dependent`): cases where one well-chosen context request resolves the decision
- **Policy / escalation review** (`policy_edge_cases`): precedent-sensitive cases where multiple signals conflict and calibrated uncertainty matters

### Three-Tier Difficulty

| Tier | Count | Characteristic | Optimal Strategy |
|------|-------|----------------|------------------|
| Easy | 117 | Text alone is sufficient | Decide immediately |
| Medium | 140 | One context request changes the answer | Targeted investigation |
| Hard | 110 | Multiple signals conflict | Multi-context + calibration |

**Why this progression?**
- Easy teaches basic policy understanding
- Medium teaches when to investigate vs. decide
- Hard teaches nuance, precedent, and appropriate uncertainty

### Canonical Benchmark vs Full Corpus

We separate the environment into two evaluation layers:

- **Canonical benchmark:** 60 curated scenarios, 20 per task, defined in `server/data/benchmark_manifest.json`
- **Full corpus:** 367 total scenarios used for broader regression testing and stress coverage

The canonical benchmark is the headline score we report in the README and baseline script. The full corpus still matters, but we do not treat metadata-only procedural variants as benchmark depth.
Manifest version `2026-04-03.2` also rebalances the hard split away from a remove-heavy concentration by adding one escalation case, one approve satire case, and one warn coded-hate case from the existing corpus.

### Decision Distribution (Hard Tier)

We maintain balanced decision distribution in hard scenarios:

| Decision | Count | Percentage |
|----------|-------|------------|
| Escalate | 35 | 31.8% |
| Warn | 32 | 29.1% |
| Remove | 24 | 21.8% |
| Approve | 19 | 17.3% |

**Rationale:** Each decision type appears at ≥15% to ensure agents learn all four outcomes. Earlier versions were skewed toward escalate/warn (75%), which let agents game by always hedging.

### Context Requirements

Each scenario specifies which context sources are actually needed:
- Easy: Usually none (text is sufficient)
- Medium: Usually one targeted source
- Hard: Usually two complementary sources

This enables trajectory reward to guide efficient investigation. A corpus-integrity rule now enforces that any context source listed in `context_needed` must also be retrievable from `available_context`.

---

## 4. Anti-Gaming Measures

### Adjacent Decision Limit

Adjacent decisions receive only 0.08 reward (was 0.12). Combined with calibration bonus of 0.02 for low-confidence adjacent decisions, the maximum "hedging" reward is:

```
0.08 (adjacent) + 0.02 (low-conf calibration) = 0.10 (11% of perfect)
```

This is low enough to discourage always-escalate strategies while still rewarding genuine uncertainty recognition.

### Repeat Offender Scenarios

17 hard scenarios feature "borderline content + repeat offender history = remove". This tests whether agents can integrate author history with surface content analysis, not just pattern-match on text.

### Calibration Mechanics

The calibration component specifically addresses hedging:
- High confidence + wrong = -0.10 penalty
- Low confidence + correct on hard case = +0.08 bonus (appropriate uncertainty)
- Low confidence + correct on easy case = 0.00 (underconfident)

This creates pressure to express appropriate certainty rather than always hedging or always being confident.

---

## 5. Policy Design

### Violation Categories

We cover the major real-world moderation categories:

| Section | Category | Real-World Prevalence |
|---------|----------|----------------------|
| 2.x | Hate Speech | High |
| 3.x | Harassment | High |
| 4.x | Violence/Threats | Medium |
| 5.x | Spam/Manipulation | High |
| 6.x | Misinformation | Medium-High |
| 7.x | Privacy | Medium |
| 8.x | Repeat Offenders | Cross-cutting |

### Exception Structure

Each violation category includes explicit exceptions:
- Quoting to condemn
- Educational/documentary context
- Gaming/competitive context
- Satire/parody
- Reclaimed language

These exceptions make the environment realistic—agents must understand context, not just keyword-match.

### Escalation Thresholds

Policy section 8.1 defines when escalation is appropriate:
- Significant ambiguity
- Conflicting policy sections
- Potential precedent-setting
- High-profile accounts
- Low confidence despite investigation

This gives agents concrete criteria for when "escalate" is the right answer rather than a cop-out.

---

## 6. Technical Decisions

### Deterministic Grading

All grading is programmatic with no LLM judges. This ensures:
- Reproducibility across runs
- Fast evaluation (no API calls)
- No evaluation cost scaling
- Clear, auditable scoring logic

### Episode Budget

8 actions maximum per episode. This is enough to gather comprehensive context but creates real tradeoffs—agents can't request everything.

### Typed Models

All actions, observations, and states use Pydantic models with explicit types. This enables:
- Validation at the boundary
- Clear API contracts
- IDE autocomplete for developers
- Automatic schema generation

### Typed State Contract

The canonical benchmark path is the typed `SafeSpaceEnv` WebSocket client, which returns the full `ModerationState` contract and is what we test in preflight. SafeSpace also overrides the default OpenEnv GET `/schema` and `/state` routes so the public HTTP contract now exposes `ModerationState` directly for validators, docs, and independent reviewers.

---

## 7. Evaluation Philosophy

### Success Metrics

We evaluate agents on:
1. **Task grade** (normalized 0-1): Weighted combination of decision, factors, efficiency, calibration
2. **Per-decision-type accuracy**: Ensures agents learn all four outcomes
3. **Calibration curve**: Confidence vs. actual correctness

### Baseline Expectations

| Tier | Expected Baseline | Rationale |
|------|-------------------|-----------|
| Easy | 0.90-0.92 | Canonical easy cases should be highly solvable for a strong frontier model |
| Medium | 0.60-0.63 | Correct context selection matters and wrong confidence still hurts |
| Hard | 0.56-0.59 | Hard cases remain meaningfully challenging despite a curated canonical split |

---

## 8. Design Tradeoffs

### What We Chose NOT To Do

1. **No LLM judges**: Reproducibility > nuance
2. **No curriculum learning**: Simplicity > sophistication
3. **No multi-agent scenarios**: Focus > complexity
4. **No real-time streaming**: Batch evaluation > latency optimization
5. **No image understanding**: Text-focused > multimodal complexity

### Why These Tradeoffs

This environment optimizes for:
- Clear, demonstrable correctness
- Reproducible benchmarking
- Efficient evaluation
- Focused scope

More complex features can be added later without architectural changes.

---

## Summary

SafeSpace is designed around these principles:

1. **Aligned incentives**: Reward and grade scores have consistent ordering
2. **Anti-gaming**: Low adjacent bonus, calibration penalties, balanced distributions
3. **Realistic complexity**: Policy exceptions, repeat offender logic, context dependencies
4. **Efficient evaluation**: Deterministic grading, typed models, clear metrics
5. **Principled defaults**: Episode budget, trajectory caps, score ranges all have explicit rationale

The goal is an environment that teaches real moderation skills, not pattern matching or hedging strategies.
