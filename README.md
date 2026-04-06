---
title: SafeSpace Content Moderation Environment
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - openenv-0.2.3
  - content-moderation
  - reinforcement-learning
  - multi-step-reasoning
---

# SafeSpace: Content Moderation OpenEnv

SafeSpace is a real-world OpenEnv environment for training and evaluating agents on content moderation workflows. The agent does not just classify text. It receives a reported post, decides whether to investigate additional evidence under an action budget, and then makes a structured moderation decision with a confidence score and cited factors.

## Project Snapshot

- Real-world task: multi-step social platform content moderation, not a toy game
- Full OpenEnv contract: typed `step()` / `reset()` / `state()` API with `openenv.yaml`
- Three graded tasks: easy, medium, and hard moderation workflows
- Meaningful reward shaping: partial-progress trajectory reward plus terminal quality reward
- Reproducible baseline: `inference.py` uses the OpenAI client and fixed seed
- Deployment ready: Docker build, OpenEnv validation, HF Space support, and a repository validator script

SafeSpace is designed around the part of moderation that is hard for static classifiers:
- deciding when the surface text is enough
- deciding when context changes the answer
- handling asymmetric costs for false negatives vs over-moderation
- learning when escalation is better than bluffing

## Why This Is an RL Environment, Not a Classifier

A one-shot classifier sees only the post and predicts a label. Real moderation work is sequential:
- moderators inspect the post
- they choose which context to fetch
- each fetch has a cost
- they stop when they have enough evidence
- they make a decision with a level of certainty

SafeSpace turns that workflow into an RL problem. Investigation actions produce trajectory-level reward, bad investigation habits are penalized, and the terminal reward measures moderation quality, calibration, and efficiency together.

## Environment Summary

- Domain: social media content moderation
- Interface: typed OpenEnv `step()` / `reset()` / `state()` API
- Tasks: 3 deterministic moderation workflows from direct triage to escalation review
- Benchmark split: 60 canonical benchmark scenarios (`server/data/benchmark_manifest.json`) plus a 367-scenario full corpus stress test
- Reward: trajectory shaping plus terminal decision reward
- Graders: deterministic task grades in `[0.0, 1.0]`, no LLM judge in the environment
- Runtime: FastAPI / OpenEnv server

## Tasks

| Task | Difficulty | Scenarios | What makes it hard |
|------|------------|-----------|--------------------|
| `clear_violations` | easy | 117 | Text alone is usually enough; over-investigation wastes reward |
| `context_dependent` | medium | 140 | One targeted context request often flips the correct answer |
| `policy_edge_cases` | hard | 110 | Multiple signals conflict; escalation, calibration, and precedent matter |

### Workflow 1: Direct Triage (`clear_violations`)

Obvious spam, explicit threats, doxxing, clear hate speech, and obvious false positives. The optimal policy usually decides immediately.

### Workflow 2: Context-Dependent Investigation (`context_dependent`)

Cases where one missing fact changes the right answer: gaming banter, appeal review, brigading victims, repeat offenders, or harmful links hidden behind harmless post text.

### Workflow 3: Policy / Escalation Review (`policy_edge_cases`)

Ambiguous cases where the agent must combine multiple evidence sources: satire vs misinformation, coded harassment, cross-cultural language, public-figure privacy edge cases, and quoted harmful claims used for correction or education.

## Action Space

### Investigation actions

Each investigation action consumes one step from the episode budget:

- `request_author_profile`
- `request_author_violations`
- `request_thread_context`
- `request_community_rules`
- `request_linked_content`
- `request_similar_precedents`
- `request_reporter_credibility`

### Terminal action

```python
ModerationAction(
    action_type="decide",
    decision="approve",          # approve | remove | warn | escalate
    primary_violation="none",    # policy section or "none"
    severity="none",             # none | low | medium | high | critical
    confidence=0.82,             # 0.0 to 1.0
    key_factors=["no_violation_found", "gaming_or_competition_context"],
)
```

## Observation Space

Each observation contains:

- `content_item`: the post being reviewed
- `trigger_info`: how it entered moderation
- `gathered_context`: evidence fetched so far
- `platform_policy`: platform-wide policy text
- `available_factors`: factors the agent can cite in its decision
- `actions_taken`, `max_actions`, `action_history`
- `feedback`
- `reward_breakdown`: typed reward payload available on both intermediate and terminal steps
- `task_grade`, `grade_breakdown`: typed normalized grading payload on terminal steps

Example:

```python
result = env.reset(scenario_id="med_001")
obs = result.observation

print(obs.content_item.text)
print(obs.trigger_info.trigger_type)
print(obs.gathered_context.thread_context)  # None until requested
```

## State Space

`ModerationState` tracks:

- `episode_id`
- `step_count`
- `task_id` (canonical benchmark task bucket)
- `scenario_id` (exact sampled scenario)
- `difficulty`
- `trigger_type`
- `actions_taken`
- `max_actions`
- `context_requested`
- `decision_made`
- `episode_reward`

The public HTTP `/schema` and `/state` endpoints expose the typed `ModerationState` contract directly. The session-backed `SafeSpaceEnv` client remains the canonical benchmark path for live per-episode state during evaluation.

## Reward Design

SafeSpace now provides reward signal over the full trajectory.

### Trajectory shaping

Investigation steps receive immediate feedback:

- positive reward for requesting context listed in `ground_truth["context_needed"]` when that context is actually retrievable
- small penalty for irrelevant context requests
- larger penalty for duplicate requests
- penalty for invalid actions
- terminal penalty if the action budget is exhausted before a decision

These rewards are intentionally small so they help the agent learn search strategy without overwhelming the terminal moderation-quality signal.
The cumulative trajectory signal is capped at `+/-0.15`, which lets 3-context hard cases benefit from all useful investigation steps without dominating the terminal reward.

### Terminal reward

When the agent takes `decide`, the final reward combines:

- decision reward
- factor overlap reward
- efficiency bonus
- calibration bonus

The decision grader is deterministic and checks:

- decision correctness
- policy violation correctness
- severity correctness
- adjacent-decision partial credit
- dangerous false negatives

Factor reward uses Jaccard similarity between the predicted factor set and the ground-truth factor set.

## Deterministic Graders

All tasks are graded programmatically. There are no LLM judges inside the environment.

- `grade_decision()` scores the moderation action against ground truth
- `grade_factors()` scores cited rationale factors
- `compute_reward()` combines terminal moderation quality
- trajectory reward helpers score investigation behavior step by step

This makes episodes reproducible and suitable for benchmarking.

## Scenario and Benchmark Stats

Canonical benchmark:

- Manifest version: `2026-04-03.2`
- Canonical split size: 60 scenarios total
- Canonical composition: 20 scenarios per benchmark task
- Canonical benchmark is the headline evaluation set used by `inference.py --mode canonical`

Full corpus:

- Total scenarios: 367
- Easy: 117
- Medium: 140
- Hard: 110

Decision distribution:

- Approve: 139
- Remove: 133
- Warn: 54
- Escalate: 41

Trigger distribution:

- `user_report`: 210
- `auto_flag`: 109
- `appeal`: 27
- `proactive_audit`: 21

Context-depth distribution:

- `0` context requests needed: 117
- `1` context request needed: 123
- `2` context requests needed: 108
- `3` context requests needed: 19

Recent targeted additions strengthen the benchmark with:

- medium cases where harmless-looking text hides a harmful link
- medium cases where benign-looking links actually leak private employee data
- medium cases that shift from public-figure criticism into borderline dogpiling
- medium medical-advice cases where community rules force escalation
- hard approve cases where harmful text is quoted in order to correct it
- hard cases where apparent whistleblowing becomes coordinated deepfake misinformation
- hard remove cases for coded harassment coordination
- hard remove cases for public-figure privacy violations that still cross the line

The full corpus includes procedural stress-test variants for broader coverage, but the 60-scenario canonical split is what we treat as benchmark depth in the README and baseline reporting.
The `2026-04-03.2` hard split intentionally rebalances away from an over-concentrated remove-heavy mix by adding a policy-conflict escalation case, a satire approve case, and a coded-hate warn case from the existing corpus.

## Quick Start

The commands below assume an activated environment where `python`, `pip`, `pytest`, and `openenv` resolve on `PATH`.

### 1. Install

```bash
pip install -e .
```

### 2. Run the full local preflight

```bash
scripts/preflight.sh
```

### 2a. Verify packaged assets in a non-editable wheel install

```bash
python scripts/check_package_assets.py
```

### 3. Run the environment server

```bash
uvicorn content_moderation_env.server.app:app --host 0.0.0.0 --port 8000
```

### 4. Connect with the typed client

```python
from content_moderation_env import SafeSpaceEnv, ModerationAction

with SafeSpaceEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(scenario_id="med_001")
    obs = result.observation

    result = env.step(ModerationAction(action_type="request_thread_context"))
    result = env.step(
        ModerationAction(
            action_type="decide",
            decision="approve",
            primary_violation="none",
            severity="none",
            confidence=0.82,
            key_factors=["gaming_or_competition_context", "no_violation_found"],
        )
    )

    print(result.reward)
    print(result.observation.reward_breakdown)
```

## Reference Evaluation

`inference.py` is the reference evaluator shipped with the project.

It:

- uses the OpenAI client for all model calls
- works with any OpenAI-compatible endpoint exposed through `API_BASE_URL`
- expects `HF_TOKEN` for the default Hugging Face Router path and still accepts `API_KEY`, `OPENAI_API_KEY`, or `AZURE_OPENAI_API_KEY` as fallbacks
- evaluates through the real `SafeSpaceEnv` client/server path
- uses a fixed `OPENAI_SEED` by default for reproducible model calls
- validates the benchmark manifest before evaluation
- supports deterministic canonical and full-dataset modes
- supports `--limit-per-task` for cheap smoke runs and `--validate-config` for preflight checks
- emits compact `[START]`, `[STEP]`, and `[END]` marker logs on stdout
- writes the aggregate JSON summary only when `--summary-json-path` is provided

### Required environment variables

```bash
export API_BASE_URL="<OpenAI-compatible endpoint>"
export MODEL_NAME="<model-id>"
export HF_TOKEN="<api-key>"
```

Accepted credential fallbacks:

```bash
export API_KEY="<api-key>"
# or
export OPENAI_API_KEY="<api-key>"
# or
export AZURE_OPENAI_API_KEY="<api-key>"
```

Additional optional variables:

```bash
export OPENAI_SEED="7"
export ENV_BASE_URL="http://localhost:8000"
export LOCAL_IMAGE_NAME="safespace:latest"
```

Connection precedence:

- if `--env-base-url` is passed, use that server
- else if `ENV_BASE_URL` is set, use that server
- else if `LOCAL_IMAGE_NAME` is set, launch the local Docker image through OpenEnv
- else fall back to `http://localhost:8000`

### Run the canonical reference baseline

```bash
python inference.py --mode canonical --summary-json-path artifacts/run-summary.json
```

### Validate config and benchmark assets before running

```bash
python inference.py --validate-config
```

### Run the full dataset evaluation

```bash
python inference.py --mode full --summary-json-path artifacts/run-summary.json
```

The script emits stdout in the following single-line format:

```text
[START] task=<task_name> env=safespace model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

When `--summary-json-path` is set, the file contains:

- evaluation mode
- benchmark manifest version
- model name
- connection mode and target
- total scenario count
- successful scenario count
- failure count
- failed scenario count
- `failure_details` for any zero-scored episodes
- `overall_average_task_grade` as the headline benchmark metric
- `overall_average_reward` as normalized public reward telemetry
- `overall_average_raw_reward` as the preserved raw RL/debugging telemetry
- per-task `average_task_grade`, `average_reward`, and `average_raw_reward`
- decision distribution per task

### Public Score Semantics

- Final episode `score` in the `[END]` line is always `task_grade`, which already lives in `[0.0, 1.0]`.
- Public `reward` values are normalized into `[0.0, 1.0]` for compatibility with tooling that expects normalized reward signals.
- Raw signed reward values are preserved in the JSON summary, state, and reward breakdown payloads for debugging and RL analysis.

### Canonical vs full evaluation

The two supported modes serve different purposes:

- `canonical` evaluates a fixed 60-scenario benchmark defined in `server/data/benchmark_manifest.json`.
- `canonical` is the score we treat as the headline benchmark because it is fast, comparable across reruns, and stable in task composition.
- `full` evaluates the entire current corpus of 367 scenarios.
- `full` is intended as a broader stress test and regression suite; it is slower and more expensive, but gives a better picture of full-corpus generalization.
- `full` runs the canonical split first and then the remainder of the corpus, so `--limit-per-task` is useful for cheap smoke checks.

### Baseline score capture

The primary reported metric is `overall_average_task_grade`.
The secondary RL/debugging metric is `overall_average_reward`.
Use `canonical` for the headline benchmark and `full` for broader regression coverage.

### Current Reference Baselines

Primary reference artifact:

`artifacts/baselines/canonical_gpt-5.4_azure_seed7_manifest_2026-04-03.2.json`

This is the main headline reference run. It uses `gpt-5.4` through an OpenAI-compatible Azure AI Foundry endpoint with `OPENAI_SEED=7` on benchmark manifest version `2026-04-03.2`. This run completed with `0` failed episodes on the current public reward surface.

```bash
export API_BASE_URL="<Azure AI Foundry OpenAI-compatible endpoint>"
export MODEL_NAME="gpt-5.4"
export AZURE_OPENAI_API_KEY="<provider key>"
export OPENAI_SEED="7"
export ENV_BASE_URL="http://localhost:8000"
python inference.py --mode canonical \
  --summary-json-path artifacts/baselines/canonical_gpt-5.4_azure_seed7_manifest_2026-04-03.2.json
```

| Task | Difficulty | Avg Task Grade | Avg Reward | Avg Raw Reward | Decision Distribution |
|------|------------|----------------|------------|----------------|----------------------|
| `clear_violations` | easy | **0.8244** | `0.7760` | `0.7200` | remove: 10, approve: 10 |
| `context_dependent` | medium | **0.4934** | `0.4914` | `0.3643` | approve: 11, warn: 5, remove: 4 |
| `policy_edge_cases` | hard | **0.4213** | `0.4695` | `0.3369` | approve: 8, escalate: 5, warn: 4, remove: 3 |

**Overall average task grade: `0.5797`**

**Overall average reward: `0.5790`**

**Overall average raw reward: `0.4737`**

Secondary open-weight reference artifact:

`artifacts/baselines/canonical_qwen2.5_72b_hf_seed7_manifest_2026-04-03.2.json`

This comparison run uses `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Router with `HF_TOKEN` and `OPENAI_SEED=7`.

| Model | Avg Task Grade | Avg Reward | Avg Raw Reward | Failed Episodes |
|-------|----------------|------------|----------------|-----------------|
| `gpt-5.4` | **0.5797** | `0.5790` | `0.4737` | `0` |
| `Qwen/Qwen2.5-72B-Instruct` | `0.4775` | `0.5098` | `0.3873` | `0` |

## Validation and Tests

Run the full verification stack from the repository root containing `openenv.yaml`, `inference.py`, and `Dockerfile`:

```bash
scripts/preflight.sh
python scripts/check_package_assets.py
scripts/validate-submission.sh https://<your-space>.hf.space .
```

The validator script is included for convenience when checking a live Space, local Docker build, and `openenv validate` status before publishing changes.

Benchmark statistics helper:

```bash
scripts/report_stats.py --format json
scripts/report_stats.py --format markdown
```

## Docker

Build and run locally:

```bash
docker build -t safespace:latest .
docker run -p 8000:8000 safespace:latest
curl http://localhost:8000/health
```

## Hugging Face Spaces

This project is structured as a containerized OpenEnv Space.

Deployment:

```bash
openenv push --repo-id <username>/safespace
```

The deployed Space exposes `/health` for liveness checks and supports the full OpenEnv API.

## Project Layout

```text
content_moderation_env/
├── artifacts/
│   └── baselines/
├── client.py
├── Dockerfile
├── inference.py
├── MANIFEST.in
├── models.py
├── openenv.yaml
├── README.md
├── scripts/
│   ├── check_package_assets.py
│   ├── validate-submission.sh
├── server/
│   ├── app.py
│   ├── environment.py
│   ├── grader.py
│   ├── policy.py
│   ├── reward.py
│   ├── scenarios.py
│   ├── Dockerfile
│   └── data/
│       ├── benchmark_manifest.json
└── tests/
```

## License

This project is licensed under the BSD 3-Clause License. See `LICENSE`.
