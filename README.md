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
---

# SafeSpace: Content Moderation OpenEnv

SafeSpace is an OpenEnv benchmark for sequential social-platform content moderation. Instead of making a one-shot label prediction, the agent reviews one reported post at a time, decides whether to gather more evidence under a fixed action budget, and then submits a structured moderation decision with calibrated confidence and cited factors.

![SafeSpace 0.2.1](https://img.shields.io/badge/SafeSpace-0.2.1-0B5FFF?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square)
![Docker Ready](https://img.shields.io/badge/Docker-Ready-2563EB?style=flat-square)
![HF Space](https://img.shields.io/badge/Hugging%20Face-Space-FFD21E?style=flat-square)
![License BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-111827?style=flat-square)

Quick links: [HF Space](https://huggingface.co/spaces/Ishangtxl/SafeSpace) · [Run locally](#run-the-environment-locally) · [Run the reference evaluator](#run-the-reference-evaluator) · [Action Space](#action-space) · [Observation Space](#observation-space) · [Example Episode](#example-episode) · [Appendix](#appendix)

## Environment Contract

| Item | SafeSpace contract |
|------|--------------------|
| Episode unit | One reported content item |
| Budget | `8` total actions per episode |
| Action space | `7` investigation actions + `1` terminal `decide` action |
| Final decisions | `approve`, `remove`, `warn`, `escalate` |
| Observation | Post content, trigger info, gathered context, platform policy, citable factors, progress fields |
| Reward | Trajectory shaping during investigation + deterministic terminal grading |
| Difficulty tiers | `easy`, `medium`, `hard` |
| Canonical benchmark | `60` scenarios total, `20` per task |
| Full corpus | `367` scenarios of broader authored + procedural regression coverage |

## Environment Description

SafeSpace models moderation as a budgeted evidence-gathering episode over one reported post. The benchmark is designed to test whether an agent can decide when the post alone is enough, when context changes the answer, and when escalation is more appropriate than overconfident guessing.

The benchmark contains three task families:

| Task | Difficulty | Scenarios | What makes it hard |
|------|------------|-----------|--------------------|
| `clear_violations` | easy | 117 | Text alone is sufficient; extra investigation wastes reward |
| `context_dependent` | medium | 140 | One targeted context request often flips the correct decision |
| `policy_edge_cases` | hard | 110 | Multiple signals conflict; calibration, precedent, and escalation matter |

## Connect To The Deployed Space

Space page: [Ishangtxl/SafeSpace](https://huggingface.co/spaces/Ishangtxl/SafeSpace)

Programmatic endpoint:

```python
from content_moderation_env import SafeSpaceEnv

with SafeSpaceEnv(base_url="https://ishangtxl-safespace.hf.space").sync() as env:
    result = env.reset()
    print(result.observation.content_item.text)
```

## Run The Environment Locally

### 1. Install dependencies

Preferred:

```bash
uv sync --all-extras
```

Alternative:

```bash
pip install -e ".[dev]"
```

SafeSpace depends on `openenv-core[core]>=0.2.3,<0.3`.

### 2. Run the local preflight

```bash
scripts/preflight.sh
```

### 3. Start the environment server

```bash
uvicorn content_moderation_env.server.app:app --host 0.0.0.0 --port 8000
```

### 4. Connect from code

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
    print(result.observation.task_grade)
```

## Run The Reference Evaluator

`inference.py` is the shipped reference evaluator. It uses the real `SafeSpaceEnv` client/server path and supports deterministic `canonical` and `full` evaluation modes.

### Required environment variables

```bash
export API_BASE_URL="<OpenAI-compatible endpoint>"
export MODEL_NAME="<model-id>"
export HF_TOKEN="<api-key>"
```

Accepted credential fallbacks:

```bash
export OPENAI_API_KEY="<api-key>"
# or
export API_KEY="<api-key>"
# or
export AZURE_OPENAI_API_KEY="<api-key>"
```

Additional optional variables:

```bash
export OPENAI_SEED="7"
export ENV_BASE_URL="http://localhost:8000"
export LOCAL_IMAGE_NAME="safespace:latest"
```

### Validate evaluator config

```bash
python inference.py --validate-config
```

### Run the canonical benchmark

```bash
python inference.py --mode canonical --summary-json-path artifacts/run-summary.json
```

### Run the full corpus evaluation

```bash
python inference.py --mode full --summary-json-path artifacts/run-summary.json
```

Connection precedence:

- if `--env-base-url` is passed, use that server
- else if `ENV_BASE_URL` is set, use that server
- else if `LOCAL_IMAGE_NAME` is set, launch the local Docker image through OpenEnv
- else fall back to `http://localhost:8000`

## Action Space

Each investigation action consumes one step from the episode budget.

| Action | Reveals | Typical use |
|--------|---------|-------------|
| `request_author_profile` | bio, account age, communities | intent, expertise, public-figure context |
| `request_author_violations` | prior moderation history | repeat-offender and escalation cases |
| `request_thread_context` | surrounding conversation | banter, quoting, dogpiling, harassment ambiguity |
| `request_community_rules` | local rule text | community-specific exceptions or stricter policy |
| `request_linked_content` | summary of off-post linked material | phishing, privacy leaks, misinformation, satire |
| `request_similar_precedents` | prior moderation examples | policy conflict and edge-case consistency |
| `request_reporter_credibility` | reporter accuracy history | brigading or unreliable reporting |

Terminal action:

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
- `task_id`
- `scenario_id`
- `difficulty`
- `trigger_type`
- `actions_taken`
- `max_actions`
- `context_requested`
- `decision_made`
- `episode_reward`
- `raw_episode_reward`
- `done`
- `last_error_code`

The public HTTP `/schema` and `/state` endpoints expose the typed `ModerationState` contract directly. The session-backed `SafeSpaceEnv` client remains the canonical benchmark path for live per-episode evaluation.

## Example Episode

`med_001` looks like a violent threat on the surface:

> "I swear I'm going to kill you next round bro, you're dead meat"

The correct policy is not obvious from the text alone, so a strong agent typically:

1. calls `reset(scenario_id="med_001")`
2. requests `request_thread_context`
3. sees friendly gaming banter in the thread
4. submits an `approve` decision with `primary_violation="none"` and factors such as `gaming_or_competition_context`
5. receives both a terminal reward breakdown and a normalized task grade

This is the core SafeSpace idea: the agent is rewarded not just for the final answer, but also for deciding when to investigate and what to investigate.

## Reward And Grading

SafeSpace optimizes moderation quality first, then explanation quality, efficiency, and calibration.

Trajectory shaping:

| Event | Raw reward |
|-------|------------|
| needed, retrievable context | `+0.05` |
| irrelevant context request | `-0.03` |
| duplicate context request | `-0.05` |
| invalid action | `-0.06` |
| budget exhausted before decision | `-0.15` |

The cumulative trajectory signal is capped at `+/-0.15`.

Terminal reward combines:

- decision reward
- factor overlap reward
- efficiency bonus
- calibration bonus

All grading is deterministic and programmatic. There are no LLM judges inside the environment.

## Research Grounding

SafeSpace is informed by prior content-moderation and explainability research, including HateXplain, especially around rationale-grounded moderation decisions, ambiguity, and explanation quality. We do not claim direct reuse of HateXplain samples in the current shipped scenario corpus.

- [OpenEnv documentation](https://meta-pytorch.org/OpenEnv/) and the [OpenEnv Quick Start](https://meta-pytorch.org/OpenEnv/auto_getting_started/) describe the environment interface conventions that SafeSpace follows.
- [HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://arxiv.org/abs/2012.10289) motivates rationale-aware moderation evaluation and explainability-oriented supervision.
- [Content moderation on social media: Does it matter who and why moderates hate speech?](https://scholars.cityu.edu.hk/en/publications/content-moderation-on-social-media-does-it-matter-who-and-why-mod/) is relevant to moderation transparency, trust, and explanation framing.

## Appendix

<details>
<summary><strong>Benchmark splits and corpus statistics</strong></summary>

Canonical benchmark:

- Manifest version: `2026-04-03.2`
- Canonical split size: `60` scenarios total
- Canonical composition: `20` scenarios per benchmark task
- Canonical benchmark is the headline evaluation set used by `inference.py --mode canonical`

Full corpus:

- Total scenarios: `367`
- Easy: `117`
- Medium: `140`
- Hard: `110`
- Includes broader authored extensions plus procedural stress/regression coverage outside the official canonical benchmark

Decision distribution:

- Approve: `139`
- Remove: `133`
- Warn: `54`
- Escalate: `41`

Trigger distribution:

- `user_report`: `210`
- `auto_flag`: `109`
- `appeal`: `27`
- `proactive_audit`: `21`

Context-depth distribution:

- `0` context requests needed: `117`
- `1` context request needed: `123`
- `2` context requests needed: `108`
- `3` context requests needed: `19`

</details>

<details>
<summary><strong>Reference evaluation details and baseline results</strong></summary>

`inference.py`:

- uses the OpenAI client for all model calls
- works with any OpenAI-compatible endpoint exposed through `API_BASE_URL`
- prefers `HF_TOKEN` and still accepts `OPENAI_API_KEY`, `API_KEY`, or `AZURE_OPENAI_API_KEY` as fallbacks
- validates the benchmark manifest before running
- emits compact `[START]`, `[STEP]`, and `[END]` logs on stdout
- writes aggregate metrics when `--summary-json-path` is provided

Public score semantics:

- Final episode `score` in the `[END]` line is always `task_grade`, which lives strictly inside `(0.0, 1.0)`
- Public `reward` values are normalized into `[0.0, 1.0]`
- Raw signed reward values are preserved in the JSON summary, state, and reward breakdown payloads

Canonical vs. full:

- `canonical` evaluates the fixed 60-scenario benchmark in `server/data/benchmark_manifest.json`
- `full` evaluates the entire current corpus of 367 scenarios
- `full` runs the canonical split first and then the remainder of the corpus

The evaluator emits stdout in the following single-line format:

```text
[START] task=<task_name> env=safespace model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Primary reference artifact:

`artifacts/baselines/canonical_gpt-5.4_azure_seed7_manifest_2026-04-08.1.json`

This canonical reference run uses `gpt-5.4` through an OpenAI-compatible Azure AI Foundry endpoint with `OPENAI_SEED=7`.

| Task | Difficulty | Avg Task Grade | Avg Reward | Avg Raw Reward | Decision Distribution |
|------|------------|----------------|------------|----------------|----------------------|
| `clear_violations` | easy | `0.8140` | `0.7665` | `0.7081` | remove: 9, approve: 9, warn: 2 |
| `context_dependent` | medium | `0.4579` | `0.4646` | `0.3308` | approve: 12, remove: 4, warn: 4 |
| `policy_edge_cases` | hard | `0.5177` | `0.5498` | `0.4372` | escalate: 8, approve: 6, warn: 3, remove: 3 |

Overall:

- Avg task grade: `0.5965`
- Avg reward: `0.5936`
- Avg raw reward: `0.4920`
- Failed episodes: `0`

Secondary open-weight comparison artifact:

- `artifacts/baselines/canonical_qwen2.5_72b_hf_seed7_manifest_2026-04-03.2.json`

This comparison run uses `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face Router with `OPENAI_SEED=7`.

| Model | Avg Task Grade | Avg Reward | Avg Raw Reward | Failed Episodes |
|-------|----------------|------------|----------------|-----------------|
| `gpt-5.4` | **`0.5965`** | `0.5936` | `0.4920` | `0` |
| `Qwen/Qwen2.5-72B-Instruct` | `0.4810` | `0.4994` | `0.3742` | `0` |

</details>

<details>
<summary><strong>Validation and deployment</strong></summary>

Run the full verification stack from the repository root:

```bash
scripts/preflight.sh
python scripts/check_package_assets.py
scripts/validate-submission.sh https://<your-space>.hf.space .
```

Additional helpers:

```bash
scripts/report_stats.py --format json
scripts/report_stats.py --format markdown
openenv validate .
```

Build and run locally:

```bash
docker build -t safespace:latest .
docker run -p 8000:8000 safespace:latest
curl http://localhost:8000/health
```

Deploy to Hugging Face Spaces:

```bash
openenv push --repo-id <username>/safespace
```

Submission-path hardening notes:

- the Docker build, health check, and typed-client smoke path are covered by `scripts/preflight.sh`
- the package asset smoke test self-heals `pip` with `ensurepip` if the active interpreter does not expose `python -m pip`
- the canonical evaluation path has been verified under a `2 CPU / 8 GB` container budget and completed well under the hackathon's `20` minute runtime limit

</details>

<details>
<summary><strong>Project layout</strong></summary>

```text
SafeSpace_Env/
├── artifacts/
│   ├── baselines/
│   └── readme/
├── client.py
├── Dockerfile
├── inference.py
├── MANIFEST.in
├── models.py
├── openenv.yaml
├── README.md
├── scripts/
│   ├── check_package_assets.py
│   ├── generate_scenarios.py
│   ├── preflight.sh
│   ├── report_stats.py
│   └── validate-submission.sh
├── server/
│   ├── app.py
│   ├── environment.py
│   ├── grader.py
│   ├── policy.py
│   ├── reward.py
│   ├── scenarios.py
│   └── data/
└── tests/
```

</details>

## License

This project is licensed under the BSD 3-Clause License. See `LICENSE`.
