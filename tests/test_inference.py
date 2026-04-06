"""Tests for the canonical baseline evaluator helpers."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import content_moderation_env.inference as inference_module
from content_moderation_env.inference import (
    EpisodeExecutionError,
    InferenceConfigurationError,
    ModelRequestError,
    OPENAI_SEED,
    SafeSpaceAgent,
    build_failed_episode_result,
    calibrate_confidence,
    compact_trigger_info_for_prompt,
    heuristic_investigation_candidates,
    infer_difficulty,
    infer_task_id,
    load_scenario_ids,
    main,
    normalize_factors,
    normalize_violation,
    parse_json_response,
    run_episode,
    run_task_evaluation,
    resolve_env_target,
    resolve_api_key_and_source,
    sanitize_prompt_text,
    summarize_task,
    validate_runtime_configuration,
)
from content_moderation_env.models import (
    ContentItem,
    GatheredContext,
    ModerationAction,
    ModerationObservation,
    TriggerInfo,
)


class FakeAgent:
    """Stub agent for inference tests."""

    def decide_investigation(self, observation, difficulty):
        return ["thread_context"]

    def make_decision(self, observation, difficulty):
        return ModerationAction(
            action_type="decide",
            decision="approve",
            primary_violation="none",
            severity="none",
            confidence=0.7,
            key_factors=["no_violation_found"],
        )


class FakeEnv:
    """Minimal async-client-like object for testing run_episode."""

    def __init__(self):
        self._reward = 0.0
        self._step_index = 0

    async def reset(self, **kwargs):
        self._reward = 0.0
        self._step_index = 0
        return SimpleNamespace(
            observation=ModerationObservation(
                content_item=ContentItem(
                    text="Looks bad at first glance",
                    post_id="p_test",
                    author_id="user_test",
                    community="general",
                    timestamp="2026-01-01T00:00:00Z",
                    media_type="text",
                    media_description=None,
                ),
                trigger_info=TriggerInfo(trigger_type="user_report"),
                gathered_context=GatheredContext(),
                platform_policy="policy",
                available_factors=["no_violation_found"],
                actions_taken=0,
                max_actions=8,
                action_history=[],
                feedback="ready",
            ),
            reward=None,
            done=False,
        )

    async def step(self, action):
        self._step_index += 1
        if action.action_type == "request_thread_context":
            self._reward += 0.04
            return SimpleNamespace(
                observation=ModerationObservation(
                    content_item=ContentItem(
                        text="Looks bad at first glance",
                        post_id="p_test",
                        author_id="user_test",
                        community="general",
                        timestamp="2026-01-01T00:00:00Z",
                        media_type="text",
                        media_description=None,
                    ),
                    trigger_info=TriggerInfo(trigger_type="user_report"),
                    gathered_context=GatheredContext(
                        thread_context=[{"author": "user_other", "text": "friendly banter"}]
                    ),
                    platform_policy="policy",
                    available_factors=["no_violation_found"],
                    actions_taken=1,
                    max_actions=8,
                action_history=["request_thread_context"],
                feedback="context",
                reward_breakdown={"score": 0.04},
                task_grade=None,
                grade_breakdown=None,
            ),
            reward=0.04,
            done=False,
        )

        self._reward += 0.60
        return SimpleNamespace(
            observation=ModerationObservation(
                content_item=ContentItem(
                    text="Looks bad at first glance",
                    post_id="p_test",
                    author_id="user_test",
                    community="general",
                    timestamp="2026-01-01T00:00:00Z",
                    media_type="text",
                    media_description=None,
                ),
                trigger_info=TriggerInfo(trigger_type="user_report"),
                gathered_context=GatheredContext(
                    thread_context=[{"author": "user_other", "text": "friendly banter"}]
                ),
                platform_policy="policy",
                available_factors=["no_violation_found"],
                actions_taken=1,
                max_actions=8,
                action_history=["request_thread_context", "decide: approve"],
                feedback="done",
                reward_breakdown={"score": 0.60},
                task_grade=0.81,
                grade_breakdown={"total": 0.81},
                done=True,
                reward=0.60,
            ),
            reward=0.60,
            done=True,
        )

    async def state(self):
        return SimpleNamespace(
            difficulty="medium",
            episode_reward=self._reward,
            raw_episode_reward=self._reward,
        )


def test_parse_json_response_handles_wrapped_json():
    """The evaluator should extract JSON from chatty model output."""
    parsed = parse_json_response("Here you go:\n{\"decision\":\"approve\"}")
    assert parsed == {"decision": "approve"}


def test_resolve_api_key_and_source_prefers_documented_precedence(monkeypatch):
    """Credential resolution should prefer HF_TOKEN, then OpenAI-compatible fallbacks."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("API_KEY", "api-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-token")
    assert resolve_api_key_and_source() == ("hf-token", "HF_TOKEN")

    monkeypatch.delenv("HF_TOKEN")
    assert resolve_api_key_and_source() == ("openai-token", "OPENAI_API_KEY")

    monkeypatch.delenv("OPENAI_API_KEY")
    assert resolve_api_key_and_source() == ("api-token", "API_KEY")

    monkeypatch.delenv("API_KEY")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-token")
    assert resolve_api_key_and_source() == ("azure-token", "AZURE_OPENAI_API_KEY")


def test_run_episode_uses_cumulative_episode_reward(monkeypatch, capsys):
    """Inference should report both episode reward and normalized task grade."""
    monkeypatch.setattr(inference_module, "MODEL_NAME", "test-model")
    result = asyncio.run(run_episode(FakeEnv(), FakeAgent(), "med_test"))
    assert result["scenario_id"] == "med_test"
    assert result["task_id"] == "context_dependent"
    assert result["episode_reward"] == 0.64
    assert result["raw_episode_reward"] == 0.64
    assert result["task_grade"] == 0.81
    assert result["investigation_plan"] == ["thread_context"]
    assert result["steps_taken"] == 2
    assert result["step_rewards"] == [0.04, 0.60]
    assert result["status"] == "success"
    assert result["failure"] is None
    stdout_lines = capsys.readouterr().out.strip().splitlines()
    assert stdout_lines == [
        "[START] task=context_dependent env=safespace model=test-model",
        "[STEP] step=1 action=request_thread_context reward=0.04 done=false error=null",
        "[STEP] step=2 action=decide:approve:none:none:0.70 reward=0.60 done=true error=null",
        "[END] success=true steps=2 score=0.81 rewards=0.04,0.60",
    ]


def test_infer_task_and_difficulty_from_scenario_prefix():
    """Scenario prefixes should resolve to canonical benchmark buckets."""
    assert infer_task_id("easy_001") == "clear_violations"
    assert infer_task_id("med_plus_001") == "context_dependent"
    assert infer_task_id("hard_001") == "policy_edge_cases"
    assert infer_difficulty(None, "hard_plus_005") == "hard"


def test_build_failed_episode_result_uses_zero_scored_fallbacks():
    """Failed episodes should contribute conservative zero scores."""
    result = build_failed_episode_result(
        task_id="context_dependent",
        scenario_id="med_fail",
        stage="make_decision",
        error="synthetic failure",
    )

    assert result["task_id"] == "context_dependent"
    assert result["difficulty"] == "medium"
    assert result["episode_reward"] == 0.0
    assert result["raw_episode_reward"] == 0.0
    assert result["task_grade"] == 0.0
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "make_decision"
    assert result["step_rewards"] == []
    assert result["steps_taken"] == 0


def test_normalize_violation_maps_verbose_labels():
    """Verbose model labels should collapse to canonical policy IDs."""
    assert normalize_violation("3.1 Targeted Harassment", "remove") == "3.1"
    assert normalize_violation("privacy doxxing", "remove") == "7.1"
    assert normalize_violation("1.1 Baseline Expectations", "warn") == "1.1"
    assert normalize_violation("bad faith participation", "warn") == "1.2"


def test_normalize_factors_maps_aliases_and_falls_back():
    """Free-form factor phrases should map to canonical factor names."""
    normalized = normalize_factors(
        ["gaming_context", "quoted_for_reporting"],
        ["gaming_or_competition_context", "quoting_or_reporting", "no_violation_found"],
        "approve",
    )
    assert normalized == ["gaming_or_competition_context", "quoting_or_reporting"]
    assert normalize_factors([], ["no_violation_found"], "approve") == ["no_violation_found"]


def test_calibrate_confidence_reduces_hard_case_overconfidence():
    """Hard cases should not stay overconfident after sparse investigation."""
    assert calibrate_confidence(0.99, "hard", "remove", 1) <= 0.65
    assert calibrate_confidence(0.99, "hard", "escalate", 3) <= 0.38


def test_sanitize_prompt_text_redacts_high_risk_literals():
    """Prompt sanitization should redact violent and private details."""
    sanitized = sanitize_prompt_text(
        "I will kill you at 123 Main St and call 206-555-0147 tonight."
    )
    assert "[violent_phrase]" in sanitized
    assert "[private_address]" in sanitized
    assert "[phone_number]" in sanitized


def test_sanitize_prompt_text_redacts_medical_claim_transport_tokens():
    """Known provider-filter-prone medical phrases should be redacted in transport."""
    sanitized = sanitize_prompt_text(
        "Vaccines cause autism. Big pharma is hiding the truth."
    )
    assert "[medical_misinformation_claim]" in sanitized
    assert "[industry_conspiracy_reference]" in sanitized


def test_compact_trigger_info_for_prompt_drops_noisy_fields():
    """Prompt trigger payload should keep only high-signal structured fields."""
    compact = compact_trigger_info_for_prompt(
        {
            "trigger_type": "user_report",
            "report_count": 12,
            "report_categories": ["privacy", "threats"],
            "sample_report_reason": "long free-form reason",
            "auto_flag_reason": None,
            "original_decision": None,
            "appeal_text": None,
            "audit_reason": None,
        }
    )

    assert compact == {
        "trigger_type": "user_report",
        "report_count": 12,
    }


def test_heuristic_investigation_prefers_links_for_suspicious_link_case():
    """Link-heavy cases should prioritize linked content investigation."""
    observation = ModerationObservation(
        content_item=ContentItem(
            post_id="p_link",
            text="Free remote job checklist, totally legit.",
            author_id="user_link",
            community="careers",
            timestamp="2026-01-01T00:00:00Z",
            media_type="text+link",
            media_description="suspicious link",
        ),
        trigger_info=TriggerInfo(
            trigger_type="auto_flag",
            auto_flag_reason="suspicious_link_score: 0.71",
        ),
        gathered_context=GatheredContext(),
        platform_policy="policy",
        available_factors=["harmful_link"],
        actions_taken=0,
        max_actions=8,
        action_history=[],
        feedback="ready",
    )
    assert "linked_content_summary" in heuristic_investigation_candidates(observation, "medium")


def test_heuristic_investigation_prefers_author_history_for_repeat_spam_case():
    """Borderline self-promo should inspect author history before anything else."""
    observation = ModerationObservation(
        content_item=ContentItem(
            post_id="p_promo",
            text="Hey everyone! Just launched my new productivity app - it's free to try.",
            author_id="user_dev_spam",
            community="productivity",
            timestamp="2026-01-01T00:00:00Z",
            media_type="text+link",
            media_description="Link to a mobile app download page",
        ),
        trigger_info=TriggerInfo(
            trigger_type="auto_flag",
            auto_flag_reason="spam_score: 0.65 (promotional language detected)",
        ),
        gathered_context=GatheredContext(),
        platform_policy="policy",
        available_factors=["repeat_offender", "spam_commercial"],
        actions_taken=0,
        max_actions=8,
        action_history=[],
        feedback="ready",
    )

    assert heuristic_investigation_candidates(observation, "medium")[0] == "author_violations"


def test_summarize_task_prefers_task_grade_as_headline_metric():
    """Task summaries should aggregate grade and reward separately."""
    summary = summarize_task(
        "context_dependent",
        [
            {
                "task_grade": 0.8,
                "episode_reward": 0.5,
                "raw_episode_reward": 0.4,
                "decision": "approve",
            },
            {
                "task_grade": 0.6,
                "episode_reward": 0.3,
                "raw_episode_reward": 0.2,
                "decision": "remove",
            },
        ],
    )

    assert summary["average_task_grade"] == 0.7
    assert summary["average_reward"] == 0.4
    assert summary["average_raw_reward"] == pytest.approx(0.3)
    assert summary["decision_distribution"] == {"approve": 1, "remove": 1}


def test_run_task_evaluation_records_failed_episode_metadata():
    """A failed scenario should be counted, surfaced, and zero-scored."""

    class FailingAgent(FakeAgent):
        def make_decision(self, observation, difficulty):
            del observation, difficulty
            raise ModelRequestError("decision failure")

    summary, failure_details = asyncio.run(
        run_task_evaluation(
            FakeEnv(),
            FailingAgent(),
            "context_dependent",
            ["med_fail"],
        )
    )

    assert summary["num_scenarios"] == 1
    assert summary["successful_scenarios"] == 0
    assert summary["failed_scenarios"] == 1
    assert summary["average_task_grade"] == 0.0
    assert summary["average_reward"] == 0.0
    assert summary["results"][0]["status"] == "failed"
    assert failure_details == [
        {
            "scenario_id": "med_fail",
            "task_id": "context_dependent",
            "stage": "make_decision",
            "error": "decision failure",
        }
    ]


def test_run_episode_wraps_stage_failures_with_episode_metadata():
    """Low-level episode failures should retain scenario and stage context."""

    class BrokenEnv(FakeEnv):
        async def reset(self, **kwargs):
            del kwargs
            raise RuntimeError("reset failed")

    with pytest.raises(EpisodeExecutionError) as exc_info:
        asyncio.run(
            run_episode(BrokenEnv(), FakeAgent(), "med_fail", task_id="context_dependent")
        )

    assert exc_info.value.scenario_id == "med_fail"
    assert exc_info.value.task_id == "context_dependent"
    assert exc_info.value.stage == "reset"


def test_safe_space_agent_passes_fixed_seed_to_openai_client():
    """OpenAI calls should use the fixed seed for reproducible outputs."""

    class StubCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"decision":"approve"}'))]
            )

    completions = StubCompletions()
    agent = object.__new__(SafeSpaceAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.model = "test-model"

    parsed = agent._call_llm("hello")  # pylint: disable=protected-access

    assert parsed == {"decision": "approve"}
    assert completions.kwargs is not None
    assert completions.kwargs["seed"] == OPENAI_SEED
    assert completions.kwargs["max_completion_tokens"] == inference_module.MAX_TOKENS


def test_safe_space_agent_falls_back_to_max_tokens_when_provider_rejects_max_completion_tokens():
    """Unsupported max_completion_tokens should retry with max_tokens."""

    class StubCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("Unsupported parameter: max_completion_tokens")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"decision":"approve"}'))]
            )

    completions = StubCompletions()
    agent = object.__new__(SafeSpaceAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.model = "test-model"

    parsed = agent._call_llm("hello")  # pylint: disable=protected-access

    assert parsed == {"decision": "approve"}
    assert "max_completion_tokens" in completions.calls[0]
    assert "max_tokens" in completions.calls[1]
    assert completions.calls[1]["seed"] == OPENAI_SEED


def test_safe_space_agent_retries_without_seed_when_provider_rejects_seed():
    """Seed rejection should retry once without the seed parameter."""

    class StubCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("Unexpected keyword argument 'seed'")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"decision":"approve"}'))]
            )

    completions = StubCompletions()
    agent = object.__new__(SafeSpaceAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.model = "test-model"

    parsed = agent._call_llm("hello")  # pylint: disable=protected-access

    assert parsed == {"decision": "approve"}
    assert completions.calls[0]["seed"] == OPENAI_SEED
    assert "seed" not in completions.calls[1]


def test_safe_space_agent_investigation_falls_back_to_heuristic_on_model_error():
    """Investigation planning should fall back to deterministic heuristics."""
    agent = object.__new__(SafeSpaceAgent)
    agent._call_llm = lambda prompt: (_ for _ in ()).throw(ModelRequestError("filtered"))  # type: ignore[attr-defined]

    observation = ModerationObservation(
        content_item=ContentItem(
            post_id="p_promo",
            text="Hey everyone! Just launched my new productivity app - it's free to try.",
            author_id="user_dev_spam",
            community="productivity",
            timestamp="2026-01-01T00:00:00Z",
            media_type="text+link",
            media_description="Link to a mobile app download page",
        ),
        trigger_info=TriggerInfo(
            trigger_type="auto_flag",
            auto_flag_reason="spam_score: 0.65 (promotional language detected)",
        ),
        gathered_context=GatheredContext(),
        platform_policy="policy",
        available_factors=["repeat_offender", "spam_commercial"],
        actions_taken=0,
        max_actions=8,
        action_history=[],
        feedback="ready",
    )

    assert agent.decide_investigation(observation, "medium") == ["author_violations"]


def test_safe_space_agent_decision_falls_back_to_heuristic_on_model_error():
    """Final decisions should fall back to deterministic heuristics."""
    agent = object.__new__(SafeSpaceAgent)
    agent._call_llm = lambda prompt: (_ for _ in ()).throw(ModelRequestError("filtered"))  # type: ignore[attr-defined]

    observation = ModerationObservation(
        content_item=ContentItem(
            post_id="p_spam",
            text="MAKE $5000 DAILY working from home! Click here for secret method!",
            author_id="user_spam",
            community="general",
            timestamp="2026-01-01T00:00:00Z",
            media_type="text",
            media_description=None,
        ),
        trigger_info=TriggerInfo(
            trigger_type="auto_flag",
            auto_flag_reason="spam_score: 0.95",
        ),
        gathered_context=GatheredContext(),
        platform_policy="policy",
        available_factors=[
            "spam_commercial",
            "clear_violation_no_exception",
            "auto_flag_high_confidence",
        ],
        actions_taken=0,
        max_actions=8,
        action_history=[],
        feedback="ready",
    )

    decision = agent.make_decision(observation, "easy")

    assert decision.decision == "remove"
    assert decision.primary_violation == "5.1"
    assert decision.severity == "high"


def test_safe_space_agent_does_not_retry_on_auth_or_network_failures():
    """Non-parameter failures should fail loudly without compatibility retries."""

    class StubCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            del kwargs
            self.call_count += 1
            raise RuntimeError("401 Unauthorized")

    completions = StubCompletions()
    agent = object.__new__(SafeSpaceAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.model = "test-model"

    with pytest.raises(ModelRequestError):
        agent._call_llm("hello")  # pylint: disable=protected-access

    assert completions.call_count == 1


def test_safe_space_agent_raises_on_unparseable_response():
    """Non-JSON model output should fail loudly."""

    class StubCompletions:
        def create(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
            )

    agent = object.__new__(SafeSpaceAgent)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions()))
    agent.model = "test-model"

    with pytest.raises(ModelRequestError):
        agent._call_llm("hello")  # pylint: disable=protected-access


def test_load_scenario_ids_reads_canonical_manifest():
    """Canonical evaluation should load the 20-scenario manifest split."""
    scenario_ids = load_scenario_ids("context_dependent", "canonical")
    assert len(scenario_ids) == 20
    assert len(set(scenario_ids)) == 20


def test_load_scenario_ids_full_mode_starts_with_canonical_split():
    """Full mode should begin with the canonical benchmark IDs for smoke friendliness."""
    canonical = load_scenario_ids("policy_edge_cases", "canonical")
    full = load_scenario_ids("policy_edge_cases", "full")
    assert full[: len(canonical)] == canonical


def test_validate_runtime_configuration_returns_manifest_metadata(monkeypatch):
    """Config validation should include manifest and seed metadata."""
    monkeypatch.setattr(inference_module, "MODEL_NAME", "test-model")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    metadata = validate_runtime_configuration("canonical")

    assert metadata["manifest_version"]
    assert metadata["canonical_task_counts"]["clear_violations"] == 20
    assert metadata["api_key_source"] == "OPENAI_API_KEY"
    assert metadata["openai_seed"] == OPENAI_SEED
    assert metadata["connection_mode"] == "base_url"


def test_validate_runtime_configuration_requires_model_name(monkeypatch):
    """Missing MODEL_NAME should fail fast."""
    monkeypatch.setattr(inference_module, "MODEL_NAME", None)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(InferenceConfigurationError):
        validate_runtime_configuration("canonical")


def test_validate_runtime_configuration_requires_api_key(monkeypatch):
    """Missing API key should fail fast."""
    monkeypatch.setattr(inference_module, "MODEL_NAME", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(InferenceConfigurationError):
        validate_runtime_configuration("canonical")


def test_resolve_env_target_prefers_local_image_when_no_url(monkeypatch):
    """A local image should be used only when no URL target is configured."""
    monkeypatch.setattr(inference_module, "ENV_BASE_URL", None)
    monkeypatch.setattr(inference_module, "LOCAL_IMAGE_NAME", "safespace:latest")

    target = resolve_env_target(None)

    assert target == {
        "connection_mode": "local_image",
        "env_base_url": None,
        "local_image_name": "safespace:latest",
    }


def test_main_validate_config_prints_manifest_metadata(monkeypatch, capsys):
    """CLI validation mode should emit the manifest version and counts."""
    monkeypatch.setattr(inference_module, "MODEL_NAME", "test-model")
    monkeypatch.setattr(inference_module, "API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(inference_module, "ENV_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr(inference_module, "LOCAL_IMAGE_NAME", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr(
        inference_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            mode="canonical",
            limit_per_task=None,
            validate_config=True,
            env_base_url="http://localhost:8000",
            summary_json_path=None,
        ),
    )

    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_version"]
    assert payload["canonical_task_counts"]["policy_edge_cases"] == 20
    assert payload["api_key_source"] == "API_KEY"
    assert payload["connection_mode"] == "base_url"


def test_main_evaluation_summary_includes_failure_metadata(monkeypatch, capsys, tmp_path):
    """CLI evaluation mode should write aggregate JSON to the requested summary path."""
    monkeypatch.setattr(inference_module, "MODEL_NAME", "test-model")
    monkeypatch.setattr(inference_module, "API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(inference_module, "ENV_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr(inference_module, "LOCAL_IMAGE_NAME", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-key")
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        inference_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            mode="canonical",
            limit_per_task=1,
            validate_config=False,
            env_base_url="http://localhost:8000",
            summary_json_path=str(summary_path),
        ),
    )

    class DummySafeSpaceEnv:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    async def fake_create_env_client(explicit_base_url):
        del explicit_base_url
        return DummySafeSpaceEnv()

    async def fake_run_task_evaluation(env, agent, task_id, scenario_ids):
        del env, agent
        result = {
            "scenario_id": scenario_ids[0],
            "task_id": task_id,
            "difficulty": inference_module.TASK_TO_DIFFICULTY[task_id],
            "episode_reward": 1.0 if task_id != "context_dependent" else 0.0,
            "raw_episode_reward": 0.8 if task_id != "context_dependent" else 0.0,
            "task_grade": 1.0 if task_id != "context_dependent" else 0.0,
            "decision": "approve" if task_id != "context_dependent" else None,
            "confidence": 0.9 if task_id != "context_dependent" else None,
            "investigation_plan": [],
            "step_rewards": [],
            "steps_taken": 0,
            "final_reward_breakdown": None,
            "final_grade_breakdown": None,
            "status": "success" if task_id != "context_dependent" else "failed",
            "failure": (
                None
                if task_id != "context_dependent"
                else {
                    "scenario_id": scenario_ids[0],
                    "task_id": task_id,
                    "stage": "make_decision",
                    "error": "synthetic failure",
                }
            ),
        }
        failures = [] if result["failure"] is None else [result["failure"]]
        summary = summarize_task(task_id, [result])
        summary["successful_scenarios"] = 1 - len(failures)
        summary["failed_scenarios"] = len(failures)
        summary["failure_details"] = failures
        return summary, failures

    monkeypatch.setattr(inference_module, "SafeSpaceAgent", lambda: object())
    monkeypatch.setattr(inference_module, "create_env_client", fake_create_env_client)
    monkeypatch.setattr(
        inference_module,
        "load_scenario_ids",
        lambda task_id, mode: [f"{task_id}_{mode}_001"],
    )
    monkeypatch.setattr(
        inference_module,
        "run_task_evaluation",
        fake_run_task_evaluation,
    )

    main()
    assert capsys.readouterr().out == ""
    payload = json.loads(summary_path.read_text())

    assert payload["total_scenarios"] == 3
    assert payload["successful_scenarios"] == 2
    assert payload["failed_scenarios"] == 1
    assert payload["failure_count"] == 1
    assert payload["api_key_source"] == "HF_TOKEN"
    assert payload["overall_average_raw_reward"] == pytest.approx((0.8 + 0.0 + 0.8) / 3)
    assert payload["failure_details"] == [
        {
            "scenario_id": "context_dependent_canonical_001",
            "task_id": "context_dependent",
            "stage": "make_decision",
            "error": "synthetic failure",
        }
    ]
