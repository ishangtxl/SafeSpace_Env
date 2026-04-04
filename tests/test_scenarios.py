# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for scenario validation."""

import pytest
from content_moderation_env.server.scenarios import (
    get_benchmark_manifest,
    get_benchmark_scenario_ids,
    get_all_scenarios,
    get_scenarios_by_task,
    get_scenario_statistics,
    get_task_id_for_scenario,
    load_scenario,
    scenario_semantic_signature,
    validate_benchmark_manifest,
    validate_scenario_record,
)
from content_moderation_env.server.policy import (
    FACTOR_LIST,
    POLICY_SECTIONS,
    VALID_DECISIONS,
    VALID_SEVERITIES,
)


class TestScenarioLoading:
    """Tests for scenario loading."""

    def test_all_scenarios_load(self):
        """Test that all scenario files load without error."""
        scenarios = get_all_scenarios()
        assert "easy" in scenarios
        assert "medium" in scenarios
        assert "hard" in scenarios

    def test_scenario_count(self):
        """Test that we have enough scenarios."""
        stats = get_scenario_statistics()
        assert stats["total"] >= 100, f"Expected at least 100 scenarios, got {stats['total']}"

    def test_difficulty_distribution(self):
        """Test that scenarios are distributed across difficulties."""
        stats = get_scenario_statistics()
        assert stats["by_difficulty"].get("easy", 0) >= 10
        assert stats["by_difficulty"].get("medium", 0) >= 10
        assert stats["by_difficulty"].get("hard", 0) >= 10


class TestScenarioSchema:
    """Tests for scenario schema validation."""

    def test_all_scenarios_have_required_fields(self):
        """Test that all scenarios have required top-level fields."""
        required_fields = [
            "scenario_id",
            "difficulty",
            "category",
            "content_item",
            "trigger_info",
            "ground_truth",
        ]

        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                for field in required_fields:
                    assert field in scenario, f"{scenario['scenario_id']} missing {field}"

    def test_all_content_items_valid(self):
        """Test that all content items have required fields."""
        required_fields = ["post_id", "text", "author_id", "community", "timestamp", "media_type"]

        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                ci = scenario["content_item"]
                for field in required_fields:
                    assert field in ci, f"{scenario['scenario_id']} content_item missing {field}"

    def test_all_trigger_types_valid(self):
        """Test that all trigger types are valid."""
        valid_triggers = ["user_report", "auto_flag", "appeal", "proactive_audit"]

        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                trigger_type = scenario["trigger_info"].get("trigger_type")
                assert trigger_type in valid_triggers, (
                    f"{scenario['scenario_id']} has invalid trigger_type: {trigger_type}"
                )

    def test_all_difficulties_valid(self):
        """Test that all difficulty values are valid."""
        valid_difficulties = ["easy", "medium", "hard"]

        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                d = scenario["difficulty"]
                assert d in valid_difficulties, f"{scenario['scenario_id']} has invalid difficulty: {d}"


class TestGroundTruthValidation:
    """Tests for ground truth validation."""

    def test_all_decisions_valid(self):
        """Test that all ground truth decisions are valid."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                gt = scenario["ground_truth"]
                decision = gt.get("correct_decision")
                assert decision in VALID_DECISIONS, (
                    f"{scenario['scenario_id']} has invalid decision: {decision}"
                )

    def test_all_violations_valid(self):
        """Test that all primary violations are valid policy sections."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                gt = scenario["ground_truth"]
                violation = gt.get("primary_violation")
                assert violation in POLICY_SECTIONS, (
                    f"{scenario['scenario_id']} has invalid violation: {violation}"
                )

    def test_all_severities_valid(self):
        """Test that all severity values are valid."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                gt = scenario["ground_truth"]
                severity = gt.get("severity")
                assert severity in VALID_SEVERITIES, (
                    f"{scenario['scenario_id']} has invalid severity: {severity}"
                )

    def test_all_factors_valid(self):
        """Test that all key factors are from FACTOR_LIST."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                gt = scenario["ground_truth"]
                factors = gt.get("key_factors", [])
                for factor in factors:
                    assert factor in FACTOR_LIST, (
                        f"{scenario['scenario_id']} has invalid factor: {factor}"
                    )

    def test_all_scenarios_have_factors(self):
        """Test that all scenarios have at least one key factor."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                gt = scenario["ground_truth"]
                factors = gt.get("key_factors", [])
                assert len(factors) >= 1, f"{scenario['scenario_id']} has no key_factors"

    def test_factor_list_is_fully_exercised(self):
        """Every published factor should appear in at least one ground-truth scenario."""
        used_factors = set()
        for scenarios in get_all_scenarios().values():
            for scenario in scenarios:
                used_factors.update(scenario["ground_truth"].get("key_factors", []))

        unused = sorted(set(FACTOR_LIST) - used_factors)
        assert not unused, f"Unused factors in FACTOR_LIST: {unused}"


class TestScenarioConsistency:
    """Tests for scenario consistency."""

    def test_unique_scenario_ids(self):
        """Test that all scenario IDs are unique."""
        seen_ids = set()
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                sid = scenario["scenario_id"]
                assert sid not in seen_ids, f"Duplicate scenario_id: {sid}"
                seen_ids.add(sid)

    def test_approve_has_no_violation(self):
        """Test that approve decisions have 'none' as primary violation."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                gt = scenario["ground_truth"]
                if gt["correct_decision"] == "approve":
                    assert gt["primary_violation"] == "none", (
                        f"{scenario['scenario_id']}: approve should have violation='none'"
                    )
                    assert gt["severity"] == "none", (
                        f"{scenario['scenario_id']}: approve should have severity='none'"
                    )

    def test_remove_has_violation(self):
        """Test that remove decisions have a violation specified."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                gt = scenario["ground_truth"]
                if gt["correct_decision"] == "remove":
                    assert gt["primary_violation"] != "none", (
                        f"{scenario['scenario_id']}: remove should have a violation"
                    )

    def test_needed_context_is_retrievable(self):
        """Any context listed as needed must be present and retrievable."""
        for difficulty, scenarios in get_all_scenarios().items():
            for scenario in scenarios:
                assert validate_scenario_record(scenario) == [], (
                    f"{difficulty}/{scenario['scenario_id']} failed integrity validation"
                )

    def test_validator_flags_unretrievable_needed_context(self):
        """Synthetic invalid scenarios should fail integrity validation."""
        scenario = {
            "scenario_id": "bad_case",
            "available_context": {"thread_context": None},
            "ground_truth": {"context_needed": ["thread_context"]},
        }
        errors = validate_scenario_record(scenario)
        assert errors
        assert "not retrievable" in errors[0]


class TestTaskMapping:
    """Tests for task-to-scenario mapping."""

    def test_clear_violations_returns_easy(self):
        """Test that clear_violations task returns easy scenarios."""
        scenarios = get_scenarios_by_task("clear_violations")
        assert len(scenarios) > 0
        for s in scenarios[:5]:  # Check first 5
            assert s["difficulty"] == "easy"

    def test_context_dependent_returns_medium(self):
        """Test that context_dependent task returns medium scenarios."""
        scenarios = get_scenarios_by_task("context_dependent")
        assert len(scenarios) > 0
        for s in scenarios[:5]:
            assert s["difficulty"] == "medium"

    def test_policy_edge_cases_returns_hard(self):
        """Test that policy_edge_cases task returns hard scenarios."""
        scenarios = get_scenarios_by_task("policy_edge_cases")
        assert len(scenarios) > 0
        for s in scenarios[:5]:
            assert s["difficulty"] == "hard"

    def test_targeted_submission_scenarios_exist(self):
        """Test that handcrafted submission-strength scenarios are loadable."""
        for scenario_id in [
            "med_plus_001",
            "med_plus_002",
            "med_plus_003",
            "med_plus_004",
            "med_plus_005",
            "hard_plus_001",
            "hard_plus_002",
            "hard_plus_003",
            "hard_plus_004",
            "hard_plus_005",
        ]:
            scenario = load_scenario(scenario_id)
            assert scenario["scenario_id"] == scenario_id

    def test_canonical_hard_split_matches_submission_rebalance(self):
        """The canonical hard split should include the documented rebalance cases."""
        canonical_hard = get_benchmark_manifest()["canonical"]["policy_edge_cases"]

        for scenario_id in ["hard_001", "hard_gen_03_00", "hard_gen_02_00"]:
            assert scenario_id in canonical_hard

        for scenario_id in ["hard_remove_15", "hard_remove_16", "hard_remove_17"]:
            assert scenario_id not in canonical_hard

        assert len(canonical_hard) == 20
        validate_benchmark_manifest()


class TestTriggerTypeDistribution:
    """Tests for trigger type distribution."""

    def test_all_trigger_types_represented(self):
        """Test that all trigger types are represented in scenarios."""
        stats = get_scenario_statistics()
        trigger_types = stats["by_trigger_type"]

        assert "user_report" in trigger_types
        assert "auto_flag" in trigger_types
        # appeal and proactive_audit may be less common


class TestScenarioReporting:
    """Tests for scenario reporting helpers."""

    def test_task_id_helper_maps_loaded_scenarios(self):
        """Loaded scenarios should map back to canonical benchmark task IDs."""
        assert get_task_id_for_scenario(load_scenario("easy_001")) == "clear_violations"
        assert get_task_id_for_scenario(load_scenario("med_001")) == "context_dependent"
        assert get_task_id_for_scenario(load_scenario("hard_001")) == "policy_edge_cases"

    def test_statistics_include_decisions_and_context_depth(self):
        """Benchmark statistics should include decision and context-depth summaries."""
        stats = get_scenario_statistics()

        assert "by_decision" in stats
        assert "context_depth_overall" in stats
        assert "context_depth_by_difficulty" in stats
        assert stats["by_decision"]["approve"] > 0
        assert stats["context_depth_overall"][0] > 0


class TestBenchmarkManifest:
    """Tests for the canonical benchmark manifest."""

    def test_manifest_loads_and_validates(self):
        """The shipped benchmark manifest should validate cleanly."""
        manifest = validate_benchmark_manifest()
        assert manifest == get_benchmark_manifest()
        assert manifest["manifest_version"]

    def test_manifest_has_exactly_twenty_ids_per_task(self):
        """Each canonical task split should contain 20 unique scenarios."""
        manifest = validate_benchmark_manifest()
        for task_id in ["clear_violations", "context_dependent", "policy_edge_cases"]:
            scenario_ids = manifest["canonical"][task_id]
            assert len(scenario_ids) == 20
            assert len(set(scenario_ids)) == 20
            assert get_benchmark_scenario_ids(task_id) == scenario_ids

    def test_manifest_canonical_scenarios_have_unique_signatures(self):
        """Canonical benchmark should not duplicate semantic signatures."""
        signatures = {}
        for task_id in ["clear_violations", "context_dependent", "policy_edge_cases"]:
            for scenario_id in get_benchmark_scenario_ids(task_id):
                scenario = load_scenario(scenario_id)
                signature = scenario_semantic_signature(scenario)
                assert signature not in signatures, (
                    f"Duplicate canonical signature for {signatures.get(signature)} "
                    f"and {scenario_id}"
                )
                signatures[signature] = scenario_id
