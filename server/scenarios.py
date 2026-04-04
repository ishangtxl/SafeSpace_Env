"""Scenario loading and validation utilities for SafeSpace."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import ScenarioCorpusError, ScenarioLookupError

DATA_DIR = Path(__file__).parent / "data"
BENCHMARK_MANIFEST_FILE = "benchmark_manifest.json"
SCENARIO_FILES = {
    "easy": "scenarios_easy.json",
    "medium": "scenarios_medium.json",
    "hard": "scenarios_hard.json",
}
TASK_TO_DIFFICULTY = {
    "clear_violations": "easy",
    "context_dependent": "medium",
    "policy_edge_cases": "hard",
}
DIFFICULTY_TO_TASK = {
    difficulty: task_id for task_id, difficulty in TASK_TO_DIFFICULTY.items()
}
VALID_CONTEXT_FIELDS = {
    "author_profile",
    "author_violations",
    "thread_context",
    "community_rules",
    "linked_content_summary",
    "similar_precedents",
    "reporter_credibility",
}

_scenario_cache: Dict[str, List[Dict[str, Any]]] = {}
_benchmark_manifest_cache: Optional[Dict[str, Any]] = None


def is_retrievable_context_value(value: Any) -> bool:
    """Return whether a context payload is actually retrievable by the agent."""
    return value is not None


def scenario_semantic_signature(scenario: Dict[str, Any]) -> Tuple[Any, ...]:
    """Build a stable signature for benchmark-diversity checks."""
    ground_truth = scenario.get("ground_truth", {})
    trigger_info = scenario.get("trigger_info", {})
    return (
        scenario.get("content_item", {}).get("text"),
        trigger_info.get("trigger_type"),
        ground_truth.get("correct_decision"),
        ground_truth.get("primary_violation"),
        ground_truth.get("severity"),
        tuple(sorted(ground_truth.get("context_needed", []))),
    )


def validate_scenario_record(scenario: Dict[str, Any]) -> List[str]:
    """Return structural or benchmark-integrity validation errors for one scenario."""
    errors: List[str] = []
    scenario_id = scenario.get("scenario_id", "<unknown>")
    available_context = scenario.get("available_context", {})
    context_needed = scenario.get("ground_truth", {}).get("context_needed", [])

    if not isinstance(available_context, dict):
        errors.append(f"{scenario_id}: available_context must be a mapping")
        return errors

    if not isinstance(context_needed, list):
        errors.append(f"{scenario_id}: ground_truth.context_needed must be a list")
        return errors

    for context_key in context_needed:
        if context_key not in VALID_CONTEXT_FIELDS:
            errors.append(f"{scenario_id}: unknown context_needed key '{context_key}'")
            continue
        if context_key not in available_context:
            errors.append(
                f"{scenario_id}: context_needed '{context_key}' is missing from available_context"
            )
            continue
        if not is_retrievable_context_value(available_context.get(context_key)):
            errors.append(
                f"{scenario_id}: context_needed '{context_key}' is not retrievable"
            )

    return errors


def _load_scenarios_from_file(filename: str) -> List[Dict[str, Any]]:
    """Load scenarios from a JSON file and fail loudly if unavailable."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise ScenarioCorpusError(
            f"Missing scenario file: {filename}",
            context={"path": str(filepath)},
        )

    with filepath.open() as handle:
        scenarios = json.load(handle)

    if not isinstance(scenarios, list) or not scenarios:
        raise ScenarioCorpusError(
            f"Scenario file is empty or invalid: {filename}",
            context={"path": str(filepath)},
        )

    return scenarios


def validate_scenario_corpus() -> Dict[str, List[Dict[str, Any]]]:
    """Load all scenario files and ensure the benchmark is present."""
    global _scenario_cache

    if not _scenario_cache:
        loaded = {
            difficulty: _load_scenarios_from_file(filename)
            for difficulty, filename in SCENARIO_FILES.items()
        }
        validation_errors: List[str] = []
        seen_ids = set()
        for difficulty, scenarios in loaded.items():
            for scenario in scenarios:
                scenario_id = scenario.get("scenario_id")
                if scenario_id in seen_ids:
                    validation_errors.append(
                        f"{difficulty}: duplicate scenario_id '{scenario_id}'"
                    )
                else:
                    seen_ids.add(scenario_id)
                validation_errors.extend(validate_scenario_record(scenario))

        if validation_errors:
            raise ScenarioCorpusError(
                "Scenario corpus failed integrity validation.",
                context={"errors": validation_errors[:50], "num_errors": len(validation_errors)},
            )

        _scenario_cache = loaded

    return _scenario_cache


def _load_benchmark_manifest() -> Dict[str, Any]:
    """Load the benchmark manifest from disk."""
    filepath = DATA_DIR / BENCHMARK_MANIFEST_FILE
    if not filepath.exists():
        raise ScenarioCorpusError(
            f"Missing benchmark manifest: {BENCHMARK_MANIFEST_FILE}",
            context={"path": str(filepath)},
        )

    with filepath.open() as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict):
        raise ScenarioCorpusError(
            "Benchmark manifest is invalid.",
            context={"path": str(filepath)},
        )

    return manifest


def get_benchmark_manifest() -> Dict[str, Any]:
    """Return the cached benchmark manifest."""
    global _benchmark_manifest_cache
    if _benchmark_manifest_cache is None:
        _benchmark_manifest_cache = _load_benchmark_manifest()
    return _benchmark_manifest_cache


def validate_benchmark_manifest(
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate the canonical benchmark manifest and return it if valid."""
    manifest = manifest or get_benchmark_manifest()
    validate_scenario_corpus()
    errors: List[str] = []

    version = manifest.get("manifest_version")
    canonical = manifest.get("canonical")
    if not isinstance(version, str) or not version:
        errors.append("manifest_version must be a non-empty string")
    if not isinstance(canonical, dict):
        errors.append("canonical must be a mapping of task IDs to scenario lists")
        canonical = {}

    all_scenarios = {
        scenario["scenario_id"]: scenario
        for scenarios in get_all_scenarios().values()
        for scenario in scenarios
    }
    all_ids: List[str] = []
    signatures: Dict[Tuple[Any, ...], str] = {}

    for task_id in TASK_TO_DIFFICULTY:
        scenario_ids = canonical.get(task_id)
        if not isinstance(scenario_ids, list):
            errors.append(f"{task_id}: canonical split must be a list")
            continue
        if len(scenario_ids) != 20:
            errors.append(f"{task_id}: canonical split must contain exactly 20 scenario IDs")
        if len(scenario_ids) != len(set(scenario_ids)):
            errors.append(f"{task_id}: canonical split contains duplicate scenario IDs")

        for scenario_id in scenario_ids:
            scenario = all_scenarios.get(scenario_id)
            if scenario is None:
                errors.append(f"{task_id}: unknown scenario_id '{scenario_id}'")
                continue
            if get_task_id_for_scenario(scenario) != task_id:
                errors.append(
                    f"{task_id}: scenario '{scenario_id}' belongs to "
                    f"'{get_task_id_for_scenario(scenario)}'"
                )
            scenario_errors = validate_scenario_record(scenario)
            if scenario_errors:
                errors.extend(
                    [
                        f"{task_id}: canonical scenario '{scenario_id}' failed integrity validation: {err}"
                        for err in scenario_errors
                    ]
                )
            signature = scenario_semantic_signature(scenario)
            prior = signatures.get(signature)
            if prior is not None:
                errors.append(
                    f"{task_id}: canonical semantic signature duplicated by "
                    f"'{prior}' and '{scenario_id}'"
                )
            else:
                signatures[signature] = scenario_id
            all_ids.append(scenario_id)

    if len(all_ids) != len(set(all_ids)):
        errors.append("Canonical benchmark contains duplicate scenario IDs across tasks")

    if errors:
        raise ScenarioCorpusError(
            "Benchmark manifest failed validation.",
            context={"errors": errors[:50], "num_errors": len(errors)},
        )

    return manifest


def get_benchmark_scenario_ids(
    task_id: str,
    split: str = "canonical",
) -> List[str]:
    """Return scenario IDs for a named benchmark split."""
    manifest = validate_benchmark_manifest()
    if split != "canonical":
        raise ScenarioLookupError(
            "unknown_benchmark_split",
            "Requested benchmark split does not exist.",
            context={"requested": split, "valid_splits": ["canonical"]},
        )
    canonical = manifest["canonical"]
    if task_id not in canonical:
        raise ScenarioLookupError(
            "unknown_benchmark_task",
            "Requested benchmark task does not exist in the manifest.",
            context={"requested": task_id, "valid_tasks": sorted(canonical)},
        )
    return list(canonical[task_id])


def get_all_scenarios() -> Dict[str, List[Dict[str, Any]]]:
    """Return the cached scenario corpus."""
    return validate_scenario_corpus()


def get_scenarios_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """Get all scenarios for a given difficulty level."""
    return get_all_scenarios().get(difficulty, [])


def get_scenarios_by_task(task_id: str) -> List[Dict[str, Any]]:
    """Get all scenarios for a benchmark task or difficulty bucket."""
    difficulty = TASK_TO_DIFFICULTY.get(task_id, task_id)
    return get_scenarios_by_difficulty(difficulty)


def get_task_id_for_scenario(scenario: Dict[str, Any]) -> Optional[str]:
    """Resolve the canonical benchmark task ID for a loaded scenario."""
    difficulty = scenario.get("difficulty")
    return DIFFICULTY_TO_TASK.get(difficulty)


def get_scenario_by_id(scenario_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific scenario by its ID."""
    for scenarios in get_all_scenarios().values():
        for scenario in scenarios:
            if scenario.get("scenario_id") == scenario_id:
                return scenario
    return None


def get_random_scenario(
    *,
    task_id: Optional[str] = None,
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Get a random scenario filtered by task or difficulty."""
    rng = random.Random(seed)

    if task_id is not None:
        scenarios = get_scenarios_by_task(task_id)
    elif difficulty is not None:
        scenarios = get_scenarios_by_difficulty(difficulty)
    else:
        scenarios = [
            scenario
            for scenario_list in get_all_scenarios().values()
            for scenario in scenario_list
        ]

    if not scenarios:
        lookup_value = task_id if task_id is not None else difficulty
        raise ScenarioLookupError(
            "unknown_task_or_difficulty",
            "Requested task or difficulty bucket does not exist.",
            context={"requested": lookup_value},
        )

    return rng.choice(scenarios)


def load_scenario(
    task_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve a scenario for an episode.

    Supports:
    - scenario_id (e.g. easy_001)
    - task_id (clear_violations, context_dependent, policy_edge_cases)
    - difficulty (easy, medium, hard)
    - None for a random scenario across the full corpus
    """
    if task_id is None:
        return get_random_scenario(seed=seed)

    scenario = get_scenario_by_id(task_id)
    if scenario is not None:
        return scenario

    if task_id in TASK_TO_DIFFICULTY or task_id in SCENARIO_FILES:
        return get_random_scenario(task_id=task_id, seed=seed)

    raise ScenarioLookupError(
        "unknown_task_or_scenario",
        "Requested task_id or scenario_id does not exist.",
        context={
            "requested": task_id,
            "valid_tasks": sorted(TASK_TO_DIFFICULTY),
            "valid_difficulties": sorted(SCENARIO_FILES),
        },
    )


def get_scenario_statistics() -> Dict[str, Any]:
    """Get descriptive statistics about the loaded scenario corpus."""
    all_scenarios = get_all_scenarios()

    stats = {
        "total": 0,
        "by_difficulty": {},
        "by_trigger_type": {},
        "by_category": {},
        "by_decision": {},
        "context_depth_overall": {},
        "context_depth_by_difficulty": {},
    }

    for difficulty, scenarios in all_scenarios.items():
        stats["by_difficulty"][difficulty] = len(scenarios)
        stats["context_depth_by_difficulty"][difficulty] = {}
        stats["total"] += len(scenarios)

        for scenario in scenarios:
            trigger_type = scenario.get("trigger_info", {}).get("trigger_type", "unknown")
            stats["by_trigger_type"][trigger_type] = (
                stats["by_trigger_type"].get(trigger_type, 0) + 1
            )

            category = scenario.get("category", "unknown")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            decision = scenario.get("ground_truth", {}).get("correct_decision", "unknown")
            stats["by_decision"][decision] = stats["by_decision"].get(decision, 0) + 1

            context_depth = len(
                scenario.get("ground_truth", {}).get("context_needed", [])
            )
            stats["context_depth_overall"][context_depth] = (
                stats["context_depth_overall"].get(context_depth, 0) + 1
            )
            stats["context_depth_by_difficulty"][difficulty][context_depth] = (
                stats["context_depth_by_difficulty"][difficulty].get(context_depth, 0)
                + 1
            )

    return stats


def clear_cache() -> None:
    """Clear the scenario cache (useful for testing)."""
    global _scenario_cache, _benchmark_manifest_cache
    _scenario_cache = {}
    _benchmark_manifest_cache = None
