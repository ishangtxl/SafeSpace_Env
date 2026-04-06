"""App and client smoke tests for SafeSpace."""

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
import uvicorn
from fastapi.testclient import TestClient

from content_moderation_env import ModerationAction, ModerationState
from content_moderation_env.client import SafeSpaceEnv
from content_moderation_env.server.app import app
from content_moderation_env.server.reward import normalize_public_reward
from content_moderation_env.server.scenarios import get_scenario_statistics


def _find_free_port() -> int:
    """Allocate an ephemeral localhost port for a temporary Uvicorn server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server_url():
    """Serve the FastAPI app over a real socket for typed WebSocket client tests."""
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    health_url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urlopen(health_url) as response:  # noqa: S310
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.1)
    else:  # pragma: no cover - defensive timeout path
        raise RuntimeError("Timed out waiting for the SafeSpace test server to start.")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_health_endpoint_returns_healthy_status():
    """The HTTP app should expose a healthy liveness endpoint."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_client_parse_result_builds_typed_breakdowns():
    """The typed client should parse reward and grade breakdown models."""
    client = SafeSpaceEnv(base_url="http://localhost:8000")
    result = client._parse_result(  # pylint: disable=protected-access
        {
            "observation": {
                "content_item": {
                    "post_id": "p_test",
                    "text": "Test content",
                    "author_id": "user_test",
                    "community": "general",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "media_type": "text",
                    "media_description": None,
                },
                "trigger_info": {"trigger_type": "user_report"},
                "gathered_context": {},
                "platform_policy": "policy",
                "available_factors": ["no_violation_found"],
                "actions_taken": 1,
                "max_actions": 8,
                "action_history": ["decide: approve"],
                "feedback": "done",
                "reward_breakdown": {
                    "reward_type": "terminal",
                    "total": 0.8,
                    "decision": {
                        "score": 0.55,
                        "max": 0.55,
                        "details": {"grade": "perfect"},
                    },
                },
                "task_grade": 0.92,
                "grade_breakdown": {
                    "decision": {
                        "weight": 0.7,
                        "score": 1.0,
                        "details": {"grade": "perfect"},
                    },
                    "total": 0.92,
                },
            },
            "reward": 0.8,
            "done": True,
        }
    )

    assert result.observation.reward_breakdown is not None
    assert result.observation.reward_breakdown.reward_type == "terminal"
    assert result.observation.reward_breakdown.decision is not None
    assert result.observation.reward_breakdown.decision.details["grade"] == "perfect"
    assert result.observation.grade_breakdown is not None
    assert result.observation.grade_breakdown.total == 0.92


def test_schema_endpoint_exposes_typed_moderation_state_schema():
    """The public schema should publish SafeSpace's typed ModerationState contract."""
    with TestClient(app) as client:
        payload = client.get("/schema").json()

    assert payload["state"]["title"] == "ModerationState"
    assert "trigger_type" in payload["state"].get("properties", {})
    assert "scenario_id" in payload["state"].get("properties", {})


def test_state_endpoint_returns_typed_moderation_state_snapshot():
    """The public state endpoint should validate as ModerationState."""
    with TestClient(app) as client:
        response = client.get("/state")

    assert response.status_code == 200
    state = ModerationState.model_validate(response.json())
    assert state.step_count == 0
    assert state.actions_taken == 0
    assert state.done is False


def test_typed_client_state_round_trips_public_fields(live_server_url):
    """The WebSocket client is the canonical typed state contract for SafeSpace."""
    with SafeSpaceEnv(base_url=live_server_url).sync() as env:
        env.reset(scenario_id="med_001")

        state = env.state()
        assert state.scenario_id == "med_001"
        assert state.task_id == "context_dependent"
        assert state.difficulty == "medium"
        assert state.trigger_type == "user_report"
        assert state.actions_taken == 0
        assert state.context_requested == []
        assert state.episode_reward == 0.0
        assert state.raw_episode_reward == 0.0

        env.step(ModerationAction(action_type="request_thread_context"))

        state = env.state()
        assert state.scenario_id == "med_001"
        assert state.task_id == "context_dependent"
        assert state.difficulty == "medium"
        assert state.trigger_type == "user_report"
        assert state.actions_taken == 1
        assert state.context_requested == ["thread_context"]
        assert state.episode_reward == pytest.approx(normalize_public_reward(0.05))
        assert state.raw_episode_reward == pytest.approx(0.05)


def test_report_stats_script_outputs_expected_sections():
    """The stats helper should emit benchmark counts as JSON."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "report_stats.py"
    output = subprocess.check_output([sys.executable, str(script_path)], text=True)
    payload = json.loads(output)

    assert payload["total"] == get_scenario_statistics()["total"]
    assert "by_decision" in payload
    assert "context_depth_overall" in payload
