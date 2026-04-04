#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PREFLIGHT_PORT:-8010}"
IMAGE_TAG="${PREFLIGHT_IMAGE_TAG:-safespace:preflight}"
CONTAINER_NAME="${PREFLIGHT_CONTAINER_NAME:-safespace-preflight}"
SCENARIO_ID="${PREFLIGHT_SCENARIO_ID:-easy_001}"
HEALTHCHECK_RETRIES="${PREFLIGHT_HEALTHCHECK_RETRIES:-20}"
DOCKERFILE_PATH="${PREFLIGHT_DOCKERFILE_PATH:-Dockerfile}"

find_bin() {
  local name="$1"
  local candidates=()

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("$VIRTUAL_ENV/bin/$name")
  fi

  candidates+=(
    "$ROOT_DIR/.venv/bin/$name"
    "$ROOT_DIR/.review-venv/bin/$name"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  command -v "$name" 2>/dev/null || return 1
}

find_python() {
  local candidates=()

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("$VIRTUAL_ENV/bin/python")
  fi

  candidates+=(
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/.review-venv/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  command -v python3 2>/dev/null || command -v python 2>/dev/null || return 1
}

cleanup() {
  if command -v docker >/dev/null 2>&1; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

PYTHON_BIN="$(find_python)"
PYTEST_BIN="$(find_bin pytest)"
OPENENV_BIN="$(find_bin openenv)"
DOCKER_BIN="$(find_bin docker || true)"

printf '\n[1/9] Running targeted runtime/client tests...\n'
"$PYTEST_BIN" tests/test_environment.py tests/test_app_client.py -q

printf '\n[2/9] Running full test suite...\n'
"$PYTEST_BIN" tests -q

printf '\n[3/9] Validating canonical benchmark manifest...\n'
"$PYTHON_BIN" - <<'PY'
from content_moderation_env.server.scenarios import validate_benchmark_manifest

manifest = validate_benchmark_manifest()
print(
    "Canonical benchmark manifest is valid: "
    f"{manifest['manifest_version']}"
)
PY

printf '\n[4/9] Running package asset smoke check...\n'
"$PYTHON_BIN" scripts/check_package_assets.py

printf '\n[5/9] Validating inference config contract...\n'
API_BASE_URL="https://example.invalid/v1" \
MODEL_NAME="test-model" \
API_KEY="test-key" \
ENV_BASE_URL="http://127.0.0.1:${PORT}" \
  "$PYTHON_BIN" inference.py --validate-config

printf '\n[6/9] Validating OpenEnv manifest...\n'
"$OPENENV_BIN" validate .

if [[ -z "$DOCKER_BIN" ]]; then
  printf '\n[7/9-9/9] Docker CLI not found on PATH. Skipping container build, health, and typed-client container smoke checks.\n'
  printf '\nPreflight completed successfully.\n'
  exit 0
fi

printf '\n[7/9] Building Docker image...\n'
"$DOCKER_BIN" build -t "$IMAGE_TAG" -f "$DOCKERFILE_PATH" .

printf '\n[8/9] Starting container and checking health...\n'
"$DOCKER_BIN" run --rm -d -p "${PORT}:8000" --name "$CONTAINER_NAME" "$IMAGE_TAG" >/dev/null

for ((attempt = 1; attempt <= HEALTHCHECK_RETRIES; attempt++)); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi

  if [[ "$attempt" -eq "$HEALTHCHECK_RETRIES" ]]; then
    printf 'Health check failed after %s attempts.\n' "$HEALTHCHECK_RETRIES" >&2
    exit 1
  fi

  sleep 1
done

curl -fsS "http://127.0.0.1:${PORT}/health"
printf '\n'

printf '\n[9/9] Running typed client smoke test...\n'
export PREFLIGHT_PORT="$PORT"
export PREFLIGHT_SCENARIO_ID="$SCENARIO_ID"
"$PYTHON_BIN" - <<'PY'
import os

from content_moderation_env import ModerationAction, SafeSpaceEnv

port = os.environ["PREFLIGHT_PORT"]
scenario_id = os.environ["PREFLIGHT_SCENARIO_ID"]

with SafeSpaceEnv(base_url=f"http://127.0.0.1:{port}").sync() as env:
    result = env.reset(scenario_id=scenario_id)
    assert env.state().scenario_id == scenario_id
    result = env.step(
        ModerationAction(
            action_type="decide",
            decision="remove",
            primary_violation="5.1",
            severity="high",
            confidence=0.95,
            key_factors=["spam_commercial"],
        )
    )
    assert result.done is True
    assert result.reward is not None
    assert result.observation.task_grade is not None
    print(
        f"Smoke test passed for {scenario_id}: "
        f"reward={result.reward:.3f}, task_grade={result.observation.task_grade:.3f}"
    )
PY

printf '\nPreflight completed successfully.\n'
