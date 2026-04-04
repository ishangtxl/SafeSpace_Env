# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SafeSpace Content Moderation Environment.

This module creates an HTTP server that exposes the SafeSpaceEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn content_moderation_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn content_moderation_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m content_moderation_env.server.app
"""

from contextlib import asynccontextmanager

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import SchemaResponse
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install with: pip install openenv-core"
    ) from e

try:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from ..models import ModerationAction, ModerationObservation, ModerationState
    from .errors import SafeSpaceError
    from .environment import SafeSpaceEnvironment
    from .scenarios import validate_benchmark_manifest, validate_scenario_corpus
except (ModuleNotFoundError, ImportError):
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from models import ModerationAction, ModerationObservation, ModerationState
    from server.errors import SafeSpaceError
    from server.environment import SafeSpaceEnvironment
    from server.scenarios import validate_benchmark_manifest, validate_scenario_corpus


# Create the app with web interface and README integration
app = create_app(
    SafeSpaceEnvironment,
    ModerationAction,
    ModerationObservation,
    env_name="safespace",
    max_concurrent_envs=10,  # Allow multiple concurrent WebSocket sessions
)


def _replace_get_route(path: str) -> None:
    """Remove OpenEnv's default GET route so SafeSpace can publish typed variants."""
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == path
            and "GET" in (getattr(route, "methods", set()) or set())
        )
    ]
    app.openapi_schema = None


def _get_state_snapshot() -> ModerationState:
    """Return a typed environment-state snapshot for the stateless HTTP route."""
    env = SafeSpaceEnvironment()
    try:
        return env.state
    finally:
        env.close()


def _get_schema_payload() -> SchemaResponse:
    """Return the action, observation, and typed state schemas for public docs."""
    return SchemaResponse(
        action=ModerationAction.model_json_schema(),
        observation=ModerationObservation.model_json_schema(),
        state=ModerationState.model_json_schema(),
    )


_replace_get_route("/state")
_replace_get_route("/schema")


@app.get(
    "/state",
    response_model=ModerationState,
    tags=["State Management"],
    summary="Get current environment state",
    description="""
Retrieve the current environment state using SafeSpace's typed ModerationState model.

The shared HTTP route remains stateless and returns a fresh environment snapshot.
Live benchmark state should still be read through a session-aware SafeSpaceEnv client.
""",
)
async def get_state() -> ModerationState:
    """Serve the typed public state schema for validators and human reviewers."""
    return _get_state_snapshot()


@app.get(
    "/schema",
    response_model=SchemaResponse,
    tags=["Environment Info"],
    summary="Get action, observation, and state schemas",
    description="""
Return the JSON schemas for the SafeSpace action, observation, and typed state models.
""",
)
async def get_schemas() -> SchemaResponse:
    """Serve typed schemas for public API consumers and documentation."""
    return _get_schema_payload()


@asynccontextmanager
async def _lifespan(_: object):
    """Fail startup early if the benchmark corpus is missing or invalid."""
    validate_scenario_corpus()
    validate_benchmark_manifest()
    yield


app.router.lifespan_context = _lifespan


@app.exception_handler(SafeSpaceError)
async def _handle_safespace_error(
    request: Request,
    exc: SafeSpaceError,
) -> JSONResponse:
    """Return structured validation errors for reset-time failures."""
    del request
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.to_payload()},
    )


def main():
    """
    Entry point for direct execution.

    This function enables running the server without Docker:
        python -m content_moderation_env.server.app
        python server/app.py

    Reads --host and --port from command line arguments if provided.
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()

    uvicorn.run(app, host=args.host, port=args.port)


# Entry point for openenv and direct execution
if __name__ == "__main__":
    main()
