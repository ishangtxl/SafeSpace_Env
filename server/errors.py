"""Structured error types for SafeSpace."""

from __future__ import annotations

from typing import Any, Dict, Optional


class SafeSpaceError(Exception):
    """Base error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.context = context or {}

    def to_payload(self) -> Dict[str, Any]:
        """Serialize the error for HTTP or test assertions."""
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class ScenarioCorpusError(SafeSpaceError):
    """Raised when the scenario corpus is missing or invalid."""

    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            "scenario_corpus_invalid",
            message,
            status_code=500,
            context=context,
        )


class ScenarioLookupError(SafeSpaceError):
    """Raised when a requested task or scenario cannot be resolved."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(code, message, status_code=422, context=context)
