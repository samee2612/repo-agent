"""LLMClient interface, GeminiClient with retry/backoff, and FakeLLMClient for tests."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any

from app.config import (
    LLM_INITIAL_BACKOFF_SECONDS,
    LLM_MAX_RETRIES,
    get_gemini_config,
)


class LLMClient(ABC):
    """Abstract interface for LLM generation with tool calling."""

    @abstractmethod
    def generate(
        self,
        history: list[Any],
        tools: list[Any],
        system: str,
    ) -> Any:
        """Generate a model response given conversation history, tools, and system prompt."""


def _is_retryable_error(exc: Exception) -> bool:
    """Return True if the exception represents an HTTP 429 or 503 error."""
    code = getattr(exc, "code", None)
    if code in (429, 503):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 503):
        return True
    msg = str(exc)
    return "429" in msg or "503" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg


class GeminiClient(LLMClient):
    """Gemini implementation of LLMClient using the google-genai SDK."""

    def __init__(
        self,
        max_retries: int = LLM_MAX_RETRIES,
        initial_backoff: float = LLM_INITIAL_BACKOFF_SECONDS,
    ) -> None:
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def generate(
        self,
        history: list[Any],
        tools: list[Any],
        system: str,
    ) -> Any:
        """Call Gemini generate_content with exponential backoff on 429/503."""
        api_key, model_name = get_gemini_config()
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools,
        )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=history,
                    config=config,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries and _is_retryable_error(exc):
                    delay = self.initial_backoff * (2**attempt)
                    time.sleep(delay)
                    continue
                raise

        assert last_exc is not None
        raise last_exc


@dataclass
class FakeFunctionCall:
    """Scripted function call for FakeLLMClient."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeResponse:
    """Scripted model response mimicking google-genai GenerateContentResponse."""

    text: str = ""
    function_calls: list[FakeFunctionCall] = field(default_factory=list)
    candidates: list[Any] = field(default_factory=list)


class FakeLLMClient(LLMClient):
    """Deterministic scripted LLMClient for unit and API tests."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        history: list[Any],
        tools: list[Any],
        system: str,
    ) -> FakeResponse:
        """Return the next scripted FakeResponse."""
        self.calls.append({"history_len": len(history), "system": system})
        if self._index >= len(self._responses):
            return FakeResponse(text="No more scripted responses.", function_calls=[])
        resp = self._responses[self._index]
        self._index += 1
        return resp
