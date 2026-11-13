"""LLMClient interface, GeminiClient with retry/backoff, and FakeLLMClient for tests."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any

from app.config import (
    LLM_INITIAL_BACKOFF_SECONDS,
    LLM_MAX_RETRIES,
    get_gemini_config,
)
from app.tools import (
    FindDependentsInput,
    ListSymbolsInput,
    ReadFileInput,
    SearchCodeInput,
)

logger = logging.getLogger(__name__)


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


@dataclass
class FakeFunctionCall:
    """Function call representation compatible with agent loop."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeResponse:
    """Model response representation compatible with agent loop."""

    text: str = ""
    function_calls: list[FakeFunctionCall] = field(default_factory=list)
    candidates: list[Any] = field(default_factory=list)


def _is_retryable_error(exc: Exception) -> bool:
    """Return True if the exception represents an HTTP 429, 503, or 504 error."""
    code = getattr(exc, "code", None)
    if code in (429, 503, 504):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 503, 504):
        return True
    msg = str(exc)
    return any(
        token in msg
        for token in (
            "429",
            "503",
            "504",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "DEADLINE_EXCEEDED",
            "timed out",
            "Timeout",
        )
    )


class GroqClient(LLMClient):
    """Free, fast OpenAI-compatible tool-calling client using Groq (or OpenRouter) via httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        ).rstrip("/")

    def _openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions matching RepoTools."""
        specs = [
            (
                "search_code",
                "Semantic search over indexed Python functions and classes.",
                SearchCodeInput.model_json_schema(),
            ),
            (
                "find_dependents",
                "Find all files and symbols that import or call a given symbol ('path::name').",
                FindDependentsInput.model_json_schema(),
            ),
            (
                "read_file",
                "Read lines from a file in the repository with 1-based line numbers.",
                ReadFileInput.model_json_schema(),
            ),
            (
                "list_symbols",
                "List all top-level functions and classes in a Python file.",
                ListSymbolsInput.model_json_schema(),
            ),
        ]
        out = []
        for name, desc, schema in specs:
            clean_schema = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": clean_schema,
                    },
                }
            )
        return out

    def _convert_history(self, history: list[Any], system: str) -> list[dict[str, Any]]:
        """Convert agent history into OpenAI chat messages."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        call_counter = 0
        pending_ids: list[str] = []

        for item in history:
            if isinstance(item, dict):
                if item.get("role") == "user" and "text" in item:
                    messages.append({"role": "user", "content": item["text"]})
                elif item.get("role") == "model" and "function_calls" in item:
                    tcs = []
                    pending_ids = []
                    for fc in item["function_calls"]:
                        call_counter += 1
                        cid = f"call_{call_counter}"
                        pending_ids.append(cid)
                        tcs.append(
                            {
                                "id": cid,
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(fc.args or {}),
                                },
                            }
                        )
                    messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
                elif item.get("role") == "tool" and "responses" in item:
                    for idx, (name, payload) in enumerate(item["responses"]):
                        cid = pending_ids[idx] if idx < len(pending_ids) else f"call_{call_counter}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": cid,
                                "name": name,
                                "content": json.dumps(payload),
                            }
                        )
                continue

            # Handle google.genai.types.Content objects
            role = getattr(item, "role", "user")
            parts = getattr(item, "parts", None) or []
            text_parts = []
            func_responses = []
            for p in parts:
                if getattr(p, "text", None):
                    text_parts.append(p.text)
                elif getattr(p, "function_response", None) is not None:
                    fr = p.function_response
                    func_responses.append((fr.name, fr.response))
            if text_parts:
                messages.append({"role": "user" if role == "user" else "assistant", "content": "\n".join(text_parts)})
            for idx, (name, payload) in enumerate(func_responses):
                cid = pending_ids[idx] if idx < len(pending_ids) else f"call_{call_counter}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "name": name,
                        "content": json.dumps(payload),
                    }
                )
        return messages

    def _discover_model(self, client: Any, headers: dict[str, str]) -> str:
        """Query Groq /models to pick an active tool-capable chat model on this account."""
        resp = client.get(f"{self.base_url}/models", headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"Groq /models HTTP {resp.status_code}: {resp.text}")
        models = [
            m.get("id", "")
            for m in (resp.json().get("data") or [])
            if m.get("id")
            and not any(
                skip in m.get("id", "").lower()
                for skip in ("whisper", "tts", "guard", "embed", "vision")
            )
        ]
        preferred = (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        )
        for p in preferred:
            if p in models:
                return p
        for m in models:
            if any(k in m.lower() for k in ("llama", "qwen", "gpt", "deepseek", "gemma", "mixtral")):
                return m
        if models:
            return models[0]
        raise RuntimeError("No active chat models found on this Groq account.")

    def generate(
        self,
        history: list[Any],
        tools: list[Any],
        system: str,
    ) -> FakeResponse:
        """Call Groq / OpenAI-compatible chat completions endpoint with auto model discovery."""
        import httpx

        api_key = (self.api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        model = (self.model or os.environ.get("GROQ_MODEL", "")).strip()
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY environment variable.")

        messages = self._convert_history(history, system)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=25.0) as client:
            if not model:
                model = self._discover_model(client, headers)
                self.model = model
                logger.info("Auto-selected Groq model: %s", model)

            payload = {
                "model": model,
                "messages": messages,
                "tools": self._openai_tools(),
                "tool_choice": "auto",
                "temperature": 0.0,
            }
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 404 or "model_not_found" in resp.text:
                model = self._discover_model(client, headers)
                self.model = model
                payload["model"] = model
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"Groq API HTTP {resp.status_code} (model={model}): {resp.text}")
            data = resp.json()

        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        fcs: list[FakeFunctionCall] = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                parsed_args = {}
            fcs.append(FakeFunctionCall(name=fn.get("name", ""), args=parsed_args))

        return FakeResponse(text=msg.get("content") or "", function_calls=fcs)


class GeminiClient(LLMClient):
    """Gemini implementation of LLMClient using the google-genai SDK."""

    def __init__(
        self,
        max_retries: int = LLM_MAX_RETRIES,
        initial_backoff: float = LLM_INITIAL_BACKOFF_SECONDS,
    ) -> None:
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self._client: Any = None
        self._cached_key: str | None = None

    def generate(
        self,
        history: list[Any],
        tools: list[Any],
        system: str,
    ) -> Any:
        """Call Gemini generate_content (or Groq if GROQ_API_KEY is set)."""
        if os.environ.get("GROQ_API_KEY", "").strip():
            return GroqClient().generate(history=history, tools=tools, system=system)

        api_key, model_name = get_gemini_config()
        from google import genai
        from google.genai import types

        if self._client is None or self._cached_key != api_key:
            self._client = genai.Client(api_key=api_key)
            self._cached_key = api_key

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools,
            temperature=0.0,
        )

        # If the primary model stalls/504s repeatedly, try fast fallback models automatically.
        models_to_try = [model_name]
        for fallback in ("gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-flash-latest"):
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            active_model = models_to_try[min(attempt, len(models_to_try) - 1)]
            try:
                return self._client.models.generate_content(
                    model=active_model,
                    contents=history,
                    config=config,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries and (
                    _is_retryable_error(exc) or "404" in str(exc) or "NOT_FOUND" in str(exc)
                ):
                    delay = self.initial_backoff * (2**attempt)
                    logger.warning(
                        "Gemini error on %s (%s); retrying in %.1fs (%d/%d)",
                        active_model,
                        exc,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise

        assert last_exc is not None
        raise last_exc


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
