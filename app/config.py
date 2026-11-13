"""Configuration constants and runtime environment helpers for repo-agent."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_REPO = (BASE_DIR / "sample_repo").resolve()

ALLOWED_INDEX_ROOT = Path(
    os.environ.get("ALLOWED_INDEX_ROOT", str(DEFAULT_SAMPLE_REPO))
).resolve()

CHROMA_PERSIST_DIR = Path(
    os.environ.get("CHROMA_PERSIST_DIR", str(BASE_DIR / ".chroma"))
).resolve()

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MAX_TOOL_ROUNDS = 8
AGENT_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES = 4
LLM_INITIAL_BACKOFF_SECONDS = 1.0
MAX_READ_LINES = 400


def get_gemini_config() -> tuple[str, str]:
    """Return (api_key, model_name) from environment, raising RuntimeError if missing."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model_name = os.environ.get("GEMINI_MODEL", "").strip()
    missing = []
    if not api_key:
        missing.append("GEMINI_API_KEY")
    if not model_name:
        missing.append("GEMINI_MODEL")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them before calling the Gemini API."
        )
    return api_key, model_name
