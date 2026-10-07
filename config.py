import os
import platform

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


OLLAMA_HOST: str = _env_str("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL: str = _env_str("MODEL", "qwen3:latest")
MAX_STEPS: int = _env_int("MAX_STEPS", 15)
MAX_HISTORY_STEPS: int = _env_int("MAX_HISTORY_STEPS", 6)
OBSERVATION_TEXT_LIMIT: int = _env_int("OBSERVATION_TEXT_LIMIT", 1600)
OBSERVATION_ELEMENT_LIMIT: int = _env_int("OBSERVATION_ELEMENT_LIMIT", 80)
HEADLESS: bool = _env_bool("HEADLESS", False)
THINKING: bool = _env_bool("THINKING", False)
OLLAMA_TEMPERATURE: float = _env_float("OLLAMA_TEMPERATURE", 0.1)
OLLAMA_NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 4096)
OLLAMA_TIMEOUT_S: float = _env_float("OLLAMA_TIMEOUT_S", 300.0)
AUTO_CONFIRM: bool = _env_bool("AUTO_CONFIRM", True)
_DEFAULT_BROWSER_CHANNEL = (
    "brave"
)
BROWSER_CHANNEL: str = _env_str("BROWSER_CHANNEL", _DEFAULT_BROWSER_CHANNEL)
BROWSER_CHANNEL_ALTERNATIVES: list[str] = [
    item.strip()
    for item in _env_str("BROWSER_CHANNEL_ALTERNATIVES", "msedge,firefox").split(",")
    if item.strip()
]
BROWSER_EXECUTABLE: str = _env_str("BROWSER_EXECUTABLE", "")
BROWSER_CDP_URL: str = _env_str("BROWSER_CDP_URL", "")
BROWSER_ALLOW_FALLBACK: bool = _env_bool("BROWSER_ALLOW_FALLBACK", False)
USER_DATA_DIR: str = _env_str("USER_DATA_DIR", "user_data")
DOWNLOAD_DIR: str = _env_str("DOWNLOAD_DIR", "downloads")
SLOW_MO_MS: int = _env_int("SLOW_MO_MS", 0)

# Reasoning/Thinking configuration
THINKING_MODE: str = _env_str("THINKING_MODE", "adaptive")
# 0 = adaptive (deep thinking only after a failed step), 1 = think every step, "auto" = auto-decide
