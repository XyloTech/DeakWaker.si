import os

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
MAX_HISTORY_STEPS: int = _env_int("MAX_HISTORY_STEPS", 10)
OBSERVATION_TEXT_LIMIT: int = _env_int("OBSERVATION_TEXT_LIMIT", 2000)
HEADLESS: bool = _env_bool("HEADLESS", False)
THINKING: bool = _env_bool("THINKING", False)
OLLAMA_TEMPERATURE: float = _env_float("OLLAMA_TEMPERATURE", 0.1)
OLLAMA_NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 4096)
OLLAMA_TIMEOUT_S: float = _env_float("OLLAMA_TIMEOUT_S", 300.0)
AUTO_CONFIRM: bool = _env_bool("AUTO_CONFIRM", True)
BROWSER_CHANNEL: str = _env_str("BROWSER_CHANNEL", "chrome")
USER_DATA_DIR: str = _env_str("USER_DATA_DIR", "user_data")
SLOW_MO_MS: int = _env_int("SLOW_MO_MS", 0 if HEADLESS else 100)
