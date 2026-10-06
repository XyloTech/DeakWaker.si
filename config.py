import os

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


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
HEADLESS: bool = _env_bool("HEADLESS", True)
THINKING: bool = _env_bool("THINKING", False)
