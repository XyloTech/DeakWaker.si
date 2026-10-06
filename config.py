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


GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY") or None
MODEL: str = _env_str("MODEL", "llama-3.3-70b-versatile")
MAX_STEPS: int = _env_int("MAX_STEPS", 15)
MAX_HISTORY_STEPS: int = _env_int("MAX_HISTORY_STEPS", 10)
OBSERVATION_TEXT_LIMIT: int = _env_int("OBSERVATION_TEXT_LIMIT", 2000)
HEADLESS: bool = _env_bool("HEADLESS", True)
