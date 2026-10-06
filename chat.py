import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent import run_turn
from browser import BrowserWrapper
from config import HEADLESS
from llm import LLMClient


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


class SessionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)

    def append(self, goal: str, transcript: list[dict]) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.log_dir / f"session-{stamp}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry in transcript:
                record = {"goal": goal, **entry}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path


def _on_confirm(prompt: str) -> bool:
    print(prompt, end="")
    try:
        return input().strip().lower() == "y"
    except EOFError:
        return False


def _on_ask(question: str) -> str:
    print(question)
    try:
        return input().strip()
    except EOFError:
        return ""


async def repl() -> None:
    print("desk.waker — describe your goal, or press Ctrl+C to quit.")
    browser = BrowserWrapper(headless=HEADLESS)
    llm = LLMClient()
    logger = SessionLogger()
    await browser.start()
    try:
        while True:
            try:
                goal = input("goal: ").strip()
            except EOFError:
                break
            if not goal:
                continue
            result = await run_turn(
                goal, browser, llm, on_confirm=_on_confirm, on_ask=_on_ask
            )
            if result.status == "error":
                print(f"Turn stopped: {result.answer}")
            else:
                print(f"Status: {result.status}")
                print(f"Answer: {result.answer}")
            path = logger.append(goal, result.transcript)
            print(f"Transcript saved: {path}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        await browser.close()


def main() -> None:
    _configure_streams()
    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
