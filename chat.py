import asyncio
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from agent import run_turn
from browser import BrowserWrapper
from config import HEADLESS, MAX_STEPS
from llm import LLMClient

BRAND = "product of Xylotech"
DEVELOPER = "Developed by Harshit"

_ACTION_COLORS = {
    "navigate": "36",
    "click": "32",
    "type": "33",
    "scroll": "35",
    "done": "1;32",
    "verify": "1;35",
    "ask_user": "1;36",
    "error": "1;31",
}


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        os.system("")


def _color_enabled() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _color_enabled():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _box(lines: list[str], title: str) -> str:
    width = max((len(line) for line in lines), default=0) + 2
    title_part = f" {title} " if title else ""
    top = "┌" + title_part + "─" * (width - len(title_part)) + "┐"
    body = ["│" + line.ljust(width) + "│" for line in lines]
    bottom = "└" + "─" * width + "┘"
    return "\n".join([top, *body, bottom])


def render_banner() -> str:
    lines = [
        "  desk.waker — autonomous browser agent  ",
        f"          {BRAND}          ",
    ]
    return _c("1;36", _box(lines, ""))


def render_footer() -> str:
    return _c("2", f"  {DEVELOPER}")


def render_step(entry: dict, page_info: dict, elements: list) -> str:
    step = entry.get("step", "?")
    action = str(entry.get("action", "?"))
    action_label = action.upper()
    action_color = _ACTION_COLORS.get(action, "1;37")
    stamp = datetime.now().strftime("%H:%M:%S")
    header = (
        f"  ◆ Step {step}/{MAX_STEPS}  "
        + _c(action_color, action_label)
        + f"  {_c('2', stamp)}"
    )
    parts = [header]
    thought = str(entry.get("thought", "") or "").strip()
    if thought:
        parts.append(
            textwrap.fill(
                thought,
                width=78,
                initial_indent="    Thought  ",
                subsequent_indent="              ",
            )
        )
    result = str(entry.get("result", "") or "")
    result_color = "32" if result.startswith("OK") else "31"
    page_bits = []
    url = str(page_info.get("url", "") or "")
    title = str(page_info.get("title", "") or "")
    if url:
        page_bits.append(url)
    if title:
        page_bits.append(title)
    page_bits.append(f"{len(elements)} elements")
    page_line = "    Result   " + _c(result_color, result) + " · " + " · ".join(page_bits)
    parts.append(page_line)
    return "\n".join(parts)


def render_status(status: str, answer) -> str:
    text = "" if answer is None else str(answer)
    if status == "done":
        return f"  {_c('1;32', '✓ DONE')}" + (f" — {text}" if text else "")
    if status == "max_steps":
        body = "  " + _c("1;33", "⚠ MAX STEPS") + " — hit the step limit"
        if text:
            body += f": {text}"
        return body
    if status == "error":
        return f"  {_c('1;31', '✗ ERROR')}" + (f" — {text}" if text else "")
    return f"  {status.upper()} — {text}"


def render_confirm_box(prompt: str) -> str:
    return _c("1;33", _box([f"  {prompt.strip()}"], "⚠ Confirmation"))


def render_ask_box(question: str) -> str:
    return _c("1;36", _box([f"  {question.strip()}"], "? Question"))


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
    print(render_confirm_box(prompt))
    try:
        return input().strip().lower() == "y"
    except EOFError:
        return False


def _on_ask(question: str) -> str:
    print(render_ask_box(question))
    try:
        return input().strip()
    except EOFError:
        return ""


def _on_step(entry: dict, page_info: dict, elements: list) -> None:
    print(render_step(entry, page_info, elements))


async def repl() -> None:
    print(render_banner())
    browser = BrowserWrapper(headless=HEADLESS)
    llm = LLMClient()
    logger = SessionLogger()
    try:
        await browser.start()
        while True:
            try:
                goal = input(_c("1;33", "goal › ")).strip()
            except EOFError:
                break
            if not goal:
                continue
            print(_c("1;36", f"  Goal  {goal}"))
            result = await run_turn(
                goal,
                browser,
                llm,
                on_confirm=_on_confirm,
                on_ask=_on_ask,
                on_step=_on_step,
                print_steps=False,
            )
            print(render_status(result.status, result.answer))
            path = logger.append(goal, result.transcript)
            print(_c("2", f"  Transcript: {path}"))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        await browser.close()
        print(render_footer())


def main() -> None:
    _configure_streams()
    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as exc:
        print(f"\nShutting down: {exc}")
        print("Fix the problem above, then start the agent again with: python chat.py")


if __name__ == "__main__":
    main()
