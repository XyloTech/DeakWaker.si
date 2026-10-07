import asyncio
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from agent import run_turn
from browser import BrowserWrapper, detect_browsers
from config import BROWSER_CHANNEL, HEADLESS, MAX_STEPS
from llm import LLMClient

BRAND = "product of Xylotech, developed by Harshit"
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
    max_width = max(48, min(shutil.get_terminal_size((100, 24)).columns - 4, 100))
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(str(line), width=max_width - 4) or [""])
    lines = wrapped
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
        "  Ollama local | Brave browser | Ctrl+C aborts the active task  ",
    ]
    return _c("1;36", _box(lines, ""))


def render_footer() -> str:
    return _c("2", f"  {DEVELOPER}")


def render_help() -> str:
    return _c(
        "2",
        _box(
            [
                "you > type a normal message or browser task",
                "/help     show these commands",
                "/status   show browser and profile status",
                "/clear    clear the terminal",
                "/quit     exit the agent",
                "Ctrl+C    abort the active task; browser stays open",
            ],
            "Commands",
        ),
    )


def render_ready(browser: BrowserWrapper) -> str:
    return _c(
        "1;32",
        _box(
            [
                "  READY",
                f"  {browser.describe()}",
                "  Type /help for commands; Ctrl+C aborts only the active task.",
            ],
            "Browser session",
        ),
    )


def render_chat_reply(reply: str) -> str:
    return _c("1;36", _box(textwrap.wrap(f"  {reply}", width=86), "AI"))


def render_task(goal: str) -> str:
    return _c("1;36", _box([f"  {goal}"], "Task"))


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
    if page_bits and len(page_bits[0]) > 72:
        page_bits[0] = page_bits[0][:32] + "..." + page_bits[0][-36:]
    page_line = "    Result   " + _c(result_color, result) + " | " + " | ".join(page_bits)
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


class ThinkingIndicator:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __call__(self, active: bool) -> None:
        if active:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        print("  AI thinking ", end="", flush=True)

        def animate() -> None:
            frames = ("·  ", "·· ", "···", " ··", "  ·", " ··")
            index = 0
            while not self._stop.wait(0.12):
                print(f"\r  AI thinking {frames[index % len(frames)]}", end="", flush=True)
                index += 1

        self._thread = threading.Thread(target=animate, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None
        print("\r  AI thinking done\n", end="", flush=True)


def _normalize_goal(raw_goal: str) -> str:
    """Keep an accidentally pasted product brief out of the task prompt."""
    goal = raw_goal.strip()
    marker = "Build a production-ready local browser automation agent"
    marker_at = goal.find(marker)
    if 0 < marker_at <= 500:
        goal = goal[:marker_at].rstrip()
        print(_c("2", "  Ignored pasted product specification after the goal."))
    return goal


def is_conversational_message(message: str) -> bool:
    words = re.findall(r"[a-z']+", message.lower())
    if not words:
        return False
    task_words = {
        "book", "check", "click", "complete", "fill", "find", "get", "go",
        "launch", "navigate", "open", "play", "search", "send", "see",
        "show", "solve", "troubleshoot", "type", "upload", "visit", "watch",
    }
    task_nouns = {
        "browser", "flight", "form", "page", "site", "song", "ticket", "video",
        "website", "youtube",
    }
    if task_words.intersection(words) or task_nouns.intersection(words):
        return False
    return True


def _needs_windows_brave_handoff() -> bool:
    return bool(
        os.getenv("WSL_DISTRO_NAME")
        and BROWSER_CHANNEL.strip().lower() == "brave"
        and any(item["name"] == "brave-windows-host" for item in detect_browsers())
    )


def _run_windows_brave_agent() -> None:
    print(
        _c(
            "2",
            "  Windows Brave detected; handing off to Windows Python "
            "so the real Brave browser and profile can be used...",
        ),
        flush=True,
    )
    try:
        completed = subprocess.run(["cmd.exe", "/c", "py", "chat.py"], check=False)
    except OSError as exc:
        raise RuntimeError(
            "Could not invoke Windows Python. Run 'py chat.py' from PowerShell "
            "or install Brave inside Kali."
        ) from exc
    if completed.returncode:
        raise RuntimeError(f"Windows Brave agent exited with code {completed.returncode}")


async def repl() -> None:
    print(render_banner())
    if _needs_windows_brave_handoff():
        _run_windows_brave_agent()
        return
    browser = BrowserWrapper(headless=HEADLESS)
    llm = LLMClient()
    logger = SessionLogger()
    thinking = ThinkingIndicator()
    try:
        print(_c("2", "  Starting local browser..."), flush=True)
        try:
            await asyncio.wait_for(browser.start(), timeout=60)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Browser startup timed out after 60 seconds. "
                "Close any running Chrome window or set BROWSER_CHANNEL=none."
            ) from exc
        print(render_ready(browser), flush=True)
        while True:
            try:
                goal = _normalize_goal(input(_c("1;33", "goal › ")))
            except EOFError:
                break
            if not goal:
                continue
            command = goal.lower()
            if command == "/help":
                print(render_help(), flush=True)
                continue
            if command == "/status":
                print(render_ready(browser), flush=True)
                continue
            if command == "/clear":
                print("\033[2J\033[H", end="", flush=True)
                print(render_banner(), flush=True)
                continue
            if command in {"/quit", "/exit"}:
                break
            if is_conversational_message(goal):
                thinking(True)
                try:
                    reply = await asyncio.to_thread(llm.chat_reply, goal)
                except Exception as exc:
                    reply = f"I’m here. Ollama conversation reply failed: {exc}"
                finally:
                    thinking(False)
                print(render_chat_reply(reply), flush=True)
                continue
            print(render_task(goal), flush=True)
            loop = asyncio.get_running_loop()
            current_task = asyncio.current_task()
            previous_sigint = signal.getsignal(signal.SIGINT)

            def abort_goal(_signum, _frame):
                if current_task is not None:
                    loop.call_soon_threadsafe(current_task.cancel)

            signal.signal(signal.SIGINT, abort_goal)
            try:
                result = await run_turn(
                    goal,
                    browser,
                    llm,
                    on_confirm=_on_confirm,
                    on_ask=_on_ask,
                    on_step=_on_step,
                    on_thinking=thinking,
                    print_steps=False,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                thinking(False)
                print(_c("1;33", "\n  ⚠ ABORTED — task cancelled by Ctrl+C; browser remains open."))
                continue
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
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
        print("  Fix the problem above, then start the agent again with: python chat.py")
        print("  Or use: .venv-linux\\bin\\python3 chat.py (recommended)")


if __name__ == "__main__":
    main()
