import io
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chat
from chat import SessionLogger

TRANSCRIPT = [
    {"step": 1, "observation": "o1", "thought": "t1", "action": "click", "result": "OK"},
    {"step": 2, "observation": "o2", "thought": "t2", "action": "done", "result": "OK"},
]


def test_session_logger_writes_jsonl_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = SessionLogger()

    path = logger.append("go to example", TRANSCRIPT)

    assert path.exists()
    assert path.parent.name == "logs"
    assert path.name.startswith("session-") and path.name.endswith(".jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["goal"] == "go to example"
    assert records[0]["step"] == 1
    assert records[1]["step"] == 2


def test_session_logger_creates_directory(tmp_path):
    log_dir = tmp_path / "custom_logs"
    logger = SessionLogger(str(log_dir))

    path = logger.append("some goal", TRANSCRIPT)

    assert log_dir.is_dir()
    assert path.exists()
    assert path.parent == log_dir


def test_main_configures_stdout_utf8(monkeypatch):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(sys, "stdout", stream)

    def fake_run(coroutine):
        coroutine.close()
        return None

    monkeypatch.setattr(chat.asyncio, "run", fake_run)

    chat.main()

    assert sys.stdout.encoding == "utf-8"


def test_on_confirm_prints_unicode_prompt_after_configure(monkeypatch):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr("builtins.input", lambda: "n")

    chat._configure_streams()
    approved = chat._on_confirm("⚠ About to: click Submit. Proceed? [y/N] ")
    stream.flush()

    assert approved is False
    assert "Proceed?" in buffer.getvalue().decode("utf-8")


def test_main_prints_clean_message_on_unexpected_error(monkeypatch, capsys):
    def exploding_run(coroutine):
        coroutine.close()
        raise RuntimeError("browser failed to start")

    monkeypatch.setattr(chat.asyncio, "run", exploding_run)

    chat.main()

    captured = capsys.readouterr()
    assert "browser failed to start" in captured.out
    assert "Traceback" not in captured.out


def test_wsl_windows_brave_handoff_is_selected(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "kali-linux")
    monkeypatch.setattr(chat, "BROWSER_CHANNEL", "brave")
    monkeypatch.setattr(
        chat,
        "detect_browsers",
        lambda: [{"name": "brave-windows-host"}],
    )
    assert chat._needs_windows_brave_handoff() is True


def test_handoff_runs_windows_python(monkeypatch):
    calls = []

    class Completed:
        returncode = 0

    monkeypatch.setattr(chat.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or Completed())
    chat._run_windows_brave_agent()
    assert calls[0][0] == (["cmd.exe", "/c", "py", "chat.py"],)


def _strip_ansi(text):
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_render_banner_shows_branding():
    banner = _strip_ansi(chat.render_banner())
    assert "desk.waker" in banner
    assert "product of Xylotech, developed by Harshit" in banner


def test_normalize_goal_removes_accidentally_pasted_product_brief(capsys):
    goal = chat._normalize_goal(
        "see the chrome it still in for testing "
        "Build a production-ready local browser automation agent\n"
        "The rest of the pasted request"
    )

    assert goal == "see the chrome it still in for testing"
    assert "Ignored pasted product specification" in capsys.readouterr().out


def test_normalize_goal_keeps_normal_long_goal():
    goal = "Open a page and extract this information: " + ("x" * 1000)

    assert chat._normalize_goal(goal) == goal


def test_conversational_message_detection():
    assert chat.is_conversational_message("hello") is True
    assert chat.is_conversational_message("hi noemal") is True
    assert chat.is_conversational_message("hello open youtube") is False
    assert chat.is_conversational_message("find a flight") is False
    assert chat.is_conversational_message("tell me a joke") is True


def test_render_footer_shows_branding():
    assert "Developed by Harshit" in _strip_ansi(chat.render_footer())


def test_render_step_shows_full_process():
    thought = "The page has loaded and I need to find the main heading on it " * 3
    entry = {"step": 7, "thought": thought, "action": "click", "result": "OK"}
    page = {"url": "https://example.com/", "title": "Example Domain"}
    elements = [{"index": i} for i in range(12)]
    card = _strip_ansi(chat.render_step(entry, page, elements))
    assert "7/15" in card
    assert "CLICK" in card
    joined = " ".join(card.split())
    assert thought.strip() in joined
    assert "OK" in card
    assert "https://example.com/" in card
    assert "Example Domain" in card
    assert "12 elements" in card


def test_render_step_shows_error_results():
    entry = {"step": 3, "thought": "", "action": "error", "result": "ERROR: stale element"}
    card = _strip_ansi(chat.render_step(entry, {"url": "about:blank", "title": ""}, []))
    assert "ERROR: stale element" in card
    assert "ERROR" in card


def test_render_step_colors_verify_action(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    entry = {"step": 4, "thought": "", "action": "verify", "result": "OK · verified"}
    card = chat.render_step(entry, {"url": "about:blank", "title": ""}, [])
    assert "\x1b[1;35mVERIFY\x1b[0m" in card


def test_render_status_contains_answer():
    done = _strip_ansi(chat.render_status("done", "Example Domain"))
    assert "DONE" in done
    assert "Example Domain" in done
    failed = _strip_ansi(chat.render_status("error", "Browser is gone"))
    assert "ERROR" in failed
    assert "Browser is gone" in failed


def test_render_status_supports_aborted_task():
    aborted = _strip_ansi(chat.render_status("aborted", "cancelled by Ctrl+C"))
    assert "ABORTED" in aborted
    assert "cancelled by Ctrl+C" in aborted


def test_goal_sigint_handler_cancels_active_async_task():
    import asyncio

    async def scenario():
        task = asyncio.current_task()
        called = []
        loop = asyncio.get_running_loop()

        def abort_goal(_signum, _frame):
            called.append(True)
            loop.call_soon_threadsafe(task.cancel)

        abort_goal(None, None)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert called

    asyncio.run(scenario())


def test_render_confirm_box_keeps_prompt_text():
    box = _strip_ansi(chat.render_confirm_box(
        "? About to: click Submit. Proceed? [y/N] "
    ))
    assert "Proceed?" in box


def test_run_turn_respects_print_steps_false(capsys):
    import asyncio

    from agent import run_turn
    from browser import BrowserWrapper
    from test_agent import FIXTURE_URL, ScriptedLLM, make_action

    script = [
        make_action("open", "navigate", url=FIXTURE_URL),
        make_action("finish", "done", answer="ok"),
    ]

    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            return await run_turn(
                "silent", browser, ScriptedLLM(script), print_steps=False
            )
        finally:
            await browser.close()

    result = asyncio.run(scenario())
    assert result.status == "done"
    assert "Step " not in capsys.readouterr().out


def test_run_turn_prints_steps_by_default(capsys):
    import asyncio

    from agent import run_turn
    from browser import BrowserWrapper
    from test_agent import FIXTURE_URL, ScriptedLLM, make_action

    script = [
        make_action("open", "navigate", url=FIXTURE_URL),
        make_action("finish", "done", answer="ok"),
    ]

    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            return await run_turn("loud", browser, ScriptedLLM(script))
        finally:
            await browser.close()

    result = asyncio.run(scenario())
    assert result.status == "done"
    assert "Step " in capsys.readouterr().out


def test_force_color_env_paints_even_without_tty(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert "\x1b[" in chat.render_banner()


def test_no_color_env_wins_over_tty(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert "\x1b[" not in chat.render_banner()
