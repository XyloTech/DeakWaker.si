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
