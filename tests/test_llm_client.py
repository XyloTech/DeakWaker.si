import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import llm
from llm import (
    ActionMessage,
    LLMClient,
    LLMOutputError,
    build_messages,
    render_observation,
)

VALID_PAYLOAD = {
    "thought": "x",
    "action": "navigate",
    "parameters": {"url": "https://example.com"},
}

PAGE_INFO = {
    "url": "https://example.com/",
    "title": "Example Domain",
    "text": "Example Domain\nThis domain is for use in examples.",
    "has_password": False,
}

ELEMENTS = [
    {"index": 0, "role": "button", "text": "Go", "selector": "button"},
    {"index": 7, "role": "link", "text": "More info", "selector": "a"},
]


def test_render_observation_formats_element_lines():
    output = render_observation(PAGE_INFO, ELEMENTS)
    assert '[E0] button "Go"' in output
    assert '[E7] link "More info"' in output
    assert "https://example.com/" in output
    assert "Example Domain" in output


def test_build_messages_contains_goal_and_observation():
    goal = "Click the Go button"
    observation = 'URL: https://example.com/\nElements:\n[E0] button "Go"'
    history = [
        {"thought": "Scroll down to find the button", "action": "scroll", "result": "OK"},
        {
            "thought": "Click the Go button now",
            "action": "click",
            "result": "ERROR: stale element",
        },
    ]

    messages = build_messages(goal, observation, history)

    assert messages[0]["role"] == "system"
    assert goal in messages[0]["content"]
    assert any(observation in message["content"] for message in messages)
    blob = "\n".join(message["content"] for message in messages)
    for step in history:
        assert step["thought"] in blob
        assert step["action"] in blob
        assert step["result"] in blob
    assert messages[-1]["content"].startswith("Decide the next single action.")
    assert "Steps used" not in messages[-1]["content"]


def test_next_action_backoff_then_fails(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    calls: list[list[dict]] = []

    def complete_fn(messages: list[dict]) -> str:
        calls.append(messages)
        raise ConnectionError("rate limited")

    client = LLMClient(complete_fn=complete_fn)
    with pytest.raises(LLMOutputError) as excinfo:
        client.next_action([{"role": "system", "content": "sys"}])

    assert len(calls) == 3
    assert "LLM API unavailable" in str(excinfo.value)
    assert "rate limited" in str(excinfo.value)
    assert len(slept) == 2
    assert slept[0] <= 1.0
    assert slept[1] > slept[0]


def test_next_action_backoff_recovers(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    calls: list[list[dict]] = []

    def complete_fn(messages: list[dict]) -> str:
        calls.append(messages)
        if len(calls) < 3:
            raise ConnectionError("rate limited")
        return json.dumps(VALID_PAYLOAD)

    client = LLMClient(complete_fn=complete_fn)
    result = client.next_action([{"role": "system", "content": "sys"}])

    assert isinstance(result, ActionMessage)
    assert result.action == "navigate"
    assert len(calls) == 3
    assert slept and slept[0] <= 1.0


def test_default_client_completes_via_ollama(monkeypatch):
    captured: dict = {}

    def fake_complete(host: str, model: str, messages: list[dict]) -> str:
        captured.update(host=host, model=model, messages=messages)
        return json.dumps(VALID_PAYLOAD)

    monkeypatch.setattr(llm, "_ollama_complete", fake_complete)

    client = LLMClient()
    result = client.next_action([{"role": "system", "content": "sys"}])

    assert isinstance(result, ActionMessage)
    assert result.action == "navigate"
    assert captured["host"] == llm.OLLAMA_HOST
    assert captured["model"] == llm.MODEL
    assert captured["messages"][0] == {"role": "system", "content": "sys"}


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("LIVE_LLM"), reason="LIVE_LLM not set")
def test_next_action_live_smoke():
    client = LLMClient()
    messages = build_messages(
        goal="Navigate to https://example.com",
        observation='URL: about:blank\nTitle: \nElements:\nVisible text:',
        history=[],
    )
    result = client.next_action(messages)
    assert isinstance(result, ActionMessage)
    assert result.action in {
        "navigate",
        "click",
        "type",
        "scroll",
        "done",
        "ask_user",
    }
    assert result.thought

def test_system_rules_forbid_browser_chrome():
    content = llm.build_messages("g", "obs", [])[0]["content"]
    assert "address bar" in content
    assert "does not exist" in content


def test_system_rules_forbid_repeat_and_require_absolute_urls():
    content = llm.build_messages("g", "obs", [])[0]["content"]
    assert "Never retry an approach" in content
    assert "must be absolute" in content


def test_build_messages_includes_step_budget():
    messages = llm.build_messages("g", "obs", [], steps_used=3, max_steps=15)
    assert messages[-1]["content"] == "Decide the next single action. Steps used: 3/15."


def test_decide_prompt_flags_repeated_failure():
    history = [
        {"thought": "", "action": "navigate", "result": "ERROR: invalid url"},
        {"thought": "", "action": "navigate", "result": "ERROR: invalid url"},
    ]
    prompt = llm.decide_prompt(4, 15, history)
    assert "repeated a failed approach" in prompt


def test_decide_prompt_silent_when_success_repeats():
    history = [
        {"thought": "", "action": "scroll", "result": "OK"},
        {"thought": "", "action": "scroll", "result": "OK"},
    ]
    prompt = llm.decide_prompt(4, 15, history)
    assert "repeated a failed approach" not in prompt


def test_decide_prompt_silent_on_single_failure():
    history = [{"thought": "", "action": "navigate", "result": "ERROR: x"}]
    prompt = llm.decide_prompt(4, 15, history)
    assert "repeated a failed approach" not in prompt

def test_chat_kwargs_disables_thinking_by_default(monkeypatch):
    monkeypatch.setattr(llm, "THINKING", False)
    kwargs = llm._chat_kwargs("qwen3:latest", [{"role": "user", "content": "hi"}])
    assert kwargs["think"] is False
    assert kwargs["format"] == "json"


def test_chat_kwargs_keeps_thinking_when_enabled(monkeypatch):
    monkeypatch.setattr(llm, "THINKING", True)
    kwargs = llm._chat_kwargs("qwen3:latest", [{"role": "user", "content": "hi"}])
    assert kwargs["think"] is True


def test_chat_kwargs_omits_think_for_truncation_control(monkeypatch):
    monkeypatch.setattr(llm, "THINKING", False)
    kwargs = llm._chat_kwargs("qwen3:latest", [{"role": "user", "content": "hi"}])
    assert "num_predict" not in kwargs


def _messages_after(result):
    history = [{"thought": "t", "action": "click", "result": result}]
    return build_messages("g", "obs", history)


def test_chat_kwargs_thinks_after_failed_step(monkeypatch):
    monkeypatch.setattr(llm, "THINKING", False)
    kwargs = llm._chat_kwargs("qwen3:latest", _messages_after("ERROR: stale element"))
    assert kwargs["think"] is True


def test_chat_kwargs_stays_fast_after_successful_step(monkeypatch):
    monkeypatch.setattr(llm, "THINKING", False)
    kwargs = llm._chat_kwargs("qwen3:latest", _messages_after("OK"))
    assert kwargs["think"] is False


def test_chat_kwargs_stays_fast_on_first_step(monkeypatch):
    monkeypatch.setattr(llm, "THINKING", False)
    kwargs = llm._chat_kwargs("qwen3:latest", build_messages("g", "obs", []))
    assert kwargs["think"] is False


def test_decide_prompt_nudges_after_failed_step():
    history = [{"thought": "", "action": "navigate", "result": "ERROR: x"}]
    prompt = llm.decide_prompt(2, 15, history)
    assert "reason carefully about why" in prompt


def test_decide_prompt_no_nudge_after_success():
    history = [{"thought": "", "action": "navigate", "result": "OK"}]
    prompt = llm.decide_prompt(2, 15, history)
    assert "reason carefully about why" not in prompt
