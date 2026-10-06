import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llm import ActionMessage, LLMClient, LLMOutputError, parse_action

VALID_PAYLOAD = {
    "thought": "x",
    "action": "navigate",
    "parameters": {"url": "https://example.com"},
}


def test_parse_action_valid_json():
    raw = json.dumps(VALID_PAYLOAD)
    result = parse_action(raw)
    assert isinstance(result, ActionMessage)
    assert result.parameters.url == "https://example.com"


def test_parse_action_accepts_fenced_json():
    raw = "```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
    result = parse_action(raw)
    assert isinstance(result, ActionMessage)
    assert result.parameters.url == "https://example.com"


def test_parse_action_accepts_trailing_prose():
    raw = json.dumps(VALID_PAYLOAD) + "\nDone!"
    result = parse_action(raw)
    assert isinstance(result, ActionMessage)
    assert result.parameters.url == "https://example.com"


def test_parse_action_rejects_truncated_json():
    with pytest.raises(LLMOutputError):
        parse_action('{"thought":"x","action":"cli')


def test_parse_action_rejects_bad_enum():
    raw = json.dumps({"thought": "x", "action": "teleport"})
    with pytest.raises(LLMOutputError):
        parse_action(raw)


def test_parse_action_rejects_missing_thought():
    raw = json.dumps({"action": "click"})
    with pytest.raises(LLMOutputError):
        parse_action(raw)


def test_next_action_success_first_try():
    calls: list[list[dict]] = []

    def complete_fn(messages: list[dict]) -> str:
        calls.append(messages)
        return json.dumps(VALID_PAYLOAD)

    client = LLMClient(complete_fn=complete_fn)
    result = client.next_action([{"role": "system", "content": "sys"}])
    assert isinstance(result, ActionMessage)
    assert len(calls) == 1


def test_next_action_repairs_then_raises():
    calls: list[list[dict]] = []

    def complete_fn(messages: list[dict]) -> str:
        calls.append(messages)
        return "sorry here's prose"

    client = LLMClient(complete_fn=complete_fn)
    with pytest.raises(LLMOutputError):
        client.next_action([{"role": "system", "content": "sys"}])
    assert len(calls) == 2
    assert len(calls[1]) == len(calls[0]) + 1
    assert calls[1][-1]["role"] == "user"
    assert calls[1][-1]["content"].startswith("Your last output was invalid:")


def test_next_action_repairs_successfully():
    calls: list[list[dict]] = []

    def complete_fn(messages: list[dict]) -> str:
        calls.append(messages)
        if len(calls) == 1:
            return '{"thought":"x","action":"cli'
        return json.dumps(VALID_PAYLOAD)

    client = LLMClient(complete_fn=complete_fn)
    result = client.next_action([{"role": "system", "content": "sys"}])
    assert isinstance(result, ActionMessage)
    assert len(calls) == 2


def test_parameters_default_to_none():
    raw = json.dumps({"thought": "t", "action": "scroll"})
    result = parse_action(raw)
    assert isinstance(result, ActionMessage)
    assert result.parameters.element is None
    assert result.parameters.direction is None
