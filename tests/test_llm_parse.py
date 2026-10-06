import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llm import (
    ActionMessage,
    LLMClient,
    LLMOutputError,
    Verdict,
    build_verify_messages,
    parse_action,
    parse_verdict,
)

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


def test_parse_verdict_complete_true():
    raw = json.dumps({"complete": True, "reason": "video is playing"})
    verdict = parse_verdict(raw)
    assert isinstance(verdict, Verdict)
    assert verdict.complete is True
    assert verdict.reason == "video is playing"


def test_parse_verdict_accepts_surrounding_prose():
    raw = 'Not yet.\n{"complete": false, "reason": "search results shown"}'
    verdict = parse_verdict(raw)
    assert verdict.complete is False
    assert verdict.reason == "search results shown"


def test_parse_verdict_defaults_reason_to_empty():
    verdict = parse_verdict('{"complete": false}')
    assert verdict.complete is False
    assert verdict.reason == ""


def test_parse_verdict_rejects_non_json():
    with pytest.raises(LLMOutputError):
        parse_verdict("the goal looks done to me")


def test_build_verify_messages_includes_goal_and_observation():
    observation = "URL: https://www.youtube.com/watch?v=1\nTitle: Yo Yo video"
    messages = build_verify_messages("Play a yo yo video", observation)
    assert messages[0]["role"] == "system"
    blob = "\n".join(message["content"] for message in messages)
    assert "Play a yo yo video" in blob
    assert observation in blob
    assert "complete" in blob


def test_client_verify_returns_verdict():
    def complete_fn(messages: list[dict]) -> str:
        return json.dumps({"complete": True, "reason": "goal met"})

    client = LLMClient(complete_fn=complete_fn)
    verdict = client.verify([{"role": "system", "content": "sys"}])
    assert isinstance(verdict, Verdict)
    assert verdict.complete is True
    assert verdict.reason == "goal met"


def test_client_verify_repairs_invalid_output():
    calls: list[list[dict]] = []

    def complete_fn(messages: list[dict]) -> str:
        calls.append(messages)
        if len(calls) == 1:
            return "hmm, hard to say"
        return json.dumps({"complete": False, "reason": "video not opened"})

    client = LLMClient(complete_fn=complete_fn)
    verdict = client.verify([{"role": "system", "content": "sys"}])
    assert verdict.complete is False
    assert len(calls) == 2
    assert calls[1][-1]["content"].startswith("Your last output was invalid:")
