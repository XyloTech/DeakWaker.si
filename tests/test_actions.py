from actions import ACTIONS, ActionContext, ActionSpec, normalize_url, parse_element
from llm import build_messages, parse_action, LLMOutputError
import pytest, json

EXPECTED_CORE = {"navigate", "click", "type", "scroll", "done", "ask_user"}


def test_registry_contains_core_actions():
    assert EXPECTED_CORE <= set(ACTIONS)


def test_prompt_lists_every_registry_action():
    system = build_messages("g", "obs", [])[0]["content"]
    for spec in ACTIONS.values():
        assert spec.name in system
        assert spec.description in system
    assert "navigate" in system and "switch_tab" not in system


def test_prompt_keeps_static_rule_sentences():
    system = build_messages("g", "obs", [])[0]["content"]
    for sentence in (
        "address bar",
        "Never retry an approach",
        "must be absolute",
        "Diagnose before retrying",
        "one-time codes",
        "Verify the outcome against the goal",
        "done only when the goal is verifiably achieved",
    ):
        assert sentence in system


def test_parse_action_rejects_action_not_in_registry():
    raw = json.dumps({"thought": "x", "action": "teleport"})
    with pytest.raises(LLMOutputError):
        parse_action(raw)


def test_normalize_url_lives_in_actions():
    assert normalize_url("youtube.com") == "https://youtube.com"