import asyncio
import pathlib
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import is_sensitive, run_turn
from browser import BrowserWrapper
from llm import ActionMessage, ActionParameters

FIXTURE_URL = (pathlib.Path(__file__).parent / "fixtures" / "site.html").resolve().as_uri()


def make_action(thought: str, action: str, **params) -> ActionMessage:
    return ActionMessage(
        thought=thought, action=action, parameters=ActionParameters(**params)
    )


class ScriptedLLM:
    def __init__(self, responses: list[ActionMessage]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def next_action(self, messages: list[dict]) -> ActionMessage:
        self.calls.append(messages)
        return self._responses.pop(0)


def last_blob(call_messages: list[dict]) -> str:
    return "\n".join(message["content"] for message in call_messages)


def index_of(elements: list[dict], **criteria) -> int:
    for element in elements:
        if all(element.get(key) == value for key, value in criteria.items()):
            return element["index"]
    raise AssertionError(f"No element matches {criteria!r}")


def test_submit_click_requires_confirmation_decline_feeds_llm():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            submit = index_of(elements, role="button", text="Submit")
            prompts: list[str] = []
            llm = ScriptedLLM(
                [
                    make_action("submit the form", "click", element=f"E{submit}"),
                    make_action("user said no, finish", "done", answer="declined"),
                ]
            )

            def on_confirm(prompt: str) -> bool:
                prompts.append(prompt)
                return False

            result = await run_turn("Submit the form", browser, llm, on_confirm=on_confirm)

            assert result.status == "done"
            assert len(prompts) == 1
            assert "click" in prompts[0]
            assert "Proceed?" in prompts[0]
            assert "USER DECLINED" in last_blob(llm.calls[1])
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_password_field_marks_login_sensitive():
    action = make_action("type secret", "type", element="E1", text="hunter2")
    page_info = {"has_password": True, "title": "Fixture Page", "url": "x", "text": ""}
    elements = [
        {"index": 1, "role": "textbox", "text": "Your name", "selector": "#name"}
    ]

    reason = is_sensitive(action, page_info, elements)

    assert reason is not None
    assert "login" in reason


def test_submit_button_text_is_sensitive():
    action = make_action("click submit", "click", element="E3")
    page_info = {"has_password": False, "title": "Fixture Page", "url": "x", "text": ""}
    elements = [
        {"index": 3, "role": "button", "text": "Submit", "selector": "button[type=submit]"}
    ]

    reason = is_sensitive(action, page_info, elements)

    assert reason is not None
    assert "form submission" in reason


def test_purchase_keyword_in_title_is_sensitive():
    action = make_action("click buy", "click", element="E0")
    page_info = {"has_password": False, "title": "Checkout — Pay now", "url": "x", "text": ""}
    elements = [{"index": 0, "role": "button", "text": "Continue", "selector": "#go"}]

    reason = is_sensitive(action, page_info, elements)

    assert reason is not None
    assert "purchase" in reason


def test_plain_link_click_auto_runs_without_confirmation():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            plain = index_of(elements, role="link", text="Plain Link")
            llm = ScriptedLLM(
                [
                    make_action("follow the link", "click", element=f"E{plain}"),
                    make_action("finished", "done", answer="followed"),
                ]
            )
            on_confirm = Mock(side_effect=AssertionError("must not confirm"))

            result = await run_turn("Follow the plain link", browser, llm, on_confirm=on_confirm)

            assert result.status == "done"
            on_confirm.assert_not_called()
            assert result.transcript[0]["result"] == "OK"
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_navigate_never_sensitive():
    action = make_action("go", "navigate", url="https://example.com")
    page_info = {"has_password": True, "title": "Checkout — Pay now", "url": "x", "text": ""}
    elements = [{"index": 0, "role": "button", "text": "Submit", "selector": "#go"}]

    assert is_sensitive(action, page_info, elements) is None


def test_submit_input_type_is_sensitive():
    action = make_action("click continue", "click", element="E4")
    page_info = {"has_password": True, "title": "Fixture Page", "url": "x", "text": ""}
    elements = [
        {
            "index": 4,
            "role": "button",
            "text": "Continue",
            "selector": "#continue",
            "submit": True,
        }
    ]

    reason = is_sensitive(action, page_info, elements)

    assert reason is not None
    assert "form submission" in reason


def test_button_label_with_submit_keyword_is_sensitive():
    action = make_action("click sign in", "click", element="E0")
    page_info = {"has_password": False, "title": "Fixture Page", "url": "x", "text": ""}
    elements = [
        {"index": 0, "role": "button", "text": "Sign in with Google", "selector": "#g"}
    ]

    reason = is_sensitive(action, page_info, elements)

    assert reason is not None
    assert "form submission" in reason


def test_submit_input_click_requires_confirmation():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            cont = index_of(elements, selector="#continue")
            prompts: list[str] = []
            llm = ScriptedLLM(
                [
                    make_action("click the continue submit input", "click", element=f"E{cont}"),
                    make_action("finish", "done", answer="done"),
                ]
            )

            def on_confirm(prompt: str) -> bool:
                prompts.append(prompt)
                return False

            result = await run_turn("Continue", browser, llm, on_confirm=on_confirm)

            assert len(prompts) == 1
            assert result.transcript[0]["result"] == "ERROR: USER DECLINED this action"
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_sensitive_without_callback_declines():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            submit = index_of(elements, role="button", text="Submit")
            llm = ScriptedLLM(
                [
                    make_action("submit", "click", element=f"E{submit}"),
                    make_action("finish", "done", answer="done"),
                ]
            )

            result = await run_turn("Submit", browser, llm, on_confirm=None)

            assert result.status == "done"
            assert result.transcript[0]["result"] == "ERROR: USER DECLINED this action"
        finally:
            await browser.close()

    asyncio.run(scenario())
