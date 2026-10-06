import asyncio
import pathlib
import sys
import threading

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import TurnResult, format_step_line, run_turn
from browser import BrowserWrapper
from llm import ActionMessage, ActionParameters, LLMOutputError, Verdict

FIXTURE_URL = (pathlib.Path(__file__).parent / "fixtures" / "site.html").resolve().as_uri()


def make_action(thought: str, action: str, **params) -> ActionMessage:
    return ActionMessage(
        thought=thought, action=action, parameters=ActionParameters(**params)
    )


class ScriptedLLM:
    def __init__(self, responses: list[ActionMessage], verdicts: list[Verdict] | None = None,
                 extract_value: object | None = None, extract_error: LLMOutputError | None = None):
        self._responses = list(responses)
        self._verdicts = list(verdicts) if verdicts is not None else None
        self.extract_calls: list[list[dict]] = []
        self.calls: list[list[dict]] = []
        self.verify_calls: list[list[dict]] = []
        self._extract_value = extract_value
        self._extract_error = extract_error

    def next_action(self, messages: list[dict]) -> ActionMessage:
        self.calls.append(messages)
        return self._responses.pop(0)

    def verify(self, messages: list[dict]) -> Verdict:
        self.verify_calls.append(messages)
        if self._verdicts is None:
            return Verdict(complete=True, reason="scripted: goal assumed complete")
        return self._verdicts.pop(0)

    def extract(self, messages: list[dict]) -> object:
        self.extract_calls.append(messages)
        if self._extract_error is not None:
            raise self._extract_error
        return self._extract_value or {"ok": True}


def last_blob(call_messages: list[dict]) -> str:
    return "\n".join(message["content"] for message in call_messages)


def index_of(elements: list[dict], **criteria) -> int:
    for element in elements:
        if all(element.get(key) == value for key, value in criteria.items()):
            return element["index"]
    raise AssertionError(f"No element matches {criteria!r}")


def test_loop_reaches_done_in_three_steps():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            go = index_of(elements, role="button", text="Go")
            name = index_of(elements, selector="#name")
            llm = ScriptedLLM(
                [
                    make_action("click the Go button", "click", element=f"E{go}"),
                    make_action("type Ada into the name field", "type", element=f"E{name}", text="Ada"),
                    make_action("goal achieved", "done", answer="Ada typed"),
                ]
            )

            result = await run_turn("Type Ada into the name field", browser, llm)

            assert isinstance(result, TurnResult)
            assert result.status == "done"
            assert result.answer == "Ada typed"
            assert result.steps_used == 3
            assert len(result.transcript) == 4
            assert [entry["action"] for entry in result.transcript] == [
                "click",
                "type",
                "verify",
                "done",
            ]
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_action_error_returned_to_llm():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("click something that does not exist", "click", element="E42"),
                    make_action("give up and finish", "done", answer="done after error"),
                ]
            )

            result = await run_turn("Click around", browser, llm)

            assert result.status == "done"
            assert result.answer == "done after error"
            assert "ERROR" in last_blob(llm.calls[1])
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_click_stale_index_returns_error_to_llm():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("leave the fixture page", "navigate", url="about:blank"),
                    make_action("click E0 from the old observation", "click", element="E0"),
                    make_action("nothing left to do", "done", answer="done"),
                ]
            )

            result = await run_turn("Navigate away then click", browser, llm)

            assert result.status == "done"
            third_call = last_blob(llm.calls[2])
            assert "ERROR" in third_call
            assert "element" in third_call
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_max_steps_returns_partial_status():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("scroll", "scroll", direction="down"),
                    make_action("scroll again", "scroll", direction="down"),
                ]
            )

            result = await run_turn("Never finish", browser, llm, max_steps=2)

            assert result.status == "max_steps"
            assert result.steps_used == 2
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_ask_user_suspends_and_resumes_with_reply():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            questions: list[str] = []
            llm = ScriptedLLM(
                [
                    make_action("need input", "ask_user", question="Which color?"),
                    make_action("scroll while waiting", "scroll", direction="down"),
                    make_action("all set", "done", answer="blue"),
                ]
            )

            def on_ask(question: str) -> str:
                questions.append(question)
                return "blue"

            result = await run_turn("Pick a color", browser, llm, on_ask=on_ask)

            assert result.status == "done"
            assert questions == ["Which color?"]
            assert "blue" in last_blob(llm.calls[2])
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_history_evicts_oldest_steps():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            replies = iter(["apple", "banana"])
            llm = ScriptedLLM(
                [
                    make_action("ask one", "ask_user", question="First?"),
                    make_action("ask two", "ask_user", question="Second?"),
                    make_action("scroll", "scroll", direction="down"),
                    make_action("finish", "done", answer="done"),
                ]
            )

            result = await run_turn(
                "Chat then scroll", browser, llm, on_ask=lambda q: next(replies), max_history=2
            )

            assert result.status == "done"
            final_blob = last_blob(llm.calls[3])
            assert "apple" not in final_blob
            assert "banana" in final_blob
            assert "Result: OK" in final_blob
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_format_step_line_truncates_thought():
    line = format_step_line(3, 15, "click", "x" * 200)
    assert line.startswith("Step 3/15 | click | ")
    assert "\n" not in line
    assert len(line) < 120


def test_element_changed_between_observation_and_dispatch():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            go = index_of(elements, role="button", text="Go")

            real_observe = browser.get_interactive_elements
            observe_calls = {"n": 0}

            async def re_rendering_observe():
                observe_calls["n"] += 1
                fresh = await real_observe()
                if observe_calls["n"] == 2:
                    fresh = [dict(entry) for entry in fresh]
                    fresh[0]["text"] = "Submit now"
                    browser._elements = fresh
                return fresh

            browser.get_interactive_elements = re_rendering_observe

            llm = ScriptedLLM(
                [
                    make_action("click Go", "click", element=f"E{go}"),
                    make_action("finished", "done", answer="done"),
                ]
            )

            result = await run_turn("Click the Go button", browser, llm)

            assert result.status == "done"
            second_call = last_blob(llm.calls[1])
            assert "ERROR" in second_call
            assert "changed" in second_call
        finally:
            await browser.close()

    asyncio.run(scenario())

def test_step_budget_sent_to_llm():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("scroll", "scroll", direction="down"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            await run_turn("Scroll then finish", browser, llm)
            assert "Steps used: 2/15." in last_blob(llm.calls[1])
        finally:
            await browser.close()

    asyncio.run(scenario())

def test_normalize_url_prepends_https_to_bare_hosts():
    from actions import normalize_url
    assert normalize_url("youtube.com") == "https://youtube.com"
    assert normalize_url("localhost:3000") == "https://localhost:3000"
    assert normalize_url("www.example.com/path?q=1") == "https://www.example.com/path?q=1"


def test_normalize_url_keeps_existing_schemes():
    from actions import normalize_url
    assert normalize_url("https://x.com") == "https://x.com"
    assert normalize_url("http://x.com") == "http://x.com"
    assert normalize_url("file:///tmp/a.html") == "file:///tmp/a.html"


def test_navigate_normalizes_url_before_dispatch():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            recorded = []

            async def fake_navigate(url):
                recorded.append(url)

            browser.navigate_to = fake_navigate
            llm = ScriptedLLM(
                [
                    make_action("go to example", "navigate", url="example.com"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            result = await run_turn("Open example.com", browser, llm)
            assert result.status == "done"
            assert recorded == ["https://example.com"]
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_invalid_element_error_coaches_llm_with_element_refs():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("click nothing", "click"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            await run_turn("Click around", browser, llm)
            blob = last_blob(llm.calls[1])
            assert "Result: ERROR: invalid element reference" in blob
            assert "you can only reference elements from the observation: [E0]" in blob
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_navigate_missing_url_error_includes_example():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("navigate with no url", "navigate"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            await run_turn("Go somewhere", browser, llm)
            assert "https://example.com" in last_blob(llm.calls[1])
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_llm_next_action_runs_off_event_loop():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            seen_threads = []

            class RecordingLLM:
                def next_action(self, messages):
                    seen_threads.append(threading.get_ident())
                    return make_action("finish", "done", answer="done")

                def verify(self, messages):
                    return Verdict(complete=True, reason="ok")

            await run_turn("Finish", browser, RecordingLLM())
            assert seen_threads and seen_threads[0] != threading.get_ident()
        finally:
            await browser.close()

    asyncio.run(scenario())

def test_on_step_receives_entry_page_info_and_elements():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            go = index_of(elements, role="button", text="Go")
            llm = ScriptedLLM(
                [
                    make_action("click Go", "click", element=f"E{go}"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            calls = []
            result = await run_turn(
                "Click Go",
                browser,
                llm,
                on_step=lambda entry, page, els: calls.append((entry, page, els)),
            )
            assert result.status == "done"
            assert [call[0]["step"] for call in calls] == [1, 2, 2]
            assert calls[0][1]["url"] == FIXTURE_URL
            assert calls[0][2] and calls[0][2][0]["index"] == 0
        finally:
            await browser.close()

    asyncio.run(scenario())

def test_on_confirm_and_on_ask_may_be_async():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            submit = index_of(elements, role="button", text="Submit")
            llm = ScriptedLLM(
                [
                    make_action("click submit", "click", element=f"E{submit}"),
                    make_action("need input", "ask_user", question="Which?"),
                    make_action("finish", "done", answer="done"),
                ]
            )

            async def decline(prompt):
                return False

            async def reply(question):
                return "hello"

            result = await run_turn(
                "Submit then ask", browser, llm, on_confirm=decline, on_ask=reply
            )
            assert result.status == "done"
            assert "USER DECLINED" in result.transcript[0]["result"]
            assert result.transcript[1]["result"] == "hello"
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_done_emits_verify_step_before_done():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [make_action("goal achieved", "done", answer="done")]
            )

            result = await run_turn("Play a yo yo video", browser, llm)

            assert result.status == "done"
            assert [entry["action"] for entry in result.transcript] == [
                "verify",
                "done",
            ]
            verify_entry = result.transcript[0]
            assert verify_entry["step"] == 1
            assert verify_entry["result"].startswith("OK · verified")
            assert len(llm.verify_calls) == 1
            blob = last_blob(llm.verify_calls[0])
            assert "Play a yo yo video" in blob
            assert FIXTURE_URL in blob
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_verifier_rejects_done_and_agent_keeps_going():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("I think it is playing", "done", answer="playing"),
                    make_action("now it is actually playing", "done", answer="playing"),
                ],
                verdicts=[
                    Verdict(complete=False, reason="video never started playing"),
                    Verdict(complete=True, reason="video is playing"),
                ],
            )

            result = await run_turn("Play a yo yo video", browser, llm)

            assert result.status == "done"
            actions = [entry["action"] for entry in result.transcript]
            assert actions == ["verify", "verify", "done"]
            first_verify = result.transcript[0]
            assert (
                first_verify["result"]
                == "ERROR: GOAL NOT COMPLETE — video never started playing"
            )
            second_call = last_blob(llm.calls[1])
            assert "GOAL NOT COMPLETE" in second_call
            assert "video never started playing" in second_call
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_verifier_error_fails_open_and_accepts_done():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)

            class BrokenVerifierLLM(ScriptedLLM):
                def verify(self, messages):
                    raise LLMOutputError("LLM API unavailable: verifier down")

            llm = BrokenVerifierLLM(
                [make_action("finish", "done", answer="done")]
            )

            result = await run_turn("Anything", browser, llm)

            assert result.status == "done"
            assert result.answer == "done"
            assert [entry["action"] for entry in result.transcript] == [
                "verify",
                "done",
            ]
            assert "verifier unavailable" in result.transcript[0]["result"]
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_verifier_runs_off_event_loop():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            seen: list[int] = []

            class ThreadingVerifyLLM(ScriptedLLM):
                def verify(self, messages):
                    seen.append(threading.get_ident())
                    return Verdict(complete=True, reason="ok")

            llm = ThreadingVerifyLLM(
                [make_action("finish", "done", answer="done")]
            )

            result = await run_turn("Finish", browser, llm)

            assert result.status == "done"
            assert seen and seen[0] != threading.get_ident()
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_step_error_appends_recent_browser_diagnostics():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            cleared: list[bool] = []
            browser.get_diagnostics = lambda: [
                "console.error: kapow",
                "requestfailed: GET x.png — net::ERR_NAME_NOT_RESOLVED",
            ]
            browser.clear_diagnostics = lambda: cleared.append(True)
            llm = ScriptedLLM(
                [
                    make_action("click something missing", "click", element="E42"),
                    make_action("finish", "done", answer="done"),
                ]
            )

            result = await run_turn("Click around", browser, llm)

            assert result.status == "done"
            blob = last_blob(llm.calls[1])
            assert "Recent browser errors:" in blob
            assert "kapow" in blob
            assert "net::ERR_NAME_NOT_RESOLVED" in blob
            assert cleared
        finally:
            await browser.close()

    asyncio.run(scenario())


def test_step_error_without_diagnostics_has_no_suffix():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("click something missing", "click", element="E42"),
                    make_action("finish", "done", answer="done"),
                ]
            )

            result = await run_turn("Click around", browser, llm)

            assert result.status == "done"
            blob = last_blob(llm.calls[1])
            assert "ERROR" in blob
            assert "Recent browser errors:" not in blob
        finally:
            await browser.close()

    asyncio.run(scenario())
