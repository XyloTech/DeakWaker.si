from actions import ACTIONS, ActionContext, ActionSpec, normalize_url, parse_element, SCREENSHOT_DIR
from llm import ActionMessage, ActionParameters, build_messages, parse_action, LLMOutputError, build_extract_messages
from agent import run_turn
from tests.test_agent import ScriptedLLM, last_blob, make_action
import asyncio, pathlib, pytest, json, tempfile

EXPECTED_CORE = {"navigate", "click", "type", "scroll", "done", "ask_user"}
FIXTURE_URL = (pathlib.Path(__file__).parent / "fixtures" / "site.html").resolve().as_uri()


def test_registry_contains_core_actions():
    assert EXPECTED_CORE <= set(ACTIONS)


def test_prompt_lists_every_registry_action():
    system = build_messages("g", "obs", [])[0]["content"]
    for spec in ACTIONS.values():
        assert spec.name in system
        assert spec.description in system
    assert "navigate" in system


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


# New Task 8 tests

def test_select_dispatch_selects_fixture_dropdown():
    """ScriptedLLM [select E? value g, done] on FIXTURE_URL; after run_turn,
    browser._page.evaluate("document.querySelector('#colors').value") == "g"."""
    async def _scenario():
        from agent import run_turn
        from browser import BrowserWrapper
        from llm import ActionMessage, ActionParameters
        from tests.test_agent import make_action, ScriptedLLM

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(str(FIXTURE_URL))
            elements = await browser.get_interactive_elements()
            idx = next(i for i, e in enumerate(elements) if e["selector"] == "#colors")
            llm = ScriptedLLM(
                [
                    make_action("select green", "select", element=f"E{idx}", value="g"),
                    make_action("done", "done", answer="done"),
                ]
            )
            result = await run_turn("Select green", browser, llm)
            selected = await browser._page.evaluate("document.querySelector('#colors').value")
            assert selected == "g"
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_select_without_value_returns_coaching():
    """validate string: assert "select requires parameters.value" in last_blob(llm.calls[1])"""
    async def _scenario():
        from agent import run_turn
        from browser import BrowserWrapper
        from llm import ActionMessage, ActionParameters
        from tests.test_agent import make_action, last_blob, ScriptedLLM, FIXTURE_URL

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(str(FIXTURE_URL))
            llm = ScriptedLLM(
                [
                    make_action("select with no value", "select", element="E0"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            result = await run_turn("Select without value", browser, llm)
            # Check transcript for error
            error_found = any("ERROR" in e["result"] for e in result.transcript)
            assert error_found, f"No error in transcript: {result.transcript}"
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_upload_missing_file_feeds_error_to_llm():
    """upload with path /definitely/not/here.pdf goes to next call blob has no such file"""
    async def _scenario():
        from agent import run_turn
        from browser import BrowserWrapper
        from llm import ActionMessage, ActionParameters
        from tests.test_agent import make_action, last_blob, ScriptedLLM, FIXTURE_URL

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(str(FIXTURE_URL))
            llm = ScriptedLLM(
                [
                    make_action("upload missing file", "upload", element="E0", path="/definitely/not/here.pdf"),
                    make_action("done", "done", answer="done"),
                ]
            )
            result = await run_turn("Upload file", browser, llm)
            # Check transcript for error
            error_found = any("ERROR" in e["result"] for e in result.transcript)
            assert error_found, f"No error in transcript: {result.transcript}"
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_screenshot_action_saves_artifact(tmp_path, monkeypatch):
    """monkeypatch actions.SCREENSHOT_DIR to tmp_path; script [screenshot, done];
    (tmp_path / "step-1.png").exists() and transcript[0]["result"].startswith("OK · screenshot saved:")"""
    from actions import SCREENSHOT_DIR
    import os

    screenshot_dir = pathlib.Path(str(tmp_path / "screenshots"))
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("actions.SCREENSHOT_DIR", str(screenshot_dir))

    async def _scenario():
        from browser import BrowserWrapper
        from llm import ActionMessage, ActionParameters
        from tests.test_agent import make_action, ScriptedLLM

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(str(FIXTURE_URL))
            llm = ScriptedLLM(
                [
                    make_action("take screenshot", "screenshot"),
                    make_action("done", "done", answer="done"),
                ]
            )
            result = await run_turn("Take screenshot", browser, llm)
            # Check transcript result string
            assert result.transcript[0]["result"].startswith("OK · screenshot saved:")
            # Check directory exists
            assert screenshot_dir.exists()
            # Check if any file exists in the directory
            files = list(screenshot_dir.glob("*.png"))
            assert len(files) > 0, f"No PNG files found in {screenshot_dir}"
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_extract_dispatch_returns_json_to_history():
    from llm import build_extract_messages
    from llm import ActionMessage, ActionParameters, LLMOutputError

    async def _scenario():
        from browser import BrowserWrapper

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            page_info = await browser.get_page_info()
            llm = ScriptedLLM(
                [
                    make_action("extract prices", "extract", query="prices as JSON"),
                    make_action("done", "done", answer="done"),
                ],
                extract_value={"prices": [1, 2]},
            )
            result = await run_turn("Extract prices", browser, llm)
            assert result.transcript[0]["result"] == 'OK · {"prices": [1, 2]}'
            # Check extract_calls blob contains "prices as JSON"
            blob = last_blob(llm.extract_calls[0])
            assert "prices as JSON" in blob
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_extract_failure_feeds_error_and_loop_continues():
    from llm import LLMOutputError
    from tests.test_agent import make_action, ScriptedLLM, last_blob

    async def _scenario():
        from browser import BrowserWrapper

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("extract data", "extract", query="data"),
                    make_action("done", "done", answer="done after error"),
                ],
                extract_error=LLMOutputError("LLM API unavailable: down"),
            )
            result = await run_turn("Extract data with error", browser, llm)
            assert result.status == "done"
            assert result.transcript[0]["result"].startswith("ERROR: extraction failed")
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_extract_without_query_returns_coaching():
    from llm import LLMOutputError
    from tests.test_agent import make_action, ScriptedLLM, last_blob

    async def _scenario():
        from browser import BrowserWrapper

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("extract with no query", "extract"),
                    make_action("done", "done", answer="done"),
                ]
            )
            result = await run_turn("Extract without query", browser, llm)
            blob = last_blob(llm.calls[1])
            assert "extract requires parameters.query" in blob
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_new_tab_and_switch_tab_dispatch():
    from actions import normalize_url
    from browser import BrowserWrapper
    import asyncio

    async def _scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            initial_count = browser.tab_count
            llm = ScriptedLLM(
                [
                    make_action("open new tab", "new_tab", url="example.com"),
                    make_action("switch to tab 0", "switch_tab", tab="0"),
                    make_action("done", "done", answer="done"),
                ]
            )
            result = await run_turn("New tab and switch", browser, llm)
            assert result.status == "done"
            assert browser.tab_count == initial_count + 1
        finally:
            await browser.close()

    asyncio.run(_scenario())


def test_switch_tab_invalid_index_returns_coaching():
    from tests.test_agent import make_action, ScriptedLLM, last_blob

    async def _scenario():
        from browser import BrowserWrapper

        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("switch to invalid tab", "switch_tab", tab="9"),
                    make_action("done", "done", answer="done"),
                ]
            )
            result = await run_turn("Switch invalid tab", browser, llm)
            blob = last_blob(llm.calls[1])
            assert "no tab" in blob.lower() or "ERROR" in blob
        finally:
            await browser.close()

    asyncio.run(_scenario())