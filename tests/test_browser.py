import asyncio
import pathlib
import sys
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from playwright.async_api import Error as PlaywrightError, Page

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import browser as browser_mod
from browser import BrowserActionError, BrowserWrapper

FIXTURE_URL = (pathlib.Path(__file__).parent / "fixtures" / "site.html").resolve().as_uri()


@asynccontextmanager
async def open_browser():
    wrapper = BrowserWrapper()
    await wrapper.start()
    try:
        yield wrapper
    finally:
        await wrapper.close()


def index_of(elements: list[dict], **criteria) -> int:
    for element in elements:
        if all(element.get(key) == value for key, value in criteria.items()):
            return element["index"]
    raise AssertionError(f"No element matches {criteria!r}")


def test_navigate_and_extract():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            info = await browser.get_page_info()
            assert "Fixture" in info["title"]
            assert "Fixture" in info["text"]
            assert len(info["text"]) <= 2000
            assert info["has_password"] is True

    asyncio.run(scenario())


def test_element_map_indices_and_roles():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            assert [element["index"] for element in elements] == list(range(len(elements)))
            assert any(
                element["role"] == "button" and element["text"] == "Go"
                for element in elements
            )
            assert all(element["selector"] for element in elements)

    asyncio.run(scenario())


def test_click_updates_dom():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            await browser.click(index_of(elements, role="button", text="Go"))
            info = await browser.get_page_info()
            assert "Clicked" in info["text"]

    asyncio.run(scenario())


def test_type_text_fills_input():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            name_index = index_of(elements, selector="#name")
            await browser.type_text(name_index, "Ada")
            assert await browser.get_value(name_index) == "Ada"

    asyncio.run(scenario())


def test_scroll_moves_position():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            position = await browser.get_scroll_position()
            assert position["y"] == 0
            await browser.scroll("down")
            position = await browser.get_scroll_position()
            assert position["y"] > 0
            assert position["y"] <= position["max"]

    asyncio.run(scenario())


def test_unknown_index_raises_action_error():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            await browser.get_interactive_elements()
            with pytest.raises(BrowserActionError):
                await browser.click(999)

    asyncio.run(scenario())


def test_stale_element_raises_action_error():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            go_index = index_of(elements, role="button", text="Go")
            await browser.click(go_index)
            await browser.navigate_to("about:blank")
            with pytest.raises(BrowserActionError):
                await browser.click(go_index)

    asyncio.run(scenario())


def test_close_is_idempotent():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        await browser.close()
        await browser.close()

    asyncio.run(scenario())


def test_observe_wraps_playwright_error():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            with patch.object(
                Page,
                "evaluate",
                side_effect=PlaywrightError("Execution context was destroyed"),
            ):
                with pytest.raises(BrowserActionError):
                    await browser.get_interactive_elements()

    asyncio.run(scenario())


def test_persistent_profile_survives_restart(tmp_path, monkeypatch):
    import functools
    import http.server
    import threading

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    handler = functools.partial(
        QuietHandler, directory=str(pathlib.Path(__file__).parent / "fixtures")
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(browser_mod, "USER_DATA_DIR", str(tmp_path / "profile"))

    async def scenario():
        first = BrowserWrapper()
        await first.start()
        try:
            await first.navigate_to(f"{origin}/site.html")
            await first._page.context.add_cookies(
                [
                    {
                        "name": "sess",
                        "value": "abc",
                        "url": origin,
                        "expires": 4102444800,
                    }
                ]
            )
        finally:
            await first.close()

        second = BrowserWrapper()
        await second.start()
        try:
            await second.navigate_to(f"{origin}/site.html")
            cookies = await second._page.context.cookies(origin)
        finally:
            await second.close()
        assert any(
            c["name"] == "sess" and c["value"] == "abc" for c in cookies
        ), "cookie from previous session must persist in the profile"

    try:
        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_constructor_resolves_headless_from_config_at_init(monkeypatch):
    monkeypatch.setattr(browser_mod, "HEADLESS", True)
    assert BrowserWrapper().headless is True
    monkeypatch.setattr(browser_mod, "HEADLESS", False)
    assert BrowserWrapper().headless is False


def test_constructor_resolves_slow_mo_from_config_at_init(monkeypatch):
    monkeypatch.setattr(browser_mod, "SLOW_MO_MS", 100)
    assert BrowserWrapper().slow_mo == 100


def test_constructor_explicit_args_win_over_config(monkeypatch):
    monkeypatch.setattr(browser_mod, "HEADLESS", True)
    monkeypatch.setattr(browser_mod, "SLOW_MO_MS", 100)
    wrapper = BrowserWrapper(headless=False, slow_mo=0)
    assert wrapper.headless is False
    assert wrapper.slow_mo == 0
