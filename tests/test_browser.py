import asyncio
import http.server
import pathlib
import sys
import time
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from playwright.async_api import Error as PlaywrightError, Page

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import browser as browser_mod
from browser import BrowserActionError, BrowserWrapper
from llm import render_observation

FIXTURE_URL = (pathlib.Path(__file__).parent / "fixtures" / "site.html").resolve().as_uri()


@asynccontextmanager
async def open_browser():
    wrapper = BrowserWrapper()
    await wrapper.start()
    try:
        yield wrapper
    finally:
        await wrapper.close()


@asynccontextmanager
async def serve_fixtures(handler_cls=None):
    import functools, http.server, threading
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
    cls = handler_cls or Quiet
    handler = functools.partial(cls, directory=str(pathlib.Path(__file__).parent / "fixtures"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


class StatusHandler(http.server.SimpleHTTPRequestHandler):  # 401/403 paths on top of fixture files
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/auth":
            self.send_error(401)
        elif self.path == "/blocked":
            self.send_error(403)
        else:
            super().do_GET()


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


def test_start_recovers_after_window_closed():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser._page.close()
            await browser.start()
            await browser.navigate_to(FIXTURE_URL)
            info = await browser.get_page_info()
            assert "Fixture" in info["title"]
        finally:
            await browser.close()

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


def test_resolve_channel_passes_chrome_through():
    assert browser_mod._resolve_channel("chrome") == "chrome"
    assert browser_mod._resolve_channel("msedge") == "msedge"


def test_resolve_channel_maps_none_tokens_to_no_channel():
    for token in ("", "none", "off", "bundled"):
        assert browser_mod._resolve_channel(token) is None


def test_constructor_resolves_channel_from_config_at_init(monkeypatch):
    monkeypatch.setattr(browser_mod, "BROWSER_CHANNEL", "chrome")
    assert BrowserWrapper().channel == "chrome"
    monkeypatch.setattr(browser_mod, "BROWSER_CHANNEL", "none")
    assert BrowserWrapper().channel == "none"


def test_constructor_explicit_channel_wins_over_config(monkeypatch):
    monkeypatch.setattr(browser_mod, "BROWSER_CHANNEL", "chrome")
    assert BrowserWrapper(channel="msedge").channel == "msedge"


def test_launch_falls_back_to_bundled_chromium_when_channel_fails():
    async def scenario():
        wrapper = BrowserWrapper(channel="chrome")
        attempts = []

        class FakeChromium:
            async def launch_persistent_context(
                self, user_data_dir, *, headless, slow_mo, channel=None
            ):
                attempts.append(channel)
                if channel == "chrome":
                    raise PlaywrightError(
                        "Chromium distribution 'chrome' is not found"
                    )
                return "bundled-context"

        wrapper._playwright = type("FakePW", (), {"chromium": FakeChromium()})()
        context = await wrapper._open_context()
        assert context == "bundled-context"
        assert attempts == ["chrome", None]

    asyncio.run(scenario())


def test_launch_does_not_fallback_when_channel_unconfigured():
    async def scenario():
        wrapper = BrowserWrapper(channel="none")

        class FakeChromium:
            async def launch_persistent_context(
                self, user_data_dir, *, headless, slow_mo, channel=None
            ):
                raise PlaywrightError("boom")

        wrapper._playwright = type("FakePW", (), {"chromium": FakeChromium()})()
        with pytest.raises(BrowserActionError, match="Failed to start browser"):
            await wrapper._open_context()

    asyncio.run(scenario())


ERRORS_FIXTURE_URL = (
    pathlib.Path(__file__).parent / "fixtures" / "errors.html"
).resolve().as_uri()


async def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


def test_diagnostics_capture_console_page_and_network_errors():
    async def scenario():
        async with open_browser() as browser:
            assert browser.get_diagnostics() == []
            await browser.navigate_to(ERRORS_FIXTURE_URL)
            await wait_for(lambda: len(browser.get_diagnostics()) >= 3)
            await wait_for(
                lambda: any("pageerror" in d for d in browser.get_diagnostics())
            )
            diags = "\n".join(browser.get_diagnostics())
            assert "console.error" in diags
            assert "kapow" in diags
            assert "pageerror" in diags
            assert "uncaught boom" in diags
            assert "requestfailed" in diags

    asyncio.run(scenario())


def test_diagnostics_buffer_keeps_only_recent_entries():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to("about:blank")
            for i in range(browser_mod.DIAGNOSTICS_LIMIT + 5):
                await browser._page.evaluate(f"console.warn('w{i}')")
            await wait_for(
                lambda: len(browser.get_diagnostics())
                == browser_mod.DIAGNOSTICS_LIMIT
            )
            assert len(browser.get_diagnostics()) == browser_mod.DIAGNOSTICS_LIMIT

    asyncio.run(scenario())


def test_clear_diagnostics_empties_buffer():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(ERRORS_FIXTURE_URL)
            await wait_for(lambda: len(browser.get_diagnostics()) > 0)
            browser.clear_diagnostics()
            assert browser.get_diagnostics() == []

    asyncio.run(scenario())


def test_page_info_reports_http_status_200_and_404():
    async def scenario():
        async with serve_fixtures() as origin:
            async with open_browser() as browser:
                await browser.navigate_to(f"{origin}/site.html")
                assert (await browser.get_page_info())["http_status"] == 200
                await browser.navigate_to(f"{origin}/missing.html")
                assert (await browser.get_page_info())["http_status"] == 404
    asyncio.run(scenario())


def test_status_listener_tracks_in_page_navigation():
    async def scenario():
        async with serve_fixtures() as origin:
            async with open_browser() as browser:
                await browser.navigate_to(f"{origin}/site.html")
                await browser._page.evaluate("window.location = '/missing.html'")
                await wait_for(lambda: browser._http_status == 404)
    asyncio.run(scenario())


def test_new_tab_switch_and_close_tracking():
    async def scenario():
        async with open_browser() as browser:
            info = await browser.get_page_info()
            assert info["tabs"] == 1 and info["active_tab"] == 0
            await browser.new_tab("about:blank")
            assert browser.tab_count == 2
            await browser.switch_tab(0)
            assert (await browser.get_page_info())["active_tab"] == 0
            await browser._pages[1].close()
            await wait_for(lambda: browser.tab_count == 1)
    asyncio.run(scenario())


def test_diagnostics_attach_to_new_tabs():
    async def scenario():
        async with open_browser() as browser:
            await browser.new_tab(ERRORS_FIXTURE_URL)
            await wait_for(lambda: any("kapow" in d for d in browser.get_diagnostics()))
    asyncio.run(scenario())


def test_switch_tab_out_of_range_reports_coaching():
    async def scenario():
        async with open_browser() as browser:
            with pytest.raises(BrowserActionError, match="tab"):
                await browser.switch_tab(7)
    asyncio.run(scenario())


def test_observation_reports_tab_state():
    async def scenario():
        async with open_browser() as browser:
            info = await browser.get_page_info()
            assert "Tabs: 1 (active 0)" in render_observation(info, [])
            await browser.new_tab("about:blank")
            info = await browser.get_page_info()
            assert info["tabs"] == 2 and info["active_tab"] == 1
            assert "Tabs: 2 (active 1)" in render_observation(info, [])
    asyncio.run(scenario())
