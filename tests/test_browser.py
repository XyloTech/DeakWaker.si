import asyncio
import pathlib
import sys
from contextlib import asynccontextmanager

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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
