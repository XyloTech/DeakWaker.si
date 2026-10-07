import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Awaitable
from pathlib import Path

from browser import BrowserActionError


ELEMENT_REF_PATTERN = re.compile(r"^\[?E(\d+)\]?$", re.IGNORECASE)
SCHEME_PATTERN = re.compile(
    r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*://|about:|file:|data:)"
)
SCREENSHOT_DIR = str(Path("artifacts/screenshots"))


def parse_element(reference: str | None) -> int | None:
    if reference is None:
        return None
    value = reference.strip()
    if value.startswith("[") != value.endswith("]"):
        return None
    match = ELEMENT_REF_PATTERN.match(value)
    if match is None:
        return None
    return int(match.group(1))


def normalize_url(url: str) -> str:
    if SCHEME_PATTERN.match(url):
        return url
    return f"https://{url}"


@dataclass
class ActionContext:
    browser: Any
    llm: Any
    goal: str
    step: int
    max_steps: int
    elements: list[dict]
    page_info: dict
    on_ask: Callable | None


@dataclass
class ActionSpec:
    name: str
    description: str
    requires_element: bool
    handler: Callable[[ActionContext, object], Awaitable[str]]
    validate: Callable[[ActionContext, object], str | None] | None = None


async def _navigate_handler(ctx: ActionContext, message) -> str:
    params = message.parameters
    if not params.url:
        return "ERROR: navigate requires parameters.url, e.g. https://example.com"
    url = normalize_url(params.url)
    try:
        await ctx.browser.navigate_to(url)
    except BrowserActionError as exc:
        if "goindigo." not in url.lower():
            raise
        fallback = "https://www.google.com/travel/flights"
        try:
            await ctx.browser.navigate_to(fallback)
        except BrowserActionError:
            raise exc
        return (
            "OK · airline site was unavailable; opened Google Flights as a fallback "
            f"({exc})"
        )
    return "OK"


async def _click_handler(ctx: ActionContext, message) -> str:
    index = parse_element(message.parameters.element)
    if index is None:
        valid = ", ".join(f'[E{element["index"]}]' for element in ctx.elements)
        return (
            f"ERROR: invalid element reference {message.parameters.element!r}. "
            f"The browser chrome (address bar, tabs, back button) does not exist — "
            f"you can only reference elements from the observation: {valid}"
        )
    await ctx.browser.click(index)
    return "OK"


async def _type_handler(ctx: ActionContext, message) -> str:
    index = parse_element(message.parameters.element)
    if index is None:
        valid = ", ".join(f'[E{element["index"]}]' for element in ctx.elements)
        return (
            f"ERROR: invalid element reference {message.parameters.element!r}. "
            f"The browser chrome (address bar, tabs, back button) does not exist — "
            f"you can only reference elements from the observation: {valid}"
        )
    if message.parameters.text is None:
        return 'ERROR: type requires parameters.text, e.g. "Ada"'
    await ctx.browser.type_text(index, message.parameters.text)
    return "OK"


async def _scroll_handler(ctx: ActionContext, message) -> str:
    await ctx.browser.scroll(message.parameters.direction or "down")
    return "OK"


async def _wait_handler(ctx: ActionContext, message) -> str:
    seconds = float(message.parameters.seconds or 1)
    await asyncio.sleep(seconds)
    return f"OK · waited {seconds:g}s for the page to settle"


async def _done_handler(ctx: ActionContext, message) -> str:
    return "OK"


async def _ask_user_handler(ctx: ActionContext, message) -> str:
    question = message.parameters.question or ""
    reply = ctx.on_ask(question) if ctx.on_ask is not None else ""
    import inspect
    if inspect.isawaitable(reply):
        reply = await reply
    return str(reply)


async def _select_handler(ctx: ActionContext, message) -> str:
    index = parse_element(message.parameters.element)
    if index is None:
        valid = ", ".join(f'[E{element["index"]}]' for element in ctx.elements)
        return (
            f"ERROR: invalid element reference {message.parameters.element!r}. "
            f"The browser chrome (address bar, tabs, back button) does not exist — "
            f"you can only reference elements from the observation: {valid}"
        )
    await ctx.browser.select_option(index, message.parameters.value)  # type: ignore
    return "OK"


async def _upload_handler(ctx: ActionContext, message) -> str:
    index = parse_element(message.parameters.element)
    if index is None:
        valid = ", ".join(f'[E{element["index"]}]' for element in ctx.elements)
        return (
            f"ERROR: invalid element reference {message.parameters.element!r}. "
            f"The browser chrome (address bar, tabs, back button) does not exist — "
            f"you can only reference elements from the observation: {valid}"
        )
    await ctx.browser.upload_file(index, message.parameters.path)  # type: ignore
    return "OK"


async def _screenshot_handler(ctx: ActionContext, message) -> str:
    screenshot_dir = Path(SCREENSHOT_DIR)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    path = screenshot_dir / f"step-{ctx.step}.png"
    await ctx.browser.take_screenshot(str(path))
    return f"OK · screenshot saved: {path}"


async def _extract_handler(ctx: ActionContext, message) -> str:
    from llm import LLMOutputError, build_extract_messages

    params = message.parameters
    if not params.query:
        return EXTRACT_VALIDATE_ERROR
    info = await ctx.browser.get_page_info()
    try:
        value = await asyncio.to_thread(
            ctx.llm.extract, build_extract_messages(info["text"], params.query)
        )
        return "OK · " + json.dumps(value, ensure_ascii=False)
    except LLMOutputError as exc:
        return f"ERROR: extraction failed: {exc}"


async def _new_tab_handler(ctx: ActionContext, message) -> str:
    params = message.parameters
    url = params.url
    await ctx.browser.new_tab(normalize_url(url) if url else None)
    return "OK"


async def _switch_tab_handler(ctx: ActionContext, message) -> str:
    params = message.parameters
    i = int(params.tab)
    await ctx.browser.switch_tab(i)
    return "OK"


def _extract_validate(ctx: ActionContext, message) -> str | None:
    if not message.parameters.query:
        return EXTRACT_VALIDATE_ERROR
    return None


def _switch_tab_validate(ctx: ActionContext, message) -> str | None:
    if not message.parameters.tab:
        return SWITCH_TAB_VALIDATE_ERROR
    try:
        int(message.parameters.tab)
    except (ValueError, TypeError):
        return SWITCH_TAB_VALIDATE_ERROR
    return None


NAVIGATE_VALIDATE_ERROR = "ERROR: navigate requires parameters.url, e.g. https://example.com"
TYPE_VALIDATE_ERROR = 'ERROR: type requires parameters.text, e.g. "Ada"'
SELECT_VALIDATE_ERROR = 'ERROR: select requires parameters.value, e.g. "g"'
UPLOAD_VALIDATE_ERROR = 'ERROR: upload requires parameters.path (absolute path to a local file)'
EXTRACT_VALIDATE_ERROR = "ERROR: extract requires parameters.query (what to extract, e.g. \"prices as JSON\")"
SWITCH_TAB_VALIDATE_ERROR = "ERROR: switch_tab requires parameters.tab as a zero-based index"


def _navigate_validate(ctx: ActionContext, message) -> str | None:
    if not message.parameters.url:
        return NAVIGATE_VALIDATE_ERROR
    return None


def _type_validate(ctx: ActionContext, message) -> str | None:
    if message.parameters.text is None:
        return TYPE_VALIDATE_ERROR
    return None


def _select_validate(ctx: ActionContext, message) -> str | None:
    if not message.parameters.value:
        return SELECT_VALIDATE_ERROR
    return None


def _upload_validate(ctx: ActionContext, message) -> str | None:
    if not message.parameters.path:
        return UPLOAD_VALIDATE_ERROR
    return None


def _wait_validate(ctx: ActionContext, message) -> str | None:
    if message.parameters.seconds is None:
        return None
    try:
        seconds = float(message.parameters.seconds)
    except (TypeError, ValueError):
        return "ERROR: wait requires parameters.seconds as a number from 0.1 to 10"
    if not 0.1 <= seconds <= 10:
        return "ERROR: wait requires parameters.seconds from 0.1 to 10"
    return None


ACTIONS: dict[str, ActionSpec] = {
    "navigate": ActionSpec(
        name="navigate",
        description="load a URL; parameters.url must be absolute (https:// or file://)",
        requires_element=False,
        handler=_navigate_handler,
        validate=_navigate_validate,
    ),
    "click": ActionSpec(
        name="click",
        description="click the element referenced as [En]",
        requires_element=True,
        handler=_click_handler,
    ),
    "type": ActionSpec(
        name="type",
        description="replace the text in an input with parameters.text",
        requires_element=True,
        handler=_type_handler,
        validate=_type_validate,
    ),
    "select": ActionSpec(
        name="select",
        description="choose an option in a dropdown; parameters.element + parameters.value",
        requires_element=True,
        handler=_select_handler,
        validate=_select_validate,
    ),
    "upload": ActionSpec(
        name="upload",
        description="attach a local file to a file input; parameters.element + parameters.path",
        requires_element=True,
        handler=_upload_handler,
        validate=_upload_validate,
    ),
    "scroll": ActionSpec(
        name="scroll",
        description="scroll the page; parameters.direction is up or down",
        requires_element=False,
        handler=_scroll_handler,
    ),
    "wait": ActionSpec(
        name="wait",
        description="wait for a dynamic page to settle; parameters.seconds is 0.1 to 10",
        requires_element=False,
        handler=_wait_handler,
        validate=_wait_validate,
    ),
    "done": ActionSpec(
        name="done",
        description="finish the task; parameters.answer holds the final answer",
        requires_element=False,
        handler=_done_handler,
    ),
"ask_user": ActionSpec(
        name="ask_user",
        description="ask the user a question; parameters.question",
        requires_element=False,
        handler=_ask_user_handler,
    ),
    "screenshot": ActionSpec(
        name="screenshot",
        description="save a screenshot artifact for the user; no parameters",
        requires_element=False,
        handler=_screenshot_handler,
    ),
    "extract": ActionSpec(
        name="extract",
        description="pull structured data out of the page as JSON; parameters.query says what",
        requires_element=False,
        handler=_extract_handler,
        validate=_extract_validate,
    ),
    "new_tab": ActionSpec(
        name="new_tab",
        description="open a new tab, optionally at parameters.url",
        requires_element=False,
        handler=_new_tab_handler,
    ),
    "switch_tab": ActionSpec(
        name="switch_tab",
        description="make another tab active; parameters.tab is the zero-based index",
        requires_element=False,
        handler=_switch_tab_handler,
        validate=_switch_tab_validate,
    ),
}
