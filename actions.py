import re
from dataclasses import dataclass
from typing import Any, Callable, Awaitable


ELEMENT_REF_PATTERN = re.compile(r"^E(\d+)$")
SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def parse_element(reference: str | None) -> int | None:
    if reference is None:
        return None
    match = ELEMENT_REF_PATTERN.match(reference.strip())
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
    await ctx.browser.navigate_to(normalize_url(params.url))
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


async def _done_handler(ctx: ActionContext, message) -> str:
    return "OK"


async def _ask_user_handler(ctx: ActionContext, message) -> str:
    question = message.parameters.question or ""
    reply = ctx.on_ask(question) if ctx.on_ask is not None else ""
    import inspect
    if inspect.isawaitable(reply):
        reply = await reply
    return str(reply)


NAVIGATE_VALIDATE_ERROR = "ERROR: navigate requires parameters.url, e.g. https://example.com"
TYPE_VALIDATE_ERROR = 'ERROR: type requires parameters.text, e.g. "Ada"'


def _navigate_validate(ctx: ActionContext, message) -> str | None:
    if not message.parameters.url:
        return NAVIGATE_VALIDATE_ERROR
    return None


def _type_validate(ctx: ActionContext, message) -> str | None:
    if message.parameters.text is None:
        return TYPE_VALIDATE_ERROR
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
    "scroll": ActionSpec(
        name="scroll",
        description="scroll the page; parameters.direction is up or down",
        requires_element=False,
        handler=_scroll_handler,
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
}