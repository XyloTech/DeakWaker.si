import asyncio
import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable

from actions import ACTIONS, ActionContext, ActionSpec, normalize_url, parse_element
from browser import BrowserActionError, BrowserWrapper
from config import AUTO_CONFIRM, MAX_HISTORY_STEPS, MAX_STEPS
from llm import (
    LLMOutputError,
    TRANSPORT_ERROR_PREFIX,
    Verdict,
    build_messages,
    build_verify_messages,
    render_observation,
)

ELEMENT_REF_PATTERN = re.compile(r"^E(\d+)$")
SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
DIAGNOSTICS_TEXT_LIMIT = 300

SUBMIT_LABEL_PATTERN = re.compile(
    r"\b(sign in|log in|submit|pay|purchase|buy|order|confirm|send)\b", re.IGNORECASE
)
PAYMENT_WORDS = frozenset({"pay", "purchase", "buy", "checkout", "price"})


def _target_text(action, elements: list[dict]) -> str:
    index = parse_element(action.parameters.element)
    if index is None:
        return ""
    for element in elements:
        if element.get("index") == index:
            return str(element.get("text", ""))
    return ""


def is_sensitive(action, page_info: dict, elements: list[dict]) -> str | None:
    if action.action not in ("click", "type"):
        return None
    target = _target_text(action, elements).lower()
    title = str(page_info.get("title", "")).lower()

    if action.action == "click":
        index = parse_element(action.parameters.element)
        is_submit_control = any(
            element.get("index") == index and element.get("submit")
            for element in elements
        )
        if is_submit_control or SUBMIT_LABEL_PATTERN.search(target):
            return "form submission"

    if any(word in target for word in PAYMENT_WORDS) or any(
        word in title for word in PAYMENT_WORDS
    ):
        return "purchase"

    if page_info.get("has_password"):
        if action.action == "type":
            return "login"
        if action.action == "click":
            index = parse_element(action.parameters.element)
            for element in elements:
                if element.get("index") == index and element.get("role") == "password":
                    return "login"
    return None


@dataclass
class TurnResult:
    status: str
    answer: str | None
    steps_used: int
    transcript: list[dict]


def format_step_line(step: int, max_steps: int, action: str, thought: str) -> str:
    return f"Step {step}/{max_steps} | {action} | {thought[:60]}"


def _recent_diagnostics(browser) -> str:
    getter = getattr(browser, "get_diagnostics", None)
    if getter is None:
        return ""
    try:
        entries = [str(entry) for entry in getter()]
    except Exception:
        return ""
    if not entries:
        return ""
    clearer = getattr(browser, "clear_diagnostics", None)
    if clearer is not None:
        try:
            clearer()
        except Exception:
            pass
    return "; ".join(entries)[-DIAGNOSTICS_TEXT_LIMIT:]


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


async def _execute_dispatch(
    message,
    browser: BrowserWrapper,
    on_ask: Callable[[str], str] | None,
    index: int | None,
    expected: tuple[str, str] | None,
    elements: list[dict],
    ctx: ActionContext,
) -> str:
    spec = ACTIONS.get(message.action)
    if spec is None:
        return f"ERROR: unknown action {message.action!r}"

    if spec.requires_element and index is None:
        valid = ", ".join(f'[E{element["index"]}]' for element in elements)
        return (
            f"ERROR: invalid element reference {message.parameters.element!r}. "
            f"The browser chrome (address bar, tabs, back button) does not exist — "
            f"you can only reference elements from the observation: {valid}"
        )

    elif (param_error := (spec.validate(ctx, message) if spec.validate else None)):
        return param_error

    else:
        reason = is_sensitive(message, ctx.page_info, ctx.elements)
        if reason is not None and not AUTO_CONFIRM:
            target = _target_text(message, ctx.elements) or message.parameters.element or ""
            prompt = f"⚠ About to: {message.action} {target}. Proceed? [y/N] "
            approved = on_confirm(prompt) if on_confirm is not None else False
            if inspect.isawaitable(approved):
                approved = await approved
            if not approved:
                return "ERROR: USER DECLINED this action"
        else:
            approved = True

        if index is not None:
            found = next(
                (e for e in await browser.get_interactive_elements() if e["index"] == index),
                None,
            )
            if found is None or (found["role"], found["text"]) != expected:
                actual = None if found is None else (found["role"], found["text"])
                return (
                    f"ERROR: element changed since observation "
                    f"(expected {expected!r}, found {actual!r})"
                )

        try:
            result = await spec.handler(ctx, message)
        except BrowserActionError as exc:
            return f"ERROR: {exc}"

    if result := (await _recent_diagnostics(browser)) and result.startswith("ERROR"):
        return f"{result}\nRecent browser errors: {_}"

    return "OK" if "result" not in locals() else result


async def run_turn(
    goal: str,
    browser: BrowserWrapper,
    llm,
    *,
    on_confirm: Callable[[str], bool] | None = None,
    on_ask: Callable[[str], str] | None = None,
    on_step: Callable[[dict, dict, list], None] | None = None,
    max_steps: int = MAX_STEPS,
    max_history: int = MAX_HISTORY_STEPS,
    print_steps: bool = True,
) -> TurnResult:
    history: list[dict] = []
    transcript: list[dict] = []
    steps_used = 0

    def emit(entry: dict) -> None:
        if on_step is not None:
            on_step(entry, page_info, elements)

    for step in range(1, max_steps + 1):
        steps_used = step
        try:
            page_info = await browser.get_page_info()
            elements = await browser.get_interactive_elements()
        except BrowserActionError as exc:
            return TurnResult("error", f"Observation failed: {exc}", steps_used, transcript)
        observation = render_observation(page_info, elements)

        try:
            message = await asyncio.to_thread(
                llm.next_action,
                build_messages(
                    goal, observation, history, steps_used=step, max_steps=max_steps
                ),
            )
        except LLMOutputError as exc:
            if str(exc).startswith(TRANSPORT_ERROR_PREFIX):
                return TurnResult("error", str(exc), steps_used, transcript)
            result = f"ERROR: {exc}"
            if print_steps:
                print(format_step_line(step, max_steps, "error", str(exc)))
            entry = {
                "step": step,
                "observation": observation,
                "thought": "",
                "action": "error",
                "result": result,
            }
            transcript.append(entry)
            history.append({"thought": "", "action": "error", "result": result})
            history = history[-max_history:]
            continue

        if print_steps:
            print(format_step_line(step, max_steps, message.action, message.thought))

        if message.action == "done":
            try:
                page_info = await browser.get_page_info()
                elements = await browser.get_interactive_elements()
                observation = render_observation(page_info, elements)
            except BrowserActionError:
                pass
            verdict: Verdict | None = None
            verify_error: str = ""
            try:
                verdict = await asyncio.to_thread(
                    llm.verify, build_verify_messages(goal, observation)
                )
            except LLMOutputError as exc:
                verify_error = str(exc)
            if verdict is None:
                verify_result = (
                    f"ERROR: goal verifier unavailable — accepting done ({verify_error})"
                )
            elif verdict.complete:
                verify_result = f"OK · verified — {verdict.reason}"
            else:
                verify_result = f"ERROR: GOAL NOT COMPLETE — {verdict.reason}"
            if print_steps:
                print(format_step_line(step, max_steps, "verify", verify_result))
            verify_entry = {
                "step": step,
                "observation": observation,
                "thought": "Check whether the goal is fully completed before finishing.",
                "action": "verify",
                "result": verify_result,
            }
            transcript.append(verify_entry)
            emit(verify_entry)
            if verdict is None or verdict.complete:
                entry = {
                    "step": step,
                    "observation": observation,
                    "thought": message.thought,
                    "action": "done",
                    "result": "OK",
                }
                transcript.append(entry)
                emit(entry)
                return TurnResult("done", message.parameters.answer, steps_used, transcript)
            history.append(
                {"thought": message.thought, "action": "verify", "result": verify_result}
            )
            history = history[-max_history:]
            continue

        index = parse_element(message.parameters.element)
        expected: tuple[str, str] | None = None
        if index is not None:
            observed = next(
                (entry for entry in elements if entry["index"] == index), None
            )
            if observed is not None:
                expected = (observed["role"], observed["text"])

        ctx = ActionContext(
            browser=browser,
            llm=llm,
            goal=goal,
            step=step,
            max_steps=max_steps,
            elements=elements,
            page_info=page_info,
            on_ask=on_ask,
        )

        if message.action == "ask_user":
            result = await _ask_user_handler(ctx, message)
            if result.startswith("ERROR"):
                diagnostics = _recent_diagnostics(browser)
                if diagnostics:
                    result = f"{result}\nRecent browser errors: {diagnostics}"
            entry = {
                "step": step,
                "observation": observation,
                "thought": message.thought,
                "action": message.action,
                "result": result,
            }
            transcript.append(entry)
            history.append(
                {"thought": message.thought, "action": message.action, "result": result}
            )
            history = history[-max_history:]
            emit(entry)
            continue

        spec = ACTIONS.get(message.action)
        if spec is None:
            result = f"ERROR: unknown action {message.action!r}"
        elif spec.requires_element and index is None:
            valid = ", ".join(f'[E{element["index"]}]' for element in elements)
            result = (
                f"ERROR: invalid element reference {message.parameters.element!r}. "
                f"The browser chrome (address bar, tabs, back button) does not exist — "
                f"you can only reference elements from the observation: {valid}"
            )
        elif (param_error := (spec.validate(ctx, message) if spec.validate else None)):
            result = param_error
        else:
            reason = is_sensitive(message, page_info, elements)
            if reason is not None and not AUTO_CONFIRM:
                target = _target_text(message, elements) or message.parameters.element or ""
                prompt = f"⚠ About to: {message.action} {target}. Proceed? [y/N] "
                approved = on_confirm(prompt) if on_confirm is not None else False
                if inspect.isawaitable(approved):
                    approved = await approved
                if not approved:
                    result = "ERROR: USER DECLINED this action"
                else:
                    if index is not None:
                        found = next(
                            (e for e in await browser.get_interactive_elements() if e["index"] == index),
                            None,
                        )
                        if found is None or (found["role"], found["text"]) != expected:
                            actual = None if found is None else (found["role"], found["text"])
                            result = (
                                f"ERROR: element changed since observation "
                                f"(expected {expected!r}, found {actual!r})"
                            )
                        else:
                            try:
                                result = await spec.handler(ctx, message)
                            except BrowserActionError as exc:
                                result = f"ERROR: {exc}"
                    else:
                        try:
                            result = await spec.handler(ctx, message)
                        except BrowserActionError as exc:
                            result = f"ERROR: {exc}"
            else:
                if index is not None:
                    found = next(
                        (e for e in await browser.get_interactive_elements() if e["index"] == index),
                        None,
                    )
                    if found is None or (found["role"], found["text"]) != expected:
                        actual = None if found is None else (found["role"], found["text"])
                        result = (
                            f"ERROR: element changed since observation "
                            f"(expected {expected!r}, found {actual!r})"
                        )
                    else:
                        try:
                            result = await spec.handler(ctx, message)
                        except BrowserActionError as exc:
                            result = f"ERROR: {exc}"
                else:
                    try:
                        result = await spec.handler(ctx, message)
                    except BrowserActionError as exc:
                        result = f"ERROR: {exc}"
        if result.startswith("ERROR"):
            diagnostics = _recent_diagnostics(browser)
            if diagnostics:
                result = f"{result}\nRecent browser errors: {diagnostics}"

        entry = {
            "step": step,
            "observation": observation,
            "thought": message.thought,
            "action": message.action,
            "result": result,
        }
        transcript.append(entry)
        history.append(
            {"thought": message.thought, "action": message.action, "result": result}
        )
        history = history[-max_history:]
        emit(entry)

    return TurnResult("max_steps", None, steps_used, transcript)