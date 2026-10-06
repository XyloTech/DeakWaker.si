import asyncio
import inspect
import re
from dataclasses import dataclass
from typing import Callable

from browser import BrowserActionError, BrowserWrapper
from config import MAX_HISTORY_STEPS, MAX_STEPS
from llm import LLMOutputError, TRANSPORT_ERROR_PREFIX, build_messages, render_observation

ELEMENT_REF_PATTERN = re.compile(r"^E(\d+)$")
SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

SUBMIT_LABEL_PATTERN = re.compile(
    r"\b(sign in|log in|submit|pay|purchase|buy|order|confirm|send)\b", re.IGNORECASE
)
PAYMENT_WORDS = frozenset({"pay", "purchase", "buy", "checkout", "price"})


def _target_text(action, elements: list[dict]) -> str:
    index = _parse_element(action.parameters.element)
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
        index = _parse_element(action.parameters.element)
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
            index = _parse_element(action.parameters.element)
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


def _parse_element(reference: str | None) -> int | None:
    if reference is None:
        return None
    match = ELEMENT_REF_PATTERN.match(reference.strip())
    if match is None:
        return None
    return int(match.group(1))


def _normalize_url(url: str) -> str:
    if SCHEME_PATTERN.match(url):
        return url
    return f"https://{url}"


async def _dispatch(
    message,
    browser: BrowserWrapper,
    on_ask: Callable[[str], str] | None,
    index: int | None,
    expected: tuple[str, str] | None,
    elements: list[dict],
) -> str:
    params = message.parameters
    if message.action in ("click", "type") and index is None:
        valid = ", ".join(f'[E{element["index"]}]' for element in elements)
        return (
            f"ERROR: invalid element reference {params.element!r}. The browser chrome "
            "(address bar, tabs, back button) does not exist — you can only reference "
            f"elements from the observation: {valid}"
        )
    try:
        if message.action == "navigate":
            if not params.url:
                return "ERROR: navigate requires parameters.url, e.g. https://example.com"
            await browser.navigate_to(_normalize_url(params.url))
        elif message.action == "scroll":
            await browser.scroll(params.direction or "down")
        elif message.action in ("click", "type"):
            if message.action == "type" and params.text is None:
                return 'ERROR: type requires parameters.text, e.g. "Ada"'
            current = await browser.get_interactive_elements()
            found = next(
                (entry for entry in current if entry["index"] == index), None
            )
            if expected is not None and (
                found is None or (found["role"], found["text"]) != expected
            ):
                actual = None if found is None else (found["role"], found["text"])
                return (
                    "ERROR: element changed since observation "
                    f"(expected {expected!r}, found {actual!r})"
                )
            if message.action == "click":
                await browser.click(index)
            else:
                await browser.type_text(index, params.text)
        elif message.action == "ask_user":
            question = params.question or ""
            reply = on_ask(question) if on_ask is not None else ""
            if inspect.isawaitable(reply):
                reply = await reply
            return str(reply)
        else:
            return f"ERROR: unknown action {message.action!r}"
    except BrowserActionError as exc:
        return f"ERROR: {exc}"
    return "OK"


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
            emit(entry)
            continue

        if print_steps:
            print(format_step_line(step, max_steps, message.action, message.thought))

        if message.action == "done":
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

        index = _parse_element(message.parameters.element)
        expected: tuple[str, str] | None = None
        if index is not None:
            observed = next(
                (entry for entry in elements if entry["index"] == index), None
            )
            if observed is not None:
                expected = (observed["role"], observed["text"])

        # Guardrail: after element parsing, before dispatch.
        reason = is_sensitive(message, page_info, elements)
        if reason is not None:
            target = _target_text(message, elements) or message.parameters.element or ""
            prompt = f"⚠ About to: {message.action} {target}. Proceed? [y/N] "
            approved = on_confirm(prompt) if on_confirm is not None else False
            if inspect.isawaitable(approved):
                approved = await approved
            if approved:
                result = await _dispatch(message, browser, on_ask, index, expected, elements)
            else:
                result = "ERROR: USER DECLINED this action"
        else:
            result = await _dispatch(message, browser, on_ask, index, expected, elements)
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
