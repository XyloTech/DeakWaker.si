import re
from dataclasses import dataclass
from typing import Callable

from browser import BrowserActionError, BrowserWrapper
from config import MAX_HISTORY_STEPS, MAX_STEPS
from llm import LLMOutputError, TRANSPORT_ERROR_PREFIX, build_messages, render_observation

ELEMENT_REF_PATTERN = re.compile(r"^E(\d+)$")


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


async def _dispatch(
    message, browser: BrowserWrapper, on_ask: Callable[[str], str] | None, index: int | None
) -> str:
    params = message.parameters
    if message.action in ("click", "type") and index is None:
        return f"ERROR: invalid element reference {params.element!r}"
    try:
        if message.action == "navigate":
            if not params.url:
                return "ERROR: navigate requires parameters.url"
            await browser.navigate_to(params.url)
        elif message.action == "scroll":
            await browser.scroll(params.direction or "down")
        elif message.action == "click":
            await browser.get_interactive_elements()
            await browser.click(index)
        elif message.action == "type":
            if params.text is None:
                return "ERROR: type requires parameters.text"
            await browser.get_interactive_elements()
            await browser.type_text(index, params.text)
        elif message.action == "ask_user":
            question = params.question or ""
            reply = on_ask(question) if on_ask is not None else ""
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
    max_steps: int = MAX_STEPS,
    max_history: int = MAX_HISTORY_STEPS,
) -> TurnResult:
    history: list[dict] = []
    transcript: list[dict] = []
    steps_used = 0

    for step in range(1, max_steps + 1):
        steps_used = step
        try:
            page_info = await browser.get_page_info()
            elements = await browser.get_interactive_elements()
        except BrowserActionError as exc:
            return TurnResult("error", f"Observation failed: {exc}", steps_used, transcript)
        observation = render_observation(page_info, elements)

        try:
            message = llm.next_action(build_messages(goal, observation, history))
        except LLMOutputError as exc:
            if str(exc).startswith(TRANSPORT_ERROR_PREFIX):
                return TurnResult("error", str(exc), steps_used, transcript)
            result = f"ERROR: {exc}"
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

        print(format_step_line(step, max_steps, message.action, message.thought))

        if message.action == "done":
            transcript.append(
                {
                    "step": step,
                    "observation": observation,
                    "thought": message.thought,
                    "action": "done",
                    "result": "OK",
                }
            )
            return TurnResult("done", message.parameters.answer, steps_used, transcript)

        index = _parse_element(message.parameters.element)

        # Task 5 guardrail hook: is_sensitive(message, page_info, elements) check
        # runs here — after element parsing, before dispatch.

        result = await _dispatch(message, browser, on_ask, index)
        transcript.append(
            {
                "step": step,
                "observation": observation,
                "thought": message.thought,
                "action": message.action,
                "result": result,
            }
        )
        history.append(
            {"thought": message.thought, "action": message.action, "result": result}
        )
        history = history[-max_history:]

    return TurnResult("max_steps", None, steps_used, transcript)
