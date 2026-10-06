import json
import time
from typing import Callable, Literal

from ollama import Client
from pydantic import BaseModel, ValidationError

from config import (
    MAX_HISTORY_STEPS,
    MODEL,
    OBSERVATION_TEXT_LIMIT,
    OLLAMA_HOST,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_S,
    THINKING,
)

REPAIR_PROMPT = (
    "Your last output was invalid: {error}. "
    "Output ONLY valid JSON matching the schema."
)

BACKOFF_ATTEMPTS = 3
BACKOFF_BASE_DELAY = 0.5
DECIDE_PROMPT = "Decide the next single action."
TRANSPORT_ERROR_PREFIX = "LLM API unavailable"

VERIFY_SYSTEM = (
    "You are a strict goal-completion verifier for a web agent. You receive the "
    "user's goal and the current page state, and you decide whether the goal is "
    "FULLY achieved. Be skeptical: if the required evidence is missing, "
    "ambiguous, or only partially satisfied, answer complete=false. "
    "Output ONLY valid JSON: {\"complete\": true|false, \"reason\": \"brief explanation\"}."
)

SYSTEM_RULES = (
    "- Respond with exactly ONE atomic action per response, as a JSON object with "
    'keys "thought", "action", and "parameters".\n'
    "- Valid actions: navigate, click, type, scroll, done, ask_user.\n"
    "- Reference elements only by the indices shown in the observation, e.g. [E7]. "
    "Never invent CSS selectors, XPath, or indices that are not in the observation.\n"
    "- Use ask_user when you are blocked and need a decision or information from "
    "the user.\n"
    "- Use done only when the goal is verifiably achieved; put the final answer in "
    "parameters.answer.\n"
    "- Check the recent history so you do not repeat actions that already failed.\n"
    "- The browser chrome (address bar, tabs, back button, anything outside the page) "
    "does not exist for you. You can only see the observation and act on the elements it lists.\n"
    "- Never retry an approach that just failed with the same parameters; change strategy "
    "or, if genuinely blocked, use ask_user or done.\n"
    "- Every navigate URL must be absolute, starting with https:// (or another scheme like file://).\n"
    "- Diagnose before retrying: read the error text and any reported console or "
    "network errors, reason about the cause, then change strategy.\n"
    "- Recover independently from real-world issues: cookie banners, popups, "
    "overlays, redirects, timeouts, and layout changes — try alternate elements "
    "or scroll to reveal content.\n"
    "- Handle everyday tasks accurately (search, bookings, shopping, messaging, "
    "accounts): fill fields carefully and double-check before submitting.\n"
    "- Use ask_user only for input only a human can supply: one-time codes, "
    "credentials, payment details, or personal choices.\n"
    "- Verify the outcome against the goal before calling done, and report "
    "failures honestly.\n"
)


class LLMOutputError(Exception):
    pass


class ActionParameters(BaseModel):
    element: str | None = None
    text: str | None = None
    url: str | None = None
    direction: str | None = None
    question: str | None = None
    answer: str | None = None


class ActionMessage(BaseModel):
    thought: str
    action: Literal["navigate", "click", "type", "scroll", "done", "ask_user"]
    parameters: ActionParameters = ActionParameters()


class Verdict(BaseModel):
    complete: bool
    reason: str = ""


def _extract_first_json_block(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def parse_action(raw: str) -> ActionMessage:
    try:
        data = _loads(raw)
    except LLMOutputError:
        block = _extract_first_json_block(raw)
        if block is None:
            raise
        data = _loads(block)
    try:
        return ActionMessage.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(f"Invalid action schema: {exc}") from exc


def parse_verdict(raw: str) -> Verdict:
    try:
        data = _loads(raw)
    except LLMOutputError:
        block = _extract_first_json_block(raw)
        if block is None:
            raise
        data = _loads(block)
    try:
        return Verdict.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(f"Invalid verdict schema: {exc}") from exc


def _loads(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"Invalid JSON: {exc}") from exc


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


HTTP_HINTS = {
    401: "authentication required; a signed-in session may be needed",
    403: "access forbidden; the site is blocking this session",
    404: "page not found; URL may be wrong or the page moved",
    429: "rate limited by the site; wait before retrying",
}
SERVER_ERROR_HINT = "server error; the site may be temporarily down"


def _http_hint(status: int) -> str | None:
    if status in HTTP_HINTS:
        return HTTP_HINTS[status]
    if status >= 500:
        return SERVER_ERROR_HINT
    return None


def render_observation(page_info: dict, elements: list[dict]) -> str:
    lines = [
        f"URL: {page_info.get('url', '')}",
        f"Title: {page_info.get('title', '')}",
    ]
    status = page_info.get("http_status")
    if status is not None:
        hint = _http_hint(status)
        if hint is not None:
            lines.append(f"HTTP: {status} — {hint}")
    lines.append("Elements:")
    for element in elements:
        lines.append(f'[E{element["index"]}] {element["role"]} "{element["text"]}"')
    lines.append("Visible text:")
    text = str(page_info.get("text", ""))[:OBSERVATION_TEXT_LIMIT]
    lines.append(text)
    return "\n".join(lines)


def _system_prompt(goal: str) -> str:
    return (
        "You are a web navigator driving a real browser to achieve the user's goal."
        f"\n\nGoal: {goal}\n\nRules:\n{SYSTEM_RULES}"
    )


def decide_prompt(
    steps_used: int | None = None,
    max_steps: int | None = None,
    history: list[dict] | None = None,
) -> str:
    prompt = DECIDE_PROMPT
    if steps_used is not None and max_steps is not None:
        prompt = f"{DECIDE_PROMPT} Steps used: {steps_used}/{max_steps}."
    if history is not None and len(history) >= 2:
        last, previous = history[-1], history[-2]
        if (
            last.get("action") == previous.get("action")
            and str(last.get("result", "")).startswith("ERROR")
            and str(previous.get("result", "")).startswith("ERROR")
        ):
            prompt += " You have repeated a failed approach — do something different."
    if history:
        if str(history[-1].get("result", "")).startswith("ERROR"):
            prompt += (
                " The previous step failed — reason carefully about why"
                " before choosing your next action."
            )
    return prompt


def build_messages(
    goal: str,
    observation: str,
    history: list[dict],
    *,
    steps_used: int | None = None,
    max_steps: int | None = None,
) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(goal)},
        {"role": "user", "content": observation},
    ]
    for step in history[-MAX_HISTORY_STEPS:]:
        thought = _as_text(step.get("thought"))
        action = _as_text(step.get("action"))
        result = _as_text(step.get("result"))
        messages.append(
            {"role": "assistant", "content": f"Thought: {thought}\nAction: {action}"}
        )
        messages.append({"role": "user", "content": f"Result: {result}"})
    messages.append(
        {"role": "user", "content": decide_prompt(steps_used, max_steps, history)}
    )
    return messages


def build_verify_messages(goal: str, observation: str) -> list[dict]:
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {
            "role": "user",
            "content": f"Goal: {goal}\n\nCurrent page state:\n{observation}",
        },
    ]


def _previous_step_failed(messages: list[dict]) -> bool:
    return (
        len(messages) >= 4
        and messages[-2].get("role") == "user"
        and str(messages[-2].get("content", "")).startswith("Result: ERROR")
    )


def _chat_kwargs(model: str, messages: list[dict]) -> dict:
    kwargs: dict = {"model": model, "messages": messages, "format": "json"}
    kwargs["think"] = THINKING or _previous_step_failed(messages)
    kwargs["options"] = {"temperature": OLLAMA_TEMPERATURE, "num_ctx": OLLAMA_NUM_CTX}
    return kwargs


def _ollama_complete(host: str, model: str, messages: list[dict], timeout: float) -> str:
    response = Client(host=host, timeout=timeout).chat(**_chat_kwargs(model, messages))
    return str(response["message"]["content"])


class LLMClient:
    def __init__(
        self,
        *,
        host: str = OLLAMA_HOST,
        model: str = MODEL,
        timeout: float | None = None,
        complete_fn: Callable[[list[dict]], str] | None = None,
    ):
        self.host = host
        self.model = model
        self.timeout = OLLAMA_TIMEOUT_S if timeout is None else timeout
        if complete_fn is None:
            complete_fn = lambda msgs: _ollama_complete(host, model, msgs, self.timeout)
        self._complete_fn = complete_fn

    def next_action(self, messages: list[dict]) -> ActionMessage:
        return self._parse_with_repair(messages, parse_action)

    def verify(self, messages: list[dict]) -> Verdict:
        return self._parse_with_repair(messages, parse_verdict)

    def _parse_with_repair(self, messages: list[dict], parser) -> object:
        raw = self._call_with_backoff(messages)
        try:
            return parser(raw)
        except LLMOutputError as exc:
            repair_message = {
                "role": "user",
                "content": REPAIR_PROMPT.format(error=exc),
            }
        raw = self._call_with_backoff([*messages, repair_message])
        try:
            return parser(raw)
        except LLMOutputError as exc:
            raise LLMOutputError(f"Model output still invalid after repair: {exc}") from exc

    def _call_with_backoff(self, messages: list[dict]) -> str:
        delay = BACKOFF_BASE_DELAY
        last_error: Exception | None = None
        for attempt in range(BACKOFF_ATTEMPTS):
            try:
                return self._complete(messages)
            except Exception as exc:
                last_error = exc
                if attempt < BACKOFF_ATTEMPTS - 1:
                    time.sleep(delay)
                    delay *= 2
        raise LLMOutputError(f"{TRANSPORT_ERROR_PREFIX}: {last_error}") from last_error

    def _complete(self, messages: list[dict]) -> str:
        return self._complete_fn(messages)
