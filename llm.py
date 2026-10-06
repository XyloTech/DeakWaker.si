import json
import time
from typing import Callable, Literal

from pydantic import BaseModel, ValidationError

from config import GROQ_API_KEY, MAX_HISTORY_STEPS, MODEL, OBSERVATION_TEXT_LIMIT

REPAIR_PROMPT = (
    "Your last output was invalid: {error}. "
    "Output ONLY valid JSON matching the schema."
)

BACKOFF_ATTEMPTS = 3
BACKOFF_BASE_DELAY = 0.5
DECIDE_PROMPT = "Decide the next single action."

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
    "- Check the recent history so you do not repeat actions that already failed."
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


def _loads(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"Invalid JSON: {exc}") from exc


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def render_observation(page_info: dict, elements: list[dict]) -> str:
    lines = [
        f"URL: {page_info.get('url', '')}",
        f"Title: {page_info.get('title', '')}",
        "Elements:",
    ]
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


def build_messages(goal: str, observation: str, history: list[dict]) -> list[dict]:
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
    messages.append({"role": "user", "content": DECIDE_PROMPT})
    return messages


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = GROQ_API_KEY,
        model: str = MODEL,
        complete_fn: Callable[[list[dict]], str] | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self._complete_fn = complete_fn
        self._client = None
        if complete_fn is None:
            from groq import Groq

            self._client = Groq(api_key=api_key)

    def next_action(self, messages: list[dict]) -> ActionMessage:
        raw = self._call_with_backoff(messages)
        try:
            return parse_action(raw)
        except LLMOutputError as exc:
            repair_message = {
                "role": "user",
                "content": REPAIR_PROMPT.format(error=exc),
            }
        raw = self._call_with_backoff([*messages, repair_message])
        try:
            return parse_action(raw)
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
        raise LLMOutputError(f"LLM API unavailable: {last_error}") from last_error

    def _complete(self, messages: list[dict]) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(messages)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except TypeError:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        return response.choices[0].message.content or ""
