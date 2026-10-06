import json
from typing import Callable, Literal

from pydantic import BaseModel, ValidationError

from config import GROQ_API_KEY, MODEL

REPAIR_PROMPT = (
    "Your last output was invalid: {error}. "
    "Output ONLY valid JSON matching the schema."
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
        try:
            return parse_action(self._complete(messages))
        except LLMOutputError as exc:
            repair_message = {
                "role": "user",
                "content": REPAIR_PROMPT.format(error=exc),
            }
        try:
            return parse_action(self._complete([*messages, repair_message]))
        except LLMOutputError as exc:
            raise LLMOutputError(f"Model output still invalid after repair: {exc}") from exc

    def _complete(self, messages: list[dict]) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(messages)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
