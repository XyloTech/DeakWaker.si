import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import TurnResult, run_turn
from browser import BrowserWrapper
from chat import SessionLogger
from config import HEADLESS, MAX_STEPS, THINKING_MODE
from llm import LLMClient

DEFAULT_CONFIRM_TIMEOUT = 300.0
STATIC_DIR = Path(__file__).parent / "static"


class GoalBody(BaseModel):
    goal: str


class RespondBody(BaseModel):
    id: str
    answer: str


class _State:
    def __init__(self, *, browser, llm, logger, confirm_timeout: float):
        self.browser = browser
        self.llm = llm
        self.logger = logger
        self.confirm_timeout = confirm_timeout
        self.turn_lock = asyncio.Lock()
        self.subscribers: set[asyncio.Queue] = set()
        self.pending: dict[str, asyncio.Future] = {}
        self.last_turn: dict | None = None
        self.turn_task: asyncio.Task | None = None


def create_app(
    *,
    browser: BrowserWrapper | None = None,
    llm=None,
    logger=None,
    confirm_timeout: float = DEFAULT_CONFIRM_TIMEOUT,
) -> FastAPI:
    state = _State(
        browser=browser, llm=llm, logger=logger, confirm_timeout=confirm_timeout
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.browser is None:
            state.browser = BrowserWrapper(headless=HEADLESS)
        if state.llm is None:
            state.llm = LLMClient()
        if state.logger is None:
            state.logger = SessionLogger()
        await state.browser.start()
        try:
            yield
        finally:
            await state.browser.close()

    app = FastAPI(title="desk.waker", lifespan=lifespan)

    def broadcast(event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        for queue in list(state.subscribers):
            queue.put_nowait(payload)

    async def _wait_reply(event_type: str, payload: dict, ids: set[str]) -> str | None:
        reply_id = uuid.uuid4().hex
        ids.add(reply_id)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        state.pending[reply_id] = future
        broadcast({**payload, "type": event_type, "id": reply_id})
        try:
            return str(await asyncio.wait_for(future, timeout=state.confirm_timeout))
        except TimeoutError:
            return None

    async def _run_goal(goal: str) -> None:
        ids: set[str] = set()

        async def _confirm(prompt: str) -> bool:
            answer = await _wait_reply("confirm", {"prompt": prompt}, ids)
            if answer is None:
                return False
            return answer.strip().lower() in {"y", "yes"}

        async def _ask(question: str) -> str:
            answer = await _wait_reply("ask", {"question": question}, ids)
            return "" if answer is None else answer.strip()

        def _on_step(entry: dict, page_info: dict, elements: list) -> None:
            broadcast(
                {
                    "type": "step",
                    "n": entry["step"],
                    "action": entry["action"],
                    "thought": entry["thought"],
                    "result": entry["result"],
                    "url": page_info.get("url", ""),
                    "title": page_info.get("title", ""),
                    "element_count": len(elements),
                }
            )

        broadcast({"type": "turn_started", "goal": goal})
        try:
            try:
                await state.browser.start()
                result = await run_turn(
                    goal,
                    state.browser,
                    state.llm,
                    on_confirm=_confirm,
                    on_ask=_ask,
                    on_step=_on_step,
                )
            except Exception as exc:
                broadcast({"type": "error", "message": str(exc)})
                result = TurnResult("error", str(exc), 0, [])
            state.last_turn = {
                "status": result.status,
                "answer": result.answer,
                "steps_used": result.steps_used,
            }
            state.logger.append(goal, result.transcript)
            broadcast({"type": "turn_finished", **state.last_turn})
        finally:
            for reply_id in ids:
                state.pending.pop(reply_id, None)
            state.turn_lock.release()

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/events")
    async def events():
        queue: asyncio.Queue = asyncio.Queue()
        state.subscribers.add(queue)

        async def stream():
            try:
                while True:
                    payload = await queue.get()
                    yield f"data: {payload}\n\n"
            finally:
                state.subscribers.discard(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/status")
    async def status():
        return {"running": state.turn_lock.locked(), "last_turn": state.last_turn}

    @app.post("/api/goal", status_code=202)
    async def set_goal(body: GoalBody):
        goal = body.goal.strip()
        if not goal:
            raise HTTPException(status_code=400, detail="Goal must not be empty")
        if state.turn_lock.locked():
            raise HTTPException(status_code=409, detail="A turn is already running")
        await state.turn_lock.acquire()
        state.turn_task = asyncio.create_task(_run_goal(goal))
        return {"status": "accepted"}

    @app.post("/api/respond")
    async def respond(body: RespondBody):
        future = state.pending.pop(body.id, None)
        if future is None or future.done():
            raise HTTPException(status_code=409, detail="Unknown or expired prompt id")
        future.set_result(body.answer)
        return {"status": "ok"}

    return app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
