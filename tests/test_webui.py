import contextlib
import json
import pathlib
import socket
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import httpx
import uvicorn

from llm import LLMOutputError
from test_agent import FIXTURE_URL, ScriptedLLM, index_of, make_action
from webui import create_app

GOAL = "Open the fixture and click the Go button"

_fixture_indices: dict | None = None


def fixture_indices() -> dict:
    global _fixture_indices
    if _fixture_indices is None:
        import asyncio

        from browser import BrowserWrapper

        async def resolve():
            browser = BrowserWrapper()
            await browser.start()
            try:
                await browser.navigate_to(FIXTURE_URL)
                elements = await browser.get_interactive_elements()
                return {
                    "go": index_of(elements, role="button", text="Go"),
                    "submit": index_of(elements, role="button", text="Submit"),
                }
            finally:
                await browser.close()

        _fixture_indices = asyncio.run(resolve())
    return _fixture_indices


def navigate_action():
    return make_action("open the fixture page", "navigate", url=FIXTURE_URL)


class GatedLLM:
    def __init__(self, responses):
        self._inner = ScriptedLLM(responses)
        self.release = threading.Event()
        self.calls = []

    def next_action(self, messages):
        self.calls.append(messages)
        self.release.wait(timeout=10)
        return self._inner._responses.pop(0)

    def verify(self, messages):
        return self._inner.verify(messages)


class FailingLLM:
    def next_action(self, messages):
        raise LLMOutputError("LLM API unavailable: boom")


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


@contextlib.contextmanager
def live_app(stream=True, **app_kwargs):
    app = create_app(**app_kwargs)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline:
            raise AssertionError("server did not start in time")
        time.sleep(0.02)

    events = []
    started = threading.Event()

    def start_stream(client):
        def reader():
            started.set()
            try:
                with client.stream("GET", "/api/events") as response:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
            except Exception:
                pass

        threading.Thread(target=reader, daemon=True).start()
        started.wait(timeout=5)
        time.sleep(0.3)

    with httpx.Client(
        base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(30.0)
    ) as client:
        if stream:
            start_stream(client)
        yield client, events, start_stream
    server.should_exit = True
    thread.join(timeout=15)


def test_index_served():
    with live_app() as (client, events, _):
        response = client.get("/")
        assert response.status_code == 200
        assert "desk.waker" in response.text


def test_goal_checks_browser_liveness_each_turn():
    from browser import BrowserWrapper

    class CountingBrowser(BrowserWrapper):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.start_calls = 0

        async def start(self):
            await super().start()
            self.start_calls += 1

    script = [navigate_action(), make_action("finish", "done", answer="ok")]
    browser = CountingBrowser()
    with live_app(llm=ScriptedLLM(script), browser=browser) as (client, events, _):
        assert browser.start_calls == 1
        response = client.post("/api/goal", json={"goal": GOAL})
        assert response.status_code == 202
        wait_for(lambda: any(e["type"] == "turn_finished" for e in events))
        assert browser.start_calls == 2
        finished = [e for e in events if e["type"] == "turn_finished"][-1]
        assert finished["status"] == "done"


def test_empty_goal_rejected():
    with live_app() as (client, events, _):
        response = client.post("/api/goal", json={"goal": "   "})
        assert response.status_code == 400


def test_status_endpoint_initial_state():
    with live_app() as (client, events, _):
        response = client.get("/api/status")
        assert response.status_code == 200
        assert response.json() == {"running": False, "last_turn": None}


def test_goal_streams_turn_events():
    indices = fixture_indices()
    script = [
        navigate_action(),
        make_action("click Go", "click", element=f"E{indices['go']}"),
        make_action("finish", "done", answer="clicked"),
    ]
    with live_app(llm=ScriptedLLM(script)) as (client, events, _):
        response = client.post("/api/goal", json={"goal": GOAL})
        assert response.status_code == 202
        wait_for(lambda: any(e["type"] == "turn_finished" for e in events))
        assert events[0] == {"type": "turn_started", "goal": GOAL}
        steps = [e for e in events if e["type"] == "step"]
        assert any(step["url"] == FIXTURE_URL for step in steps)
        finished = events[-1]
        assert finished["type"] == "turn_finished"
        assert finished["status"] == "done"
        assert finished["answer"] == "clicked"
        assert finished["steps_used"] == 3


def test_second_goal_conflicts_and_status_resyncs():
    gated = GatedLLM([make_action("finish", "done", answer="done")])
    with live_app(llm=gated, stream=False) as (client, events, _):
        response = client.post("/api/goal", json={"goal": GOAL})
        assert response.status_code == 202
        wait_for(lambda: client.get("/api/status").json()["running"] is True)
        second = client.post("/api/goal", json={"goal": "another goal"})
        assert second.status_code == 409
        gated.release.set()
        wait_for(lambda: client.get("/api/status").json()["running"] is False)
        status = client.get("/api/status").json()
        assert status["last_turn"]["status"] == "done"
        assert status["last_turn"]["answer"] == "done"


def test_confirm_round_trip_decline():
    indices = fixture_indices()
    script = [
        navigate_action(),
        make_action("click submit", "click", element=f"E{indices['submit']}"),
        make_action("finish", "done", answer="after decline"),
    ]
    with live_app(llm=ScriptedLLM(script)) as (client, events, _):
        assert client.post("/api/goal", json={"goal": GOAL}).status_code == 202
        wait_for(lambda: any(e["type"] == "confirm" for e in events))
        confirm = next(e for e in events if e["type"] == "confirm")
        assert "submit" in confirm["prompt"].lower()
        response = client.post(
            "/api/respond", json={"id": confirm["id"], "answer": "n"}
        )
        assert response.status_code == 200
        wait_for(lambda: any(e["type"] == "turn_finished" for e in events))
        assert any(
            "USER DECLINED" in e.get("result", "")
            for e in events
            if e["type"] == "step"
        )
        again = client.post(
            "/api/respond", json={"id": confirm["id"], "answer": "y"}
        )
        assert again.status_code == 409


def test_ask_round_trip():
    script = [
        make_action("need input", "ask_user", question="Which color?"),
        make_action("finish", "done", answer="blue"),
    ]
    with live_app(llm=ScriptedLLM(script)) as (client, events, _):
        assert client.post("/api/goal", json={"goal": GOAL}).status_code == 202
        wait_for(lambda: any(e["type"] == "ask" for e in events))
        ask = next(e for e in events if e["type"] == "ask")
        assert ask["question"] == "Which color?"
        assert (
            client.post(
                "/api/respond", json={"id": ask["id"], "answer": "blue"}
            ).status_code
            == 200
        )
        wait_for(lambda: any(e["type"] == "turn_finished" for e in events))
        finished = next(e for e in events if e["type"] == "turn_finished")
        assert finished["answer"] == "blue"


def test_late_subscriber_receives_late_events():
    indices = fixture_indices()
    gated = GatedLLM(
        [
            navigate_action(),
            make_action("click Go", "click", element=f"E{indices['go']}"),
            make_action("finish", "done", answer="late"),
        ]
    )
    with live_app(llm=gated, stream=False) as (client, events, start_stream):
        assert client.post("/api/goal", json={"goal": GOAL}).status_code == 202
        start_stream(client)
        gated.release.set()
        wait_for(lambda: any(e["type"] == "turn_finished" for e in events))
        assert any(e["type"] == "step" for e in events)
        assert events[-1]["type"] == "turn_finished"
        assert events[-1]["answer"] == "late"


def test_confirm_timeout_declines():
    indices = fixture_indices()
    script = [
        navigate_action(),
        make_action("click submit", "click", element=f"E{indices['submit']}"),
        make_action("finish", "done", answer="done"),
    ]
    with live_app(llm=ScriptedLLM(script), confirm_timeout=0.3) as (
        client,
        events,
        _,
    ):
        assert client.post("/api/goal", json={"goal": GOAL}).status_code == 202
        wait_for(lambda: any(e["type"] == "turn_finished" for e in events), timeout=20)
        assert any(
            "USER DECLINED" in e.get("result", "")
            for e in events
            if e["type"] == "step"
        )


def test_transport_error_finishes_turn():
    with live_app(llm=FailingLLM()) as (client, events, _):
        assert client.post("/api/goal", json={"goal": GOAL}).status_code == 202
        wait_for(lambda: any(e["type"] == "turn_finished" for e in events))
        finished = next(e for e in events if e["type"] == "turn_finished")
        assert finished["status"] == "error"
        assert "LLM API unavailable" in finished["answer"]
