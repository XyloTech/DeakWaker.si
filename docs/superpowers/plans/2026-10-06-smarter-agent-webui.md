# Smarter Agent + Live Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the desk.waker agent markedly smarter (prompt hardening, dispatch coaching, URL normalization, model bake-off) and add a FastAPI + SSE web UI that shows the agent working step-by-step with in-page confirm/ask prompts.

**Architecture:** v1 core (browser.py / llm.py / agent.py / chat.py) stays intact; v2 extends it via new optional callbacks and a new `webui.py` that reuses `run_turn` unchanged. A `tools/model_bench.py` script runs the model bake-off and its winner becomes the default `MODEL`. Frontend is a single vanilla-JS `static/index.html` fed by SSE.

**Tech Stack:** Python 3.14, Playwright (sync API inside asyncio + `asyncio.to_thread`), local Ollama via `ollama` package, FastAPI + Uvicorn + SSE, pytest (+ httpx for TestClient).

**Spec:** `docs/superpowers/specs/2026-10-06-smarter-agent-webui-design.md` (includes 2026-10-06 planning amendments: to_thread, 300 s confirm/ask timeout, `/api/status`, `on_step(entry, page_info, elements)`)

## Global Constraints

- Run all commands from repo root in PowerShell; venv python is `.venv\Scripts\python.exe`; full suite: `.venv\Scripts\python.exe -m pytest -q` must end `passed` (baseline: 47 passed, 1 skipped).
- Local Ollama only, default host `http://127.0.0.1:11434`; no cloud APIs ever.
- Runtime deps exactly: `playwright, ollama, pydantic, python-dotenv, fastapi, uvicorn` (requirements.txt); dev-only: `pytest, httpx`.
- No tracebacks, exception types, or internal paths ever reach a user (terminal or UI) — spec §6; unexpected exceptions surface as `{type:"error"}` / `turn_finished{status:"error"}` with the exception message only.
- `run_turn` public behavior for existing callers is preserved: signature only gains optional keyword-only `on_step=None`; existing tests must stay green untouched.
- `llm.next_action` is always invoked via `asyncio.to_thread` inside `run_turn`.
- SSE event JSON shapes and endpoint routes are exactly as spec §4.1 (`turn_started`, `step{n,action,thought,result,url,title,element_count}`, `confirm{id,prompt}`, `ask{id,question}`, `turn_finished{status,answer,steps_used}`, `error{message}`; `GET /api/events`, `GET /api/status`, `POST /api/goal`, `POST /api/respond`).
- One active turn at a time (`409` otherwise); confirm/ask futures time out after `confirm_timeout` (default `300.0` s) resolving as decline / empty string.
- Max steps 15, observation text limit 2000 (unchanged).
- Windows/PowerShell; commit messages in repo style (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

## Review Focus

1. **SSE client connects mid-turn** (late joiner) — it must still receive subsequent events including `turn_finished`. → Task 5, `test_late_subscriber_receives_late_events`.
2. **Confirm/ask never answered** (closed tab) — server must not wedge into permanent `409`s; turn continues as decline/empty after timeout. → Task 5, `test_confirm_timeout_declines`.
3. **Two tabs / double submit** — second goal while running returns `409`, and after SSE drop/reconnect `/api/status` resyncs the UI (`running` false, `last_turn` set). → Task 5, `test_second_goal_conflicts_and_status`.
4. **URL normalization must not corrupt valid URLs** — `file://` fixture navigation and any scheme'd URL pass through untouched; bare hosts get `https://`. → Task 2, `test_normalize_url_*` + full suite regression.
5. **Repeat-failure hint false positives** — hint must not fire on two repeated *successful* actions or on a single failure. → Task 1, `test_decide_prompt_silent_*`.

---

### Task 1: Prompt hardening (`llm.py` + agent call site)

**Files:**
- Modify: `llm.py` (SYSTEM_RULES ~lines 19–30, `build_messages` ~line 128)
- Modify: `agent.py:150` (single `build_messages` call)
- Test: `tests/test_llm_client.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: existing `DECIDE_PROMPT = "Decide the next single action."`, `build_messages(goal, observation, history)`, history entries `{"thought","action","result"}` where failures start with `"ERROR"`.
- Produces (relied on by Tasks 2–5):
  - `SYSTEM_RULES` gains three bullets (exact text below).
  - `decide_prompt(steps_used: int | None = None, max_steps: int | None = None, history: list[dict] | None = None) -> str`
  - `build_messages(goal: str, observation: str, history: list[dict], *, steps_used: int | None = None, max_steps: int | None = None) -> list[dict]` — last message content is `decide_prompt(steps_used, max_steps, history)`; with no keyword args this equals today's `DECIDE_PROMPT` exactly.
  - `run_turn` passes `steps_used=step, max_steps=max_steps`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_client.py`:

```python
def test_system_rules_forbid_browser_chrome():
    content = llm.build_messages("g", "obs", []) [0]["content"]
    assert "address bar" in content
    assert "does not exist" in content

def test_system_rules_forbid_repeat_and_require_absolute_urls():
    content = llm.build_messages("g", "obs", []) [0]["content"]
    assert "Never retry an approach" in content
    assert "must be absolute" in content

def test_build_messages_includes_step_budget():
    messages = llm.build_messages("g", "obs", [], steps_used=3, max_steps=15)
    assert messages[-1]["content"] == "Decide the next single action. Steps used: 3/15."

def test_decide_prompt_flags_repeated_failure():
    history = [
        {"thought": "", "action": "navigate", "result": "ERROR: invalid url"},
        {"thought": "", "action": "navigate", "result": "ERROR: invalid url"},
    ]
    prompt = llm.decide_prompt(4, 15, history)
    assert "repeated a failed approach" in prompt

def test_decide_prompt_silent_when_success_repeats():
    history = [
        {"thought": "", "action": "scroll", "result": "OK"},
        {"thought": "", "action": "scroll", "result": "OK"},
    ]
    prompt = llm.decide_prompt(4, 15, history)
    assert "repeated a failed approach" not in prompt

def test_decide_prompt_silent_on_single_failure():
    history = [{"thought": "", "action": "navigate", "result": "ERROR: x"}]
    prompt = llm.decide_prompt(4, 15, history)
    assert "repeated a failed approach" not in prompt
```

Append to `tests/test_agent.py` (imports already present):

```python
def test_step_budget_sent_to_llm():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("scroll", "scroll", direction="down"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            await run_turn("Scroll then finish", browser, llm)
            assert "Steps used: 2/15." in last_blob(llm.calls[1])
        finally:
            await browser.close()

    asyncio.run(scenario())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_agent.py::test_step_budget_sent_to_llm -v`
Expected: FAIL — `AttributeError: module 'llm' has no attribute 'decide_prompt'` / `"address bar"` not in system content / `"Steps used"` not found.

- [ ] **Step 3: Implement in `llm.py`**

Append three bullets to the `SYSTEM_RULES` tuple (exact copy):

```python
"- The browser chrome (address bar, tabs, back button, anything outside the page) "
"does not exist for you. You can only see the observation and act on the elements it lists.\n"
"- Never retry an approach that just failed with the same parameters; change strategy "
"or, if genuinely blocked, use ask_user or done.\n"
"- Every navigate URL must be absolute, starting with https:// (or another scheme like file://).\n"
```

Add (above `build_messages`):

```python
def decide_prompt(steps_used=None, max_steps=None, history=None) -> str:
    ...
```

Body: start `base = DECIDE_PROMPT`; if `steps_used is not None and max_steps is not None`: `base = f"{DECIDE_PROMPT} Steps used: {steps_used}/{max_steps}."`; if `history` has ≥2 entries, last two share the same `action`, and both `result`s start with `"ERROR"`: append `" You have repeated a failed approach — do something different."`.

Change `build_messages` to keyword-only `steps_used=None, max_steps=None` and replace the final `messages.append({"role": "user", "content": DECIDE_PROMPT})` with `decide_prompt(steps_used, max_steps, history)`.

In `agent.py:150` change the call to `build_messages(goal, observation, history, steps_used=step, max_steps=max_steps)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_agent.py -q`
Expected: all PASS (existing `test_build_messages_contains_goal_and_observation` still pins the default prompt exactly).

- [ ] **Step 5: Commit**

```bash
git add llm.py agent.py tests/test_llm_client.py tests/test_agent.py
git commit -m "feat: harden system rules and add step-budget/repeat-failure decide prompt"
```

---

### Task 2: Dispatch coaching, URL normalization, off-loop LLM (`agent.py`)

**Files:**
- Modify: `agent.py` (`_dispatch` ~lines 79–123, `run_turn` ~line 150)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 1's `build_messages(..., steps_used=, max_steps=)`; observation element dicts `{"index","role","text","selector","submit"}`.
- Produces:
  - `_normalize_url(url: str) -> str` (module-level, importable for tests).
  - `_dispatch(message, browser, on_ask, index, expected, elements: list[dict]) -> str` — new trailing `elements` param (both call sites updated).
  - `run_turn` invokes `llm.next_action` via `asyncio.to_thread`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
def test_normalize_url_prepends_https_to_bare_hosts():
    from agent import _normalize_url
    assert _normalize_url("youtube.com") == "https://youtube.com"
    assert _normalize_url("localhost:3000") == "https://localhost:3000"
    assert _normalize_url("www.example.com/path?q=1") == "https://www.example.com/path?q=1"

def test_normalize_url_keeps_existing_schemes():
    from agent import _normalize_url
    assert _normalize_url("https://x.com") == "https://x.com"
    assert _normalize_url("http://x.com") == "http://x.com"
    assert _normalize_url("file:///tmp/a.html") == "file:///tmp/a.html"

def test_navigate_normalizes_url_before_dispatch():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            recorded = []
            async def fake_navigate(url):
                recorded.append(url)
            browser.navigate_to = fake_navigate
            llm = ScriptedLLM(
                [
                    make_action("go to example", "navigate", url="example.com"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            result = await run_turn("Open example.com", browser, llm)
            assert result.status == "done"
            assert recorded == ["https://example.com"]
        finally:
            await browser.close()

    asyncio.run(scenario())

def test_invalid_element_error_coaches_llm_with_element_refs():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("click nothing", "click"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            await run_turn("Click around", browser, llm)
            blob = last_blob(llm.calls[1])
            assert "invalid element reference" in blob
            assert "address bar" in blob
            assert "[E0]" in blob
        finally:
            await browser.close()

    asyncio.run(scenario())

def test_navigate_missing_url_error_includes_example():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            llm = ScriptedLLM(
                [
                    make_action("navigate with no url", "navigate"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            await run_turn("Go somewhere", browser, llm)
            assert "https://example.com" in last_blob(llm.calls[1])
        finally:
            await browser.close()

    asyncio.run(scenario())

def test_llm_next_action_runs_off_event_loop():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            seen_threads = []

            class RecordingLLM:
                def next_action(self, messages):
                    seen_threads.append(threading.get_ident())
                    return make_action("finish", "done", answer="done")

            await run_turn("Finish", browser, RecordingLLM())
            assert seen_threads == [threading.get_ident()] and seen_threads[0] != threading.main_thread().ident
        finally:
            await browser.close()

    asyncio.run(scenario())
```

Add `import threading` at the top of `tests/test_agent.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py -v`
Expected: FAIL — `ImportError: cannot import name '_normalize_url'` / coaching assertions / thread assertion (next_action currently runs on the main thread).

- [ ] **Step 3: Implement in `agent.py`**

- Add `import asyncio` and `SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")`; add `_normalize_url(url)` → return `url` if `SCHEME_PATTERN.match(url)` else `"https://" + url`.
- `_dispatch` signature gains `elements: list[dict]`; the three coaching messages replace the current bare errors:
  - invalid element: `valid = ", ".join(f'[E{e["index"]}]' for e in elements)` → `f"ERROR: invalid element reference {params.element!r}. The browser chrome (address bar, tabs, back button) does not exist — you can only reference elements from the observation: {valid}"`
  - missing url: `"ERROR: navigate requires parameters.url, e.g. https://example.com"`
  - missing text: `"ERROR: type requires parameters.text, e.g. \"Ada\""`
- Navigate branch: `await browser.navigate_to(_normalize_url(params.url))`.
- `run_turn`: pass `elements` to both `_dispatch` call sites; change line 150 to `message = await asyncio.to_thread(llm.next_action, build_messages(goal, observation, history, steps_used=step, max_steps=max_steps))`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py tests/test_guardrails.py -q`
Expected: all PASS (guardrail tests untouched and green).

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: coach llm on dispatch errors, normalize bare urls, run llm off event loop"
```

---

### Task 3: `on_step` callback (`agent.py`)

**Files:**
- Modify: `agent.py` (`run_turn` signature + 3 transcript-append sites)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 2's `run_turn`.
- Produces: `run_turn(..., on_step: Callable[[dict, dict, list], None] | None = None)` — called as `on_step(entry, page_info, elements)` immediately after every `transcript.append` (LLM-error path, done path, normal path). Task 5's webui is the sole consumer; default `None` must not change terminal behavior.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent.py`:

```python
def test_on_step_receives_entry_page_info_and_elements():
    async def scenario():
        browser = BrowserWrapper()
        await browser.start()
        try:
            await browser.navigate_to(FIXTURE_URL)
            elements = await browser.get_interactive_elements()
            go = index_of(elements, role="button", text="Go")
            llm = ScriptedLLM(
                [
                    make_action("click Go", "click", element=f"E{go}"),
                    make_action("finish", "done", answer="done"),
                ]
            )
            calls = []
            result = await run_turn(
                "Click Go", browser, llm,
                on_step=lambda entry, page, els: calls.append((entry, page, els)),
            )
            assert result.status == "done"
            assert [call[0]["step"] for call in calls] == [1, 2]
            assert calls[0][1]["url"] == FIXTURE_URL
            assert calls[0][2] and calls[0][2][0]["index"] == 0
        finally:
            await browser.close()

    asyncio.run(scenario())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py::test_on_step_receives_entry_page_info_and_elements -v`
Expected: FAIL — `TypeError: run_turn() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Implement in `agent.py`**

Add `on_step=None` to `run_turn`'s keyword-only params; define local helper `def emit(entry): if on_step is not None: on_step(entry, page_info, elements)`; call `emit(entry)` immediately after each of the three `transcript.append(...)` sites (LLM-error path appends `entry` at ~line 163; done path ~line 175; normal path ~line 210 — note the done path returns right after, so emit before `return`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 48+ passed, 1 skipped (full regression).

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add on_step callback to run_turn for live ui streaming"
```

---

### Task 4: Model bake-off + default model (`tools/model_bench.py`, `config.py`)

**Files:**
- Create: `tools/model_bench.py`, `docs/superpowers/bench/2026-10-06-model-bakeoff.md` (script output)
- Modify: `config.py:26` (`MODEL` default)
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `run_turn`, `BrowserWrapper`, `LLMClient(model=...)`, `ollama.Client().list()` for available models.
- Produces:
  - `summarize_transcript(transcript: list[dict]) -> dict` → `{"steps": int, "repairs": int}` where repairs counts transcript results containing `"Model output still invalid after repair"`.
  - `format_table(results: list[dict]) -> str` → markdown table, columns `| model | goal | status | steps | repairs | seconds |`.
  - `run_goal(model: str, goal: str) -> dict` → `{"model","goal","status","answer","steps","repairs","seconds"}` (fresh `BrowserWrapper` per run, wall-clock seconds).
  - `main()` runs models × goals and writes the markdown doc; models = intersection of `["llama3.1:8b", "qwen3:latest", "qwen3.5:latest"]` with `ollama.Client().list()` model names (fallback: all available), goals fixed:
    1. `f"Open the local fixture at {pathlib.Path(__file__).parents[1] / 'tests/fixtures/site.html'} and click the Go button"` (use `.as_uri()`)
    2. `"Open https://example.com and report the main heading"`
    3. `"Open YouTube and tell me the name of a currently trending video"`
- Winner rule (recorded in the generated doc): goal achievement → fewest average steps on achieved goals → fewest JSON/repair failures.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench.py`:

```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.model_bench import format_table, summarize_transcript


def test_summarize_transcript_counts_steps_and_repairs():
    transcript = [
        {"step": 1, "result": "OK"},
        {"step": 2, "result": "ERROR: Model output still invalid after repair: Invalid JSON"},
        {"step": 3, "result": "OK"},
    ]
    assert summarize_transcript(transcript) == {"steps": 3, "repairs": 1}


def test_format_table_includes_header_and_rows():
    results = [
        {"model": "m1", "goal": "g1", "status": "done", "steps": 3, "repairs": 0, "seconds": 12.5},
        {"model": "m1", "goal": "g2", "status": "max_steps", "steps": 15, "repairs": 2, "seconds": 90.1},
    ]
    table = format_table(results)
    assert "| model | goal | status | steps | repairs | seconds |" in table
    assert "| m1 | g1 | done | 3 | 0 | 12.5 |" in table
    assert "max_steps" in table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bench.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.model_bench'`.

- [ ] **Step 3: Implement `tools/model_bench.py`**

Create `tools/__init__.py` (empty) and `tools/model_bench.py` with the four functions above; `main()` orchestrates: import models from ollama, iterate models × goals, `asyncio.run(run_goal(...))` sequentially, `Path("docs/superpowers/bench").mkdir(parents=True, exist_ok=True)`, write `2026-10-06-model-bakeoff.md` containing the winner rule paragraph plus `format_table(results)` plus per-model average steps for `done` goals.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bench.py -q`
Expected: 2 passed.

- [ ] **Step 5: Run the bake-off against live Ollama**

Run: `.venv\Scripts\python.exe tools\model_bench.py`
Expected: doc written under `docs/superpowers/bench/` with one row per (model, goal) — takes several minutes; failures record as their status, never a traceback.

- [ ] **Step 6: Apply the winner and verify**

Read the doc, apply the winner rule, edit `config.py` line 26: `MODEL: str = _env_str("MODEL", "<winner>")`. Run: `.venv\Scripts\python.exe -m pytest -q` → passed.

- [ ] **Step 7: Commit**

```bash
git add tools/ tests/test_bench.py config.py docs/superpowers/bench/
git commit -m "feat: model bake-off script and set winning default model"
```

---

### Task 5: Web UI backend (`webui.py`, deps, `tests/test_webui.py`)

**Files:**
- Create: `webui.py`, `tests/test_webui.py`
- Modify: `requirements.txt` (add `fastapi`, `uvicorn`; dev: `httpx`)

**Interfaces:**
- Consumes: Task 2's `run_turn` (to_thread), Task 3's `on_step`, `SessionLogger.append(goal, transcript) -> Path`, `LLMClient`, `BrowserWrapper`.
- Produces:
  - `create_app(*, browser: BrowserWrapper | None = None, llm=None, logger=None, confirm_timeout: float = 300.0) -> FastAPI` — lifespan creates+starts defaults for `None` inputs and closes the browser on shutdown (closes an injected browser too).
  - `main()` entry: `python webui.py` → uvicorn on `127.0.0.1:8000`.
  - Routes and event shapes per spec §4.1 / Global Constraints. Confirm resolution: `answer.strip().lower() in {"y", "yes"}`; ask returns the raw stripped string; timeout → `False` / `""`.
  - `GET /` serves `static/index.html` (path resolved from `Path(__file__).parent`, not cwd).

- [ ] **Step 1: Install deps**

Run: `.venv\Scripts\python.exe -m pip install fastapi uvicorn httpx`
Add to `requirements.txt`: `fastapi`, `uvicorn` in the main block; `httpx` under `# dev-only`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_webui.py`. Import helpers from the sibling test module (`pytest` puts `tests/` on `sys.path`): `from test_agent import FIXTURE_URL, ScriptedLLM, index_of, make_action`. Structure:

```python
import json, pathlib, sys, threading, time
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from llm import LLMOutputError
from webui import create_app
from test_agent import FIXTURE_URL, ScriptedLLM, index_of, make_action

GOAL = "Open the fixture and click the Go button"


class GatedLLM:
    """ScriptedLLM whose first next_action blocks on a threading.Event."""
    def __init__(self, responses):
        self._inner = ScriptedLLM(responses)
        self.release = threading.Event()
        self.calls = []
    def next_action(self, messages):
        self.calls.append(messages)
        self.release.wait(timeout=10)
        return self._inner._responses.pop(0)


def collect_events(client):
    events = []
    def reader():
        with client.stream("GET", "/api/events") as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    time.sleep(0.2)  # let the subscription land
    return events


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met in time")
```

Test list (each in its own `def`, each opening `with TestClient(create_app(...)) as client:`):

1. `test_index_served` — `client.get("/")` → 200, `"desk.waker"` in `response.text`.
2. `test_empty_goal_rejected` — `client.post("/api/goal", json={"goal": "   "})` → 400.
3. `test_status_endpoint_initial_state` — `GET /api/status` → `{"running": False, "last_turn": None}`.
4. `test_goal_streams_turn_events` — script `[click Go, done(answer="clicked")]`; start `collect_events`, post goal `{"goal": GOAL}` → 202; `wait_for` any `turn_finished`; assert: first event is `turn_started` with this goal; ≥1 `step` with `url == FIXTURE_URL`; last is `turn_finished{status:"done", answer:"clicked", steps_used:2}`.
5. `test_second_goal_conflicts_and_status_resyncs` — `GatedLLM([done(answer="done")])`, `create_app(llm=gated)`; post goal → 202; `wait_for` `GET /api/status` reports `running True`; post second goal → 409; `gated.release.set()`; wait for `turn_finished`; `GET /api/status` → `running False`, `last_turn.status == "done"`.
6. `test_confirm_round_trip_decline` — script `[click submit button (index of role=button text="Submit"), done(answer="after decline")]`; collect events; post goal; `wait_for` a `confirm` event; `client.post("/api/respond", json={"id": confirm["id"], "answer": "n"})` → 200; wait `turn_finished`; assert some `step` event has `"USER DECLINED"` in `result`; re-post the same id → 409.
7. `test_ask_round_trip` — script `[ask_user(question="Which color?"), done(answer="blue")]`; respond `{"answer": "blue"}`; assert `turn_finished` `answer == "blue"`.
8. `test_late_subscriber_receives_late_events` — `GatedLLM` with script `[click Go, done(answer="late")]`; post goal (no subscriber yet), then start `collect_events`, then `gated.release.set()`; wait `turn_finished`; assert the late subscriber received ≥1 `step` and the `turn_finished`.
9. `test_confirm_timeout_declines` — `create_app(llm=scripted_submit_click_done, confirm_timeout=0.3)`; post goal; no respond; wait `turn_finished`; assert a `step` event with `"USER DECLINED"` in `result`.
10. `test_transport_error_finishes_turn` — `create_app(llm=FailingLLM)` where `FailingLLM.next_action` raises `LLMOutputError("LLM API unavailable: boom")`; post goal; wait `turn_finished` with `status == "error"` and `"LLM API unavailable"` in `answer`.

(For scripts needing element indices, resolve them in the test against the fixture via a throwaway `BrowserWrapper` navigate + `index_of`, exactly as `tests/test_agent.py` does — or use stable `E0/E1/...` ordering: the fixture's first elements are Go button / name / pw / Submit, so precompute at test time for robustness.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_webui.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webui'`.

- [ ] **Step 4: Implement `webui.py`**

- `create_app` builds `FastAPI(title="desk.waker")` with `lifespan` (`@asynccontextmanager`): resolve `browser`/`llm`/`logger` defaults, `await browser.start()`, yield, `await browser.close()`. Attach `app.state`: `browser`, `llm`, `logger`, `turn_lock = asyncio.Lock()`, `subscribers: set[asyncio.Queue]`, `pending: dict[str, asyncio.Future]`, `last_turn = None`, `confirm_timeout`.
- `_broadcast(event)` puts `json.dumps(event)` on every subscriber queue.
- `GET /api/events`: create queue, add to subscribers, `finally` discard; response generator yields `f"data: {payload}\n\n"` per queue item; `Response(media_type="text/event-stream", headers={"Cache-Control": "no-cache"})`.
- `POST /api/goal` (body `{goal: str}`): `goal.strip()` empty → 400; if `turn_lock.locked()` → 409 `{"detail": "A turn is already running"}` (check before `await acquire()` — no `await` between, so it's atomic on the single-threaded loop); else `await turn_lock.acquire()`, `asyncio.create_task(_run_goal(goal))`, return 202 `{"status": "accepted"}`.
- `_run_goal(goal)`:
  1. `broadcast {"type":"turn_started","goal":goal}`
  2. helpers `_confirm(prompt) -> bool`: new `id` (`uuid4().hex`), future into `pending`, `broadcast {"type":"confirm","id","prompt"}`, `await asyncio.wait_for(fut, confirm_timeout)` with `TimeoutError → False`, then answer logic `strip().lower() in {"y","yes"}`; `_ask(question) -> str` same but timeout → `""`, returns stripped answer.
  3. `on_step`: `broadcast {"type":"step","n":entry["step"],"action":entry["action"],"thought":entry["thought"],"result":entry["result"],"url":page_info.get("url",""),"title":page_info.get("title",""),"element_count":len(elements)}`
  4. `result = await run_turn(goal, state.browser, state.llm, on_confirm=_confirm, on_ask=_ask, on_step=on_step)` inside `try`; unexpected `Exception as exc` → `broadcast {"type":"error","message":str(exc)}` and synthesize `result = TurnResult("error", str(exc), 0, [])`.
  5. `state.last_turn = {"status": result.status, "answer": result.answer, "steps_used": result.steps_used}`; `state.logger.append(goal, result.transcript)`; `broadcast {"type":"turn_finished", **state.last_turn}`.
  6. `finally`: clear matching `pending` futures, `state.turn_lock.release()`.
- `GET /api/status` → `{"running": state.turn_lock.locked(), "last_turn": state.last_turn}`.
- `POST /api/respond` (`{id, answer}`): `fut = state.pending.pop(id, None)`; `None` → 409; `fut.set_result(answer)`; return `{"status":"ok"}`.
- `GET /` → `FileResponse` of `Path(__file__).parent / "static" / "index.html"` (guard: file may not exist yet until Task 6 — tests 1 depends on it; therefore implement a placeholder `static/index.html` containing the string `desk.waker` in this task).
- `if __name__ == "__main__": uvicorn.run(create_app(), host="127.0.0.1", port=8000)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_webui.py -q`
Expected: 10 passed.

- [ ] **Step 6: Full regression + commit**

Run: `.venv\Scripts\python.exe -m pytest -q` → all passed (baseline + new).

```bash
git add webui.py tests/test_webui.py requirements.txt static/
git commit -m "feat: fastapi web ui with sse streaming, confirm/ask round trips, status"
```

---

### Task 6: Frontend, docs, live verification

**Files:**
- Create/Replace: `static/index.html` (full vanilla JS UI; placeholder from Task 5 replaced)
- Modify: `README.md` (web UI run section), `desk.waker.llm.md` (append M5–M7)

**Interfaces:**
- Consumes: Task 5's four routes and six event types; `GET /api/status` resync.
- Produces: a single-page UI — no other files, no build step, no JS deps.

- [ ] **Step 1: Build `static/index.html`**

One HTML file with inline CSS/JS implementing spec §4.2 exactly:
- Layout: chat pane (left), step feed (right), status line, goal form (`input#goal-input` + `button#run-btn`), hidden prompt card (`#prompt-card` with `#prompt-text`, `#prompt-yes`, `#prompt-reply`, `#prompt-answer`).
- `const es = new EventSource("/api/events")`; `es.onopen = es.onerror = resync` → `fetch("/api/status")` sets status line + `running` flag (turn_running = `running` or turn_started without turn_finished).
- Event handlers: `turn_started` → clear feed, status `working…`, user bubble with goal; `step` → append row `Step {n}/15 | {action} | {thought[:60]}` + result line (and title/url); `confirm` → show card, Yes/No → `POST /api/respond {id, answer: "y"|"n"}`; `ask` → show card with input, Reply → `POST /api/respond {id, answer}`; `turn_finished` → status map `{done:"done", max_steps:"max_steps", error:"error"}`, assistant bubble with `answer`, hide card; `error` → status `error`, assistant bubble with message (never a stack).
- Run button disabled while `working…`.

- [ ] **Step 2: Automated check**

Run: `.venv\Scripts\python.exe -m pytest -q` → all passed (Task 5's `test_index_served` covers serving the real page).

- [ ] **Step 3: Live web UI verification (M5/M6)**

Run in one terminal: `.venv\Scripts\python.exe webui.py`, open `http://127.0.0.1:8000`.
Verify and record:
- Goal `Open https://example.com and report the main heading` → steps stream into the feed, status ends `done`, answer appears in chat, no tracebacks anywhere.
- Goal `Open YouTube and tell me the name of a currently trending video` → URL-normalized navigation visible in steps; answer or `max_steps` ends cleanly; confirm/ask (if raised) renders as an in-page card and a reply resumes the run.
- Two tabs: submit a long goal in both → second gets immediate busy feedback, no crash.

- [ ] **Step 4: Terminal regression (M7)**

Run: `@("Open the local fixture at '<FIXTURE_URL>' and click the Go button", "y") | .venv\Scripts\python.exe chat.py` (substitute the absolute `tests/fixtures/site.html` file URI) → expect `Status: done` / `Answer:` lines, confirm prompt honored, clean EOF exit.

- [ ] **Step 5: Docs + blueprint + commit**

README: add "## Web UI" — `python webui.py`, open `http://127.0.0.1:8000`, note SSE live feed and in-page prompts. `desk.waker.llm.md`: append milestones `M5 Web UI`, `M6 Live streaming + in-page prompts`, `M7 Regression: terminal + suite green` — mark Complete with one-line results from Steps 3–4.

```bash
git add static/index.html README.md desk.waker.llm.md
git commit -m "feat: vanilla js web frontend with sse feed and in-page prompts"
```

---

## Plan self-review (executor may skip)

- **Spec coverage:** §3.1 URL normalization → Task 2; §3.2 coaching errors → Task 2; §3.3 prompt rules/budget/repeat hint → Task 1; §3.4 bake-off → Task 4; §4.1 app/routes/callbacks/events/status/timeout → Task 5; §4.2 frontend → Task 6; §6 error philosophy → Global Constraints + Task 5 step 4.4; §7 in-page confirm/ask → Task 5 tests 6/7/9 + Task 6; §8 out-of-scope items touched by none. Milestones M5–M7 → Task 6.
- **Type consistency checked:** `on_step(entry, page_info, elements)` (Task 3 → Task 5), `_dispatch(..., elements)` (Task 2), `build_messages(..., steps_used=, max_steps=)` (Task 1 → Tasks 2–5), `summarize_transcript`/`format_table` names identical in test and implementation (Task 4).
- **Proportion:** the plan carries names, exact strings, and assertions; algorithm bodies are left to the implementer except where the spec fixes copy.
