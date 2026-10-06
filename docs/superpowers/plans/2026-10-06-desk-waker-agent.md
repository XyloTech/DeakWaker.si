# desk.waker Autonomous Browser Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terminal-chat agent where a Llama LLM (Groq) drives a Playwright browser through an Observe → Think → Act loop until a natural-language goal is done.

**Architecture:** Modular Python package, one asyncio run. `browser.py` owns all Playwright; `llm.py` owns all Groq + JSON schema/repair; `agent.py` owns the loop and guardrails behind one `run_turn()` entry point; `chat.py` is the REPL. Element targeting uses a numbered element map — the LLM references `E7`, never CSS selectors.

**Tech Stack:** Python 3.10+, Playwright (asyncio), Groq SDK, Pydantic, python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-desk-waker-agent-design.md` — executors read the spec alongside this plan; every requirement below argues from it.

## Global Constraints

- Dependencies: ONLY `playwright`, `groq`, `pydantic`, `python-dotenv` (plus `pytest` for tests)
- LLM backend: Groq API only, default model `llama-3.3-70b-versatile` (env-overridable); no Ollama
- Interface: terminal REPL only; no Streamlit/web UI
- `max_steps` default: 15 (overridable per call); observation visible text capped at 2000 chars
- Element references in observations: `[E0] role "text"` format; LLM output `parameters.element` is that index string (e.g. `"E7"`)
- Sensitive keywords (exact): button text ∈ {submit, sign in, log in, pay, purchase, buy, order, confirm, send}; payment keywords ∈ {pay, purchase, buy, checkout, price}; password field + click/type = login
- Confirmation default: **N** (decline when no `on_confirm` supplied); decline feeds `ERROR: USER DECLINED this action` back to the LLM
- Groq transport failure: exponential backoff ×2 (3 attempts total), then abort turn with a clear message
- Transcripts: `logs/session-<timestamp>.jsonl`; one console line per step: `Step n/max | action | thought`
- No exception reaches the user as a traceback (spec §7)
- Live-Groq tests are manual/optional — excluded from default `pytest` run
- Platform: Windows/PowerShell; all commands must run in PowerShell

## Review Focus

1. **LLM wraps JSON in markdown fences or trailing prose** — must still parse; garbage after one repair must raise `LLMOutputError`, never crash the loop → test in Task 2 (`test_parse_action_accepts_fenced_json`, `test_next_action_repairs_then_raises`)
2. **Element index stale at dispatch time** (page changed after observation) — agent must re-observe, then return `ERROR` to the LLM if still stale → test in Task 4 (`test_click_stale_index_returns_error_to_llm`)
3. **Sensitive action runs without confirmation** — click on a submit button must call `on_confirm` before dispatch and honor a decline → test in Task 5 (`test_submit_click_requires_confirmation_decline_feeds_llm`)
4. **Non-sensitive action blocked by over-eager confirmation** — plain link click and `navigate` must auto-run with `on_confirm` never called → test in Task 5 (`test_plain_link_click_auto_runs_without_confirmation`)
5. **`ask_user` must suspend and resume with the exact user reply** — the reply appears in history sent back to the LLM → test in Task 4 (`test_ask_user_suspends_and_resumes_with_reply`)

---

### Task 1: Scaffolding + BrowserWrapper (Milestone 1)

**Files:**
- Create: `requirements.txt`, `.env.example`, `config.py`, `browser.py`, `tests/fixtures/site.html`, `tests/test_browser.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces (later tasks rely on these exactly):
  - `config.py` constants: `GROQ_API_KEY: str | None`, `MODEL: str = "llama-3.3-70b-versatile"`, `MAX_STEPS: int = 15`, `MAX_HISTORY_STEPS: int = 10`, `OBSERVATION_TEXT_LIMIT: int = 2000`, `HEADLESS: bool = True` (all env-overridable)
  - `class BrowserActionError(Exception)`
  - `class BrowserWrapper`:
    - `def __init__(self, *, headless: bool = HEADLESS)`
    - `async def start(self) -> None`
    - `async def close(self) -> None` (idempotent — safe to call twice)
    - `async def navigate_to(self, url: str) -> None`
    - `async def get_page_info(self) -> dict` → `{"url": str, "title": str, "text": str, "has_password": bool}` (text ≤ 2000 chars)
    - `async def get_interactive_elements(self) -> list[dict]` → `[{"index": int, "role": str, "text": str, "selector": str}]` (also stores list internally as the dispatch table)
    - `async def click(self, index: int) -> None` (resolves via stored table; raises `BrowserActionError` if index unknown or element gone/timeout)
    - `async def type_text(self, index: int, text: str) -> None`
    - `async def scroll(self, direction: str = "down") -> None`
    - `async def get_scroll_position(self) -> dict` → `{"y": int, "max": int}`
    - `async def take_screenshot(self, path: str) -> None`

- [ ] **Step 1: Initialize repo and environment**

```powershell
git init
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install playwright pydantic groq python-dotenv pytest
playwright install chromium
```
Expected: all installs succeed; `python -c "import playwright, groq, pydantic; print('ok')"` prints `ok`.

- [ ] **Step 2: Write `requirements.txt`, `.env.example`, fixture**

`requirements.txt`: `playwright`, `groq`, `pydantic`, `python-dotenv`; separate line group or comment marking `pytest` as dev-only.
`.env.example`: `GROQ_API_KEY=`.
`tests/fixtures/site.html`: a page with `<h1>Fixture</h1>`, a `<button id="go">Go</button>` that appends `<p id="clicked">Clicked</p>` on click, an `<input id="name">`, a `<form>` with `<input type="password" id="pw">` + `<button type="submit">Submit</button>`, an `<a id="plain" href="#dest">Plain Link</a>`, and enough filler content (~2500 chars in divs) to make the page scrollable.

- [ ] **Step 3: Write the failing tests in `tests/test_browser.py`**

Test names with exact assertions:
- `test_navigate_and_extract` — start browser, `navigate_to(file://…site.html)`, `get_page_info()["title"]` contains `Fixture`, `["text"]` contains `Fixture` and is `len <= 2000`, `["has_password"] is True`
- `test_element_map_indices_and_roles` — `get_interactive_elements()` returns sequential indices starting at 0; contains an entry with `role == "button"` and `text == "Go"`; every entry has a non-empty `selector`
- `test_click_updates_dom` — observe, `click(index of "Go")`, then `get_page_info()["text"]` contains `Clicked`
- `test_type_text_fills_input` — `type_text(index of name input, "Ada")`; assert input value via the stored selector is `Ada` (use `page` through a small test-only accessor OR assert via `get_page_info` — implementer's choice; prefer exposing `async def get_value(self, index: int) -> str` on the wrapper if needed)
- `test_scroll_moves_position` — `get_scroll_position()["y"] == 0`, `scroll("down")`, `y > 0` and `y <= max`
- `test_unknown_index_raises_action_error` — `click(999)` raises `BrowserActionError`
- `test_stale_element_raises_action_error` — observe, click `Go` (page mutation), call `click(index of "Go")` again on the re-observed index of a now-removed element after removing it via a second scripted action (use the form's submit: navigate away with `navigate_to` to `about:blank`, then `click(old_index)`) → raises `BrowserActionError`
- `test_close_is_idempotent` — `await close(); await close()` does not raise

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_browser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'browser'`

- [ ] **Step 5: Implement `config.py` and `browser.py`**

Approach notes: launch Chromium via `await playwright.chromium.launch(headless=headless)`; build element map by querying `a, button, input, select, textarea, [role=button], [role=link]` in DOM order, skipping invisible/disabled; `selector` = prefer element `id` (`#id`), else construct a unique-enough CSS path; `text` = trimmed `innerText` or `value`/`placeholder`, max 80 chars; `page_info["text"]` = `document.body.innerText[:2000]`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_browser.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```powershell
git add -A; git commit -m "feat: browser wrapper with element map (Milestone 1)"
```

---

### Task 2: Action schema + JSON parse/repair (Milestone 2, no API)

**Files:**
- Create: `llm.py`, `tests/test_llm_parse.py`

**Interfaces:**
- Consumes: `config.MODEL`
- Produces:
  - `class LLMOutputError(Exception)`
  - `class ActionParameters(BaseModel)`: `element: str | None`, `text: str | None`, `url: str | None`, `direction: str | None`, `question: str | None`, `answer: str | None` (all default `None`)
  - `class ActionMessage(BaseModel)`: `thought: str`, `action: Literal["navigate","click","type","scroll","done","ask_user"]`, `parameters: ActionParameters = ActionParameters()`
  - `def parse_action(raw: str) -> ActionMessage` — raises `LLMOutputError`
  - `class LLMClient`:
    - `def __init__(self, *, api_key: str | None = GROQ_API_KEY, model: str = MODEL, complete_fn: Callable[[list[dict]], str] | None = None)` — `complete_fn` injects transport for tests; Groq SDK used when `None`
    - `def next_action(self, messages: list[dict]) -> ActionMessage` — parse; on `LLMOutputError` resend once with repair message; raise `LLMOutputError` if still invalid

- [ ] **Step 1: Write the failing tests in `tests/test_llm_parse.py`**

- `test_parse_action_valid_json` — minimal `{"thought":"x","action":"navigate","parameters":{"url":"https://example.com"}}` → returns `ActionMessage`, `parameters.url == "https://example.com"`
- `test_parse_action_accepts_fenced_json` — same payload wrapped in ` ```json … ``` ` → parses
- `test_parse_action_accepts_trailing_prose` — payload followed by `\nDone!` → parses (extract first balanced `{…}` block)
- `test_parse_action_rejects_truncated_json` — `{"thought":"x","action":"cli` → raises `LLMOutputError`
- `test_parse_action_rejects_bad_enum` — `"action":"teleport"` → raises `LLMOutputError`
- `test_parse_action_rejects_missing_thought` — no `thought` key → raises `LLMOutputError`
- `test_next_action_success_first_try` — `complete_fn` returns valid JSON; result is `ActionMessage`; `complete_fn` called exactly once
- `test_next_action_repairs_then_raises` — `complete_fn` returns `"sorry here's prose"` both calls → raises `LLMOutputError`; called exactly twice (repair retry)
- `test_next_action_repairs_successfully` — first call returns truncated JSON, second returns valid → returns `ActionMessage`; called twice
- `test_parameters_default_to_none` — JSON with only `thought`+`action:"scroll"` → `parameters.element is None`, `parameters.direction is None`

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 3: Implement schema + parsing in `llm.py`**

Approach: `parse_action` tries `json.loads(raw)`; on failure, extracts the first `{…}` (balanced-brace scan) and retries; on success, validates via `ActionMessage.model_validate` — Pydantic raises for bad enum/missing field, wrap everything in `LLMOutputError`. Repair message text: `"Your last output was invalid: {error}. Output ONLY valid JSON matching the schema."` appended as a `{"role":"user","content":…}` message for the retry call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_parse.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```powershell
git add llm.py tests/test_llm_parse.py; git commit -m "feat: action schema with JSON parse and repair"
```

---

### Task 3: Prompt building + Groq transport with backoff

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm_client.py` (new)

**Interfaces:**
- Consumes: `LLMClient`, `ActionMessage`, `LLMOutputError` from Task 2; `config.OBSERVATION_TEXT_LIMIT`, `config.MAX_HISTORY_STEPS`
- Produces:
  - `def render_observation(page_info: dict, elements: list[dict]) -> str` → `URL: …\nTitle: …\nElements:\n[E0] button "Go"\n…\nVisible text:\n…` (element lines exactly `[E{index}] {role} "{text}"`)
  - `def build_messages(goal: str, observation: str, history: list[dict]) -> list[dict]` → `[system, user(observation), *history_as_messages, user("Decide the next single action.")]`; history items are `{"thought", "action", "result"}` rendered as assistant/user turns (implementer's exact rendering is free; it MUST include goal, observation, and every history thought/action/result)
  - `LLMClient.next_action` now: `_call_with_backoff` (3 attempts, exponential delay starting ≤1s) around transport; on final transport failure raise `LLMOutputError("LLM API unavailable: …")`
  - Groq call passes `response_format={"type": "json_object"}`; parse fallback covers SDKs/versions where the field errors

- [ ] **Step 1: Write the failing tests in `tests/test_llm_client.py`**

- `test_render_observation_formats_element_lines` — given elements `[{index:0, role:"button", text:"Go"},…]` and page_info, output contains `[E0] button "Go"`, the URL, and the title
- `test_build_messages_contains_goal_and_observation` — messages[0] role `system` contains the goal; some message contains the observation string; history thought/action/result strings all appear
- `test_next_action_backoff_then_fails` — `complete_fn` raises `ConnectionError("rate limited")` every call → `next_action` raises `LLMOutputError`; fake tracks calls == 3
- `test_next_action_backoff_recovers` — `complete_fn` raises `ConnectionError` twice, returns valid JSON third time → returns `ActionMessage`; calls == 3
- `test_next_action_live_smoke` (marked `@pytest.mark.live`, skipped unless `GROQ_API_KEY` set) — one real call returning a `navigate` action for a trivial prompt

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_messages'`

- [ ] **Step 3: Implement `render_observation`, `build_messages`, `_call_with_backoff` in `llm.py`**

System prompt content (from spec §4): role as web navigator; the fixed goal; rules — one atomic action per response, reference element indices only, never invent selectors, `ask_user` when blocked, `done` only when goal verifiably achieved. Create `pytest.ini` registering the `markers = live` option; the live test uses `@pytest.mark.live` plus `skipif(not os.getenv("GROQ_API_KEY"))` so default runs show `SKIPPED`, not error.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: all PASS, live test SKIPPED (or passing if key present)

- [ ] **Step 5: Commit**

```powershell
git add llm.py tests/test_llm_client.py pytest.ini; git commit -m "feat: prompt building and groq transport with backoff"
```

---

### Task 4: Agent loop — observe/think/act (Milestone 3)

**Files:**
- Create: `agent.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `BrowserWrapper` (Task 1), `render_observation`/`build_messages` (Task 3), `ActionMessage`/`LLMOutputError` (Task 2)
- Produces:
  - `@dataclass TurnResult`: `status: str` (`"done" | "max_steps" | "error"`), `answer: str | None`, `steps_used: int`, `transcript: list[dict]` (each `{"step": int, "observation": str, "thought": str, "action": str, "result": str}`)
  - `async def run_turn(goal: str, browser: BrowserWrapper, llm, *, on_confirm: Callable[[str], bool] | None = None, on_ask: Callable[[str], str] | None = None, max_steps: int = MAX_STEPS, max_history: int = MAX_HISTORY_STEPS) -> TurnResult`
  - `def format_step_line(step: int, max_steps: int, action: str, thought: str) -> str` → `f"Step {step}/{max_steps} | {action} | {thought[:60]}"` (single line, `< 120` chars); `run_turn` prints it per step
  - `llm` is duck-typed: any object with `next_action(messages) -> ActionMessage`
  - Element dispatch: parse `parameters.element` (`"E7"` → index `7`) via regex `^E(\d+)$`; unparseable → `ERROR: invalid element reference …` to LLM (no crash)

- [ ] **Step 1: Write the failing tests in `tests/test_agent.py`**

All tests: real `BrowserWrapper(headless=True)` vs `fixtures/site.html` (`file://` URL), scripted fake LLM class `ScriptedLLM(responses: list[ActionMessage])` whose `next_action` pops the next response and records the `messages` it was given; observation built by the REAL `build_messages`.

- `test_loop_reaches_done_in_three_steps` — script: `click E(goto button)`, `type E(name, "Ada")`, `done` → `status == "done"`, `answer == "Ada typed"`, `steps_used == 3`, transcript has 3 entries with matching actions
- `test_action_error_returned_to_llm` — script a click on a bogus valid index `E42` first → second scripted response's received `messages` contain `ERROR` in the last result; then `done` → `status == "done"`
- `test_click_stale_index_returns_error_to_llm` — observe inside run_turn, but mutate: first scripted action navigates to `about:blank`, second scripted action clicks `E0` (from pre-navigation observation — re-observe should yield empty/new page, so index invalid) → the LLM receives `ERROR` mentioning element; then `done`
- `test_max_steps_returns_partial_status` — `run_turn(..., max_steps=2)` with scripted responses that never `done` → `status == "max_steps"`, `steps_used == 2`
- `test_ask_user_suspends_and_resumes_with_reply` — scripted `ask_user` with `parameters.question == "Which color?"`; `on_ask` captures question, returns `"blue"` → transcript/result history fed to the third call contains `"blue"`; final `done` → `status == "done"`
- `test_history_evicts_oldest_steps` — `run_turn(..., max_history=2)` with 4 steps → the `messages` seen by the fake on the final call contain results from steps 3–4 only (assert step-1 result string absent)
- `test_format_step_line_truncates_thought` — `format_step_line(3, 15, "click", "x"*200)` → starts with `Step 3/15 | click | `, single line, `len < 120`

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Implement `agent.py`**

Loop per spec §3: observe (`get_page_info` + `get_interactive_elements` → `render_observation`) → `build_messages` → `llm.next_action` (catch `LLMOutputError` → result `ERROR: <msg>`, step counts) → parse element index → dispatch (`navigate` uses `parameters.url`; `scroll` uses `parameters.direction`; `type` uses `parameters.text`; `done` returns; `ask_user` calls `on_ask(question)` and stores reply as result) → catch `BrowserActionError` as result `ERROR: <msg>` → append transcript, print `format_step_line(...)` → evict history beyond `max_history` → on transport/`LLMOutputError` that aborts: `status == "error"` with clear message in `answer`. Guardrail hook (sensitivity) is Task 5 — leave a clearly-named insertion point: after parse, before dispatch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: Tasks 1–4 all PASS, live test SKIPPED

- [ ] **Step 6: Commit**

```powershell
git add agent.py tests/test_agent.py; git commit -m "feat: agent observe-think-act loop (Milestone 3)"
```

---

### Task 5: Sensitive-action guardrails (Milestone 4, part 1)

**Files:**
- Modify: `agent.py`
- Test: `tests/test_guardrails.py` (new)

**Interfaces:**
- Consumes: `run_turn` insertion point from Task 4; `ActionMessage`, `BrowserWrapper.get_page_info`
- Produces:
  - `def is_sensitive(action: ActionMessage, page_info: dict, elements: list[dict]) -> str | None` — returns reason string (e.g. `"form submission"`, `"login"`, `"purchase"`) or `None`
  - Sensitive flow inside `run_turn`: reason found → prompt `f"⚠ About to: {action} {target}. Proceed? [y/N] "` → `approved = on_confirm(prompt) if on_confirm else False` → decline: dispatch skipped, result `ERROR: USER DECLINED this action` (step counts, loop continues)

- [ ] **Step 1: Write the failing tests in `tests/test_guardrails.py`**

Fixture already has password + submit form (Task 1).

- `test_submit_click_requires_confirmation_decline_feeds_llm` — scripted: click submit-button index → `on_confirm` returns `False` → assert `on_confirm` called once, prompt contains `"click"` and `"Proceed?"`; next scripted `next_action` receives messages containing `USER DECLINED`; then `done` → `status == "done"`
- `test_password_field_marks_login_sensitive` — `is_sensitive` for a `type` action when `page_info["has_password"]` is `True` returns a reason containing `login`
- `test_submit_button_text_is_sensitive` — `is_sensitive` for click on the element whose text is `Submit` returns reason containing `form submission`
- `test_purchase_keyword_in_title_is_sensitive` — `is_sensitive` for click with `page_info["title"] == "Checkout — Pay now"` returns a reason containing `purchase`
- `test_plain_link_click_auto_runs_without_confirmation` — scripted click on `Plain Link` → `on_confirm` mock never called, result `OK`
- `test_navigate_never_sensitive` — `is_sensitive` for `navigate` action on any fixture page returns `None`
- `test_sensitive_without_callback_declines` — `run_turn` called with `on_confirm=None`, script clicks submit → result contains `USER DECLINED`, `on_confirm`-less run never raises

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_guardrails.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_sensitive'`

- [ ] **Step 3: Implement `is_sensitive` and wire into `run_turn` at the Task-4 insertion point**

Detection order: (1) password-field login — `page_info["has_password"]` and action ∈ {click, type}; (2) form submission — target element text (click) or `type=submit` role matches the exact button-word set; (3) purchase — payment keywords in target element text OR page title. Else `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_guardrails.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all PASS except live test SKIPPED. Confirm Task 4 tests still green (no regression from the guardrail hook).

- [ ] **Step 6: Commit**

```powershell
git add agent.py tests/test_guardrails.py; git commit -m "feat: sensitive-action confirmation guardrails"
```

---

### Task 6: Terminal REPL + session logging (Milestone 4)

**Files:**
- Create: `chat.py`, `tests/test_chat.py`
- Modify: `desk.waker.llm.md` (milestone statuses)

**Interfaces:**
- Consumes: `run_turn` (Task 4/5), `BrowserWrapper`, `LLMClient`, `config`, `format_step_line` (already printed by `run_turn`)
- Produces:
  - `class SessionLogger`: `def __init__(self, log_dir: str = "logs")`; `def append(self, goal: str, transcript: list[dict]) -> Path` → creates dir, writes `session-<UTCtimestamp>.jsonl` (one JSON object per line: `{"goal": …, "step": …}`), returns path
  - `async def repl() -> None` / `def main() -> None` (entry: `asyncio.run`, `KeyboardInterrupt` → print goodbye, browser closed in `finally`)

- [ ] **Step 1: Write the failing tests in `tests/test_chat.py`**

- `test_session_logger_writes_jsonl_roundtrip` — `append("go to example", transcript_2steps)` creates `logs/session-*.jsonl`; reading it yields 2 lines, each `json.loads` clean, first line `goal == "go to example"`
- `test_session_logger_creates_directory` — init with `tmp_path / "custom_logs"` → file exists after append

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chat'`

- [ ] **Step 3: Implement `chat.py`**

REPL behavior: greeting; loop reads `goal:` line; builds shared `BrowserWrapper` + `LLMClient` once; `on_confirm` = print prompt, `input().strip().lower() == "y"`; `on_ask` = print question, `input()`; after each turn print `status` + `answer`, `SessionLogger.append`; on `status == "error"` print `answer` as a friendly message (no traceback); `Ctrl+C`/EOF → close browser in `finally`, exit cleanly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat.py -v`
Expected: all PASS

- [ ] **Step 5: Full suite + manual Milestone-4 smoke**

Run: `pytest -v` → all PASS, live SKIPPED.
Manual (with `GROQ_API_KEY` in `.env`): `python chat.py`, run goal `Open example.com and tell me the page heading` → observe ≥3 step lines, `done` with correct heading; run a goal ending in a form submit on a test site → confirmation prompt appears, answer `N` → agent adapts. Check `logs/session-*.jsonl` written.

- [ ] **Step 6: Mark milestones + commit**

Update `desk.waker.llm.md` Milestone Tracker: all four `Status: Pending` → `(Status: Complete ✓)`.

```powershell
git add -A; git commit -m "feat: terminal REPL with session logging (Milestone 4)"
```
