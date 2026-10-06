# Design Spec: desk.waker v2 — Smarter Agent + Live Web UI

**Date:** 2026-10-06
**Predecessor spec:** `2026-10-06-desk-waker-agent-design.md` (v1 — unchanged except where amended here)
**Status:** Draft for user review

## 1. Intent

Two user-reported needs from a real v1 run (`goal: open youtube.com and play a random video`, session `20261006T104219195040Z`):

1. **The agent is not smart enough.** Diagnosed failure patterns: (a) `navigate "youtube.com"` without a scheme → Playwright invalid-URL error; (b) the LLM fixated on a phantom "URL bar" (browser chrome, never in the element map) and fired `type`/`click` with `element: None` ten times without adapting; (c) the error string `invalid element reference None` gave it no actionable alternative; (d) `llama3.1:8b` may be a weak agentic reasoner vs the locally available `qwen3.5:latest`.
2. **The user cannot see what the agent is doing.** v1 is terminal-only; step lines scroll by with no page state, no live thought/result pairing, no in-page confirmations.

**Success criteria:**
- **S1:** The YouTube goal completes (status `done`) within 15 steps with no repeated identical failed approach, using the bake-off-winning default model.
- **S2:** `python webui.py` opens a browser page where goals can be entered, every step streams live (thought, action, result, page state), and confirmation/`ask_user` prompts are answerable inline.
- **S3:** Full suite green; v1 terminal REPL unchanged in behavior; no exception reaches the user as a traceback (v1 spec §7 applies to the UI too).

**Stated constraints (user decisions):**
- UI approach: lightweight web UI — FastAPI + SSE single page (user chose over Streamlit and terminal TUI)
- Smartness scope: targeted fixes + model bake-off (user chose over reflection loop and screenshot vision)
- Local Ollama only (v1 directive) — no cloud APIs
- Out of scope (YAGNI): screenshots/vision, session-history browser, parallel turns, Streamlit

**Planning amendments (2026-10-06, during plan mapping — surfaced by spec self-review):** `llm.next_action` runs via `asyncio.to_thread`; confirm/ask futures time out at 300 s (replacing the earlier "no timeout" line) and resolve as decline/empty; `GET /api/status` added for SSE-reconnect resync; `on_step` signature is `(entry, page_info, elements)`.

## 2. Architecture

```
desk.waker/
├── llm.py        # + prompt rules, coaching errors, dynamic decide-prompt   (modified)
├── agent.py      # + URL normalization, repeat-failure hint, step budget    (modified)
├── browser.py    # unchanged
├── chat.py       # unchanged (terminal REPL keeps working)
├── webui.py      # NEW: FastAPI app factory + SSE + lifespan-managed browser
├── static/
│   └── index.html  # NEW: single-page chat + live step feed (vanilla JS)
├── tools/
│   └── model_bench.py  # NEW: local-model bake-off driver (dev tool, not imported by the app)
├── tests/
│   ├── test_agent.py       # + URL normalization, coaching error, prompt-rule tests
│   ├── test_webui.py       # NEW: endpoint + SSE tests (fake LLM, real headless browser)
│   └── ...
└── requirements.txt        # + fastapi, uvicorn
```

**Boundaries preserved from v1:** `browser.py` is the only Playwright importer; `llm.py` is the only `ollama` importer; `agent.run_turn` remains the single loop entry point; the UI consumes `run_turn` through the same duck-typed `llm` and callback hooks (`on_confirm`, `on_ask`) plus one new `on_step` callback.

**New dependency constraint:** v1's "ONLY playwright, ollama, pydantic, python-dotenv" becomes "… + fastapi, uvicorn" (pytest remains dev-only).

## 3. Smarter-agent changes (deterministic)

### 3.1 URL normalization (`agent.py`)

Before dispatching `navigate`: if `parameters.url` does not match a scheme-with-`://` prefix (`^[a-zA-Z][a-zA-Z0-9+.-]*://`), prepend `https://`. This normalizes `youtube.com`, `www.example.com/path`, and `localhost:3000` (no `://` → `https://localhost:3000`); URLs already carrying any scheme (`https://`, `http://`, `file://`) pass through untouched. Anything still invalid after normalization surfaces as `ERROR` to the LLM as today.

### 3.2 Coaching error messages (`agent.py`)

| Current | Replacement (pattern) |
|---|---|
| `ERROR: invalid element reference None` | `ERROR: parameters.element must be one of [E0]…[En] shown in the observation. There is no address bar or browser chrome — only page elements. To open a site use action navigate with a full URL (https://…).` (indices listed from the current observation) |
| `ERROR: navigate requires parameters.url` | same + `Example: {"url": "https://example.com"}` |
| `ERROR: type requires parameters.text` | same + `Example: {"element": "E2", "text": "hello"}` |
| Playwright invalid-URL (post-normalization, still bad) | existing `ERROR: Failed to navigate …` unchanged — normalization prevents the common case |

Existing error strings that tests and the decline path depend on (`ERROR: USER DECLINED this action`, `ERROR: element changed since observation`, `ERROR: <BrowserActionError>`) are unchanged.

### 3.3 Prompt hardening (`llm.py`)

Added to `SYSTEM_RULES`:
- The browser chrome (address bar, tabs, back button, anything outside the page) does not exist for you. You can only see the observation and act on the elements it lists.
- Never retry an approach that just failed with the same parameters; change strategy or, if genuinely blocked, use ask_user or done.
- Every navigate URL must be absolute (`https://…`).

Dynamic decide prompt: `DECIDE_PROMPT` becomes a function `decide_prompt(steps_used, max_steps, history)` → `"Decide the next single action. Steps used: {u}/{m}."` plus when the two most recent results share the same action AND both are `ERROR`: `" You have repeated a failed approach — do something different."`. `build_messages` gains optional `steps_used`/`max_steps` params (defaults preserve v1 call sites and tests).

### 3.4 Model bake-off (`tools/model_bench.py`)

- Driver script (run manually; excluded from default `pytest`): for each model in `OLLAMA_MODELS` (default: models returned by `ollama list` at run time, at minimum `llama3.1:8b`, `qwen3.5:latest`) × each goal in a fixed goal list (fixture goal, `Open https://example.com and tell me the page heading`, `open youtube.com and play a random video`), run `run_turn` headless with a fresh browser and record: status, steps used, JSON-repair events (count of `Model output still invalid after repair` in transcript thoughts/results), wall time.
- Output: Markdown table printed to stdout and written to `docs/superpowers/bench/2026-10-06-model-bakeoff.md`.
- Winner selection rule (decided before running, to avoid cherry-picking): **goal achievement first; among achievers, fewest average steps; tie-break fewer JSON failures.** If no model achieves all goals, pick the one with most achievements.
- The winner becomes `MODEL`'s default in `config.py`; the losing model remains usable via `MODEL` env.
- **Risk:** bake-off results depend on the live sites (YouTube personalization/consent screens). The YouTube goal's pass criterion is S1 (reaches `done` with a coherent answer), not "video actually playing" — consent screens make exact playback nondeterministic.

## 4. Web UI changes

### 4.1 Server (`webui.py`)

- `create_app(browser=None, llm=None, logger=None)` factory (dependency-injected for tests); `python webui.py` runs `uvicorn` on `127.0.0.1:8000` with a real `BrowserWrapper` + `LLMClient` + `SessionLogger` created in FastAPI lifespan and closed on shutdown.
- **Single active turn:** a module-level `asyncio.Lock`; `POST /api/goal` while a turn runs → `409 {"error": "a turn is already running"}`.
- **Endpoints:**
  - `GET /` → `static/index.html`
  - `POST /api/goal` `{"goal": str}` → `202 {"started": true}`; spawns `run_turn` as a task
  - `GET /api/events` → `text/event-stream`; streams JSON events (below) to every connected client (broadcast from the active turn; clients connecting mid-turn receive subsequent events only)
  - `GET /api/status` → `{"running": bool, "last_turn": {status, answer, steps_used} | null}`
  - `POST /api/respond` `{"id": str, "answer": str}` → resolves a pending confirm (`"y"`/`"n"`/free text) or `ask_user` reply; unknown/expired id → `409`
- **Event types** (each a JSON `data:` frame): `turn_started {goal}`, `step {n, action, thought, result, url, title, element_count}`, `confirm {id, prompt}`, `ask {id, question}`, `turn_finished {status, answer, steps_used}`, `error {message}`.
- **Callbacks bridge:** `run_turn(..., on_confirm=..., on_ask=..., on_step=...)` where `on_confirm`/`on_ask` publish the event and await an `asyncio.Future` resolved by `/api/respond`; `on_step` is invoked as `on_step(entry, page_info, elements)` after each transcript append, where `entry` is the transcript dict (`step`, `thought`, `action`, `result`), `page_info` is the step's observation snapshot (`url`, `title`, …), and `elements` is the observation's element list (the UI derives `element_count` from it). `run_turn` passes `on_step` to nothing else; the terminal REPL simply omits it (default `None`, no behavior change).
- **Prompt responsiveness:** `run_turn` invokes `llm.next_action` via `asyncio.to_thread`, so the event loop (SSE, `/api/respond`, `/api/status`) stays live while the local model thinks.
- **Orphaned prompts:** confirm/`ask_user` futures time out after `confirm_timeout` seconds (default **300**; injectable for tests). On timeout the turn treats it as a decline / empty reply and continues — a closed tab must never wedge the server into permanent `409`s.
- **Resync:** `GET /api/status` → `{"running": bool, "last_turn": {status, answer, steps_used} | null}`; the frontend calls it whenever the SSE stream (re)opens so a dropped connection cannot leave the UI stuck on "working".
- **v1 spec §7 applies:** unhandled exceptions become `error` events + a FastAPI exception handler returning clean JSON; SSE connections never emit tracebacks. Session transcripts still written via `SessionLogger` after each turn.

### 4.2 Frontend (`static/index.html`)

Single page, vanilla JS, no build step:
- **Left/main: chat pane** — goal input box; each turn appends the goal and, on `turn_finished`, the final answer/status.
- **Right/side: live step feed** — cards appended on each `step` event: header `Step n/15 | action`, thought text, result (green `OK` / red `ERROR`), and page state line `URL · title · N elements`. Pending `confirm` events render an inline card with **Proceed (y)** / **Decline (n)** buttons; `ask` events render the question with a text input + Send. Only one pending prompt at a time.
- Status line shows `working…` / `done` / `max_steps` / `error`; reconnects SSE on drop (EventSource auto-retry) and calls `GET /api/status` whenever the stream (re)opens to resync the status line and turn state.
- No screenshots, no history persistence beyond the open page (out of scope).

## 5. Testing

| Test | Scope | Needs |
|---|---|---|
| `test_agent.py` additions | `navigate` bare-host normalization; coaching error strings (unparseable element ref, missing url/text); decide-prompt contains step budget; repeat-failure hint appears on two consecutive identical ERROR results; system rules contain no-chrome rule | none beyond existing fixtures |
| `test_webui.py` | `POST /api/goal` starts a turn (scripted fake LLM, real headless browser via lifespan); second goal during a turn → 409; SSE stream yields `turn_started`, ≥1 `step`, `turn_finished`; `confirm` event emitted for a fixture submit click and resolves via `/api/respond` with decline (`USER DECLINED` visible in next step's flow); `ask_user` round-trip; `/` serves the page; error path → `error` event, no traceback text | FastAPI TestClient, fake LLM, headless Chromium |
| `test_llm_parse`/`test_llm_client` | unchanged behavior; `build_messages` new params default-compatible | none |
| Manual | `python webui.py` → S1 YouTube goal watched live in UI; terminal `chat.py` regression; bake-off run | local Ollama, internet |

**Milestone mapping (v2):** M5 = smartness unit tests green + YouTube goal passes locally (S1); M6 = bake-off executed, default model set (documented table); M7 = web UI live demo (S2) + suite green (S3).

## 6. Error philosophy (inherited)

No traceback ever reaches the user — terminal or browser. Every failure becomes an `ERROR` result the LLM can react to, a clean turn-abort, a `turn_finished {status: "error"}` event, or a clean shutdown message.
