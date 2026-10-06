# Design Spec: desk.waker — Llama-Powered Autonomous Browser Agent

**Date:** 2026-10-06
**Source blueprint:** `desk.waker.llm.md`
**Status:** Approved (design reviewed section-by-section, 2026-10-06)

## 1. Intent

Build a Python agent that accepts natural-language goals in a terminal chat and executes multi-step browser tasks autonomously via an Observe → Think → Act loop: Playwright observes and acts, a Llama LLM on Groq decides the next atomic action as structured JSON.

**Success = all four blueprint milestones pass:** Playwright interaction works, LLM emits valid JSON actions, 3+ autonomous closed-loop steps, chat + guardrails functional.

**Stated constraints (user decisions):**
- Groq API is the primary (only) LLM backend — no Ollama abstraction in v1
- Terminal chat only; Streamlit deferred to a later iteration
- Element targeting via numbered interactive-element map — LLM never invents CSS selectors
- Confirmations only before sensitive actions (logins, form submissions, purchases); other actions auto-run

**Out of scope (YAGNI):** FastAPI/web transport, vision screenshots, Ollama fallback, multi-tab handling, downloads, browser extensions.

## 2. Architecture (Approach 1 — modular package, single asyncio run)

```
desk.waker/
├── desk.waker.llm.md          # existing blueprint
├── requirements.txt           # playwright, groq, pydantic, python-dotenv
├── .env.example               # GROQ_API_KEY=
├── config.py                  # model name, max_steps=15, timeouts, headless flag
├── browser.py                 # BrowserWrapper class (async Playwright)
│     navigate_to / click / type_text / scroll / extract_page_content /
│     take_screenshot / get_interactive_elements() -> {index, role, text, selector}
├── llm.py                     # Groq client; build prompt, parse + repair JSON
├── agent.py                   # run_turn(goal, on_confirm) — loop, guardrails, logging
├── chat.py                    # terminal REPL (entry point)
├── logs/                      # session-<timestamp>.jsonl transcripts
└── tests/
    ├── fixtures/              # static HTML test page
    ├── test_llm_parse.py      # JSON repair cases (no API calls)
    ├── test_browser.py        # element map vs fixture (headless, no LLM)
    └── test_agent.py          # scripted fake LLM + real headless browser
```

**Execution model:** Playwright's Python API is async; `chat.py` starts one `asyncio` event loop and runs each turn on it. Modules call each other as plain functions — no HTTP, no threads, no background tasks.

**Boundaries:**
- `browser.py` is the only module importing Playwright; all other modules consume plain dicts/strings
- `llm.py` is the only module importing `groq`; accepts injected client for tests
- `agent.py` exposes one entry point: `run_turn(goal, on_confirm=callable) -> TurnResult`
- One browser/page instance persists across turns within a chat session (cookies retained)

**Dependencies:** `playwright`, `groq`, `pydantic`, `python-dotenv` — nothing else.

## 3. Data Flow (agent loop)

```
chat.py REPL
  └─ run_turn(goal)
       ├─ OBSERVE → get_interactive_elements() + page URL/title + visible text (≤2000 chars)
       ├─ THINK   → llm.next_action(history, observation) -> ActionMessage
       ├─ PARSE   → pydantic validate; on failure: 1 repair retry, else ERROR to LLM
       ├─ ACT     → dispatch: navigate|click|type|scroll → browser
       │            done → finish with final answer
       │            ask_user → suspend; chat.py prints Q, stdin answer resumes loop
       ├─ LOG     → console line "Step n/15 | <action> | <thought>"
       │            + append to in-memory history + jsonl transcript
       └─ LOOP until done | max_steps | unrecoverable error
```

- **History:** rolling list of `{observation_summary, thought, action, result}` resent each step; oldest evicted beyond token budget
- **Action results:** `OK` or `ERROR: <msg>` fed back so the LLM adapts (timeout, stale selector, user declined)
- **Element map:** observation lists entries like `[E7] button "Submit"`; LLM references `E7`; `browser.py` maps index → selector internally; observation is re-extracted immediately before dispatch to minimize staleness
- **Termination:** `done` with final `answer` string; max_steps=15 reports partial status; errors report clearly

## 4. LLM Schema & Prompt

**Action schema (Pydantic):**

```json
{
  "thought": "string — reasoning about current state",
  "action": "navigate | click | type | scroll | done | ask_user",
  "parameters": {
    "element": "E7 (index from observation; click/type)",
    "text": "text to type (type)",
    "url": "https://... (navigate)",
    "direction": "up|down (scroll, optional)",
    "question": "string (ask_user)",
    "answer": "final summary (done)"
  }
}
```

All parameters optional; validation failure triggers the repair path.

**System prompt sections:** (1) role as web navigator, (2) fixed user goal, (3) rules — one atomic action per response, reference element indices only, never invent selectors, `ask_user` when blocked, `done` only when goal verifiably achieved, (4) observation block (URL/title, element list, text excerpt), (5) recent history.

**Groq:** model `llama-3.3-70b-versatile` (overridable in `config.py`), `response_format={"type": "json_object"}` with parse fallback.

**Repair path:** invalid output → resend with pydantic error message and instruction to output only valid JSON → one retry → step fails with `ERROR` to loop.

## 5. Guardrails & Error Handling

**Sensitive detection (`is_sensitive`)** — if matched, call `on_confirm(prompt)` before dispatch; terminal default answer **N**; decline feeds `USER DECLINED` back to LLM:

| Signal | Detected via |
|---|---|
| Form submission | `type=submit` or button text ∈ {submit, sign in, log in, pay, purchase, buy, order, confirm, send} |
| Login | password field on page + click/type action |
| Purchase/payment | keyword ∈ {pay, purchase, buy, checkout, price} in target or page title |

**Failure handling:**

| Failure | Handling |
|---|---|
| Playwright timeout | `ERROR: timed out` to LLM; step counts |
| Invalid/stale element index | re-observe before dispatch; else `ERROR` to LLM |
| Groq API error | exponential backoff ×2, then abort turn with message |
| Unparseable JSON after repair | `ERROR` to LLM; step counts |
| max_steps=15 exhausted | stop; report steps + best-so-far status |
| Browser crash/page closed | abort cleanly, close session, tell user to restart |
| Ctrl+C in REPL | close Playwright cleanly, exit |

**Logging:** one console line per step; full transcript to `logs/session-<timestamp>.jsonl`.

## 6. Testing & Milestones

| Test | Scope | Needs |
|---|---|---|
| `test_llm_parse.py` | JSON repair: valid, fenced, trailing prose, truncated, bad enum | neither API nor browser |
| `test_browser.py` | element-map indices, click/type/navigate vs static fixture | Playwright headless |
| `test_agent.py` | scripted fake LLM + real headless browser: dispatch order, confirmation trigger, ask_user suspension, max-steps, done | Playwright headless |
| Manual smoke | one live Groq end-to-end run on a simple site | API key |

Live-Groq tests are manual/optional — excluded from default `pytest`.

**Milestone mapping:** M1 = `test_browser.py`; M2 = `test_llm_parse.py` + live parse; M3 = `test_agent.py` + live multi-step run; M4 = REPL end-to-end + observed confirmation prompt.

**Build order:** Phase 1 browser wrapper → Phase 2 schema/prompt → Phase 3 loop → Phase 4 guardrails → Phase 5 REPL, tests written alongside each phase (TDD).

## 7. Error Philosophy

No exception ever reaches the user as a traceback. Every failure becomes one of: a `ERROR` message the LLM can react to, a clear turn-abort message, or a clean shutdown instruction.
