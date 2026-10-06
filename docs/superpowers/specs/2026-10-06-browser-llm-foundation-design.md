# Browser + LLM Foundation — design (sub-project #1 of 4)

**Date:** 2026-10-06
**Status:** approved design; awaiting user review, then implementation planning.
**Context:** first sub-project of the four-way decomposition of the production
Browser-Use-style agent spec (2026-10-06). Sibling sub-projects: #2 agent
hardening, #3 trust layer (permissions/tools/memory), #4 control UI + docs.

## Goal

Give the agent the browser capabilities and LLM configuration a professional
local agent needs — new page actions, multi-tab awareness, HTTP-status-aware
observations, structured extraction, and full Ollama parameter control — built
on a thin action-registry skeleton that sub-projects #2–#4 will extend rather
than rework.

**Success criteria:**
- The model can `select`, `upload`, `screenshot`, `extract`, `new_tab`,
  `switch_tab`; downloads are captured automatically; all with model-facing
  coaching errors on misuse.
- Observations show HTTP status (401/403/404 hints), tab state, and recent
  downloads.
- `temperature`, `num_ctx`, per-call timeout, and reasoning are env-configurable.
- Full existing suite stays green; new tests cover every new surface.

## Non-goals (per the approved four-way split)

Recovery matrix behaviors (CAPTCHA, session expiry, redirect policies),
rate limiting, cancellation, structured logging → #2. Permission levels,
tool-system audit fields, memory/task-state, secrets handling → #3. UI
screenshots/pause/stop → #4. Vision observation stays out entirely (user
decision: text/DOM only); screenshots are artifacts for the human, never
model input.

## 1. Action registry (`actions.py`, new)

- `ActionSpec` dataclass: `name: str`, `description: str` (one line, fed to
  the prompt), `validate(ctx, message) -> str | None` (pre-dispatch param
  coaching; `None` = pass), `handler(ctx, message) -> str` (async; returns
  `"OK…"` or `"ERROR: …"`). Structured so #3 can add `permission`, `timeout`,
  `retry`, `audit` fields without reshaping entries.
- `ActionContext` dataclass passed to every handler: `browser`, `llm`
  (duck-typed; only `.extract` is used), `goal`, `step`, `max_steps`,
  `elements`, `page_info`, `on_ask`.
- `ACTIONS: dict[str, ActionSpec]` — entries for the existing six actions
  (`navigate`, `click`, `type`, `scroll`, `done`, `ask_user`) plus new.
  `done` is registered for prompt/validation purposes only: `run_turn`
  intercepts it **before** handler dispatch for the verify gate (unchanged);
  `ask_user`'s handler answers through `ctx.on_ask`. New actions:
  - `select` — `element` + `value` (new field): option value first, then
    visible label; mismatch → coaching listing available options.
  - `upload` — `element` + `path` (new field): `Path(path).is_file()` check
    first; any local path allowed (local-first). Uses `set_input_files`.
  - `screenshot` — no params: saves `artifacts/screenshots/step-{step}.png`
    (handler receives step via context).
  - `extract` — `query` (new field, required): page-text snapshot → dedicated
    LLM extraction call → history result `OK · <json>`; failure →
    `ERROR: extraction failed: <reason>`.
  - `new_tab` — optional `url`: opens a page, navigates if given.
  - `switch_tab` — new `tab` field (int index): bounds-checked, active page
    re-collected on next observation.
  - Downloads have **no action**: a listener auto-saves to `DOWNLOAD_DIR` and
    the next observation reports it (non-blocking by design).
- Import discipline: `actions.py` never imports `llm` at runtime (`llm` →
  `actions` must stay acyclic); `ActionMessage` referenced under
  `TYPE_CHECKING`. Shared helpers (`_normalize_url`, `ELEMENT_REF_PATTERN`,
  `_parse_element`) move from `agent.py` into `actions.py`; `agent` imports
  them from there.
- `llm.py`: `ActionMessage.action` widens to `str`; `parse_action` rejects
  names not in `ACTIONS` with `LLMOutputError` (existing repair path; the
  bad-enum test keeps passing). The system-prompt line
  `"Valid actions: …"` is generated from `ACTIONS` keys; every other static
  rule line stays byte-identical so prompt tests stay green.

## 2. Agent loop (`agent.py`)

Loop shape unchanged (observe → decide → validate → execute → capture →
verify-gate → loop). Dispatch becomes: parse → `spec = ACTIONS[message.action]`
→ `spec.validate` → guardrails (`is_sensitive`, `AUTO_CONFIRM`,
diagnostics suffix — all unchanged and automatically covering new actions)
→ `await spec.handler(ctx, message)`. Guardrail/diagnostics/verify-gate code
paths are not modified in this sub-project beyond extracting shared helpers.

## 3. Browser layer (`browser.py`)

- **HTTP status:** `navigate_to` captures the goto `Response` (final URL after
  redirects + status); a main-frame `response` listener keeps `self._http_status`
  fresh. Exposed via `get_page_info`.
- **Tabs:** track `self._pages`; `context.on("page")` registers new pages,
  `page.on("close")` drops them. `new_tab(url=None)`, `switch_tab(index)`,
  `tab_count`. Element indices always refer to the active page only.
- **Page listeners:** existing diagnostics attach is generalized to
  `_attach_page_listeners(page)` — console/pageerror/requestfailed/response +
  download listener — called for the initial page and every new tab.
- **Downloads:** `download` listener saves to `DOWNLOAD_DIR` (config,
  default `downloads/`, gitignored), sanitized filename, keeps last 10 in
  `self._downloads`; surfaced in `get_page_info`. Never blocks a click.
- **Select/upload:** `select_option(index, value)` (value then label) and
  `upload_file(index, path)` with existence check; both raise
  `BrowserActionError` with coaching text.
- **`get_page_info` additions:** `http_status: int | None`, `tabs: int`,
  `active_tab: int`, `recent_downloads: list[str]`.
- **`render_observation` (llm.py):** adds `HTTP: 404 — page not found; URL may
  be wrong or the page moved` style hint lines for 401/403/404/429/5xx,
  `Tabs: 2 (active 0)`, and `Downloads: …` when present.

## 4. Extract + Ollama configuration (`llm.py`, `config.py`)

- `EXTRACT_SYSTEM` prompt ("output ONLY valid JSON…"), builder
  `build_extract_messages(page_text, query)` with `page_text` capped at
  `OBSERVATION_TEXT_LIMIT`.
- `LLMClient.extract(messages) -> object` — reuses `_parse_with_repair` with a
  JSON parser; runs through `asyncio.to_thread` in the handler.
- Config (all env-overridable; `_env_float` helper added):
  `OLLAMA_TEMPERATURE=0.1`, `OLLAMA_NUM_CTX=4096`, `OLLAMA_TIMEOUT_S=300`,
  `DOWNLOAD_DIR=downloads`. `_chat_kwargs` adds
  `options={"temperature": …, "num_ctx": …}`; `LLMClient` passes
  `timeout=OLLAMA_TIMEOUT_S` to the Ollama `Client`. Existing kwargs tests
  only assert individual keys, so this is additive-compatible.

## 5. Testing (TDD, hermetic)

- New `tests/test_actions.py`: registry completeness (prompt list ==
  registry keys), unknown-action repair path, per-action validation coaching,
  dispatch of each new action against the fixture.
- Fixture work: `site.html` gains a `<select>` and `<input type=file>`
  **appended at the end of `<body>`** (existing index-based tests use
  `index_of` lookups and `[E0]` coaching assertions depend on the first
  element staying first); new `download.html` fixture with an `<a download>`
  link; status-serving HTTP handler variant for 401/403 (404 = missing file
  on the existing `SimpleHTTPRequestHandler` pattern).
- `test_browser.py`: select value/label/mismatch, upload missing-file error,
  HTTP status in `get_page_info`, tab tracking + diagnostics on new tabs,
  download auto-save into a tmp `DOWNLOAD_DIR`.
- `test_llm_*`: extract prompt content, `extract()` success + repair-then-raise,
  `_chat_kwargs` options, `_env_float`.
- Full suite green is the completion gate.

## 6. Docs

`.env.example` + README config table gain the four new vars; README lists the
agent's actions; this spec is committed; implementation plan follows via the
writing-plans skill.

## Contracts & compatibility

- Handler result strings keep the `"OK…"` / `"ERROR: …"` contract every
  existing test asserts.
- `ActionParameters` only gains optional fields — old payloads parse
  unchanged.
- Transcript/step/verify entries, `TurnResult`, SSE step events: unchanged
  schemas.
- `browser.close()` resets tabs/downloads state along with diagnostics.

## Risks

- **Prompt drift:** generated "Valid actions" line changes model-facing text;
  mitigated by keeping all other rules byte-identical and running the live
  smoke only as manual verification.
- **Download timing:** surfacing is observation-based (next step), not
  click-blocking — documented, deliberate.
- **Small-model JSON for extract:** may need the repair path frequently;
  reuse of `_parse_with_repair` keeps it graceful.
- **ollama client kwargs:** `timeout`/`options` support must be verified
  against the installed `ollama` package during implementation (test will pin
  it).
