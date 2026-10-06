# Browser + LLM Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the sub-project #1 spec: a thin action registry driving the agent loop, six new browser capabilities (select, upload, screenshot, extract, new_tab, switch_tab), auto-captured downloads, HTTP-status-aware observations, and full Ollama parameter configuration.

**Architecture:** `actions.py` becomes the single source of truth for actions (registry entries with name/description/validate/handler); `llm.py` validates model output against the registry and generates the prompt's action block from it; `browser.py` grows status/tab/download/select/upload support behind the existing `BrowserWrapper` façade; the `run_turn` loop keeps its shape and its guardrails, replacing `if/elif` dispatch with registry lookup.

**Tech Stack:** Python 3.10+, Playwright (async), Ollama local client, pydantic, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-browser-llm-foundation-design.md` (read it first; this plan argues from it)

## Global Constraints

- Local-first: Ollama only (`_ollama_complete`), no cloud LLM calls; model inference and browser data stay local.
- Observation mode is text/DOM only — screenshots are artifacts for the human, never model input (spec non-goal: vision).
- Handler result contract: every action returns a string starting `OK` or `ERROR: ` (existing tests assert exact strings listed per task; keep them verbatim).
- Tests stay hermetic: `tests/conftest.py` forces `HEADLESS=1`, `SLOW_MO_MS=0`, `BROWSER_CHANNEL=none`, `AUTO_CONFIRM=0`; default suite runs no live LLM.
- `requirements.txt` unchanged — no new packages.
- Existing static system-prompt rule sentences stay byte-identical (only the action block becomes generated).
- Fixture changes to `site.html` are append-only at end of `<body>` (index-based tests rely on `[E0]` staying first).
- Commit at the end of every task; do not commit anything else.

## Review Focus

(spec inputs no single task's tests fully pin — each line's owning test is named)

1. **Prompt drift from the generated action block** — the model's action list changes shape; all previously asserted rule sentences must survive. Test: `test_prompt_keeps_static_rule_sentences` (Task 7).
2. **Stale-element guard must cover the new element actions** — select/upload firing on a changed DOM should return the existing `ERROR: element changed` string, not act blindly. Test: `test_select_stale_index_returns_error_to_llm` (Task 8).
3. **Download save is async** — click returns before the file lands; observation must eventually report it and the file must exist on disk. Test: `test_click_download_is_saved_and_reported` (Task 5, condition-based wait).
4. **Extract failure must not kill the turn** — a broken extraction returns `ERROR: extraction failed: …` and the loop continues. Test: `test_extract_failure_feeds_error_and_loop_continues` (Task 9).
5. **Installed `ollama` package kwargs compatibility** — `Client(timeout=…)` and `chat(options=…)` must actually be accepted. Test: `test_ollama_complete_passes_timeout_and_options` (Task 1, fake client records kwargs).

---

### Task 1: Ollama configuration knobs

**Files:**
- Modify: `config.py` (add `_env_float` + three settings)
- Modify: `llm.py` (`_chat_kwargs`, `_ollama_complete`, `LLMClient.__init__`, move `from ollama import Client` to module top)
- Modify: `.env.example`
- Test: `tests/test_config.py` (new), `tests/test_llm_client.py`

**Interfaces:**
- Consumes: `config._env_int`/`_env_str` patterns.
- Produces: `config._env_float(name: str, default: float) -> float`; `config.OLLAMA_TEMPERATURE: float` (default `0.1`), `config.OLLAMA_NUM_CTX: int` (default `4096`), `config.OLLAMA_TIMEOUT_S: float` (default `300.0`); `llm.Client` (module-level import); `_chat_kwargs(model, messages)` now includes `"options": {"temperature": …, "num_ctx": …}`; `LLMClient()` passes `timeout=OLLAMA_TIMEOUT_S` into `Client(host=…, timeout=…)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
import config

def test_env_float_parses_decimal(monkeypatch):
    monkeypatch.setenv("FLOAT_X", "0.25")
    assert config._env_float("FLOAT_X", 0.1) == 0.25

def test_env_float_missing_or_invalid_returns_default(monkeypatch):
    monkeypatch.delenv("FLOAT_X", raising=False)
    assert config._env_float("FLOAT_X", 0.1) == 0.1
    monkeypatch.setenv("FLOAT_X", "junk")
    assert config._env_float("FLOAT_X", 0.1) == 0.1
```

In `tests/test_llm_client.py` add:

```python
def test_chat_kwargs_include_temperature_and_context(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_TEMPERATURE", 0.7)
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 8192)
    kwargs = llm._chat_kwargs("qwen3:latest", [{"role": "user", "content": "hi"}])
    assert kwargs["options"] == {"temperature": 0.7, "num_ctx": 8192}

def test_ollama_complete_passes_timeout_and_options(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *, host, timeout):
            captured.update(host=host, timeout=timeout)
        def chat(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"message": {"content": json.dumps(VALID_PAYLOAD)}}

    monkeypatch.setattr(llm, "Client", FakeClient)
    monkeypatch.setattr(llm, "OLLAMA_TIMEOUT_S", 42.0)
    monkeypatch.setattr(llm, "OLLAMA_TEMPERATURE", 0.7)
    monkeypatch.setattr(llm, "OLLAMA_NUM_CTX", 8192)
    client = llm.LLMClient()
    result = client.next_action([{"role": "user", "content": "x"}])
    assert result.action == "navigate"
    assert captured["timeout"] == 42.0
    assert captured["kwargs"]["options"] == {"temperature": 0.7, "num_ctx": 8192}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_config.py tests/test_llm_client.py -q`
Expected: FAIL — `_env_float` missing, `options` absent, `Client` not module-level.

- [ ] **Step 3: Implement**

`config.py`: `_env_float` = try `float(value)` except (`TypeError`, `ValueError`) → default, same empty-string-to-default convention as `_env_str`. Add the three settings with the spec's exact names/defaults after `THINKING`.

`llm.py`: hoist `from ollama import Client` to module top; `_chat_kwargs` adds `"options": {"temperature": OLLAMA_TEMPERATURE, "num_ctx": OLLAMA_NUM_CTX}` (import both from `config` alongside existing names); `_ollama_complete(host, model, messages, timeout)` builds `Client(host=host, timeout=timeout)`; `LLMClient.__init__` gains keyword `timeout: float = OLLAMA_TIMEOUT_S` and its default lambda passes it.

`.env.example`: append the three vars with one-line comments (defaults as above).

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_config.py tests/test_llm_client.py -q`
Expected: PASS (all, including pre-existing chat-kwargs tests).

- [ ] **Step 5: Commit**

```bash
git add config.py llm.py .env.example tests/test_config.py tests/test_llm_client.py
git commit -m "feat: configurable ollama temperature, context, and timeout"
```

---

### Task 2: HTTP status awareness

**Files:**
- Modify: `browser.py` (status capture + main-frame listener + `get_page_info`)
- Modify: `llm.py` (`render_observation`)
- Test: `tests/test_browser.py`, `tests/test_llm_client.py`

**Interfaces:**
- Produces: `browser.BrowserWrapper._http_status: int | None`; `get_page_info()["http_status"]`; observation line `HTTP: {status} — {hint}` where hints are exactly: 401 → `authentication required; a signed-in session may be needed`, 403 → `access forbidden; the site is blocking this session`, 404 → `page not found; URL may be wrong or the page moved`, 429 → `rate limited by the site; wait before retrying`, 5xx → `server error; the site may be temporarily down` (no HTTP line when status is `None`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_browser.py`, add a shared helper (reused by Task 5) and the status tests:

```python
@asynccontextmanager
async def serve_fixtures(handler_cls=None):
    import functools, http.server, threading
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
    cls = handler_cls or Quiet
    handler = functools.partial(cls, directory=str(pathlib.Path(__file__).parent / "fixtures"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)

class StatusHandler:  # 401/403 paths on top of fixture files
    ...
```

(`StatusHandler` subclasses the quiet handler: `do_GET` returns `send_error(401)` for `/auth`, `send_error(403)` for `/blocked`, else delegates to `super().do_GET()`.)

```python
def test_page_info_reports_http_status_200_and_404():
    async def scenario():
        async with serve_fixtures() as origin:
            async with open_browser() as browser:
                await browser.navigate_to(f"{origin}/site.html")
                assert (await browser.get_page_info())["http_status"] == 200
                await browser.navigate_to(f"{origin}/missing.html")
                assert (await browser.get_page_info())["http_status"] == 404
    asyncio.run(scenario())

def test_status_listener_tracks_in_page_navigation():
    async def scenario():
        async with serve_fixtures() as origin:
            async with open_browser() as browser:
                await browser.navigate_to(f"{origin}/site.html")
                await browser._page.evaluate("window.location = '/missing.html'")
                await wait_for(lambda: True)  # replaced below by condition
                # real assertion: poll get_page_info until status != 200
    asyncio.run(scenario())
```

(Write the real polling with the existing `wait_for` helper and a lambda reading `browser._http_status == 404`.)

In `tests/test_llm_client.py`:

```python
def test_render_observation_includes_http_hint():
    page = {"url": "https://x/404", "title": "t", "text": "", "http_status": 404}
    out = llm.render_observation(page, [])
    assert "HTTP: 404 — page not found; URL may be wrong or the page moved" in out

def test_render_observation_omits_http_line_when_unknown():
    out = llm.render_observation(PAGE_INFO, ELEMENTS)  # no http_status key
    assert "HTTP:" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py -k http tests/test_llm_client.py -k "render_observation" -q`
Expected: FAIL — `http_status` never set, no hint line.

- [ ] **Step 3: Implement**

`browser.py`: `self._http_status: int | None = None` in `__init__`; `navigate_to` assigns `response.status if response else None` from `page.goto`; a `response` listener attached in `_attach_diagnostics` (renamed in Task 3 — for now attach alongside): fire when `response.request.is_navigation_request()` and `response.request.frame == page.main_frame`; `get_page_info` returns `"http_status": self._http_status`.

`llm.py` `render_observation`: after the Title line, if `page_info.get("http_status")` is not None, append the `HTTP: {status} — {hint}` line using the exact hint map above (5xx prefix-match for ≥ 500).

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py tests/test_llm_client.py -q`
Expected: PASS (suite files, all existing included).

- [ ] **Step 5: Commit**

```bash
git add browser.py llm.py tests/test_browser.py tests/test_llm_client.py
git commit -m "feat: http status in page info and observation hints"
```

---

### Task 3: Multi-tab tracking

**Files:**
- Modify: `browser.py` (`_pages`, page/close listeners, `new_tab`, `switch_tab`, diagnostics rename, `get_page_info`)
- Modify: `llm.py` (`render_observation` tabs line)
- Test: `tests/test_browser.py`

**Interfaces:**
- Produces: `browser.BrowserWrapper.tab_count -> int`; `async new_tab(url: str) -> None` (url absolute — callers normalize); `async switch_tab(index: int) -> None`; `get_page_info()["tabs"]: int`, `["active_tab"]: int` (0-based); observation line always present: `Tabs: {n} (active {i})`. `_attach_diagnostics` is renamed to `_attach_page_listeners(page)` (Task 5 adds the download listener there); it is called for the initial page, every `new_tab`, and every page raised by the context `page` event (registration via an idempotent `_register_page`).

- [ ] **Step 1: Write the failing tests**

```python
def test_new_tab_switch_and_close_tracking():
    async def scenario():
        async with open_browser() as browser:
            info = await browser.get_page_info()
            assert info["tabs"] == 1 and info["active_tab"] == 0
            await browser.new_tab("about:blank")
            assert browser.tab_count == 2
            await browser.switch_tab(0)
            assert (await browser.get_page_info())["active_tab"] == 0
            await browser._pages[1].close()
            await wait_for(lambda: browser.tab_count == 1)
    asyncio.run(scenario())

def test_diagnostics_attach_to_new_tabs():
    async def scenario():
        async with open_browser() as browser:
            await browser.new_tab(ERRORS_FIXTURE_URL)
            await wait_for(lambda: any("kapow" in d for d in browser.get_diagnostics()))
    asyncio.run(scenario())

def test_switch_tab_out_of_range_reports_coaching():
    async def scenario():
        async with open_browser() as browser:
            with pytest.raises(BrowserActionError, match="tab"):
                await browser.switch_tab(7)
    asyncio.run(scenario())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py -k "tab" -q`
Expected: FAIL — attributes/methods missing.

- [ ] **Step 3: Implement**

`browser.py`: `_pages: list[Page]`, idempotent `_register_page(page)` (append if absent + `_attach_page_listeners(page)` + `page.on("close", lambda: self._pages.remove(page) if page in self._pages else None)`); context `page` event in `start()`; rename `_attach_diagnostics` → `_attach_page_listeners` (behavior unchanged); `new_tab(url)`: `page = await context.new_page()` → `_register_page` → `page.goto(url)` (wrap `PlaywrightError` → `BrowserActionError`); `switch_tab(i)`: bounds → `BrowserActionError(f"no tab {i}; {n} tabs open")`, else `self._page = self._pages[i]`; `tab_count` property; `get_page_info` gains `tabs`/`active_tab`; `close()` resets `_pages`.

`llm.py` `render_observation`: append `Tabs: {int(page_info.get('tabs', 1))} (active {int(page_info.get('active_tab', 0))})`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add browser.py llm.py tests/test_browser.py
git commit -m "feat: multi-tab tracking with per-page listeners"
```

---

### Task 4: Select and upload browser methods

**Files:**
- Modify: `browser.py` (`select_option`, `upload_file`)
- Modify: `tests/fixtures/site.html` (append `<select>` + `<input type=file>` at end of `<body>`)
- Test: `tests/test_browser.py`

**Interfaces:**
- Produces: `async select_option(index: int, value: str) -> None` (matches option value, falls back to visible label; mismatch raises `BrowserActionError` whose message lists `available:` options); `async upload_file(index: int, path: str) -> None` (missing file raises `BrowserActionError` with `no such file: {path}`).

- [ ] **Step 1: Append fixture elements and write the failing tests**

Fixture (before `</body>`): `<select id="colors"><option value="r">Red</option><option value="g">Green</option><option value="b">Blue</option></select>` and `<input type="file" id="uploader">`.

```python
def test_select_option_by_value_and_label():
    async def scenario():
        async with open_browser() as browser:
            await browser.navigate_to(FIXTURE_URL)
            idx = index_of(browser._elements or await browser.get_interactive_elements(), selector="#colors")
            await browser.select_option(idx, "g")
            # assert selected via evaluate on the active page
            selected = await browser._page.evaluate("document.querySelector('#colors').value")
            assert selected == "g"
            await browser.select_option(idx, "Blue")
            selected = await browser._page.evaluate("document.querySelector('#colors').value")
            assert selected == "b"
    asyncio.run(scenario())

def test_select_option_mismatch_lists_available():
    ...  # pytest.raises(BrowserActionError, match="available") on value "nope"

def test_upload_missing_file_reports_no_such_file():
    ...  # pytest.raises(BrowserActionError, match="no such file")

def test_upload_file_lands_in_input():
    ...  # tmp_path file; after upload_file, evaluate input.files[0].name == name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py -k "select or upload" -q`
Expected: FAIL — methods missing.

- [ ] **Step 3: Implement**

`select_option`: resolve selector; try `page.select_option(selector, value=value, timeout=ACTION_TIMEOUT_MS)`; on `PlaywrightError`, re-read options via `page.evaluate("sel => Array.from(document.querySelector(sel).options).map(o => o.value + '|' + o.text)", selector)` and raise `BrowserActionError(f"select_option({index}) failed: no option matching {value!r}; available: {opts}")`; if that also fails, re-raise a plain `BrowserActionError`. Second path for label matching: on value-miss, retry `select_option(..., label=value)` before declaring failure (Playwright's `select_option` accepts `label=`).

`upload_file`: `Path(path).is_file()` guard → `BrowserActionError(f"upload_file: no such file: {path}")`; else `page.set_input_files(selector, str(path), timeout=ACTION_TIMEOUT_MS)` wrapped in the existing `BrowserActionError` style.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py -q`
Expected: PASS; also confirm `tests/test_agent.py -q` still green (fixture gained elements).

- [ ] **Step 5: Commit**

```bash
git add browser.py tests/fixtures/site.html tests/test_browser.py
git commit -m "feat: select option and file upload browser methods"
```

---

### Task 5: Automatic downloads

**Files:**
- Modify: `config.py` (`DOWNLOAD_DIR`), `browser.py` (listener, `recent_downloads`), `llm.py` (`render_observation`), `.env.example`, `.gitignore`
- Test: `tests/test_browser.py`

**Interfaces:**
- Consumes: `serve_fixtures` helper (Task 2), `_attach_page_listeners` (Task 3).
- Produces: `config.DOWNLOAD_DIR: str` (default `"downloads"`; tests monkeypatch `browser_mod.DOWNLOAD_DIR`); `get_page_info()["recent_downloads"]: list[str]` (last saved paths); observation lines `Downloads: {name} → {path}`.

- [ ] **Step 1: Write the failing tests**

New fixture `tests/fixtures/download.html`: `<a id="dl" download href="/site.html">Get it</a>`.

```python
def test_click_download_is_saved_and_reported():
    async def scenario():
        async with serve_fixtures() as origin, tempfile context DOWNLOAD_DIR:
            monkeypatch browser_mod.DOWNLOAD_DIR to tmp dir
            async with open_browser() as browser:
                await browser.navigate_to(f"{origin}/download.html")
                idx = index_of(await browser.get_interactive_elements(), text="Get it")
                await browser.click(idx)
                await wait_for(lambda: (await browser.get_page_info())["recent_downloads"])
    asyncio.run(scenario())
```

(Write it with a sync lambda over `browser._downloads` — condition-based wait — then assert the file exists at the recorded path and `get_page_info()["recent_downloads"]` is non-empty.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py -k download -q`
Expected: FAIL — no listener / key.

- [ ] **Step 3: Implement**

`config.py`: `DOWNLOAD_DIR: str = _env_str("DOWNLOAD_DIR", "downloads")`. `browser.py`: in `start()` `Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)`; `self._downloads: list[str] = []`; in `_attach_page_listeners`: `page.on("download", on_download)` where the async handler saves `await download.save_as(str(Path(DOWNLOAD_DIR) / Path(download.suggested_filename).name))` and appends the path (cap `self._downloads` to last 10). `get_page_info` gains `recent_downloads: list(self._downloads[-3:])`; `close()` clears it. `llm.render_observation` appends `Downloads: {Path(p).name} → {p}` lines when the list is non-empty. `.env.example` + `.gitignore` (`downloads/`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_browser.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py browser.py llm.py .env.example .gitignore tests/fixtures/download.html tests/test_browser.py
git commit -m "feat: automatic download capture with observation reporting"
```

---

### Task 6: Extract LLM path

**Files:**
- Modify: `llm.py` (`EXTRACT_SYSTEM`, `build_extract_messages`, `_parse_json_value`, `LLMClient.extract`)
- Test: `tests/test_llm_parse.py`

**Interfaces:**
- Consumes: `LLMClient._parse_with_repair` (exists).
- Produces: `llm.EXTRACT_SYSTEM: str`; `build_extract_messages(page_text: str, query: str) -> list[dict]` (page_text truncated to `OBSERVATION_TEXT_LIMIT`); `LLMClient.extract(messages: list[dict]) -> object` (any JSON value; raises `LLMOutputError` after repair).

- [ ] **Step 1: Write the failing tests**

```python
def test_build_extract_messages_caps_text_and_carries_query():
    long_text = "x" * (llm.OBSERVATION_TEXT_LIMIT + 500)
    msgs = build_extract_messages(long_text, "prices as JSON")
    blob = "\n".join(m["content"] for m in msgs)
    assert "prices as JSON" in blob
    assert llm.OBSERVATION_TEXT_LIMIT * "x"[:0] or long_text[:llm.OBSERVATION_TEXT_LIMIT] in blob
    assert long_text[:llm.OBSERVATION_TEXT_LIMIT + 10] not in blob

def test_extract_returns_parsed_json():
    client = LLMClient(complete_fn=lambda m: json.dumps({"total": 9}))
    assert client.extract([{"role": "user", "content": "x"}]) == {"total": 9}

def test_extract_recovers_json_behind_prose():
    calls = []
    def fn(m):
        calls.append(m)
        return 'Sure!\n{"total": 9}' if len(calls) == 1 else json.dumps({"total": 9})
    ...

def test_extract_repairs_then_raises():
    client = LLMClient(complete_fn=lambda m: "no json here")
    with pytest.raises(LLMOutputError):
        client.extract([{"role": "user", "content": "x"}])
```

(Fix the `in blob` assertion to the simple two-line form: prefix in blob, longer slice not in blob.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_llm_parse.py -q`
Expected: FAIL — names missing.

- [ ] **Step 3: Implement**

In `llm.py` beside the action prompt: `EXTRACT_SYSTEM` = `"You extract structured data from web page content. Output ONLY valid JSON matching the user's request. No prose, no markdown fences."`; `build_extract_messages` returns `[system, {"role": "user", "content": f"Request: {query}\n\nPage content:\n{page_text[:OBSERVATION_TEXT_LIMIT]}"}]`; `_parse_json_value(raw)` = `_loads` with the `_extract_first_json_block` fallback (returns any JSON value); `LLMClient.extract` = `self._parse_with_repair(messages, _parse_json_value)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_llm_parse.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm_parse.py
git commit -m "feat: llm extraction path with json repair"
```

---

### Task 7: Action registry + loop migration

**Files:**
- Create: `actions.py`
- Modify: `llm.py` (`ActionMessage.action: str`, registry validation, generated action block)
- Modify: `agent.py` (registry dispatch; delete `_dispatch`; import shared helpers from `actions`)
- Test: `tests/test_actions.py` (new); update `_normalize_url` imports in `tests/test_agent.py`

**Interfaces:**
- Consumes: browser methods (existing), `ActionParameters`/`ActionMessage` (llm).
- Produces (Task 8/9 build on these exact names):
  - `actions.ELEMENT_REF_PATTERN`, `actions.parse_element(reference: str | None) -> int | None`, `actions.normalize_url(url: str) -> str` (moved from `agent.py`, renamed without underscore)
  - `actions.ActionContext(browser, llm, goal, step, max_steps, elements, page_info, on_ask)` (dataclass; `llm`/`on_ask` typed `Any`/`Callable | None`)
  - `actions.ActionSpec(name: str, description: str, requires_element: bool, handler: Callable[[ActionContext, ActionMessage], Awaitable[str]], validate: Callable[[ActionContext, ActionMessage], str | None] | None = None)`
  - `actions.ACTIONS: dict[str, ActionSpec]` containing exactly `navigate, click, type, scroll, done, ask_user` after this task
  - `llm.parse_action` raises `LLMOutputError` for actions not in `ACTIONS`; system prompt contains `- Actions:` block generated from registry (`name: description` per line)

- [ ] **Step 1: Write the failing tests**

`tests/test_actions.py`:

```python
from actions import ACTIONS, ActionContext, ActionSpec, normalize_url, parse_element
from llm import build_messages, parse_action
import pytest, json

EXPECTED_CORE = {"navigate", "click", "type", "scroll", "done", "ask_user"}

def test_registry_contains_core_actions():
    assert EXPECTED_CORE <= set(ACTIONS)

def test_prompt_lists_every_registry_action():
    system = build_messages("g", "obs", [])[0]["content"]
    for spec in ACTIONS.values():
        assert spec.name in system
        assert spec.description in system
    assert "navigate" in system and "switch_tab" not in system  # Task 9 flips this

def test_prompt_keeps_static_rule_sentences():
    system = build_messages("g", "obs", [])[0]["content"]
    for sentence in (
        "address bar", "Never retry an approach", "must be absolute",
        "Diagnose before retrying", "one-time codes", "Verify the outcome against the goal",
        "done only when the goal is verifiably achieved",
    ):
        assert sentence in system

def test_parse_action_rejects_action_not_in_registry():
    raw = json.dumps({"thought": "x", "action": "teleport"})
    with pytest.raises(LLMOutputError):
        parse_action(raw)

def test_normalize_url_lives_in_actions():
    assert normalize_url("youtube.com") == "https://youtube.com"
```

(Keep the existing `test_parse_action_rejects_bad_enum` — it must stay green through the registry check. In `tests/test_agent.py`, change the two `from agent import _normalize_url` tests to `from actions import normalize_url` and use that name.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_actions.py -q`
Expected: FAIL — `actions` module missing.

- [ ] **Step 3: Implement `actions.py` with the six core entries**

- Move `ELEMENT_REF_PATTERN`, `SCHEME_PATTERN`, element parsing and URL normalization here (public names `parse_element`, `normalize_url`); `agent.py` imports them.
- `ActionContext`/`ActionSpec` as specified; `ACTIONS` a plain dict literal of six entries with these descriptions:
  - navigate: `load a URL; parameters.url must be absolute (https:// or file://)`
  - click: `click the element referenced as [En]`
  - type: `replace the text in an input with parameters.text`
  - scroll: `scroll the page; parameters.direction is up or down`
  - done: `finish the task; parameters.answer holds the final answer`
  - ask_user: `ask the user a question; parameters.question`
- Handlers receive `(ctx, message)` and return result strings; move the verbatim strings from `_dispatch` into handlers/validates:
  - navigate validate: `ERROR: navigate requires parameters.url, e.g. https://example.com`
  - type validate: `ERROR: type requires parameters.text, e.g. "Ada"`
  - ask_user handler: existing `question or ""` + await-if-awaitable reply logic (uses `ctx.on_ask`).
  - done handler: returns `"OK"` — never called; `run_turn` intercepts `done` before dispatch (leave that branch exactly where it is).
- `requires_element`: click/type True; others False.

- [ ] **Step 4: Migrate `llm.py` and `agent.py`**

`llm.py`: `action: str` on `ActionMessage`; `parse_action` after `model_validate`: `if data.get("action") not in ACTIONS: raise LLMOutputError(f"Unknown action {action!r}; valid: {', '.join(ACTIONS)}")`; import `from actions import ACTIONS` at module top (actions never imports llm at runtime — `TYPE_CHECKING` only). Replace the static `"- Valid actions: …"` line in `SYSTEM_RULES` with a generated block built in `_system_prompt` from `ACTIONS` (`"- Actions:\n" + "".join(f"  - {n}: {s.description}\n" …)`).

`agent.py`: delete `_dispatch`; in the loop build `ctx = ActionContext(browser=browser, llm=llm, goal=goal, step=step, max_steps=max_steps, elements=elements, page_info=page_info, on_ask=on_ask)`; then the existing shell, registry-shaped:

```python
if spec.requires_element and index is None:
    result = (existing invalid-element coaching string, verbatim)
elif (param_error := (spec.validate(ctx, message) if spec.validate else None)):
    result = param_error
else:
    reason = is_sensitive(message, page_info, elements)      # unchanged guardrail
    approved = True if AUTO_CONFIRM or reason is None else on_confirm(...)  # existing logic
    if not approved:
        result = "ERROR: USER DECLINED this action"
    else:
        if index is not None:                                 # stale check, verbatim strings
            found = next((e for e in await browser.get_interactive_elements() if e["index"] == index), None)
            ... "ERROR: element changed since observation ..."
        try:
            result = await spec.handler(ctx, message)
        except BrowserActionError as exc:
            result = f"ERROR: {exc}"
if result.startswith("ERROR"):                                # diagnostics suffix, unchanged
    ...
```

Preserve verbatim: `ERROR: invalid element reference … you can only reference elements from the observation: {valid}`, `ERROR: element changed since observation (expected …, found …)`, `ERROR: USER DECLINED this action`.

- [ ] **Step 5: Run the new tests, then the full suite**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_actions.py -q` → PASS
Run: `& ".venv\Scripts\python.exe" -m pytest -q` → PASS (124+ baseline; any failure = a contract string drifted — fix the code, not the test, unless the test imports a moved helper).

- [ ] **Step 6: Commit**

```bash
git add actions.py agent.py llm.py tests/test_actions.py tests/test_agent.py
git commit -m "feat: action registry drives prompt, validation, and dispatch"
```

---

### Task 8: select, upload, screenshot actions

**Files:**
- Modify: `actions.py` (three entries), `llm.py` (`ActionParameters` gains `value`, `path`)
- Test: `tests/test_actions.py`

**Interfaces:**
- Consumes: `browser.select_option`, `browser.upload_file` (Task 4), `browser.take_screenshot` (exists).
- Produces: `actions.SCREENSHOT_DIR: str = "artifacts/screenshots"` (tests monkeypatch); parameter fields `ActionParameters.value: str | None`, `.path: str | None`; result string of screenshot: `OK · screenshot saved: {path}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_dispatch_selects_fixture_dropdown():
    # ScriptedLLM [select E? value g, done] on FIXTURE_URL; after run_turn,
    # browser._page.evaluate("document.querySelector('#colors').value") == "g"

def test_select_without_value_returns_coaching():
    # validate string: assert "select requires parameters.value" in last_blob(llm.calls[1])

def test_upload_missing_file_feeds_error_to_llm():
    # upload with path "/definitely/not/here.pdf" → next call blob has "no such file"

def test_screenshot_action_saves_artifact(tmp_path, monkeypatch):
    # monkeypatch actions.SCREENSHOT_DIR to tmp_path; script [screenshot, done];
    # (tmp_path / "step-1.png").exists() and transcript[0]["result"].startswith("OK · screenshot saved:")
```

(Reuse `ScriptedLLM`, `make_action`, `FIXTURE_URL`, `index_of` from `test_agent`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_actions.py -q`
Expected: FAIL — actions unknown (`parse_action` rejects them) / fields missing.

- [ ] **Step 3: Implement**

`llm.ActionParameters`: add `value: str | None = None`, `path: str | None = None`.
`actions.py` entries (descriptions exactly):
- select: `choose an option in a dropdown; parameters.element + parameters.value`
- upload: `attach a local file to a file input; parameters.element + parameters.path`
- screenshot: `save a screenshot artifact for the user; no parameters`
`validate` strings: `ERROR: select requires parameters.value, e.g. "g"`; `ERROR: upload requires parameters.path (absolute path to a local file)`.
Handlers: select → `browser.select_option(index, value)` (element required — `requires_element=True`); upload → `browser.upload_file(index, path)` (`requires_element=True`); screenshot → `browser.take_screenshot(str(Path(SCREENSHOT_DIR) / f"step-{ctx.step}.png"))` (mkdir parents first), return `f"OK · screenshot saved: {path}"`. Wrap `BrowserActionError` handling in the run_turn shell (Task 7) — handlers let it propagate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_actions.py tests/test_agent.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add actions.py llm.py tests/test_actions.py
git commit -m "feat: select, upload, and screenshot actions"
```

---

### Task 9: extract, new_tab, switch_tab actions + README

**Files:**
- Modify: `actions.py` (three entries), `llm.py` (`ActionParameters` gains `query`, `tab`), `README.md`
- Test: `tests/test_actions.py`; `ScriptedLLM` in `tests/test_agent.py` gains `extract`

**Interfaces:**
- Consumes: `build_extract_messages`/`LLMClient.extract` (Task 6), `browser.new_tab`/`switch_tab` (Task 3), `normalize_url` (Task 7).
- Produces: `ActionParameters.query: str | None`, `.tab: str | None`; extract result `OK · {json.dumps(value, ensure_ascii=False)}`; error `ERROR: extraction failed: {exc}`; `ScriptedLLM.extract(messages) -> object` recording into `extract_calls` (default `{"ok": True}` when unscripted).

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_dispatch_returns_json_to_history():
    # ScriptedLLM(extract_value={"prices": [1, 2]}) script [extract query, done];
    # transcript[0]["result"] == 'OK · {"prices": [1, 2]}'
    # llm.extract_calls[0] blob contains "prices as JSON" and the page URL text

def test_extract_failure_feeds_error_and_loop_continues():
    # ScriptedLLM(extract_error=LLMOutputError("LLM API unavailable: down"));
    # script [extract, done] → status "done"; transcript[0]["result"].startswith("ERROR: extraction failed")

def test_extract_without_query_returns_coaching():
    # blob of next call contains "extract requires parameters.query"

def test_new_tab_and_switch_tab_dispatch():
    # script [new_tab url FIXTURE_URL, switch_tab tab "0", done] → real browser;
    # result.status == "done"; browser.tab_count back... assert via final page_info tabs == 2

def test_switch_tab_invalid_index_returns_coaching():
    # tab "9" → next blob contains "no tab"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_actions.py -q`
Expected: FAIL — actions rejected.

- [ ] **Step 3: Implement**

`llm.ActionParameters`: `query: str | None = None`, `tab: str | None = None`.
`actions.py` entries (descriptions exactly):
- extract: `pull structured data out of the page as JSON; parameters.query says what`
- new_tab: `open a new tab, optionally at parameters.url`
- switch_tab: `make another tab active; parameters.tab is the zero-based index`
validate strings: `ERROR: extract requires parameters.query (what to extract, e.g. "prices as JSON")`; `ERROR: switch_tab requires parameters.tab as a zero-based index`.
Handlers:
- extract: `info = await ctx.browser.get_page_info()` → `value = await asyncio.to_thread(ctx.llm.extract, build_extract_messages(info["text"], query))` → `return "OK · " + json.dumps(value, ensure_ascii=False)`; `except LLMOutputError as exc: return f"ERROR: extraction failed: {exc}"`.
- new_tab: `await ctx.browser.new_tab(normalize_url(url) if url else None)` → handler signature `async def` returns `"OK"`; when no url, `browser.new_tab(None)` opens blank (Task 3 signature takes `str | None` — keep).
- switch_tab: `int(tab)` (validate pre-checks parseability) → `await ctx.browser.switch_tab(i)` → `"OK"`.

`ScriptedLLM` (tests/test_agent.py): add `extract_calls` list, `extract_value`/`extract_error` constructor kwargs, `extract(messages)` method (record; raise `extract_error` if set; else return `extract_value or {"ok": True}`).

README: add a short **Actions** list (registry names + one-liners) and the four new env vars to the config paragraph.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_actions.py -q` → PASS
Run: `& ".venv\Scripts\python.exe" -m pytest -q` → PASS (full suite).

- [ ] **Step 5: Commit**

```bash
git add actions.py llm.py README.md tests/test_actions.py tests/test_agent.py
git commit -m "feat: extract, new tab, and switch tab actions"
```

---

### Task 10: Final verification

**Files:** none new — verification only.

- [ ] **Step 1: Full suite green**

Run: `& ".venv\Scripts\python.exe" -m pytest -q`
Expected: all PASS, 1 skipped (live LLM). Report any failure by name before proceeding.

- [ ] **Step 2: Registry smoke**

Run: `& ".venv\Scripts\python.exe" -c "from actions import ACTIONS; print(sorted(ACTIONS))"`
Expected: exactly the 12 names: ask_user, click, done, extract, navigate, new_tab, screenshot, scroll, select, switch_tab, type, upload.

- [ ] **Step 3: Real-Chrome launch smoke**

Run: `& ".venv\Scripts\python.exe" -c "import asyncio; from browser import BrowserWrapper; exec('async def m():\n b=BrowserWrapper()\n await b.start()\n print((await b._page.evaluate(\"navigator.userAgent\")))\n await b.close()\nasyncio.run(m())')"`
Expected: UA contains `Chrome/` (installed Chrome on Windows).

- [ ] **Step 4: Commit any residual test/doc fixes (if Steps 1–3 produced fixes)**

```bash
git add -u && git commit -m "chore: final fixes for browser+llm foundation plan" --allow-empty
```
