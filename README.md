# desk.waker

A terminal-chat autonomous browser agent: you type a natural-language goal, and a Llama model running locally on Ollama drives Playwright through an Observe → Think → Act loop until the task is done.

## Development environment

- **GitHub Codespaces:** repo page → **Code** → **Codespaces** → **Create codespace on main**. The container builds unattended (`postCreateCommand` installs dependencies and Chromium).
- **VS Code:** install the Dev Containers extension, open the folder, then **Reopen in Container**. Docker Desktop is required for local VS Code use only (Codespaces needs nothing installed).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

## Run tests

```
pytest -v
```

The live-LLM test skips unless `LIVE_LLM=1` is set (`LIVE_LLM=1 python -m pytest -m live -v`, requires Ollama running).

## Actions

- ask_user: ask the user a question; parameters.question
- click: click the element referenced as [En]
- done: finish the task; parameters.answer holds the final answer
- extract: pull structured data out of the page as JSON; parameters.query says what
- navigate: load a URL; parameters.url must be absolute (https:// or file://)
- new_tab: open a new tab, optionally at parameters.url
- screenshot: save a screenshot artifact for the user; no parameters
- scroll: scroll the page; parameters.direction is up or down
- select: choose an option in a dropdown; parameters.element + parameters.value
- switch_tab: make another tab active; parameters.tab is the zero-based index
- type: replace the text in an input with parameters.text
- upload: attach a local file to a file input; parameters.element + parameters.path
- wait: wait briefly for a dynamic page to settle; parameters.seconds is 0.1 to 10

## Run the agent

1. Ensure Ollama is running (`ollama serve`) and the model is pulled: `ollama pull qwen3:latest`.
2. Optionally copy `.env.example` to `.env` to override `OLLAMA_HOST`, `MODEL`, `THINKING` (model reasoning; off by default for speed), `HEADLESS`, `USER_DATA_DIR`, `SLOW_MO_MS`, `BROWSER_CHANNEL`, `BROWSER_EXECUTABLE`, `BROWSER_CDP_URL`, `BROWSER_ALLOW_FALLBACK`, `AUTO_CONFIRM`, `EXTRACT_QUERY`, `NEW_TAB_URL`, `SWITCH_TAB_INDEX`, or `EXTRACT_TIMEOUT`.
3. Run `python chat.py`.

A visible window of the explicitly selected browser opens by default (`BROWSER_CHANNEL=brave`). The agent reports the actual browser and profile used and does not silently switch to Chrome or another browser; set `BROWSER_CDP_URL=http://127.0.0.1:9222` to connect to a Brave window you opened and authenticated manually with remote debugging. Brave is detected from standard Windows, Linux, and WSL locations; in WSL, a Windows `.exe` is reported but is not launched by Linux Playwright, so install Linux Brave or use the Windows agent. `USER_DATA_DIR` selects the persistent profile, so manual sign-in is reused across runs. Fast mode defaults to `SLOW_MO_MS=0`, six history steps, a 1600-character page observation, and 80 interactive elements per Ollama prompt; omitted elements remain clickable after a later observation. Set `HEADLESS=1` for invisible runs. Sensitive actions require confirmation when `AUTO_CONFIRM=0`; keep that setting for real workflows.

## Web UI

1. Run `python webui.py` (same setup as the terminal agent).
2. Open `http://127.0.0.1:8000`, type a goal, and press **Run**.

Every step streams live over SSE (thought, action, result, page state), and `ask_user` prompts appear as cards you can answer in-page (confirmation cards appear only when `AUTO_CONFIRM=0`). Only one goal runs at a time; the terminal REPL (`chat.py`) keeps working as before.

## Project layout

- `config.py` — model name, max steps, timeouts, headless flag (env-overridable).
- `browser.py` — `BrowserWrapper`: async Playwright navigation, clicks, typing, element map.
- `llm.py` — Ollama client: builds prompts, parses and repairs JSON actions.
- `agent.py` — `run_turn()`: observe–think–act loop with guardrails and logging.
- `chat.py` — terminal REPL entry point.
- `webui.py` — FastAPI + SSE web UI entry point.
- `static/` — web UI frontend (single-page vanilla JS).
- `tools/model_bench.py` — model bake-off script.
- `tests/` — unit and integration tests with a static HTML fixture.

## Specs

- Blueprint: [`desk.waker.llm.md`](desk.waker.llm.md)
- Design specs: [`docs/superpowers/specs/`](docs/superpowers/specs/)
