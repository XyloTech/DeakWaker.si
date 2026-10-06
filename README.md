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

## Run the agent

1. Ensure Ollama is running (`ollama serve`) and the model is pulled: `ollama pull qwen3:latest`.
2. Optionally copy `.env.example` to `.env` to override `OLLAMA_HOST`, `MODEL`, `THINKING` (model reasoning; off by default for speed), `HEADLESS`, `USER_DATA_DIR`, or `SLOW_MO_MS`.
3. Run `python chat.py`.

A visible Chromium window opens by default so you can watch every action (`SLOW_MO_MS` defaults to 100 ms of human-like delay per action when headful). Logins/cookies persist across runs in `./user_data/` (gitignored) — sign into a site once in the agent's window and it stays signed in. Set `HEADLESS=1` for invisible runs (required in Codespaces/devcontainer, which have no display).

## Web UI

1. Run `python webui.py` (same setup as the terminal agent).
2. Open `http://127.0.0.1:8000`, type a goal, and press **Run**.

Every step streams live over SSE (thought, action, result, page state), and confirmation / `ask_user` prompts appear as cards you can answer in-page (Proceed/Decline/Send). Only one goal runs at a time; the terminal REPL (`chat.py`) keeps working as before.

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
