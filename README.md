# desk.waker

A terminal-chat autonomous browser agent: you type a natural-language goal, and a Llama model running locally on Ollama drives Playwright through an Observe → Think → Act loop until the task is done.

## Development environment

- **GitHub Codespaces:** repo page → **Code** → **Codespaces** → **Create codespace on main**. The container builds unattended (`postCreateCommand` installs dependencies and Chromium).
- **VS Code:** install the Dev Containers extension, open the folder, then **Reopen in Container**. Docker Desktop is required for local VS Code use only (Codespaces needs nothing installed).

## Run tests

```
pytest -v
```

The live-LLM test skips unless `LIVE_LLM=1` is set (`LIVE_LLM=1 python -m pytest -m live -v`, requires Ollama running).

## Run the agent

1. Ensure Ollama is running (`ollama serve`) and the model is pulled: `ollama pull llama3.1:8b`.
2. Optionally copy `.env.example` to `.env` to override `OLLAMA_HOST` or `MODEL`.
3. Run `python chat.py`.

## Project layout

- `config.py` — model name, max steps, timeouts, headless flag (env-overridable).
- `browser.py` — `BrowserWrapper`: async Playwright navigation, clicks, typing, element map.
- `llm.py` — Ollama client: builds prompts, parses and repairs JSON actions.
- `agent.py` — `run_turn()`: observe–think–act loop with guardrails and logging.
- `chat.py` — terminal REPL entry point.
- `tests/` — unit and integration tests with a static HTML fixture.

## Specs

- Blueprint: [`desk.waker.llm.md`](desk.waker.llm.md)
- Design specs: [`docs/superpowers/specs/`](docs/superpowers/specs/)
