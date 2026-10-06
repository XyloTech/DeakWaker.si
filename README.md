# desk.waker

A terminal-chat autonomous browser agent: you type a natural-language goal, and a Llama model on Groq drives Playwright through an Observe → Think → Act loop until the task is done.

## Development environment

- **GitHub Codespaces:** repo page → **Code** → **Codespaces** → **Create codespace on main**. The container builds unattended (`postCreateCommand` installs dependencies and Chromium).
- **VS Code:** install the Dev Containers extension, open the folder, then **Reopen in Container**. Docker Desktop is required for local VS Code use only (Codespaces needs nothing installed).

## Run tests

```
pytest -v
```

The live-Groq test skips without a key.

## Run the agent

1. Copy `.env.example` to `.env`.
2. Set `GROQ_API_KEY` in `.env`.
3. Run `python chat.py`.

## Project layout

- `config.py` — model name, max steps, timeouts, headless flag (env-overridable).
- `browser.py` — `BrowserWrapper`: async Playwright navigation, clicks, typing, element map.
- `llm.py` — Groq client: builds prompts, parses and repairs JSON actions.
- `agent.py` — `run_turn()`: observe–think–act loop with guardrails and logging.
- `chat.py` — terminal REPL entry point.
- `tests/` — unit and integration tests with a static HTML fixture.

## Specs

- Blueprint: [`desk.waker.llm.md`](desk.waker.llm.md)
- Design specs: [`docs/superpowers/specs/`](docs/superpowers/specs/)
