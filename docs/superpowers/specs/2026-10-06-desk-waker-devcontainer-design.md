# Design Spec: desk.waker Devcontainer (GitHub Codespaces / VS Code)

**Date:** 2026-10-06
**Source request:** "use docker for it" — user-selected Approach A (devcontainer feature) in chat design review, approved 2026-10-06.
**Status:** Approved (chat design approved 2026-10-06; pending written-spec review)

## 1. Intent

Give the desk.waker project a secure, configurable, dedicated development environment: opening the repo in GitHub Codespaces or VS Code Dev Containers spins up a container with Python, Playwright, and Chromium preinstalled, so `pytest` and the agent run identically for every developer with zero host setup.

**Success criteria:**
- Opening the repo in Codespaces/VS Code yields a working container where `pytest -v` passes without any manual installs.
- Fresh-container setup runs unattended (`postCreateCommand`); no interactive steps.
- The host machine needs nothing installed — Docker/Codespaces provides everything.

**Stated constraints (user decisions):**
- Approach A: `devcontainer.json` + Microsoft Playwright devcontainer feature — no custom Dockerfile.
- Scope: development environment only. No runtime image, no docker-compose, no CI.
- A short `README.md` is included (user approved inclusion by not objecting).

**Out of scope (YAGNI):** Dockerfile, docker-compose, GitHub Actions/Codespaces workflows, runtime packaging, devcontainer customizations beyond what setup requires.

## 2. Files

```
.devcontainer/
└── devcontainer.json     # container definition
README.md                 # project intro + devcontainer + test/agent instructions
```

No application code, `requirements.txt`, `.gitignore`, or spec/plan files are modified.

## 3. devcontainer.json (exact values)

```json
{
  "name": "desk.waker",
  "image": "mcr.microsoft.com/devcontainers/python:1-3.11",
  "features": {
    "ghcr.io/devcontainers/features/playwright:1": {}
  },
  "postCreateCommand": "pip install -r requirements.txt && playwright install chromium",
  "remoteUser": "vscode"
}
```

- **Image:** `mcr.microsoft.com/devcontainers/python:1-3.11` — satisfies the plan's Python 3.10+ floor; `1-3.11` is the stable major-1 tag for Python 3.11.
- **Feature:** `ghcr.io/devcontainers/features/playwright:1` with default options (latest Playwright, Chromium + system deps). No `version` pin — `requirements.txt` (Task 1) is unpinned, so both sides track latest; `playwright install chromium` in `postCreateCommand` reconciles the Python-side browser revision if the feature's revision differs.
- **postCreateCommand:** idempotent. `pip install` is a no-op when cached; `playwright install chromium` skips any revision already present in `~/.cache/ms-playwright`.
- **remoteUser:** `vscode` — the image's default non-root user (explicit for clarity).

## 4. README.md (required sections, in order)

1. **desk.waker** — one-paragraph description: terminal-chat autonomous browser agent, Llama via Groq + Playwright.
2. **Development environment** — "Open in Codespaces" (repo → Code → Codespaces) and VS Code "Reopen in Container" instructions; note Docker Desktop required for local VS Code use only.
3. **Run tests** — `pytest -v` (live-Groq test skips without key).
4. **Run the agent** — ensure Ollama is running (`ollama pull llama3.1:8b`), optionally set `OLLAMA_HOST`/`MODEL` in `.env`, run `python chat.py`.
5. **Project layout** — the six modules from the existing spec (`config.py`, `browser.py`, `llm.py`, `agent.py`, `chat.py`, `tests/`), one line each.
6. **Spec/blueprint links** — pointers to `desk.waker.llm.md` and `docs/superpowers/specs/`.

## 5. Verification

| Check | Command / action | Pass condition |
|---|---|---|
| Config validity | parse `.devcontainer/devcontainer.json` as JSON | no parse error |
| Configuration dump (if Docker available locally) | `devcontainer read-configuration --workspace-folder .` | exits 0 |
| No host regression | `pytest -v` on host | same results as pre-change (8/8 passing) |
| Real environment proof (manual) | open repo in Codespaces or VS Code | container builds; `pytest -v` green |

The manual check is the definitive gate and is performed by the user; automated checks are the implementer's evidence.

## 6. Execution & GitHub

- Implemented as **Task 7** of the existing plan `docs/superpowers/plans/2026-10-06-desk-waker-agent.md` via the running SDD pipeline (brief → implementer → review), after the queued Task 1 review.
- After Task 7 merges: `git remote add origin https://github.com/XyloTech/DeakWaker.si.git && git push -u origin main`.
- If the remote already contains commits (GitHub UI-created README), fetch first and reconcile without force-push — the container README wins for content, no history rewriting.
- Security: `.env` is gitignored; no secrets are committed or pushed. The repo is private (per user's pasted GitHub UI).

## 7. Error Philosophy

Container setup failures surface at codespace creation time with the failing command visible (`postCreateCommand` output) — the user never gets a silently half-provisioned environment. Nothing in the application's runtime error handling (spec §7 of the main design) changes.
