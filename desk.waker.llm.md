# Project Blueprint: Llama-Powered Autonomous Browser Agent

## 1. Project Overview
* **Objective:** Build an autonomous AI agent powered by a Llama LLM that accepts natural language commands via a chat interface and executes multi-step tasks inside a custom-controlled browser environment.
* **Core Tech Stack:**
  * **Brain (LLM):** Llama 3 (via Ollama locally or Groq API for speed) with Structured JSON / Function Calling.
  * **Automation Engine:** Playwright (Python).
  * **Orchestrator:** Python (FastAPI or custom event loop).
  * **Frontend (Optional/Later):** Streamlit or a lightweight web chat UI.

---

## 2. System Architecture & Agentic Loop
The agent operates on an **Observe $\rightarrow$ Think $\rightarrow$ Act** feedback loop:
1. **User Input:** Chat message detailing the objective.
2. **Observation:** Browser extracts page text, DOM elements, or screenshots.
3. **Thought (Llama):** LLM analyzes the current state against the goal and decides the next atomic command.
4. **Action (Playwright):** Executes the command (`navigate`, `click`, `type`, `scroll`, etc.).
5. **Loop / Termination:** Repeats until goal is achieved, max steps reached, or human intervention is required.

---

## 3. Step-by-Step Implementation Roadmap

### Phase 1: Environment Setup & Playwright Wrapper
* [ ] Initialize Python virtual environment.
* [ ] Install dependencies: `playwright`, `ollama` (or `openai`/`requests`), `pydantic`.
* [ ] Install Playwright browsers (`playwright install`).
* [ ] Create `browser_wrapper.py` with primitive functions:
  * `navigate_to(url)`
  * `click(selector)`
  * `type_text(selector, text)`
  * `extract_page_content()`
  * `take_screenshot()`

### Phase 2: LLM Action Schema & Prompt Engineering
* [ ] Define the JSON schema for LLM outputs. Example:
  ```json
  {
    "thought": "Reasoning about current state",
    "action": "click | type | navigate | done | ask_user",
    "parameters": {
      "selector": "CSS selector or XPath",
      "text": "Text to type if applicable",
      "url": "URL if navigating"
    }
  }
  ```
* [ ] Write the system prompt instructing Llama on its role as a web navigator, available tools, and constraints.

### Phase 3: The Orchestration Engine (`agent.py`)
* [ ] Implement the main execution loop with step counters (e.g., max 15 steps to prevent infinite loops).
* [ ] Handle parsing of LLM JSON responses with fallback/error correction logic.
* [ ] Log each step's thought process and action execution to the console.

### Phase 4: Safety, Guardrails & Human-in-the-Loop
* [ ] Add a confirmation hook before sensitive actions (e.g., form submissions, logins, purchasing).
* [ ] Handle timeout errors and unexpected page states gracefully.

### Phase 5: Chat Interface Integration
* [ ] Connect the agentic loop to a simple interface (Terminal chat first, then optional Streamlit UI).

---

## 4. Milestone Tracker
* **Milestone 1:** Basic Playwright script successfully navigates and interacts with a test website. *(Status: Complete ✓)*
* **Milestone 2:** Local Llama successfully parses a task and outputs valid JSON action commands. *(Status: Complete ✓)*
* **Milestone 3:** Closed-loop execution (Llama drives Playwright autonomously for 3+ consecutive steps). *(Status: Complete ✓)*
* **Milestone 4:** Full chat integration and safety guardrails. *(Status: Complete ✓)*
* **Milestone 5:** Smarter agent — hardened prompt rules, step-budget coaching, URL normalization; smartness tests green and the YouTube goal ends cleanly with no tracebacks. *(Status: Complete ✓)*
* **Milestone 6:** Model bake-off executed — `qwen3:latest` wins (3/3 goals) and is the default `MODEL`; report in `docs/superpowers/bench/`. *(Status: Complete ✓)*
* **Milestone 7:** Web UI live demo (S2: goal entry, SSE step streaming, in-page confirm card) + terminal regression (`chat.py` confirm honored, `Status: done`) + suite green (77 passed, 1 skipped). *(Status: Complete ✓)*