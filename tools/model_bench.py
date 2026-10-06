import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import run_turn
from browser import BrowserWrapper
from llm import LLMClient

CANDIDATE_MODELS = ["llama3.1:8b", "qwen3:latest", "qwen3.5:latest"]
REPAIR_MARKER = "Model output still invalid after repair"
RESULTS_PATH = pathlib.Path(__file__).parents[1] / "docs" / "superpowers" / "bench" / "2026-10-06-model-bakeoff.md"

WINNER_RULE = (
    "Winner rule (applied in this order): (1) most goals achieved with status `done`; "
    "(2) fewest average steps across achieved goals; (3) fewest JSON/repair failures."
)


def summarize_transcript(transcript: list[dict]) -> dict:
    return {
        "steps": len(transcript),
        "repairs": sum(
            1 for entry in transcript if REPAIR_MARKER in str(entry.get("result", ""))
        ),
    }


def format_table(results: list[dict]) -> str:
    lines = ["| model | goal | status | steps | repairs | seconds |", "| --- | --- | --- | --- | --- | --- |"]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['goal']} | {row['status']} | {row['steps']} "
            f"| {row['repairs']} | {row['seconds']} |"
        )
    return "\n".join(lines)


def goals() -> list[str]:
    fixture = pathlib.Path(__file__).parents[1] / "tests" / "fixtures" / "site.html"
    return [
        f"Open the local fixture at {fixture.as_uri()} and click the Go button",
        "Open https://example.com and report the main heading",
        "Open YouTube and tell me the name of a currently trending video",
    ]


def available_models() -> list[str]:
    from ollama import Client

    response = Client().list()
    raw = response.get("models", []) if isinstance(response, dict) else response.models
    names = []
    for model in raw:
        name = model.get("name") if isinstance(model, dict) else model.model
        if name:
            names.append(name)
    matched = [name for name in CANDIDATE_MODELS if name in names]
    return matched or names


async def run_goal(model: str, goal: str) -> dict:
    browser = BrowserWrapper()
    started = time.monotonic()
    status, answer = "error", "unknown"
    try:
        await browser.start()
        try:
            result = await run_turn(goal, browser, LLMClient(model=model))
            status, answer = result.status, result.answer
            summary = summarize_transcript(result.transcript)
        except Exception as exc:
            summary = {"steps": 0, "repairs": 0}
            answer = str(exc)
    finally:
        await browser.close()
    return {
        "model": model,
        "goal": _short_goal(goal),
        "status": status,
        "answer": answer,
        "steps": summary["steps"],
        "repairs": summary["repairs"],
        "seconds": round(time.monotonic() - started, 1),
    }


def _short_goal(goal: str) -> str:
    if goal.startswith("Open the local fixture"):
        return "fixture-go"
    if "example.com" in goal:
        return "example-heading"
    return "youtube-trending"


def _averages(results: list[dict]) -> str:
    lines = ["| model | done goals | avg steps (done) | total repairs |", "| --- | --- | --- | --- |"]
    by_model: dict[str, list[dict]] = {}
    for row in results:
        by_model.setdefault(row["model"], []).append(row)
    for model, rows in sorted(by_model.items()):
        done = [row for row in rows if row["status"] == "done"]
        avg = (
            round(sum(row["steps"] for row in done) / len(done), 1) if done else "n/a"
        )
        repairs = sum(row["repairs"] for row in rows)
        lines.append(f"| {model} | {len(done)}/{len(rows)} | {avg} | {repairs} |")
    return "\n".join(lines)


def write_report(results: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = (
        "# Model bake-off — 2026-10-06\n\n"
        f"{WINNER_RULE}\n\n"
        f"{format_table(results)}\n\n"
        "## Per-model summary (achieved goals and repairs)\n\n"
        f"{_averages(results)}\n"
    )
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    models = available_models()
    print(f"Models: {models}")
    results: list[dict] = []
    for model in models:
        for goal in goals():
            print(f"--- {model} :: {goal[:60]}")
            results.append(asyncio.run(run_goal(model, goal)))
            print(f"    -> {results[-1]['status']} in {results[-1]['seconds']}s")
    write_report(results, RESULTS_PATH)
    print(f"Report written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
