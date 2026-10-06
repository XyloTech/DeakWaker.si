import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.model_bench import format_table, summarize_transcript


def test_summarize_transcript_counts_steps_and_repairs():
    transcript = [
        {"step": 1, "result": "OK"},
        {"step": 2, "result": "ERROR: Model output still invalid after repair: Invalid JSON"},
        {"step": 3, "result": "OK"},
    ]
    assert summarize_transcript(transcript) == {"steps": 3, "repairs": 1}


def test_format_table_includes_header_and_rows():
    results = [
        {"model": "m1", "goal": "g1", "status": "done", "steps": 3, "repairs": 0, "seconds": 12.5},
        {"model": "m1", "goal": "g2", "status": "max_steps", "steps": 15, "repairs": 2, "seconds": 90.1},
    ]
    table = format_table(results)
    assert "| model | goal | status | steps | repairs | seconds |" in table
    assert "| m1 | g1 | done | 3 | 0 | 12.5 |" in table
    assert "max_steps" in table
