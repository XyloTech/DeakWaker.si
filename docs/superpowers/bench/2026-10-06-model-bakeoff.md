# Model bake-off — 2026-10-06

Winner rule (applied in this order): (1) most goals achieved with status `done`; (2) fewest average steps across achieved goals; (3) fewest JSON/repair failures.

| model | goal | status | steps | repairs | seconds |
| --- | --- | --- | --- | --- | --- |
| llama3.1:8b | fixture-go | done | 6 | 0 | 27.6 |
| llama3.1:8b | example-heading | max_steps | 15 | 0 | 50.9 |
| llama3.1:8b | youtube-trending | max_steps | 15 | 1 | 50.5 |
| qwen3:latest | fixture-go | done | 3 | 0 | 78.0 |
| qwen3:latest | example-heading | done | 2 | 0 | 59.3 |
| qwen3:latest | youtube-trending | done | 5 | 0 | 309.7 |
| qwen3.5:latest | fixture-go | max_steps | 15 | 0 | 775.4 |
| qwen3.5:latest | example-heading | done | 2 | 0 | 68.3 |
| qwen3.5:latest | youtube-trending | max_steps | 15 | 1 | 908.2 |

## Per-model summary (achieved goals and repairs)

| model | done goals | avg steps (done) | total repairs |
| --- | --- | --- | --- |
| llama3.1:8b | 1/3 | 6.0 | 1 |
| qwen3.5:latest | 1/3 | 2.0 | 1 |
| qwen3:latest | 3/3 | 3.3 | 0 |
