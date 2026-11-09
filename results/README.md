# Results: recap vs no memory (rows and aggregates only)

No task text, traces, patches or tokens are stored here.

| File | What it is |
|---|---|
| `run_rows.jsonl` | One row per (backend, arm, run, task): outcome, overflow, caps, compactions, repeat measures, H-a/H-b detector values, and plumbing for memory arms. Built by `build_rows.py` from the raw run outputs, which are not committed. |
| `analyze_rows.py` | Regenerates every reported number from `run_rows.jsonl` and writes `analysis.json`. The notebook's last cell runs it. |
| `posthoc_tests.md` | Every post-hoc test run so far, with its p value. |
| `pooled_run1_v2.md` / `.json`, `aa_calibration.json` | The analysis outputs as first reported. `analyze_rows.py` reproduces them from the rows. |
| `provenance/` | The scripts that produced the first reports from the raw traces. |

Rows in `run_rows.jsonl`, by backend:

| Backend | No memory | Recap |
|---|---|---|
| Modal (H100) | runs 1-2, ff9bf75d | run 1 (16 tasks) and run 2 (7 tasks, partial), 59f10a6 |
| Kaggle (4x L4) | runs 1-2, dee8321a | runs 1-2, 16e71dab = 59f10a6 |

The environment hash is 7f74aeda for all. The 7 Modal recap run-2 rows carry `exploratory_partial=true` and enter no registered figure.
