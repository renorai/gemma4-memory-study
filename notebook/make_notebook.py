"""Generate gemma4_graph_memory.ipynb: the offline reproduction notebook for the Writeup.

It regenerates every results table in paper/WRITEUP.md (Tables 1-6, same numbers and headers)
from the arm-run kernels' rows.jsonl files, and re-runs the frozen preregistered analysis
(stats/analyze.py). With no rows attached it runs in FAKE-DATA mode: it writes synthetic
rows.jsonl and runs the identical code on them, so every table can be produced today. Fake
output is labelled on every table and is never a result. Internet is never used.

The table code lives in paper_tables.py and is inlined here, so the notebook needs only this
repo's `stats` package (shipped in the Kaggle code dataset).

    python notebook/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STRIP = "if __name__ == \"__main__\":  # NOTEBOOK-STRIP"


def inlined_tables():
    src = (HERE / "paper_tables.py").read_text(encoding="utf-8")
    body = src[:src.index(STRIP)].rstrip() + "\n"
    return "# notebook/paper_tables.py, inlined by make_notebook.py (edit that file, not this cell)\n" + body


MD, CODE = "markdown", "code"
CELLS = [
    (MD, """# Told once: offline Renor memory for small Gemma 4 coding agents (reproduction)

Every number in the Writeup's results tables comes from this notebook. It reads the arm-run
kernels' `rows.jsonl` files and applies the rules committed before any memory-arm result
(`DESIGN-CHANGELOG-addendum.md`) and the frozen preregistration (`PREREG.md`).

**Inputs.** Attach the kernels' outputs (any `rows.jsonl` under `/kaggle/input`), or set
`ROWS_GLOB`. Optionally attach the committed 48-task list as `floor48.txt` (or set `FLOOR_LIST`).

**Fake-data mode.** With no rows attached, the notebook writes synthetic rows and runs the same
code on them. Every table it prints is headed FAKE DATA. Nothing it prints is a result."""),
    (CODE, """import glob, json, os, sys
from pathlib import Path
ROOT = Path.cwd().parent if (Path.cwd().parent / 'stats').exists() else Path.cwd()
for p in glob.glob('/kaggle/input/**/stats/analyze.py', recursive=True):
    ROOT = Path(p).parent.parent
    break
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('NB_OUT', '/kaggle/working/nb-out' if Path('/kaggle/working').exists() else 'nb-out'))
OUT.mkdir(parents=True, exist_ok=True)
REAL = sorted(glob.glob(os.environ.get('ROWS_GLOB', '/kaggle/input/**/rows.jsonl'), recursive=True))
FLOOR_LIST = os.environ.get('FLOOR_LIST') or next(iter(sorted(glob.glob('/kaggle/input/**/floor48.txt', recursive=True))), None)
USE_FAKE = not REAL
print('fake harness (NOT results): synthetic rows, no model was run' if USE_FAKE else REAL)"""),
    (CODE, "@@TABLES@@"),
    (CODE, """# Rows: the real kernels' outputs, or synthetic rows written and read back through the same path
if USE_FAKE:
    write_rows(OUT / 'fake' / 'rows.jsonl', fake_rows())
    rows, MODEL = read_rows([OUT / 'fake' / 'rows.jsonl']), FAKE_MODEL
else:
    rows, MODEL = read_rows(REAL), MODEL_31B
write_rows(OUT / 'rows-all.jsonl', rows)
mine = [r for r in rows if r['model'] == MODEL]
ALL_TASKS = sorted({r['task_id'] for r in mine})
FLOOR, HOW = floor_set(mine, Path(FLOOR_LIST).read_text().split() if FLOOR_LIST else None)
print(len(rows), 'rows;', len(ALL_TASKS), 'tasks;', 'floor set:', len(FLOOR), 'tasks,', HOW)"""),
    (CODE, """# Rule 3 first: the recap headline switch, from no-memory rows only
RECAP = recap_decision(mine, MODEL, set(FLOOR))
json.dump(RECAP, open(OUT / 'recap-decision.json', 'w'), indent=2)
RECAP"""),
    (CODE, """# Tables 1-6 of the Writeup (copy cells from tables.md; nothing else)
TABLES, SUMMARY = build(rows, MODEL, FLOOR, ALL_TASKS)
(OUT / 'tables.md').write_text(f'Floor set: {HOW}\\n\\n' + TABLES)
json.dump(SUMMARY, open(OUT / 'summary.json', 'w'), indent=2, default=str)
print(TABLES)"""),
    (CODE, """# The frozen preregistered analysis (PREREG.md section 5) on the headline set, for its verdict words
from stats.analyze import analyze
HEAD = set(ALL_TASKS) if SUMMARY['full_129']['headline'] == '129' else set(FLOOR)
# headline (DESIGN.md section 10): recap vs none; fake data has only node, so it falls back
HEADLINE = ('recap', 'none') if any(r['arm'] == 'recap' for r in rows) else ('node', 'none')
res = analyze([r for r in rows if r['task_id'] in HEAD], primary_models=(MODEL,), headline=HEADLINE)
json.dump(res, open(OUT / 'analysis.json', 'w'), indent=2, default=str)
json.dump(res['moderators'], open(OUT / 'moderators.json', 'w'), indent=2)
for m, h in res['primary'].items():
    print(f"H1 {m} {HEADLINE[0]}-none: mean={h['mean']:.3f} CI=[{h['lo']:.3f}, {h['hi']:.3f}] "
          f"wins/losses={h['wins']}/{h['losses']} p_holm={h['p_holm']:.4f} -> {h['verdict']}")"""),
]


def main():
    cells = [(t, inlined_tables() if src == "@@TABLES@@" else src) for t, src in CELLS]
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == CODE else {})} for t, src in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "gemma4_graph_memory.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
