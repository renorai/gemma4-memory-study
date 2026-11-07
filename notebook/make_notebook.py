"""Generate gemma4_graph_memory.ipynb: the offline reproduction notebook for the Writeup.

It recomputes every number in the Writeup from the arm-run kernel's outputs (rows.jsonl per
model). With no real outputs attached (USE_FAKE), it runs the whole protocol on the fake harness
instead, which proves the mechanics and produces NO result. Internet is never used.

    python notebook/make_notebook.py
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
CELLS = [
    (MD, """# Transferable node-anchored memory for Gemma 4 coding agents: reproduction

Every number in the Writeup comes from this notebook, which re-runs the preregistered analysis
(PREREG.md, frozen; its sha256 is in the PR body) on the arm-run kernel's `rows.jsonl` files.
With no kernel output attached it runs the protocol on the **fake harness**: that checks the
mechanics and produces **no result**."""),
    (CODE, """import glob, json, os, sys, tempfile
from pathlib import Path
ROOT = Path.cwd().parent if (Path.cwd().parent / 'memory').exists() else Path.cwd()
for p in glob.glob('/kaggle/input/**/memory/keys.py', recursive=True):
    ROOT = Path(p).parent.parent
    break
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('NB_OUT', '/kaggle/working/nb-out' if Path('/kaggle/working').exists() else 'nb-out'))
OUT.mkdir(parents=True, exist_ok=True)
REAL = sorted(glob.glob('/kaggle/input/**/out/*/rows.jsonl', recursive=True))
USE_FAKE = not REAL
print('fake harness (NOT results)' if USE_FAKE else REAL)"""),
    (CODE, """# Rows: the real kernels' outputs, or the fake protocol end to end
from pipeline.experiment import read_jsonl, run_protocol, write_jsonl
if USE_FAKE:
    import fake_harness as fh
    work = Path(tempfile.mkdtemp())
    fh.write_repos(work / 'repos')
    rows, store, dstats = run_protocol(fh.FakeEval(work / 'repos'), fh.fake_generate, fh.tasks(),
                                       lambda t: work / 'repos' / t.repo, fh.FAKE_MODEL, work / 'out')
    PRIMARY = (fh.FAKE_MODEL,)
else:
    rows = [r for p in REAL for r in read_jsonl(p)]
    PRIMARY = ('gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-12b-it-qat-w4a16-ct')
write_jsonl(OUT / 'rows-all.jsonl', rows)
print(len(rows), 'rows')"""),
    (CODE, """# The preregistered analysis (stats/analyze.py)
from stats.analyze import analyze
from stats import markdown_table
res = analyze(rows, primary_models=PRIMARY)
json.dump(res, open(OUT / 'analysis.json', 'w'), indent=2, default=str)
for m, h in res['primary'].items():
    print(f"H1 {m} node-none: mean={h['mean']:.3f} CI=[{h['lo']:.3f}, {h['hi']:.3f}] "
          f"wins/losses={h['wins']}/{h['losses']} p_holm={h['p_holm']:.4f} -> {h['verdict']}")
tables = '\\n\\n'.join([
    markdown_table(list(res['primary'].values()), ['model', 'x', 'y', 'mean', 'lo', 'hi', 'wins', 'losses', 'p_holm', 'verdict']),
    markdown_table(res['exploratory'], ['model', 'x', 'y', 'mean', 'lo', 'hi', 'wins', 'losses', 'sign_p']),
    markdown_table(res['process'], ['model', 'arm', 'runs', 'n', 'success_rate', 'hit@1', 'hit@3', 'hit@5',
                                    'tool_calls_median', 'model_calls_median', 'tokens_in_mean', 'tokens_out_mean',
                                    'tokens_per_success', 'wall_min_median']),
    markdown_table([{'cell': k, **v} for k, v in res['flip_rate'].items()], ['cell', 'runs', 'n', 'pairwise', 'any_flip']),
])
(OUT / 'tables.md').write_text(tables)
print(tables)"""),
    (CODE, """# When does it help? Exploratory moderators (PREREG.md section 6)
json.dump(res['moderators'], open(OUT / 'moderators.json', 'w'), indent=2)
res['moderators']"""),
]


def main():
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == CODE else {})} for t, src in CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).with_name("gemma4_graph_memory.ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
