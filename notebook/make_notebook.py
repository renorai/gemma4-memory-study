"""Regenerates gemma4_graph_memory.ipynb from the cells below (keeps the notebook diffable).

    python notebook/make_notebook.py
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# Graph-anchored experience memory for small coding agents: reproduction notebook

This notebook reproduces every number in the Writeup (`paper/WRITEUP.md`).

**How it runs**
- It runs on a Kaggle **L4x4** notebook, **with internet off**.
- It makes no network calls. Cell 2 blocks outbound sockets, so an accidental call fails loudly.
- The protocol is preregistered in `DESIGN.md`. This notebook implements it and does not change it.

**Before the first real run**
1. Attach the competition dataset and this code, as a Kaggle dataset or a utility script.
2. Implement `memory/adapter.py:KaggleHarnessAdapter` and `make_model()` below. Both are marked
   **ADAPTER**.
3. Fill in DESIGN.md section 9, "Adapter deviations".

**Checking the notebook without the data.** Set `USE_FAKE = True`. The whole pipeline then runs on
the offline fake harness and a scripted policy. Those numbers are **not results**."""),
    (CODE, """# 1. Configuration
import os, sys, json, time
from pathlib import Path

USE_FAKE = True                                   # False once the dataset and the model adapter exist
CODE_DIR = Path('/kaggle/input/renor-gemma4-memory/kaggle-gemma4')   # ADAPTER: where this folder is attached
if not CODE_DIR.exists():
    CODE_DIR = Path.cwd().parent if (Path.cwd().parent / 'memory').exists() else Path.cwd()
HARNESS_DIR = Path('/kaggle/input/gemma-4-developer-agent')          # ADAPTER: competition dataset mount
OUT = Path(os.environ.get('NB_OUT') or ('/kaggle/working/out' if Path('/kaggle/working').exists() else CODE_DIR / 'out'))
MODELS = ['gemma-4-e4b', 'gemma-4-12b', 'gemma-4-31b']                # run order: cheapest first (DESIGN.md section 6)
TIME_LIMIT_H = 11.0
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE_DIR))
print(CODE_DIR, OUT)"""),
    (CODE, """# 2. Offline guard: refuse every non-loopback connection for the rest of the session
import socket
_orig_connect = socket.socket.connect
def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and host not in ('127.0.0.1', 'localhost', '::1') and not host.startswith('/'):
        raise RuntimeError(f'network call blocked by the offline guard: {host}')
    return _orig_connect(self, address)
socket.socket.connect = _guarded_connect
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'"""),
    (CODE, """# 3. Harness and models
from memory import RetrievalConfig, LessonStore
from runner.experiment import run_loro, run_arm, write_jsonl, read_jsonl
from runner.budget import plan, research_phases
from runner.probes import cross_repo_embedding_probe, flip_rate
from stats.analyze import analyze
from stats import markdown_table

def make_model(model_name, adapter):
    \"\"\"ADAPTER: return (agent, generate) for one Gemma 4 model served locally (e.g. vLLM on the 4 L4s).

    agent(task, arm, run_id=...) -> attempt dict in the schema of memory/distill.py
        (it runs the ADK agent with arm.for_task(task) injected and recall_lessons bound to arm)
    generate(prompt) -> str, the SAME model, greedy, used only for lesson distillation
    \"\"\"
    if USE_FAKE:
        from fake_harness import FakeAgent, fake_generate
        a = FakeAgent(adapter)
        return a, fake_generate
    raise NotImplementedError('ADAPTER: serve the model and wrap the ADK agent here')

if USE_FAKE:
    from fake_harness import FakeHarness
    harness = FakeHarness()
    MODELS = ['fake-policy']
else:
    from memory.adapter import KaggleHarnessAdapter
    harness = KaggleHarnessAdapter(str(HARNESS_DIR))
tasks = list(harness.tasks())
print(len(tasks), 'tasks;', sorted({t.repo for t in tasks}))"""),
    (CODE, """# 4. Pre-run probe (DESIGN.md section 8, threat 3): do node embeddings share one space across repos?
# Matched pairs are chosen by NAME before any cosine is looked at.
if USE_FAKE:
    from fake_harness import REPOS, ROLES, node
    pairs = [(node('alpha', r), node('beta', r)) for r in ROLES]
    all_nodes = [node(rp, r) for rp in REPOS for r in ROLES]
else:
    pairs = []      # ADAPTER: e.g. requests Session.send <-> httpx Client.send, listed by hand from names
    all_nodes = []  # ADAPTER: every node id in the four dev repos
probe = cross_repo_embedding_probe(harness, pairs, all_nodes)
json.dump(probe, open(OUT / 'probe_embeddings.json', 'w'), indent=2)
print(probe)
if not probe['pass']:
    print('PROBE FAILED: disable the emb route for LORO (emb_weight=0) and log a deviation in DESIGN.md section 10')"""),
    (CODE, """# 5. Calibration and budget: measure seconds per episode on 5 tasks, refuse to start if over the limit
cfg = RetrievalConfig()
from memory import make_arm
calib = {}
for m in MODELS:
    agent, generate = make_model(m, harness)
    t0 = time.time()
    run_arm(tasks[:5], agent, make_arm('none', LessonStore(), harness, cfg), run_id=f'{m}-calib')
    sec_ep = (time.time() - t0) / 5
    calib[m] = plan(research_phases(m, len(tasks), sec_ep, sec_distill=sec_ep / 4), TIME_LIMIT_H)
    print(m, round(calib[m]['total_h'], 2), 'h projected; fits =', calib[m]['fits'])
json.dump(calib, open(OUT / 'budget_plan.json', 'w'), indent=2)"""),
    (CODE, """# 6. Main runs: baseline (none) -> self-distillation -> LORO flat and graph, per model
all_rows = []
for m in MODELS:
    if not calib[m]['fits']:
        print(f'SKIP {m}: projected over {TIME_LIMIT_H} h; run it per fold in its own session (DESIGN.md section 6)')
        continue
    agent, generate = make_model(m, harness)
    rows, store, dstats = run_loro(harness, agent, generate, m, cfg, out_dir=OUT)
    json.dump({'model': m, 'attempts': dstats.attempts, 'lessons': dstats.lessons,
               'dropped': dict(dstats.dropped)}, open(OUT / f'distill-{m}.json', 'w'), indent=2)
    all_rows += rows
write_jsonl(OUT / 'rows-all.jsonl', all_rows)
print(len(all_rows), 'rows')"""),
    (CODE, """# 7. Run-to-run noise: 20 tasks, arm none, run twice
noise = {}
for m in MODELS:
    agent, _ = make_model(m, harness)
    none = make_arm('none', LessonStore(), harness, cfg)
    sub = tasks[:20]
    r1 = [dict(model=m, task_id=a['task_id'], success=int(a['outcome'] == 'success')) for a in run_arm(sub, agent, none, f'{m}-noise1')]
    r2 = [dict(model=m, task_id=a['task_id'], success=int(a['outcome'] == 'success')) for a in run_arm(sub, agent, none, f'{m}-noise2')]
    noise[m] = flip_rate(r1, r2)
json.dump(noise, open(OUT / 'noise.json', 'w'), indent=2)
noise"""),
    (CODE, """# 8. Preregistered analysis (stats/analyze.py) and the Writeup tables
res = analyze(read_jsonl(OUT / 'rows-all.jsonl'))
json.dump(res, open(OUT / 'analysis.json', 'w'), indent=2, default=str)
for h in ('H1', 'H2'):
    r = res[h]
    print(f"{h} {r['x']}-{r['y']}: mean={r['mean']:.3f} CI=[{r['lo']:.3f}, {r['hi']:.3f}] "
          f"wins/losses={r['wins']}/{r['losses']} p_holm={r['p_holm']:.4f} -> {r['verdict']}")
tables = '\\n\\n'.join([
    markdown_table(res['per_model'], ['model', 'x', 'y', 'mean', 'lo', 'hi', 'wins', 'losses', 'sign_p']),
    markdown_table(res['cost'], ['model', 'arm', 'n', 'success_rate', 'tokens_in_mean', 'tokens_out_mean',
                                 'tokens_per_success', 'wall_h', 'gpu_h']),
])
(OUT / 'tables.md').write_text(tables)
print(tables)"""),
    (CODE, """# 9. Graph-arm retrieval routes (self / hop / emb), from the attempt logs
from collections import Counter
routes = Counter()
for m in MODELS:
    p = OUT / f'attempts-{m}.jsonl'
    if p.exists():
        for a in read_jsonl(p):
            if a.get('arm') == 'graph':
                routes[(m, len(a.get('lesson_ids', [])) > 0)] += 1
routes  # ADAPTER: once the real agent logs Retrieved.via, count routes by via"""),
    (CODE, """# 10. Leaderboard submissions, one per arm (hidden test set). Lessons: all four dev repos, same model.
from runner.build_submission import build
SUBMIT_MODEL = MODELS[0]  # chosen after the dev analysis; the choice is reported, not hidden
for arm in ('none', 'flat', 'graph'):
    lessons = OUT / f'lessons-{SUBMIT_MODEL}.jsonl'
    if not USE_FAKE:
        names = build(arm, lessons if arm != 'none' else None, SUBMIT_MODEL + '-it-qat-w4a16', OUT / f'submission-{arm}.zip')
        print(arm, len(names), 'files')"""),
]


def main():
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == CODE else {})}
            for t, src in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).with_name("gemma4_graph_memory.ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
