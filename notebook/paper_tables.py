"""Every results table in paper/WRITEUP.md and paper/arxiv/main.tex, from rows.jsonl files.

Standard library plus this folder's `stats` package. make_notebook.py inlines this file into the
notebook, so the notebook and `python notebook/paper_tables.py` run the same code.

One row is one episode: {model, arm, run, task_id, repo, success, ...}. Arms use the code's names
(none, node, shuffled, recap); the paper calls `node` "memory". Optional fields, reported as n/a
when absent: context_overflow, compactions, repeats_after_compaction, repeat_rate (all three from
stats/repeats.py, the single definition of the repeated-action measure), tool_calls, model_calls, tokens_in, tokens_out, wall_s, recalls, lessons_shown, finished_at.

The rules applied here are DESIGN-CHANGELOG-addendum.md's, written before any memory-arm result:
  - the 48-task floor set always runs; 129 tasks x 2 arms x 3 runs is the headline only if it is
    complete by HEADLINE_DEADLINE, otherwise 48 is the headline and 129 is exploratory;
  - the recap headline switch reads no-memory rows only (recap_decision);
  - primary = resolved rate as a per-task mean over runs; paired bootstrap 95% CI over tasks and
    exact McNemar on per-run pairs; a CI that crosses 0 is a pilot effect size, no significance.

Fake mode (fake_rows) writes SYNTHETIC rows so every table can be produced today. Its model name
starts with FAKE- and every table it produces is headed "FAKE DATA". It is never a result.
"""

import hashlib
import json
import random
from collections import defaultdict
import sys
from pathlib import Path

if __name__ == "__main__" and "__file__" in globals():   # CLI: make the package root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stats import flip_rates, mcnemar_exact, paired_bootstrap_ci, run_matrix
from stats.repeats import per_task_rates

SEED = 20261107
N_BOOT = 10_000
MODEL_31B = "gemma-4-31b-it-qat-w4a16-ct"
FAKE_MODEL = "FAKE-gemma-4-31b-synthetic"
HEADLINE_DEADLINE = "2026-11-11T12:00:00Z"
FLOOR_N = 48
FULL_N = 129
FULL_RUNS = 3
REPOS = ("fastapi", "rich", "requests", "httpx")
ARMS = (("none", "no memory"), ("node", "memory"), ("shuffled", "shuffled memory"), ("recap", "recap"))
LABEL = dict(ARMS)
# Recap headline switch (DESIGN-CHANGELOG-addendum.md rule 3): no-memory rows only.
RECAP_MIN_COMPACTED = 0.30      # share of no-memory episodes compacted at least once
RECAP_MIN_REPEAT = 0.10         # mean per-task repeated actions per compaction (stats/repeats.py)


# ---------------------------------------------------------------- rows

def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def fake_rows(seed=SEED, n_tasks=FULL_N, runs=FULL_RUNS):
    """SYNTHETIC episodes with every field the tables read. Shape only; the numbers mean nothing.
    httpx gets one task, as in the real development set."""
    rng = random.Random(seed)
    sizes = {"fastapi": 52, "rich": 44, "requests": n_tasks - 52 - 44 - 1, "httpx": 1}
    tasks = [(f"FAKE-{repo}-{i:03d}", repo) for repo in REPOS for i in range(sizes[repo])]
    lift = {"none": 0.0, "node": 0.03, "shuffled": 0.01, "recap": 0.02}
    rows = []
    for tid, repo in tasks:
        p = rng.random() ** 1.5
        for arm in lift:
            n_runs = runs if arm in ("none", "node") else 1
            for run in range(1, n_runs + 1):
                compactions = rng.choice((0, 0, 1, 1, 2, 3))
                reps = rng.randint(0, 3 * compactions)
                rows.append({
                    "model": FAKE_MODEL, "arm": arm, "run": run, "task_id": tid, "repo": repo,
                    "success": int(rng.random() < min(1.0, p + lift[arm])),
                    "context_overflow": rng.random() < 0.19, "compactions": compactions,
                    "repeats_after_compaction": reps, "repeat_rate": reps / compactions if compactions else None,
                    "tool_calls": rng.randint(15, 100), "model_calls": rng.randint(20, 90),
                    "tokens_in": rng.randint(150_000, 900_000), "tokens_out": rng.randint(5_000, 40_000),
                    "wall_s": rng.uniform(300, 3600),
                    "recalls": rng.randint(1, 3) if arm in ("node", "shuffled") else 0,
                    "lessons_shown": rng.randint(0, 5) if arm in ("node", "shuffled") else 0,
                    "finished_at": "2026-11-10T00:00:00Z"})
    return rows


class _T:
    def __init__(self, task_id, repo):
        self.task_id, self.repo = task_id, repo


def floor_set(rows, listed=None):
    """The 48-task floor set. `listed` is the committed task-id list (the addendum requires one
    before any memory-arm result). Without it, fall back to the runner's outcome-blind
    runner.budget.stratified_subset and say so."""
    if listed:
        return list(listed), "committed list"
    repo = {r["task_id"]: r["repo"] for r in rows}
    try:
        from runner.budget import stratified_subset
    except ImportError:
        stratified_subset = None
    if stratified_subset is None:
        ids = sorted(repo, key=lambda t: hashlib.sha256(t.encode()).hexdigest())[:FLOOR_N]
        return ids, "WARNING: no committed list and no runner; sha256 order"
    return ([t.task_id for t in stratified_subset([_T(t, repo[t]) for t in sorted(repo)], FLOOR_N)],
            "WARNING: no committed list; runner.budget.stratified_subset")


# ---------------------------------------------------------------- statistics

def _sel(rows, model, arm, tasks=None):
    return [r for r in rows if r["model"] == model and r["arm"] == arm and (tasks is None or r["task_id"] in tasks)]


def task_means(rows, model, arm, tasks=None, metric="success"):
    g = defaultdict(list)
    for r in _sel(rows, model, arm, tasks):
        if r.get(metric) is not None:
            g[r["task_id"]].append(float(r[metric]))
    return {t: sum(v) / len(v) for t, v in g.items()}


def n_runs(rows, model, arm, tasks=None):
    return len({int(r["run"]) for r in _sel(rows, model, arm, tasks)})


def contrast(rows, model, x, y, tasks=None):
    """Per-task mean over runs, paired over the tasks both arms have; bootstrap CI over tasks;
    exact McNemar for every run number present in both arms. Arms are compared only on the run
    numbers both have, so a 1-run arm is paired with run 1 of the other (PREREG.md section 6)."""
    runs = ({int(r["run"]) for r in _sel(rows, model, x, tasks)} & {int(r["run"]) for r in _sel(rows, model, y, tasks)})
    rows = [r for r in rows if int(r["run"]) in runs]
    mx, my = task_means(rows, model, x, tasks), task_means(rows, model, y, tasks)
    common = sorted(set(mx) & set(my))
    if not common:
        return None
    d = [mx[t] - my[t] for t in common]
    ci = paired_bootstrap_ci(d, n_boot=N_BOOT, alpha=0.05, seed=SEED)
    rx, ry = run_matrix(_sel(rows, model, x, tasks), model, x), run_matrix(_sel(rows, model, y, tasks), model, y)
    mcn = []
    for run in sorted(set(rx) & set(ry)):
        ts = sorted(set(rx[run]) & set(ry[run]))
        if ts:
            mcn.append({"run": run, **mcnemar_exact({t: rx[run][t] for t in ts}, {t: ry[run][t] for t in ts})})
    crosses = ci["lo"] <= 0 <= ci["hi"]
    return {"x": x, "y": y, "n_tasks": len(common), "x_rate": sum(mx[t] for t in common) / len(common),
            "y_rate": sum(my[t] for t in common) / len(common), "mean": ci["mean"], "lo": ci["lo"],
            "hi": ci["hi"], "mcnemar": mcn, "runs": (n_runs(rows, model, x, tasks), n_runs(rows, model, y, tasks)),
            "reading": ("pilot effect size; CI crosses 0; no significance claim" if crosses
                        else "CI excludes 0")}


def full_status(rows, model, all_tasks, deadline=HEADLINE_DEADLINE):
    """Rule 2: is 129 x {none, node} x 3 runs complete, and finished by the deadline?"""
    need = {(t, a, r) for t in all_tasks for a in ("none", "node") for r in range(1, FULL_RUNS + 1)}
    have = {(r["task_id"], r["arm"], int(r["run"])): r for r in rows
            if r["model"] == model and r["arm"] in ("none", "node")}
    complete = len(all_tasks) >= FULL_N and need <= set(have)
    stamps = [have[k].get("finished_at") for k in need if k in have]
    if not complete:
        on_time = False
    elif any(s is None for s in stamps):
        on_time = None            # cannot tell from the rows: the operator must state it
    else:
        on_time = max(stamps) <= deadline
    return {"complete": complete, "on_time": on_time, "episodes": len(need & set(have)), "needed": len(need),
            "headline": "129" if complete and on_time else "48"}


def context_stats(rows, model, arm, tasks=None):
    rs = _sel(rows, model, arm, tasks)
    if not rs:
        return None

    def share(key, pred):
        v = [pred(r[key]) for r in rs if r.get(key) is not None]
        return sum(v) / len(v) if v else None
    comp = [r["compactions"] for r in rs if r.get("compactions") is not None]
    rates, _no_compaction = per_task_rates(rs, model, arm)      # single definition: stats/repeats.py
    return {"episodes": len(rs), "compacted": share("compactions", lambda c: c >= 1),
            "compactions_mean": sum(comp) / len(comp) if comp else None,
            "repeat_share": (sum(rates.values()) / len(rates)) if rates else None,
            "repeat_tasks": len(rates),
            "overflow": share("context_overflow", bool),
            "overflow_n": sum(1 for r in rs if r.get("context_overflow"))}


def recap_decision(rows, model, tasks):
    """Rule 3. Reads ONLY no-memory rows: the arm filter is the first thing done."""
    base = [r for r in rows if r["arm"] == "none"]
    s = context_stats(base, model, "none", tasks)
    if not s or s["compacted"] is None or s["repeat_share"] is None:
        return {"decision": "TBD: no-memory rows lack compactions / repeat_rate fields", **(s or {})}
    promote = s["compacted"] >= RECAP_MIN_COMPACTED and s["repeat_share"] >= RECAP_MIN_REPEAT
    return {"decision": "recap promoted to headline" if promote else "recap stays exploratory",
            "thresholds": {"compacted": RECAP_MIN_COMPACTED, "repeat_share": RECAP_MIN_REPEAT}, **s}


# ---------------------------------------------------------------- tables

def _f(v, pct=False):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{100 * v:.1f}%" if pct else f"{v:.3f}"
    return str(v)


def _ci(c):
    return "n/a" if c is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def _mcn(c):
    if c is None or not c["mcnemar"]:
        return "n/a"
    return "; ".join(f"r{m['run']}: {m['b']}/{m['c']}, p={m['p']:.3f}" for m in c["mcnemar"])


def _table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _median(v):
    v = sorted(x for x in v if x is not None)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else None


def build(rows, model, floor_ids, all_tasks):
    """All results tables, numbered as in the Writeup. Returns (markdown, summary dict)."""
    floor, full = set(floor_ids), set(all_tasks)
    status = full_status(rows, model, all_tasks)
    sets = (("48-task floor", floor, "headline" if status["headline"] == "48" else "secondary"),
            ("129 tasks", full, "headline" if status["headline"] == "129" else "exploratory"))
    out, summary = [], {"model": model, "full_129": status, "contrasts": {}}

    t1 = []
    for name, ts, role in sets:
        c = contrast(rows, model, "node", "none", ts)
        summary["contrasts"][f"{name}: memory-none"] = c
        t1.append([f"{name} ({role})", c["n_tasks"] if c else "n/a", "/".join(map(str, c["runs"])) if c else "n/a",
                   _f(c["y_rate"]) if c else "n/a",
                   _f(c["x_rate"]) if c else "n/a", _ci(c), _mcn(c), c["reading"] if c else "n/a"])
    out.append("Table 1: resolved rate, mean over runs per task\n\n" + _table(
        ["Task set", "tasks", "runs (memory/no memory)", "no memory", "memory", "Δ [95% CI]",
         "McNemar per run (b/c, p)", "reading"], t1))

    rec = recap_decision(rows, model, floor)
    summary["recap"] = rec
    t2 = []
    for name, ts, _role in sets:
        cs = [contrast(rows, model, x, y, ts) for x, y in (("node", "shuffled"), ("shuffled", "none"), ("recap", "none"))]
        for (x, y), c in zip((("node", "shuffled"), ("shuffled", "none"), ("recap", "none")), cs):
            summary["contrasts"][f"{name}: {LABEL[x]}-{LABEL[y]}"] = c
        t2.append([name] + [_ci(c) for c in cs] + [rec["decision"]])
    out.append("Table 2: controls and the recap arm\n\n" + _table(
        ["Task set", "memory − shuffled [95% CI]", "shuffled − no memory [95% CI]",
         "recap − no memory [95% CI]", "recap status"], t2))

    t3 = []
    for arm, label in ARMS:
        s = context_stats(rows, model, arm, floor)
        t3.append([label] + (["n/a"] * 4 if s is None else
                             [_f(s["compacted"], True), _f(s["compactions_mean"]), _f(s["repeat_share"]),
                              f"{s['overflow_n']}/{s['episodes']} ({_f(s['overflow'], True)})"]))
    out.append("Table 3: context measures per arm, 48-task floor set, all runs\n\n" + _table(
        ["Arm", "episodes compacted ≥1×", "compactions per episode",
         "repeated actions per compaction (mean of per-task rates)",
         "context overflows"], t3))

    t4 = []
    for name, ts, _role in sets:
        for arm in ("none", "node"):
            fr = flip_rates(run_matrix(_sel(rows, model, arm, ts), model, arm))
            t4.append([f"{LABEL[arm]} / {name.split('-')[0].split()[0]}", fr["runs"], _f(fr["pairwise"], True),
                       _f(fr["any_flip"], True)])
    out.append("Table 4: run-to-run flip rate between identical runs\n\n" + _table(
        ["Arm / task set", "runs", "pairwise flip rate", "tasks not identical in all runs"], t4))

    headline = full if status["headline"] == "129" else floor
    t5 = []
    repo_of = {r["task_id"]: r["repo"] for r in rows}
    for repo in REPOS:
        ts = {t for t in headline if repo_of.get(t) == repo}
        a, b = task_means(rows, model, "none", ts), task_means(rows, model, "node", ts)
        common = sorted(set(a) & set(b))
        if common:
            na, nb = sum(a[t] for t in common) / len(common), sum(b[t] for t in common) / len(common)
            t5.append([repo, len(common), _f(na), _f(nb), f"{nb - na:+.3f}"])
        else:
            t5.append([repo, 0, "n/a", "n/a", "n/a"])
    out.append(f"Table 5: memory − no memory by evaluated repository (headline set: {status['headline']})\n\n" + _table(
        ["Fold", "tasks", "no memory", "memory", "Δ"], t5))

    t6 = []
    for arm, label in ARMS:
        rs = _sel(rows, model, arm, floor)
        if not rs:
            t6.append([label] + ["n/a"] * 6)
            continue
        succ = sum(int(r["success"]) for r in rs)
        tok = sum((r.get("tokens_in") or 0) + (r.get("tokens_out") or 0) for r in rs)
        mean = lambda k: (sum(r[k] for r in rs if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in rs))
                          if any(r.get(k) is not None for r in rs) else None)
        wall = _median([r.get("wall_s") for r in rs])
        t6.append([label, _f(_median([r.get("tool_calls") for r in rs])), _f(_median([r.get("model_calls") for r in rs])),
                   f"{tok / succ:,.0f}" if succ and tok else "n/a", _f(wall / 60 if wall is not None else None),
                   _f(mean("recalls")), _f(mean("lessons_shown"))])
    out.append("Table 6: process and cost per arm, 48-task floor set\n\n" + _table(
        ["Arm", "tool calls (median)", "model calls (median)", "tokens per resolved task", "agent minutes (median)",
         "recalls per task", "lessons shown per task"], t6))

    head = (f"Model: {model}. Headline set: {status['headline']} "
            f"(129-task run complete: {status['complete']}, on time: {status['on_time']}, "
            f"{status['episodes']}/{status['needed']} episodes).")
    if model.startswith("FAKE"):
        head = "FAKE DATA: synthetic rows, no model was run. NOT RESULTS.\n\n" + head
    return head + "\n\n" + "\n\n".join(out) + "\n", summary


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rows", nargs="*", help="rows.jsonl files; none = fake mode")
    ap.add_argument("--floor-list", help="file with the committed 48 task ids, one per line")
    ap.add_argument("--model", default=MODEL_31B)
    ap.add_argument("--out", default="nb-out")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.rows:
        rows, model = read_rows(a.rows), a.model
    else:
        write_rows(out / "fake" / "rows.jsonl", fake_rows())
        rows, model = read_rows([out / "fake" / "rows.jsonl"]), FAKE_MODEL
    listed = Path(a.floor_list).read_text().split() if a.floor_list else None
    all_tasks = sorted({r["task_id"] for r in rows if r["model"] == model})
    floor_ids, how = floor_set([r for r in rows if r["model"] == model], listed)
    md, summary = build(rows, model, floor_ids, all_tasks)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper-tables.md").write_text(f"Floor set: {how}\n\n" + md, encoding="utf-8")
    sys.stdout.write(md)


if __name__ == "__main__":  # NOTEBOOK-STRIP: make_notebook.py drops everything from this line
    main()
