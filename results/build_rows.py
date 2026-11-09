"""Build the committed result rows from raw run outputs (operator side; the raw traces are NOT
committed). One row per (backend, arm, run, task): outcomes and aggregates only, no task text, no
patch, no tool arguments.

    python results/build_rows.py <out.jsonl> backend=arm=run=<results_dir>[,<results_dir>] ...
                                 [--recap-submission <dir>] [--exploratory-partial backend=arm=run ...]

Rows of a (backend, arm, run) named with --exploratory-partial get exploratory_partial=true: the 7
finished Modal recap run-2 task-runs (a biased subset of short requests tasks), declared exploratory
in DESIGN.md section 10 (Modal pairing rule). No registered figure uses them (results/analyze_rows.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.swegemma import read_trace  # noqa: E402
from memory import plumbing as pl  # noqa: E402
from stats.caps import cap_hits  # noqa: E402
from stats.hypotheses import compactions_before_first_submit, leaked_thought  # noqa: E402
from stats.repeats import compaction_turns, repeated_actions  # noqa: E402

KEEP_PLUMBING = ("skill_status", "skill_error", "notes_written", "writes_in_task_dir", "explorer_injections",
                 "injected_share", "plumbing_failed")


def drop_post_submit(steps):
    first, kept = None, []
    for s in steps:
        if s["tool"] == "submit_patch" and first is not None:
            continue
        kept.append(s)
        if s["tool"] == "submit_patch" and '"status": "ok"' in (s["output"] or ""):
            first = s["turn"]
    return kept


def row(backend, arm, run, d, r, sub):
    t = r["instance_id"]
    p = d / "traces" / f"trace_{t}.json"
    tr = read_trace(p)
    pt = tr["root_prompt_tokens"]
    c = compaction_turns(pt)
    err = str(r.get("error") or "")
    out = {"backend": backend, "arm": arm, "run": run, "task_id": t, "repo": t.split("_")[0],
           "success": int(bool(r["resolved"])), "context_overflow": "ContextWindowExceeded" in err,
           "overflow_phase": (None if "ContextWindowExceeded" not in err else
                              "before first compaction" if not c else "after first compaction"),
           "overflow_last_prompt": pt[-1] if "ContextWindowExceeded" in err and pt else None,
           "compactions": len(c), "tool_calls": r.get("tool_calls"), "model_calls": tr["model_calls"],
           "wall_s": r.get("duration_seconds"), "tokens_in": tr["tokens_in"], "tokens_out": tr["tokens_out"]}
    for tag, steps in (("", tr["steps"]), ("_posthoc", drop_post_submit(tr["steps"]))):
        rep = repeated_actions(steps, c)
        after = [s for s in steps if s.get("turn") is not None and c and s["turn"] >= c[0]]
        out[f"repeats_after_compaction{tag}"] = rep
        out[f"repeat_rate{tag}"] = rep / len(c) if c else None
        out[f"repeats_per_call_after_compaction{tag}"] = rep / len(after) if after else None
    out["post_submit_submits"] = sum(1 for s in tr["steps"] if s["tool"] == "submit_patch") - \
        sum(1 for s in drop_post_submit(tr["steps"]) if s["tool"] == "submit_patch")
    out["submit_loop"] = out["post_submit_submits"] >= 10
    out["compactions_before_first_submit"] = compactions_before_first_submit(p)[0]
    out["leaked_thought"] = leaked_thought(p)
    out.update({f"cap_{k}": v for k, v in cap_hits(err, p).items()})
    if arm != "none" and sub:
        patch_f = d / "patches" / f"{t}.patch"
        plumb = pl.plumbing(p, arm, patch_f.read_text() if patch_f.exists() else "", pl.injecting_agents(sub))
        out.update({k: plumb.get(k) for k in KEEP_PLUMBING})
    return out


def main(argv):
    out_path, specs, sub, partial = Path(argv[0]), [], None, set()
    i = 1
    while i < len(argv):
        if argv[i] == "--recap-submission":
            sub = Path(argv[i + 1]); i += 2; continue
        if argv[i] == "--exploratory-partial":
            b_, a_, r_ = argv[i + 1].split("="); partial.add((b_, a_, int(r_))); i += 2; continue
        backend, arm, run, dirs = argv[i].split("=", 3)
        specs.append((backend, arm, int(run), [Path(x) for x in dirs.split(",")])); i += 1
    rows = []
    for backend, arm, run, dirs in specs:
        for d in dirs:
            for line in open(d / "task_results.jsonl"):
                r = row(backend, arm, run, d, json.loads(line), sub)
                r["exploratory_partial"] = (backend, arm, run) in partial
                rows.append(r)
    with open(out_path, "w") as f:
        for r in sorted(rows, key=lambda r: (r["backend"], r["arm"], r["run"], r["task_id"])):
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(len(rows), "rows ->", out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
