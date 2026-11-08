"""Key secondary outcome (DESIGN.md section 10, 2026-11-08, fixed before any memory-arm run):
the repeated-action rate after context compaction.

Per task (one attempt), counting only the root agent's own tool calls (sub-agent calls run in
their own context):
  - a compaction is a root-agent turn whose prompt tokens fall below COMPACTION_DROP x the previous
    turn's, when the previous turn had more than COMPACTION_MIN prompt tokens (the traces carry no
    explicit compaction event; the harness compacts at about 14k tokens);
  - a repeat is a tool call, at or after the first compaction's turn, with the same tool and
    identical normalized arguments (whitespace collapsed, paths normalized) as an earlier call in
    the same task; a read_file repeat counts only if that file was not edited (edit_file or
    write_file) since the most recent identical read;
  - repeat_rate = repeats / compactions; undefined (None) for an attempt with no compaction.
Standard library only.
"""
from __future__ import annotations

import json
import re

COMPACTION_DROP = 0.7
COMPACTION_MIN = 8000
READ_TOOLS = ("read_file",)
EDIT_TOOLS = ("edit_file", "write_file")
PATH_KEYS = ("filepath", "file_path", "path", "filename")


def compaction_turns(prompt_tokens):
    pt = [int(p or 0) for p in prompt_tokens]
    return [i for i in range(1, len(pt)) if pt[i - 1] > COMPACTION_MIN and pt[i] < COMPACTION_DROP * pt[i - 1]]


def norm_path(p):
    p = " ".join(str(p).split())
    p = re.sub(r"/{2,}", "/", p)
    p = re.sub(r"(^|(?<=\s))/workspace/", r"\1", p)
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/") or p


def _norm(v, key=None):
    if isinstance(v, dict):
        return {k: _norm(x, k) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, str):
        return norm_path(v) if key in PATH_KEYS else re.sub(r"(^|(?<=\s))/workspace/", r"\1",
                                                               " ".join(v.split()))
    return v


def _path(step):
    a = step.get("args") or {}
    for k in PATH_KEYS + ("_",):
        if isinstance(a.get(k), str):
            return norm_path(a[k])
    return None


def repeated_actions(steps, compactions):
    """Number of repeated tool calls at or after the first compaction (module docstring)."""
    if not compactions:
        return 0
    first = compactions[0]
    root = [s for s in steps if s.get("turn") is not None]
    seen, last_edit, n = {}, {}, 0
    for i, s in enumerate(root):
        key = (s["tool"], json.dumps(_norm(s.get("args") or {}), sort_keys=True))
        prev = seen.get(key)
        if prev is not None and s["turn"] >= first:
            if s["tool"] in READ_TOOLS:
                if last_edit.get(_path(s), -1) < prev:
                    n += 1
            else:
                n += 1
        if s["tool"] in EDIT_TOOLS:
            last_edit[_path(s)] = i
        seen[key] = i
    return n


def repeat_metrics(attempt):
    """{'compactions', 'repeats_after_compaction', 'repeat_rate'} for one read_run attempt."""
    c = compaction_turns(attempt.get("root_prompt_tokens") or [])
    rep = repeated_actions(attempt.get("steps") or [], c)
    return {"compactions": len(c), "repeats_after_compaction": rep,
            "repeat_rate": (rep / len(c)) if c else None}


# ---------------------------------------------------------------------------------------------
# Pre-registered statistics for the key secondary outcome (DESIGN.md section 10, 2026-11-09):
#   unit       the per-task rate: repeats / compactions, averaged over that task's runs that
#              compacted (a run with 0 compactions has no rate);
#   exclusion  a task with no rate in either arm is excluded, and the exclusions are counted;
#   key        the mean of the per-task paired differences (x - y), with a paired bootstrap 95% CI
#              over tasks (10,000 resamples, seed 20261107);
#   robust     the exact Wilcoxon signed-rank test on those differences (zeros dropped, mid-ranks
#              for ties) and the 10% trimmed mean of the differences.
# ---------------------------------------------------------------------------------------------

def trimmed_mean(values, prop=0.10):
    v = sorted(float(x) for x in values)
    if not v:
        raise ValueError("no values")
    k = int(prop * len(v))
    v = v[k:len(v) - k] if len(v) - 2 * k > 0 else v
    return sum(v) / len(v)


def wilcoxon_signed_rank(diffs):
    """Exact two-sided Wilcoxon signed-rank test. Zeros are dropped; tied |d| get mid-ranks.
    The exact null distribution is computed on doubled ranks (integers), so ties stay exact."""
    d = [float(x) for x in diffs if float(x) != 0.0]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "p": 1.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks2 = [0] * n                      # doubled mid-ranks
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = (i + 1) + (j + 1)          # 2 x mid-rank
        i = j + 1
    w2_plus = sum(r for r, x in zip(ranks2, d) if x > 0)
    total2 = sum(ranks2)
    counts = {0: 1}                       # exact distribution of the doubled W+ under H0
    for r in ranks2:
        nxt = dict(counts)
        for s, c in counts.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    allc = 2 ** n
    lo = min(w2_plus, total2 - w2_plus)
    p = min(1.0, 2 * sum(c for s, c in counts.items() if s <= lo) / allc)
    return {"n": n, "w_plus": w2_plus / 2, "w_minus": (total2 - w2_plus) / 2, "p": p}


def per_task_rates(rows, model, arm):
    """{task_id: mean repeat_rate over the runs of (model, arm) that compacted}, and the
    task ids whose runs all had 0 compactions."""
    g, seen = {}, set()
    for r in rows:
        if r["model"] == model and r["arm"] == arm:
            seen.add(r["task_id"])
            if r.get("repeat_rate") is not None:
                g.setdefault(r["task_id"], []).append(float(r["repeat_rate"]))
    rates = {t: sum(v) / len(v) for t, v in g.items()}
    return rates, sorted(seen - set(rates))


def paired_repeat_stats(rows, x, y, model, n_boot=10_000, seed=20261107):
    """The pre-registered report for the repeated-action rate, x against y."""
    from . import paired_bootstrap_ci
    rx, zx = per_task_rates(rows, model, x)
    ry, zy = per_task_rates(rows, model, y)
    both = sorted(set(rx) & set(ry))
    out = {"model": model, "x": x, "y": y, "label": "key secondary: repeated actions per compaction",
           "mean_" + x: (sum(rx.values()) / len(rx)) if rx else None,
           "mean_" + y: (sum(ry.values()) / len(ry)) if ry else None,
           "n_tasks_paired": len(both),
           "excluded_no_compaction": {x: len(zx), y: len(zy),
                                      "either_arm": len((set(rx) | set(zx) | set(ry) | set(zy)) - set(both))}}
    if both:
        diffs = [rx[t] - ry[t] for t in both]
        out.update(paired_bootstrap_ci(diffs, n_boot=n_boot, alpha=0.05, seed=seed))
        out["trimmed_mean_10"] = trimmed_mean(diffs, 0.10)
        out["wilcoxon"] = wilcoxon_signed_rank(diffs)
    return out
