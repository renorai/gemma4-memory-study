"""Detectors for two directional secondary hypotheses (DESIGN.md section 10, 2026-11-09). They were
frozen as code before any Kaggle recap-arm result was read. Modal (H100) is the discovery set,
Kaggle's 32 tasks (L4x4) the replication set, run 1 primary.

  H-a  recap has more compactions up to the first ok submit_patch than none.
       Unit: per task, `compactions_before_first_submit` of run 1; paired one-sided Wilcoxon
       signed-rank (recap > none) over tasks.
  H-b  recap has more runs with leaked thought text (`<|channel>`) in the coder's output than none.
       Unit: run (run 1); one-sided Fisher exact (recap > none). Also the task-level version
       (a task counts if any of its runs leaked).

Both detectors read the raw ATIF trace file. Standard library only.
"""
from __future__ import annotations

import json
import math

LEAK = "<|channel>"


def _coder_steps(trace):
    agent = [s for s in trace.get("steps", []) if s.get("source", "agent") == "agent"]
    root = (agent[0].get("extra") or {}).get("author") if agent else None
    return [s for s in agent if (s.get("extra") or {}).get("author") == root]


def leaked_thought(trace_path):
    """True when any root-agent (coder) message carries `<|channel>`: reasoning text that leaked
    into the visible output."""
    t = json.load(open(trace_path, encoding="utf-8"))
    return any(isinstance(s.get("message"), str) and LEAK in s["message"] for s in _coder_steps(t))


def _ok(content):
    if content is None:
        return False
    if isinstance(content, (dict, list)):
        content = json.dumps(content)
    return '"status": "ok"' in content or "'status': 'ok'" in content


def compactions_before_first_submit(trace_path):
    """Compactions (stats/repeats.py:compaction_turns over coder prompt tokens) at coder turns up to
    and including the turn of the first submit_patch call whose result is ok. With no ok submit,
    every compaction of the run counts. Returns (count, first_ok_submit_turn or None)."""
    from .repeats import compaction_turns
    t = json.load(open(trace_path, encoding="utf-8"))
    steps = _coder_steps(t)
    pt, first = [], None
    for i, s in enumerate(steps):
        pt.append(int((s.get("metrics") or {}).get("prompt_tokens") or 0))
        if first is not None:
            continue
        obs = s.get("observation") or {}
        res = {r.get("source_call_id"): r.get("content") for r in obs.get("results", []) or []}
        for k, c in enumerate(s.get("tool_calls", []) or []):
            name = c.get("function_name") or c.get("name") or (c.get("function") or {}).get("name")
            if name != "submit_patch":
                continue
            out = res.get(c.get("tool_call_id") or c.get("id"))
            if out is None and not res and len(s.get("tool_calls") or []) == 1:
                out = obs.get("content")
            if _ok(out):
                first = i
                break
    comps = compaction_turns(pt)
    n = len([c for c in comps if first is None or c <= first])
    return n, first


def fisher_one_sided(a, n1, b, n2):
    """P(X >= a) for X ~ hypergeometric: a of n1 'recap' runs leaked, b of n2 'none' runs leaked."""
    k, n = a + b, n1 + n2
    def h(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    return sum(h(x) for x in range(a, min(n1, k) + 1))


def wilcoxon_one_sided(diffs):
    """Exact one-sided Wilcoxon signed-rank, H1: diffs > 0. Zeros dropped, mid-ranks for ties; the
    exact null distribution is computed on doubled ranks. Returns n, W+, W- and p = P(W+ >= observed)."""
    d = [float(x) for x in diffs if float(x) != 0.0]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "p_one_sided": 1.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks2 = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = (i + 1) + (j + 1)
        i = j + 1
    w2 = sum(r for r, x in zip(ranks2, d) if x > 0)
    counts = {0: 1}
    for r in ranks2:
        nxt = dict(counts)
        for s, c in counts.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    p = sum(c for s, c in counts.items() if s >= w2) / 2 ** n
    return {"n": n, "w_plus": w2 / 2, "w_minus": (sum(ranks2) - w2) / 2, "p_one_sided": p}
