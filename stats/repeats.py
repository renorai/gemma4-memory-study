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
