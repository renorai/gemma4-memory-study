"""Per-task plumbing assertions for the memory arms, read from one ATIF trace (operator side; not
shipped with the skill). They make a broken memory setup visible per run instead of silently
scoring it as "memory did not help" (DESIGN.md section 10, 2026-11-09).

    plumbing(trace_path, arm, patch="") -> dict of row fields:
      skill_status            first status line of the first recall: "on", "off", "error" or None
                              (recall never ran)
      skill_error             the first `memory error:` line any graph-memory script printed, or None
      notes_written           workmem `note` acknowledgements
      writes_in_task_dir      every note landed in ONE notes file inside the task's own tmp folder
                              (the file the status line named, under renor-notes/, not /workspace),
                              and the patch holds no notes or task file
      recaps                  recap pages shown ("where was I"), on request or every 6th note
      recall_runs_after_compaction
                              after EACH compaction (stats/repeats.py:compaction_turns) a recap page
                              was shown before the next compaction or the end; None without
                              compactions
      recap_tokens            tokens of all recap pages shown (memory/renor.py:count_tokens)
      plumbing_failed         recap arms only: status not "on", a memory error, a write outside the
                              task folder, or a compaction not followed by a recap; None elsewhere

It reads the trace file itself because harness.swegemma.read_trace takes tool outputs from
observation.results[], and the real swegemma traces carry a single tool output in
observation.content; that shared reader is frozen, so this reader is the additive one.
"""
from __future__ import annotations

import json
import re

from .renor import count_tokens
from .workmem import RECAP_ARMS

_NOTES_PATH = re.compile(r"\[([^\]\n]*/renor-notes/[^\]\n]*notes\.jsonl)\]")
_STATUS_NOTES = re.compile(r"working notes: on \(notes (\S+notes\.jsonl);")


def _stdout(out):
    """A skill script's printed text: run_skill_script returns JSON with `stdout` (and `stderr`)."""
    try:
        d = json.loads(out)
    except (TypeError, ValueError):
        return out or ""
    if isinstance(d, dict) and ("stdout" in d or "stderr" in d):
        err = str(d.get("stderr") or "")
        return str(d.get("stdout") or "") + ("\n" + err if err else "")
    return out if isinstance(out, str) else json.dumps(d)


def read_steps(trace_path):
    """[(turn or None, tool, args dict, output text)] and the root agent's prompt tokens per turn,
    with the same root/turn rule as harness.swegemma.read_trace."""
    with open(trace_path, encoding="utf-8") as f:
        t = json.load(f)
    steps, root, turn, root_pt = [], None, -1, []
    for s in t.get("steps", []):
        if s.get("source", "agent") != "agent":
            continue
        author = (s.get("extra") or {}).get("author")
        if root is None:
            root = author
        is_root = author == root
        if is_root:
            turn += 1
            root_pt.append(int((s.get("metrics") or {}).get("prompt_tokens") or 0))
        obs = s.get("observation") or {}
        results = {r.get("source_call_id"): r.get("content") for r in obs.get("results", []) or []}
        calls = s.get("tool_calls", []) or []
        for c in calls:
            name = c.get("function_name") or c.get("name") or (c.get("function") or {}).get("name")
            args = c.get("arguments", (c.get("function") or {}).get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_": args}
            out = results.get(c.get("tool_call_id") or c.get("id"))
            if out is None and len(calls) == 1:
                out = obs.get("content")
            out = out if isinstance(out, str) else json.dumps(out) if out is not None else ""
            steps.append((turn if is_root else None, name, args or {}, out))
    return steps, root_pt


def _is_skill(name, args, script):
    return name == "run_skill_script" and str(args.get("file_path", "")).strip('"').endswith(script)


def plumbing(trace_path, arm, patch=""):
    from stats.repeats import compaction_turns
    steps, root_pt = read_steps(trace_path)
    status, error, status_notes = None, None, None
    acks, recap_turns, recap_tokens = [], [], 0
    for turn, name, args, out in steps:
        recall = _is_skill(name, args, "recall.py")
        wm = _is_skill(name, args, "workmem.py")
        if not (recall or wm):
            continue
        text = _stdout(out)
        first = text.strip().splitlines()[0] if text.strip() else ""
        if recall and status is None and first:
            status = "on" if first.startswith("working notes: on") else \
                "off" if first.startswith("working notes: off") else "error" if first.startswith("memory error") else None
            m = _STATUS_NOTES.search(first)
            status_notes = m.group(1) if m else None
        for line in text.splitlines():
            if line.startswith("memory error") and error is None:
                error = line.strip()[:300]
        if wm and first.startswith("noted #"):
            m = _NOTES_PATH.search(first)
            acks.append(m.group(1) if m else None)
        if "where was I" in text:
            recap_turns.append(turn)
            recap_tokens += count_tokens(text[text.index("where was I"):])
    comps = compaction_turns(root_pt)
    after = None
    if comps:
        bounds = comps[1:] + [float("inf")]
        after = all(any(t is not None and c <= t < b for t in recap_turns) for c, b in zip(comps, bounds))
    files = set(acks)
    leaked = any(w in (patch or "") for w in ("renor-notes", "notes.jsonl", "task.md"))
    in_dir = bool(acks) and None not in files and len(files) == 1 and not leaked and all(
        "/renor-notes/" in f and "/workspace/" not in f and (status_notes is None or f == status_notes)
        for f in files)
    row = {"skill_status": status, "skill_error": error, "notes_written": len(acks),
           "notes_file": next(iter(files)) if len(files) == 1 else None,
           "writes_in_task_dir": in_dir, "recaps": len(recap_turns),
           "recall_runs_after_compaction": after, "recap_tokens": recap_tokens}
    if arm in RECAP_ARMS:
        row["plumbing_failed"] = bool(status != "on" or error or not in_dir or after is False)
    else:
        row["plumbing_failed"] = None
    return row
