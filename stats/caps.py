"""Budget-cap hits per run, with the harness's exact wording (swegemma 0.2.11, read from its source:
harness/agent_runner.py and context.py). Descriptive, reported per arm next to overflow
(DESIGN.md section 10). A cap can end a run through the run's error, or show only as a tool
result while the agent stops on its own, so both places are read.

  calls   "Agent exceeded tool call budget (N calls)"  /  tool result "Tool call budget exhausted (N calls)"
  time    "Agent exceeded session timeout (N min)"     /  tool result "Session time budget exhausted (N minutes)"
  turns   "Agent exceeded turns budget (N turns)"      /  tool result "Turn budget exhausted (N turns)"
"""
from __future__ import annotations

import json

RUN_ERRORS = {"calls": "Agent exceeded tool call budget", "time": "Agent exceeded session timeout",
              "turns": "Agent exceeded turns budget"}
TOOL_RESULTS = {"calls": "Tool call budget exhausted", "time": "Session time budget exhausted",
                "turns": "Turn budget exhausted"}


def cap_hits(error, trace_path=None):
    """{'calls', 'time', 'turns'} -> bool, from a task_results error and (optionally) the trace's tool
    results (observation.results[] or observation.content)."""
    err = str(error or "")
    out = {k: v in err for k, v in RUN_ERRORS.items()}
    if trace_path is not None:
        try:
            t = json.load(open(trace_path, encoding="utf-8"))
        except (OSError, ValueError):
            return out
        text = []
        for s in t.get("steps", []):
            obs = s.get("observation") or {}
            for r in obs.get("results", []) or []:
                text.append(str(r.get("content")))
            if obs.get("content") is not None:
                text.append(str(obs.get("content")))
        blob = "\n".join(text)
        for k, v in TOOL_RESULTS.items():
            out[k] = out[k] or v in blob
    return out
