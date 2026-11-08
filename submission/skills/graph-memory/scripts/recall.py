#!/usr/bin/env python3
"""graph-memory skill entry point: print the lessons that apply to the code being worked on.

    python3 recall.py --task-file /tmp/task.md --repo /workspace [--symbol NAME ...]

Runs inside the task sandbox: standard library only, no network, no model call, and it writes
no file (no bytecode either). The retrieval code is `memory/` next to this file, copied verbatim
from research/kaggle-gemma4/memory/ by runner/build_submission.py. The arm and its parameters
come from ../assets/config.yaml, the lessons from ../assets/lessons.txt (JSON lines; the harness
allows .txt but not .jsonl).

Two ways it is run, and both work:
- ADK `run_skill_script`: ADK copies only the skill's scripts/, assets/ and references/ into a
  fresh temporary folder and runs this file with that folder as the working directory. The data
  therefore lives in assets/, and the repository is not the working directory: pass --repo, or
  it defaults to /workspace (the task's working folder).
- `run_command` from the repository root: --repo defaults to the working directory.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.dont_write_bytecode = True  # never leave __pycache__ in the skill, the repo or anywhere else

WORKSPACE = "/workspace"
HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from memory import LessonStore, RepoIndex, RetrievalConfig, make_arm, render  # noqa: E402

INT_FIELDS = {"k", "char_budget", "per_task_cap", "n_seeds", "hops", "max_region"}
FLOAT_FIELDS = {"hop_decay", "bm25_k1", "bm25_b"}


def read_config(path):
    """Flat `key: value` YAML subset; comments and blank lines ignored."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = (x.strip() for x in line.split(":", 1))
            out[k] = int(v) if k in INT_FIELDS else float(v) if k in FLOAT_FIELDS else v
    return out


def data_file(skill_dir, name):
    """assets/<name> (the layout ADK materializes); the skill root as a fallback."""
    for d in (os.path.join(skill_dir, "assets"), skill_dir):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return os.path.join(skill_dir, "assets", name)


def default_repo(cwd=None, skill_dir=SKILL, workspace=WORKSPACE):
    """The working directory, unless it is the skill's own folder (ADK's temporary copy)."""
    cwd = os.path.realpath(cwd or os.getcwd())
    skill = os.path.realpath(skill_dir)
    if cwd == skill or cwd.startswith(skill + os.sep):
        return workspace
    return cwd


def split_symbols(values):
    """--symbol may arrive as one value per flag, or as one list-like or comma-separated string."""
    out = []
    for v in values:
        for s in re.split(r"[\s,\[\]'\"]+", str(v)):
            if s and s not in out:
                out.append(s)
    return out


def recall(task_text, symbols, repo, skill_dir=SKILL):
    cfg = read_config(data_file(skill_dir, "config.yaml"))
    arm = cfg.pop("arm", "none")
    lessons = data_file(skill_dir, "lessons.txt")
    store = LessonStore.load(lessons) if os.path.exists(lessons) else LessonStore()
    rcfg = RetrievalConfig(**{k: v for k, v in cfg.items() if k in INT_FIELDS | FLOAT_FIELDS})
    memory = make_arm(arm, store, rcfg)
    index = RepoIndex.build(repo) if arm != "text" else None
    return memory.recall(index, task_text, list(symbols))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-file")
    ap.add_argument("--task", default="")
    ap.add_argument("--symbol", action="append", default=[])
    ap.add_argument("--repo", default=None)
    a = ap.parse_args(argv)
    repo = a.repo or default_repo()
    if not os.path.isdir(repo):
        print(f"memory unavailable (no repository at {repo}; pass --repo)")
        return 0
    text = a.task
    if a.task_file and os.path.exists(a.task_file):
        with open(a.task_file, encoding="utf-8", errors="replace") as f:
            text = (text + "\n" + f.read())[:20000]
    try:
        got = recall(text, split_symbols(a.symbol)[:5], repo)
    except Exception as e:  # never break the agent's episode over memory
        print(f"memory unavailable ({type(e).__name__})")
        return 0
    print(render(got) or "no lessons for this code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
