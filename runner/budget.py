"""Compute planner: arithmetic on MEASURED throughput (kernels/throughput.py), never a guess.

Cut order, fixed in PREREG.md section 7 before any arm run: exploratory arms on 12B, then
exploratory arms on 31B, then tasks (stratified by repo, down to MIN_TASKS). Headline runs
(3 x none, 3 x node, per model) are never cut.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

HEADLINE_RUNS = 6            # 3 x none + 3 x node
EXPLORATORY_ARMS = ("text", "node_khop", "shuffled")
MIN_TASKS = 48
MODELS = ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct")


@dataclass
class Scope:
    n_tasks: int
    exploratory: dict = field(default_factory=dict)   # model -> tuple of arms
    hours: float = 0.0
    fits: bool = True
    cuts: list = field(default_factory=list)


def episodes(n_tasks, n_exploratory):
    return n_tasks * (HEADLINE_RUNS + n_exploratory)


def hours_needed(n_tasks, exploratory, tasks_per_hour, distill_h_per_model=0.5):
    """tasks_per_hour: {model: measured agent episodes per wall-hour at the chosen concurrency}."""
    return sum(episodes(n_tasks, len(exploratory.get(m, ()))) / tasks_per_hour[m] + distill_h_per_model
               for m in tasks_per_hour)


def choose_scope(tasks_per_hour, hours_available, n_all=129, min_tasks=MIN_TASKS):
    models = [m for m in MODELS if m in tasks_per_hour]
    s = Scope(n_all, {m: EXPLORATORY_ARMS for m in models})
    s.hours = hours_needed(s.n_tasks, s.exploratory, tasks_per_hour)
    for m in reversed(models):                 # 12B first, then 31B
        if s.hours <= hours_available:
            break
        s.exploratory[m] = ()
        s.cuts.append(f"exploratory arms dropped for {m}")
        s.hours = hours_needed(s.n_tasks, s.exploratory, tasks_per_hour)
    while s.hours > hours_available and s.n_tasks - 4 >= min_tasks:
        s.n_tasks -= 4
        s.hours = hours_needed(s.n_tasks, s.exploratory, tasks_per_hour)
    if s.n_tasks < n_all:
        s.cuts.append(f"tasks cut to {s.n_tasks} (stratified by repo)")
    s.fits = s.hours <= hours_available
    return s


def stratified_subset(tasks, n):
    """Deterministic, outcome-blind: per repo, order tasks by sha256(task_id); take round-robin
    across repos (repos in name order) until n tasks."""
    by = {}
    for t in tasks:
        by.setdefault(t.repo, []).append(t)
    for r in by:
        by[r].sort(key=lambda t: hashlib.sha256(t.task_id.encode()).hexdigest())
    out, i = [], 0
    while len(out) < min(n, len(tasks)):
        for r in sorted(by):
            if i < len(by[r]) and len(out) < n:
                out.append(by[r][i])
        i += 1
    return out
