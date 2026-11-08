"""Compute planner: arithmetic on MEASURED throughput (kernels/throughput.py), never a guess.

Cut order, fixed in PREREG.md section 7 before any arm run: exploratory arms on 12B, then
exploratory arms on 31B, then tasks (stratified by repo, down to MIN_TASKS). Headline runs are
never cut by this rule. Deviation (DESIGN.md section 10, 2026-11-08): headline runs are 2 x none
+ 2 x node per model, not 3 + 3, because measured throughput and the free GPU quota left before
the deadline do not fit 3.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

HEADLINE_RUNS = 4            # 2 x none + 2 x node (DESIGN.md section 10, 2026-11-08; PREREG said 3 + 3)
EXPLORATORY_ARMS = ("text", "node_khop", "shuffled")
MIN_TASKS = 48
SUBSET_SEED = 20261107      # published; the PREREG.md statistics seed
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


def split_every_third(tasks):
    """Deterministic split of a subset in its seed order: every third task (positions 3, 6, 9, ...)
    and the rest. Used to run one subset on two runners (DESIGN.md section 10)."""
    third = [t for i, t in enumerate(tasks) if i % 3 == 2]
    rest = [t for i, t in enumerate(tasks) if i % 3 != 2]
    return rest, third


def stratified_subset(tasks, n, seed=SUBSET_SEED):
    """Seeded random, outcome-blind and length-blind: per repo, order tasks by
    sha256(f"{seed}:{task_id}") (a random permutation fixed by the published seed); take
    round-robin across repos (repos in name order) until n tasks. Every repo keeps tasks, so
    leave-one-repo-out still works, and every arm and run gets the same subset."""
    by = {}
    for t in tasks:
        by.setdefault(t.repo, []).append(t)
    for r in by:
        by[r].sort(key=lambda t: hashlib.sha256(f"{seed}:{t.task_id}".encode()).hexdigest())
    out, i = [], 0
    while len(out) < min(n, len(tasks)):
        for r in sorted(by):
            if i < len(by[r]) and len(out) < n:
                out.append(by[r][i])
        i += 1
    return out
