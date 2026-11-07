"""Pre-run probe whose outcome is reported next to the headline (PREREG.md section 6).

Key recurrence: for each dev task, do the strong transferable keys of its gold fix site
(symbols enclosing the gold patch's hunks) also occur at the fix site of some task in ANOTHER
repo? This uses gold patches and source only, never a model outcome, so it can run before any
arm. It bounds what the node arms can do under leave-one-repo-out: a lesson can only reach a
task whose code shares a strong key with code lessons were written on.
"""
from __future__ import annotations

from harness.swegemma import patch_old_lines
from memory.keys import STRONG, key_type, keys_for


def fix_site_keys(task, index):
    keys = set()
    for path, lines in patch_old_lines(task.gold_patch).items():
        for q in index.symbols_at(path, lines):
            keys.update(k for k in keys_for(index, q) if key_type(k) in STRONG)
    return keys


def key_recurrence_probe(tasks, index_of):
    keys = {t.task_id: fix_site_keys(t, index_of(t)) if index_of(t) else set() for t in tasks}
    by_repo = {}
    for t in tasks:
        by_repo.setdefault(t.repo, set()).update(keys[t.task_id])
    per_repo, hits = {}, 0
    for t in tasks:
        others = set().union(*(v for r, v in by_repo.items() if r != t.repo)) if len(by_repo) > 1 else set()
        hit = bool(keys[t.task_id] & others)
        hits += hit
        pr = per_repo.setdefault(t.repo, [0, 0])
        pr[0] += hit
        pr[1] += 1
    n = len(tasks)
    return {"n": n, "with_fix_site_keys": sum(bool(v) for v in keys.values()),
            "recurs_in_other_repo": hits, "rate": hits / n if n else None,
            "per_repo": {r: {"hits": a, "n": b} for r, (a, b) in sorted(per_repo.items())}}
