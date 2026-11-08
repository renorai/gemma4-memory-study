"""The experiment protocol (PREREG.md section 3), independent of how an eval is executed.

    1. baseline   arm none, R_HEADLINE runs, every task
    2. distil     run 1 of the baseline -> this model's lessons (self-distillation)
    3. headline   arm node, R_HEADLINE runs, leave-one-repo-out: for each repo, a submission
                  built with the OTHER repos' lessons only, evaluated on that repo's tasks
    4. explore    arms text, node_khop, shuffled: 1 run each, same LORO folds
    5. rows       one row per (model, arm, run, task) for stats/

An `Evaluator` is any callable (submission_dir, tasks, results_dir) -> None that leaves a
swegemma-format results directory behind: kernels/common.py runs the real `swegemma eval`;
fake_harness runs a scripted policy. Everything else is shared.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.swegemma import files_opened, is_recall, lesson_ids_shown, patch_files, read_run
from memory import LessonStore, RepoIndex, RetrievalConfig
from pipeline.distill import DistillStats, distill_attempt
from runner.build_submission import build
from stats.repeats import repeat_metrics

R_HEADLINE = 2                     # DESIGN.md section 10, 2026-11-08 (PREREG said 3)
HEADLINE = ("none", "node")
EXPLORATORY = ("text", "node_khop", "shuffled")
HIT_KS = (1, 3, 5)


def repos_of(tasks):
    return sorted({t.repo for t in tasks})


def eval_arm(evaluator, arm, tasks, store, model, run, out_root, cfg=RetrievalConfig()):
    """One run of one arm over `tasks`. Memory arms are evaluated fold by fold (LORO)."""
    out_root = Path(out_root)
    attempts = []
    folds = [None] if arm == "none" else repos_of(tasks)
    for repo in folds:
        tag = "all" if repo is None else f"fold-{repo}"
        base = out_root / model / arm / f"run{run}" / tag
        fold_tasks = tasks if repo is None else [t for t in tasks if t.repo == repo]
        report = build(arm, store if arm != "none" else None, model, base / "submission",
                       exclude_repos=() if repo is None else (repo,), source_model=model, cfg=cfg)
        (base / "build.json").write_text(json.dumps({k: v for k, v in report.items() if k != "files"},
                                                    indent=2, sort_keys=True))
        evaluator(base / "submission", fold_tasks, base / "results")
        attempts += read_run(base / "results", fold_tasks, model=model, arm=arm, run=run,
                             fold=repo or "all")
    return attempts


class IndexCache:
    def __init__(self, snapshot_of):
        self.snapshot_of, self._c = snapshot_of, {}

    def __call__(self, task):
        p = self.snapshot_of(task)
        if p is None:
            return None
        key = str(p)
        if key not in self._c:
            self._c[key] = RepoIndex.build(key)
        return self._c[key]


def distil(attempts, generate, index_of, stats=None):
    stats = stats if stats is not None else DistillStats()
    store = LessonStore()
    for a in attempts:
        ix = index_of(a)
        if ix is None:
            stats.dropped["no_snapshot"] += 1
            continue
        for lesson in distill_attempt(a, generate, ix, stats=stats):
            store.remember(lesson)   # Renor-style write: subject, time, replaces-link (memory/store.py)
    return store


def to_row(a, gold_files=()):
    opened = files_opened(a["steps"])
    row = {k: a[k] for k in ("model", "arm", "run", "task_id", "repo", "fold", "success",
                             "model_calls", "tokens_in", "tokens_out", "wall_s")}
    row["tool_calls"] = len(a["steps"])
    row["context_overflow"] = int(bool(a.get("context_overflow")))   # its own outcome, not a plain failure
    row.update(repeat_metrics(a))        # key secondary outcome (stats/repeats.py)
    row["recalls"] = sum(1 for s in a["steps"] if is_recall(s))
    row["lessons_shown"] = len(lesson_ids_shown(a["steps"]))
    gold = set(gold_files)
    for k in HIT_KS:
        row[f"hit@{k}"] = int(bool(gold) and bool(gold & set(opened[:k]))) if gold else None
    return row


def run_protocol(evaluator, generate, tasks, snapshot_of, model, out_root, *, runs=R_HEADLINE,
                 exploratory=EXPLORATORY, cfg=RetrievalConfig()):
    out_root = Path(out_root)
    gold = {t.task_id: patch_files(t.gold_patch) for t in tasks}
    by_id = {t.task_id: t for t in tasks}
    attempts = []
    for r in range(1, runs + 1):
        attempts += eval_arm(evaluator, "none", tasks, None, model, r, out_root, cfg)
    stats = DistillStats()
    cache = IndexCache(snapshot_of)
    store = distil([a for a in attempts if a["run"] == 1], generate, lambda a: cache(by_id[a["task_id"]]), stats)
    store.save(out_root / model / "lessons.jsonl")
    (out_root / model / "distill.json").write_text(json.dumps(
        {**stats.as_dict(), "key_types": dict(store.key_counts())}, indent=2, sort_keys=True))
    for r in range(1, runs + 1):
        attempts += eval_arm(evaluator, "node", tasks, store, model, r, out_root, cfg)
    for arm in exploratory:
        attempts += eval_arm(evaluator, arm, tasks, store, model, 1, out_root, cfg)
    rows = [to_row(a, gold[a["task_id"]]) for a in attempts]
    write_jsonl(out_root / model / "rows.jsonl", rows)
    return rows, store, stats


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
