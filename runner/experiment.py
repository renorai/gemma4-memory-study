"""The experiment protocol, independent of the harness and of the model.

    1. baseline: arm "none" on every dev task, per model -> attempt logs
    2. distil:   each model's own attempts -> that model's lesson store
    3. LORO:     for each held-out repo R, arms "flat" and "graph" on R's tasks with
                 memory = lessons from the OTHER repos only (same model)
    4. rows:     one row per (model, arm, task) for stats/

Arm "none" rows are the baseline attempts themselves (no memory, so nothing
leaks into them; DESIGN.md section 3 records this reuse before any run).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from memory import DistillStats, LessonStore, RetrievalConfig, distill_attempt, make_arm
from memory.adapter import HarnessAdapter, Task

Agent = Callable[..., dict]          # agent(task, arm, run_id=...) -> attempt dict
Generate = Callable[[str], str]


def run_arm(tasks: Iterable[Task], agent: Agent, arm, run_id: str) -> list[dict]:
    return [agent(t, arm, run_id=run_id) for t in tasks]


def distil(attempts: Iterable[dict], generate: Generate, adapter: HarnessAdapter,
           stats: DistillStats | None = None) -> LessonStore:
    store = LessonStore()
    for a in attempts:
        for lesson in distill_attempt(a, generate, repo_of=adapter.repo_of, stats=stats):
            store.add(lesson)
    return store


def loro_memory(store: LessonStore, held_out_repo: str, model: str) -> LessonStore:
    fold = store.filtered(exclude_repos=[held_out_repo], source_model=model)
    leaked = [l for l in fold if l.repo == held_out_repo or l.source_model != model]
    assert not leaked, f"leakage guard failed: {len(leaked)} lessons from {held_out_repo}"
    return fold


def to_row(attempt: dict, arm: str, fold: str) -> dict:
    return {
        "model": attempt["model"], "arm": arm, "task_id": attempt["task_id"],
        "repo": attempt["repo"], "fold": fold, "success": int(attempt["outcome"] == "success"),
        "tokens_in": attempt.get("tokens_in", 0), "tokens_out": attempt.get("tokens_out", 0),
        "wall_s": attempt.get("wall_s", 0.0), "n_lessons": len(attempt.get("lesson_ids", [])),
    }


def run_loro(adapter: HarnessAdapter, agent: Agent, generate: Generate, model: str,
             cfg: RetrievalConfig = RetrievalConfig(), arms=("flat", "graph"),
             out_dir: str | Path | None = None) -> tuple[list[dict], LessonStore, DistillStats]:
    tasks = list(adapter.tasks())
    repos = sorted({t.repo for t in tasks})
    none_arm = make_arm("none", LessonStore(), adapter, cfg)
    baseline = run_arm(tasks, agent, none_arm, run_id=f"{model}-none")
    stats = DistillStats()
    store = distil(baseline, generate, adapter, stats)
    rows = [to_row(a, "none", a["repo"]) for a in baseline]
    attempts = list(baseline)
    for repo in repos:
        fold_tasks = [t for t in tasks if t.repo == repo]
        memory = loro_memory(store, repo, model)
        for arm_name in arms:
            arm = make_arm(arm_name, memory, adapter, cfg)
            got = run_arm(fold_tasks, agent, arm, run_id=f"{model}-{arm_name}-loro-{repo}")
            attempts += got
            rows += [to_row(a, arm_name, repo) for a in got]
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        store.save(out / f"lessons-{model}.jsonl")
        write_jsonl(out / f"attempts-{model}.jsonl", attempts)
        write_jsonl(out / f"rows-{model}.jsonl", rows)
    return rows, store, stats


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
