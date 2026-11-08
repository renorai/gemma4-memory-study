"""Kaggle kernel: one model through the preregistered protocol (PREREG.md section 3), or part of
it, in one 12 h session. Internet off.

Edit CONFIG (or set ARM_RUN_CONFIG to a JSON string) before `kaggle kernels push`:
    model       one of the four aliases (31B first, then 12B)
    phases      any of "baseline", "distill", "headline", "explore", "analyze", in order
    n_tasks     from runner/budget.py:choose_scope on the throughput kernel's numbers
    exploratory arms for the "explore" phase ([] when the scope cut them)
    concurrency from the throughput kernel
    resume_from a /kaggle/input path holding an earlier session's /kaggle/working/out
                (attach that session's output as a dataset), or null

Each phase is idempotent: a results directory whose task_results.jsonl has a row for every task is
skipped (an incomplete one is rerun),
so a session that hits the 12 h wall can be resumed by the next one.
"""
import glob
import json
import os
import shutil
import sys

for p in sorted(glob.glob("/kaggle/input/**/kernels/common.py", recursive=True)):
    sys.path.insert(0, os.path.dirname(p))
    break
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from common import WORK, SwegemmaEval, chat, log, setup, validated_exclusions  # noqa: E402

CONFIG = {                       # defaults = the scope recorded in DESIGN.md section 10 (2026-11-08)
    "model": "gemma-4-31b-it-qat-w4a16-ct",
    "phases": ["baseline", "distill", "headline", "explore", "analyze"],
    "n_tasks": 48,
    "exploratory": [],
    "concurrency": 8,
    "runs": 2,
    "resume_from": None,
}


class Resumable(SwegemmaEval):
    def __call__(self, submission_dir, tasks, results_dir):
        from pathlib import Path
        tr = Path(results_dir) / "task_results.jsonl"
        if tr.exists():
            done = sum(1 for line in tr.read_text().splitlines() if line.strip())
            if done >= len(tasks):
                log("skip (done):", results_dir)
                return
            log(f"incomplete ({done}/{len(tasks)}), rerunning:", results_dir)   # a session cut mid-run
            shutil.rmtree(results_dir)
        super().__call__(submission_dir, tasks, results_dir)


def main():
    cfg = dict(CONFIG, **json.loads(os.environ.get("ARM_RUN_CONFIG", "{}")))
    alias = cfg["model"]
    code, comp, my, proc = setup(alias)
    from harness.swegemma import read_tasks, snapshot_for
    from memory import LessonStore
    from pipeline.experiment import IndexCache, distil, eval_arm, read_jsonl, to_row, write_jsonl
    from harness.swegemma import patch_files
    from runner.budget import stratified_subset
    from stats.analyze import analyze
    out = WORK / "out"
    if cfg["resume_from"] and not out.exists():
        shutil.copytree(cfg["resume_from"], out)
    excluded = validated_exclusions()
    tasks = stratified_subset([t for t in read_tasks(comp / "tasks.jsonl") if t.task_id not in excluded],
                              cfg["n_tasks"])
    (out / alias).mkdir(parents=True, exist_ok=True)
    with open(out / alias / "sessions.jsonl", "a") as f:     # one line per session: code, env, config, tasks
        f.write(json.dumps({"code_sha256": (WORK / "code_sha256.txt").read_text().strip(),
                            "scorer_env": json.loads((WORK / "scorer_env.json").read_text()),
                            "config": cfg, "excluded": excluded,
                            "task_ids": [t.task_id for t in tasks]}) + "\n")
    log("tasks:", len(tasks), [t.task_id for t in tasks])
    by_id = {t.task_id: t for t in tasks}
    ev = Resumable(comp, my, concurrency=cfg["concurrency"])
    lessons_path = out / alias / "lessons.jsonl"
    attempts = []
    if "baseline" in cfg["phases"]:
        for r in range(1, cfg["runs"] + 1):
            attempts += eval_arm(ev, "none", tasks, None, alias, r, out)
    if "distill" in cfg["phases"]:
        from harness.swegemma import read_run
        base = out / alias / "none" / "run1" / "all" / "results"
        run1 = read_run(base, tasks, model=alias, arm="none", run=1)
        cache = IndexCache(lambda t: snapshot_for(comp / "snapshots", t))
        from pipeline.distill import DistillStats
        st = DistillStats()
        store = distil(run1, lambda p: chat(alias, p), lambda a: cache(by_id[a["task_id"]]), st)
        store.save(lessons_path)
        (out / alias / "distill.json").write_text(json.dumps(
            {**st.as_dict(), "key_types": dict(store.key_counts())}, indent=2, sort_keys=True))
        log("distilled", len(store), "lessons", st.as_dict())
    store = LessonStore.load(lessons_path) if lessons_path.exists() else None
    if "headline" in cfg["phases"]:
        for r in range(1, cfg["runs"] + 1):
            attempts += eval_arm(ev, "node", tasks, store, alias, r, out)
    if "explore" in cfg["phases"]:
        for arm in cfg["exploratory"]:
            attempts += eval_arm(ev, arm, tasks, store, alias, 1, out)
    if "analyze" in cfg["phases"]:
        from harness.swegemma import read_run
        rows = []
        for res in sorted((out / alias).glob("*/run*/*/results")):
            arm, run, fold = res.parts[-4], int(res.parts[-3][3:]), res.parts[-2]
            fold_tasks = tasks if fold == "all" else [t for t in tasks if f"fold-{t.repo}" == fold]
            for a in read_run(res, fold_tasks, model=alias, arm=arm, run=run, fold=fold):
                rows.append(to_row(a, patch_files(by_id[a["task_id"]].gold_patch)))
        write_jsonl(out / alias / "rows.jsonl", rows)
        res = analyze(rows)
        (out / alias / "analysis.json").write_text(json.dumps(res, indent=2, default=str))
        log(json.dumps(res["primary"], default=str))
    proc.terminate()


if __name__ == "__main__":
    main()
