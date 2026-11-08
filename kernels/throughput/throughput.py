"""Kaggle kernel: throughput of gemma-4-31b-it-qat-w4a16-ct on L4x4 under `swegemma eval
--sandbox subprocess`, at concurrency 8 and 16, on one 16-task shard (4 per dev repo).

Outputs /kaggle/working/throughput.json: per concurrency, tasks per hour, median agent minutes
per task, model calls per task (median and mean), plus the session's CPU count and GPUs.
These are THROUGHPUT measurements for planning (runner/budget.py), not results: the arm is
`none`, and the success counts are printed only so a broken setup is visible.

Optional (SKILL_SMOKE=1, default on): 2 tasks with the `node` arm and a one-lesson pool, to
check that the agent can find and run the skill script (DESIGN.md section 9, VERIFY rows).
Internet off. Inputs: competition data, the model, vLLM wheels, and this code as a dataset.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for p in sorted(glob.glob("/kaggle/input/**/kernels/common.py", recursive=True)):
    sys.path.insert(0, os.path.dirname(p))
    break
import json  # noqa: E402
import statistics  # noqa: E402

from common import WORK, SwegemmaEval, cpu_info, log, setup  # noqa: E402

ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
CONCURRENCY = (8, 16)
N_TASKS = 16


def main():
    code, comp, my, proc = setup(ALIAS)
    from harness.swegemma import read_run, read_tasks
    from runner.budget import stratified_subset
    from runner.build_submission import build
    tasks = stratified_subset(read_tasks(comp / "tasks.jsonl"), N_TASKS)
    log("shard:", [t.task_id for t in tasks])
    sub = WORK / "sub-none"
    build("none", None, ALIAS, sub)
    out = {"model": ALIAS, "arm": "none", "sandbox": "subprocess", "n_tasks": len(tasks),
           "task_ids": [t.task_id for t in tasks], "machine": cpu_info(), "runs": []}
    for c in CONCURRENCY:
        ev = SwegemmaEval(comp, my, concurrency=c)
        res = WORK / f"tp-c{c}" / "results"
        ev(sub, tasks, res)
        att = read_run(res, tasks, model=ALIAS, arm="none", run=1)
        wall_h = ev.walls[-1]["wall_s"] / 3600
        mins = [a["wall_s"] / 60 for a in att if a["wall_s"]]
        calls = [a["model_calls"] for a in att]
        out["runs"].append({
            "concurrency": c, "scored": len(att), "wall_h": wall_h,
            "tasks_per_hour": len(att) / wall_h if wall_h else None,
            "agent_minutes_median": statistics.median(mins) if mins else None,
            "model_calls_median": statistics.median(calls) if calls else None,
            "model_calls_mean": statistics.mean(calls) if calls else None,
            "tool_calls_median": statistics.median([len(a["steps"]) for a in att]) if att else None,
            "resolved_not_a_result": sum(a["success"] for a in att)})
        json.dump(out, open(WORK / "throughput.json", "w"), indent=2)
        log(json.dumps(out["runs"][-1]))
    if os.environ.get("SKILL_SMOKE", "1") == "1":
        from memory import Lesson, LessonStore
        smoke = LessonStore([Lesson(title="smoke", text="Smoke-test lesson: check the callers first.",
                                    outcome="success", keys=("name:__init__",), anchor="smoke.X.__init__",
                                    repo="smoke-repo", source_task="smoke", source_model=ALIAS)])
        build("node", smoke, ALIAS, WORK / "sub-smoke")
        res = WORK / "smoke" / "results"
        SwegemmaEval(comp, my, concurrency=2)(WORK / "sub-smoke", tasks[:2], res)
        att = read_run(res, tasks[:2], model=ALIAS, arm="node", run=1)
        from harness.swegemma import is_recall
        ran = [any(is_recall(s) for s in a["steps"]) for a in att]
        outs = [s.get("output", "")[:200] for a in att for s in a["steps"] if is_recall(s)]
        saved = [any(s["tool"] == "run_command" and "/tmp/task.md" in json.dumps(s["args"]) for s in a["steps"])
                 for a in att]
        via = sorted({s["tool"] for a in att for s in a["steps"] if is_recall(s)})
        # pass = every task saved its issue and recalled, and no recall failed to load the skill
        bad = [o for o in outs if o.startswith(("Error", "memory unavailable")) or "Traceback" in o]
        out["skill_smoke"] = {"tasks": len(att), "issue_saved": saved, "recall_ran": ran, "recall_via": via,
                              "recall_outputs": outs[:4], "recall_errors": bad[:4],
                              "pass": bool(att) and all(saved) and all(ran) and not bad}
        json.dump(out, open(WORK / "throughput.json", "w"), indent=2)
        log("skill smoke:", out["skill_smoke"])
    proc.terminate()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
