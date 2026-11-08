"""Kaggle kernel (CPU only, internet ON, no GPU quota): build the scorer-like sandbox environment
(kernels/scorer_sandbox.py) and validate every public task under it with the harness's own
`verify_task`, the function that scores agent patches.

For each of the 129 tasks, in the seeded order of runner.budget.stratified_subset:
  base  no agent patch, task.test_patch applied: the target tests must be collected and FAIL
        (pytest exit 1, 'failed' in the summary, no collection error, nothing only skipped);
  gold  the gold patch applied: the harness must score it resolved (exit 0 and its JUnit checks).

Outputs (attach this kernel's output to the GPU kernels; they read both directories):
  scorer_env/   py313.tar.gz, image_sp.tar.gz, manifest.json   (the sandbox environment)
  envcheck/     validation.jsonl (one row per task), summary.json (counts, exclusions with
                reasons, and the validated 48-task subset)
"""
import asyncio
import glob
import json
import os
import re
import shutil
import sys
import time

for p in sorted(glob.glob("/kaggle/input/**/kernels/common.py", recursive=True)):
    sys.path.insert(0, os.path.dirname(p))
    break
from common import WORK, code_sha256, find_code_root, find_competition, install_wheelhouse, log  # noqa: E402

CONCURRENCY = int(os.environ.get("ENVCHECK_CONCURRENCY", "4"))
N_SUBSET = 48
_SUMMARY = re.compile(r"^=*\s*((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)"
                      r"(?:, )?)+) in [\d.]+s", re.M)
_NO_TESTS = ("no tests ran", "collected 0 items")


def pytest_summary(out):
    hits = _SUMMARY.findall(out or "")
    return hits[-1] if hits else ""


def classify(base, gold):
    s, out = base["summary"], base["out"]
    import_err = bool(re.search(r"ImportError|cannot import name|ModuleNotFoundError", out))
    if base["exit"] == 124:                     # rule (DESIGN.md section 10): base timeout
        if gold["resolved"]:
            return "valid_base_timeout", ("tests time out on the base commit; the gold patch resolves "
                                          "within the same 300 s verification timeout")
        return "base_timeout", "tests time out on the base commit and the gold patch does not resolve"
    if not gold["resolved"]:
        return "gold_fails", f"gold patch not resolved: {gold['error'] or gold['summary'] or gold['exit']}"
    if base["resolved"] or base["exit"] == 0:
        return "base_passes", "tests pass without any patch, so they cannot tell a fix from no fix"
    if any(k in out.lower() for k in _NO_TESTS):
        return "base_no_tests", "no tests collected on the base commit"
    if base["exit"] == 1 and "failed" in s and "error" not in s:
        return "valid", ""
    if import_err:
        return "valid_import", ("tests fail on the base commit at import (collection error), most likely "
                                "of a symbol the gold patch adds; gold passes")
    return "base_other", f"base exit {base['exit']}, summary '{s}'"


async def run(tasks, comp, sandbox, cfg):
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.verification import verify_task
    sem = asyncio.Semaphore(CONCURRENCY)
    rows = []
    out_f = open(WORK / "envcheck" / "validation.jsonl", "w")

    async def one(rank, t):
        async with sem:
            snap, base_snap, patch = resolve_task_snapshot_paths(comp / "snapshots", t.instance_id, t.repo)
            res = {}
            for kind, agent_patch in (("base", ""), ("gold", t.patch or "")):
                t0 = time.perf_counter()
                try:
                    r = await verify_task(sandbox, cfg, t, snap, base_snapshot_path=base_snap,
                                          patch_path=patch, agent_patch=agent_patch, start_time=t0)
                    out = r.test_output or ""
                    res[kind] = {"resolved": bool(r.resolved), "exit": r.test_exit_code, "error": r.error,
                                 "summary": pytest_summary(out), "out": out}
                except Exception as e:  # a harness setup failure is a validation failure, not a crash
                    res[kind] = {"resolved": False, "exit": None, "error": f"{type(e).__name__}: {e}",
                                 "summary": "", "out": ""}
                res[kind]["seconds"] = round(time.perf_counter() - t0, 1)
            status, reason = classify(res["base"], res["gold"])
            row = {"rank": rank, "task_id": t.instance_id, "repo": t.repo, "status": status, "reason": reason}
            for kind in ("base", "gold"):
                row[kind] = {k: v for k, v in res[kind].items() if k != "out"}
                row[kind]["out_tail"] = res[kind]["out"][-2500:]
            rows.append(row)
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            log(f"[{len(rows)}/{len(tasks)}] {t.instance_id}: {status} (base exit {res['base']['exit']} "
                f"'{res['base']['summary']}', gold {'ok' if res['gold']['resolved'] else 'FAIL'})")

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    out_f.close()
    return sorted(rows, key=lambda r: r["rank"])


def main():
    code = find_code_root()
    sys.path.insert(0, str(code))
    comp = find_competition()
    install_wheelhouse()
    import scorer_sandbox                     # kernels/, already on sys.path with common.py
    scorer_sandbox.patch_harness()            # post_setup after every install_test_dependencies
    (WORK / "envcheck").mkdir(parents=True, exist_ok=True)
    scorer_sandbox.build_env(WORK / "scorer_env", comp)
    env = scorer_sandbox.load_env(WORK / "scorer_env")
    digest = code_sha256(code)
    log("code sha256:", digest, "| scorer env:", env["python"], env["image_dists"])

    from harness.swegemma import read_tasks
    from runner.budget import SUBSET_SEED, stratified_subset
    from swegemma.config import EvalConfig
    from swegemma.models import load_tasks
    from adk_submission import ModelRegistry
    ours = read_tasks(comp / "tasks.jsonl")
    order = [t.task_id for t in stratified_subset(ours, len(ours))]          # the seeded order
    by_id = {t.instance_id: t for t in load_tasks(comp / "tasks.jsonl")}
    tasks = [by_id[i] for i in order]
    cfg = EvalConfig(tasks_path=comp / "tasks.jsonl", snapshots_dir=comp / "snapshots",
                     results_dir=WORK / "envcheck" / "results", submission_dir=WORK,
                     models=ModelRegistry(), sandbox="subprocess", wheels_dir=comp / "wheels",
                     graph_dir=str(comp / "graphs"), embeddings_dir=str(comp / "embeddings"),
                     verification_timeout_seconds=300, display_mode="quiet")
    sandbox = scorer_sandbox.DockerSandbox(timeout_seconds=300)
    t0 = time.time()
    rows = asyncio.run(run(tasks, comp, sandbox, cfg))

    usable = {"valid", "valid_import", "valid_base_timeout"}
    excluded = {r["task_id"]: f"{r['status']}: {r['reason']}" for r in rows if r["status"] not in usable}
    first48 = [t.task_id for t in stratified_subset(ours, N_SUBSET)]
    validated = [t.task_id for t in stratified_subset([t for t in ours if t.task_id not in excluded], N_SUBSET)]
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    in48 = {}
    for r in rows:
        if r["task_id"] in first48:
            in48[r["status"]] = in48.get(r["status"], 0) + 1
    summary = {
        "code_sha256": digest, "scorer_env": env, "seed": SUBSET_SEED, "n_tasks": len(rows),
        "minutes": round((time.time() - t0) / 60, 1), "counts_all": counts, "counts_first48": in48,
        "first48_seeded": first48,
        "first48_failing": {i: excluded[i] for i in first48 if i in excluded},
        "excluded": excluded,
        "validated_subset48": validated,
        "replacements": [i for i in validated if i not in first48],
        "replacement_rule": ("drop excluded tasks, then take stratified_subset (seed 20261107) of the "
                             "rest: each repo's next task in its seeded order, round-robin across repos"),
    }
    by_repo = {}
    for r in rows:
        d = by_repo.setdefault(r["repo"], {"tasks": 0, "valid": 0, "excluded": {}})
        d["tasks"] += 1
        if r["status"] in usable:
            d["valid"] += 1
        else:
            d["excluded"][r["status"]] = d["excluded"].get(r["status"], 0) + 1
    summary["by_repo"] = by_repo
    summary["n_valid"] = sum(d["valid"] for d in by_repo.values())
    (WORK / "envcheck" / "summary.json").write_text(json.dumps(summary, indent=2))
    # appendix files: the validated task list (seeded order, with status) and the env manifest
    (WORK / "envcheck" / "validated_tasks.json").write_text(json.dumps(
        {"seed": SUBSET_SEED, "valid_in_seed_order": [r["task_id"] for r in rows if r["status"] in usable],
         "subset48": validated, "excluded": excluded, "by_repo": by_repo}, indent=2))
    shutil.copy(WORK / "scorer_env" / "manifest.json", WORK / "envcheck" / "env_manifest.json")
    log(json.dumps({k: summary[k] for k in ("counts_all", "counts_first48", "first48_failing", "replacements")},
                   indent=2))


if __name__ == "__main__":
    main()
