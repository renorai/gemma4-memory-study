"""Kaggle run-1 table, recap vs none. argv: code_root(branch head) pin59_root none_results recap_results recap_submission tasks.jsonl"""
import json, sys, statistics
from pathlib import Path
code, pin, none_res, rec_res, rec_sub, tp = sys.argv[1:7]
sys.path.insert(0, code)
from harness.swegemma import read_tasks, read_run, patch_files, trace_path, read_trace
from pipeline.experiment import to_row
from stats import paired_bootstrap_ci, sign_test, mcnemar_exact
from stats.repeats import paired_repeat_stats, repeat_metrics, compaction_turns
from stats.hypotheses import compactions_before_first_submit, leaked_thought, wilcoxon_one_sided, fisher_one_sided
from stats.caps import cap_hits
import memory.plumbing as pl          # identical at the branch head and at the pin 59f10a6 (checked by diff)
alias = "gemma-4-31b-it-qat-w4a16-ct"
by = {t.task_id: t for t in read_tasks(tp)}

def rows_for(res, arm):
    res = Path(res); tr = {json.loads(l)["instance_id"]: json.loads(l) for l in open(res / "task_results.jsonl")}
    tasks = [by[t] for t in tr]
    out = {}
    for a in read_run(res, tasks, model=alias, arm=arm, run=1):
        r = to_row(a, patch_files(by[a["task_id"]].gold_patch)); t = a["task_id"]; p = trace_path(res, t)
        # post-hoc variant: drop submit_patch calls after the first ok submit
        tr_ = read_trace(p); first = None; kept = []
        for s in tr_["steps"]:
            if s["tool"] == "submit_patch" and first is not None:
                continue
            kept.append(s)
            if s["tool"] == "submit_patch" and '"status": "ok"' in (s["output"] or ""):
                first = s["turn"]
        post_submit = sum(1 for s in tr_["steps"] if s["tool"] == "submit_patch") - sum(1 for s in kept if s["tool"] == "submit_patch")
        ph = repeat_metrics({"steps": kept, "root_prompt_tokens": tr_["root_prompt_tokens"]})
        r["repeat_rate_posthoc"] = ph["repeat_rate"]; r["repeats_posthoc"] = ph["repeats_after_compaction"]
        r["post_submit_submits"] = post_submit; r["loop"] = post_submit >= 10
        r["cbfs"] = compactions_before_first_submit(p)[0]; r["leak"] = leaked_thought(p)
        err = tr[t].get("error"); r["caps"] = cap_hits(err, p)
        pt = tr_["root_prompt_tokens"]; c = compaction_turns(pt)
        r["overflow_phase"] = (None if not r["context_overflow"] else "before first compaction" if not c else "after first compaction")
        r["overflow_last_prompt"] = pt[-1] if r["context_overflow"] and pt else None
        if arm == "recap":
            patch = (res / "patches" / f"{t}.patch").read_text() if (res / "patches" / f"{t}.patch").exists() else ""
            r.update(pl.plumbing(p, "recap", patch, pl.injecting_agents(rec_sub)))
        out[t] = r
    return out

n1, r1 = rows_for(none_res, "none"), rows_for(rec_res, "recap")
ts = sorted(set(n1) & set(r1))
print("paired tasks:", len(ts), "| none-only:", sorted(set(n1) - set(r1)), "| recap-only:", sorted(set(r1) - set(n1)))
d = [r1[t]["success"] - n1[t]["success"] for t in ts]
ci = paired_bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=20261107)
print("RESOLVED run1: none", sum(n1[t]["success"] for t in ts), "recap", sum(r1[t]["success"] for t in ts), "of", len(ts),
      "| recap better/worse/same", sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d),
      f"| mean diff {sum(d)/len(d):+.3f} CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]", "| sign", sign_test(d),
      "| McNemar", mcnemar_exact({t: r1[t]["success"] for t in ts}, {t: n1[t]["success"] for t in ts}))
rows = [n1[t] for t in ts] + [r1[t] for t in ts]
reg = paired_repeat_stats(rows, "recap", "none", alias)
print("REPEAT registered:", {k: reg.get(k) for k in ("mean_recap", "mean_none", "n_tasks_paired", "excluded_no_compaction", "mean", "lo", "hi", "trimmed_mean_10")}, "wilcoxon", reg.get("wilcoxon"))
for r in rows: r["repeat_rate"] = r["repeat_rate_posthoc"]
ph = paired_repeat_stats(rows, "recap", "none", alias)
print("REPEAT post-hoc (no post-submit submit_patch):", {k: ph.get(k) for k in ("mean_recap", "mean_none", "n_tasks_paired", "mean", "lo", "hi", "trimmed_mean_10")}, "wilcoxon", ph.get("wilcoxon"))
ha = [r1[t]["cbfs"] - n1[t]["cbfs"] for t in ts]
print("H-a: recap more/fewer/equal", sum(x > 0 for x in ha), sum(x < 0 for x in ha), sum(x == 0 for x in ha), "one-sided Wilcoxon", wilcoxon_one_sided(ha))
la, lb = sum(r1[t]["leak"] for t in ts), sum(n1[t]["leak"] for t in ts)
print(f"H-b: leaks recap {la}/{len(ts)} none {lb}/{len(ts)} run-level one-sided Fisher p {fisher_one_sided(la, len(ts), lb, len(ts)):.4f} (task-level == run-level with 1 run)")
for name, R in (("none", n1), ("recap", r1)):
    rs = [R[t] for t in ts]; ov = [r for r in rs if r["context_overflow"]]
    from collections import Counter
    print(f"{name}: overflow {len(ov)} {dict(Counter(r['overflow_phase'] for r in ov))} last prompts {[r['overflow_last_prompt'] for r in ov]}"
          f" | caps calls {sum(r['caps']['calls'] for r in rs)} time {sum(r['caps']['time'] for r in rs)} turns {sum(r['caps']['turns'] for r in rs)}"
          f" | loops (>=10 post-submit submits) {sum(r['loop'] for r in rs)} | compacted {sum(1 for r in rs if r['compactions'])}, compactions {sum(r['compactions'] for r in rs)}"
          f" | median agent min {statistics.median([r['wall_s']/60 for r in rs if r['wall_s']]):.1f}, model calls median {statistics.median([r['model_calls'] for r in rs])}")
rs = [r1[t] for t in ts]
mach = [t for t in ts if r1[t].get("skill_status") != "on" or r1[t].get("skill_error") or r1[t].get("writes_in_task_dir") is False]
noth = [t for t in ts if r1[t].get("plumbing_failed") and t not in mach]
notes = [r.get("notes_written") or 0 for r in rs]
sh = [r["injected_share"] for r in rs if r.get("injected_share") is not None]
print("ADHERENCE: notes per run", notes, "runs with any note", sum(1 for x in notes if x), "| explorer injections", sum(r.get("explorer_injections") or 0 for r in rs),
      "| injected_share mean", round(statistics.mean(sh), 3) if sh else None)
print("PLUMBING: failed", sum(1 for r in rs if r.get("plumbing_failed")), "| machinery", [(t, r1[t].get("skill_status"), r1[t].get("skill_error")) for t in mach], "| nothing-to-inject", noth)
json.dump({"none": n1, "recap": r1}, open(Path(rec_res).parents[3] / "run1_rows.json", "w"), indent=1, default=str)
