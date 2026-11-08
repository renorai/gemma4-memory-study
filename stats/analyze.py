"""The preregistered analysis (PREREG.md section 5). Changing this file after the first arm run
is a protocol deviation and must be logged in DESIGN.md section 10 with a date and a reason."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from . import (flip_rates, holm, mcnemar_exact, mean_over_runs, paired_bootstrap_ci, paired_diffs,
               process_table, run_matrix, sign_test)

# Headline (DESIGN.md section 10, 2026-11-08, before any memory-arm run): the in-task working-memory
# recap after compaction ("recap") against no memory. PREREG.md had node against none; that is now
# arm 2, run if the budget allows, and is analysed through `headline=("node", "none")`.
HEADLINE = ("recap", "none")
PRIMARY_MODELS = ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct")
EXPLORATORY = (("text", "none"), ("node", "text"), ("node_khop", "node"), ("node", "shuffled"),
               ("node_khop", "none"))
ALPHA = 0.05
N_BOOT = 10_000
SEED = 20261107
VERDICTS = ("SUPPORTED", "CONTRADICTED", "NOT SUPPORTED (inconclusive)")


def contrast(rows: list[Mapping], x: str, y: str, model: str) -> dict:
    """Per-task mean over runs, then the task-level paired bootstrap and exact sign test."""
    d = paired_diffs(mean_over_runs(rows), x, y, model=model)
    ci = paired_bootstrap_ci(d, n_boot=N_BOOT, alpha=ALPHA, seed=SEED)
    st = sign_test([round(v, 12) for _m, _t, v in d])
    return {"x": x, "y": y, "model": model, **ci, "sign_p": st["p"], "wins": st["pos"], "losses": st["neg"]}


def per_run_mcnemar(rows, x, y, model):
    mx, my = run_matrix(rows, model, x), run_matrix(rows, model, y)
    return [{"run": r, **mcnemar_exact(mx[r], my[r])} for r in sorted(set(mx) & set(my))]


def first_run(rows, arm):
    return [r for r in rows if r["arm"] != arm or int(r["run"]) == 1]


def moderators(rows, model):
    """Exploratory 'when does it help' splits for node - none (PREREG.md section 6)."""
    m = mean_over_runs([r for r in rows if r["model"] == model])
    by = {(r["arm"], r["task_id"]): r["success"] for r in m}
    shown = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["arm"] == "node":
            shown[r["task_id"]].append(r.get("lessons_shown", 0) > 0)
    tasks = sorted(t for (a, t) in by if a == "none" and ("node", t) in by)
    out = {}
    split_hit = {"lesson shown": [t for t in tasks if any(shown[t])],
                 "no lesson shown": [t for t in tasks if not any(shown[t])]}
    base = {t: by[("none", t)] for t in tasks}
    split_diff = {"none-arm solved in no run": [t for t in tasks if base[t] == 0],
                  "none-arm solved in some runs": [t for t in tasks if 0 < base[t] < 1],
                  "none-arm solved in every run": [t for t in tasks if base[t] == 1]}
    for name, ts in list(split_hit.items()) + list(split_diff.items()):
        diffs = [by[("node", t)] - by[("none", t)] for t in ts]
        out[name] = {"n": len(ts), "mean_diff": sum(diffs) / len(diffs) if diffs else None}
    return out


def repeat_rate_secondary(rows, x, y, model):
    """Key secondary outcome; the pre-registered statistics live in stats/repeats.py."""
    from .repeats import paired_repeat_stats
    return paired_repeat_stats(rows, x, y, model, n_boot=N_BOOT, seed=SEED)


def analyze(rows: Iterable[Mapping], primary_models=PRIMARY_MODELS, headline=HEADLINE) -> dict:
    rows = list(rows)
    models = sorted({r["model"] for r in rows})
    prim = [m for m in primary_models if m in models]
    res = {"headline": list(headline), "primary": {}, "per_run_mcnemar": {}, "secondary": {},
           "exploratory": [], "flip_rate": {}, "moderators": {}, "process": process_table(rows)}
    for m in prim:
        res["primary"][m] = contrast(rows, *headline, m)
        res["per_run_mcnemar"][m] = per_run_mcnemar(rows, *headline, m)
        res["secondary"][m] = repeat_rate_secondary(rows, *headline, m)
    adj = holm({m: h["sign_p"] for m, h in res["primary"].items()}, ALPHA) if prim else {}
    for m, h in res["primary"].items():
        h["p_holm"] = adj[m]["p_holm"]
        h["supported"] = bool(h["lo"] > 0 and adj[m]["reject"])
        h["verdict"] = (VERDICTS[0] if h["supported"]
                        else VERDICTS[1] if h["hi"] < 0 and adj[m]["reject"] else VERDICTS[2])
    for m in models:
        arms = {r["arm"] for r in rows if r["model"] == m}
        for x, y in EXPLORATORY:
            if x in arms and y in arms:
                # single-run arms are paired with run 1 of the headline arms, fixed in advance
                sub = first_run(first_run(rows, "node"), "none")
                try:
                    c = contrast(sub, x, y, m)
                except ValueError:
                    continue
                c["label"] = "exploratory, 1 run, no multiplicity correction"
                res["exploratory"].append(c)
        for arm in sorted(arms):
            res["flip_rate"][f"{m}/{arm}"] = flip_rates(run_matrix(rows, m, arm))
        if {"node", "none"} <= arms:
            res["moderators"][m] = moderators(rows, m)
    return res
