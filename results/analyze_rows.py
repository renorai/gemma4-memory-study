"""Regenerate every reported recap-vs-none number from results/run_rows.jsonl alone (no traces).

    python results/analyze_rows.py [run_rows.jsonl] [out.json]

Primary (DESIGN.md section 10): run 1 pooled over 48 pairs (Modal 16 + Kaggle 32), ITT, bootstrap
stratified by backend. Also: Newcombe paired score interval and exact McNemar; the registered repeat
rate and its post-hoc variant; the post-hoc denominator checks; H-a/H-b (Kaggle replication, one-sided);
per-protocol; the A/A calibration (none run 2 vs none run 1); and the list of post-hoc tests run.
"""
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stats import mcnemar_exact  # noqa: E402
from stats.hypotheses import fisher_one_sided, wilcoxon_one_sided  # noqa: E402
from stats.repeats import trimmed_mean, wilcoxon_signed_rank  # noqa: E402

SEED, NB = 20261107, 10_000
BACKENDS = ("modal", "kaggle")


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_paired(a, b, c, d):
    """Newcombe (1998) method 10, p1 - p2 for paired binary data (b: x only, c: y only)."""
    n = a + b + c + d; p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n); l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return p1 - p2 - dl, p1 - p2 + du


def strat_ci(groups):
    rng = random.Random(SEED); st = []
    for _ in range(NB):
        s = n = 0
        for g in groups:
            for _ in range(len(g)):
                s += g[rng.randrange(len(g))]; n += 1
        st.append(s / n)
    st.sort()
    return st[int(0.025 * NB)], st[int(0.975 * NB) - 1]


def index(rows):
    return {(r["backend"], r["arm"], r["run"], r["task_id"]): r for r in rows}


def pairs(I, backend, x, y, keep=None):
    """[(task, row_x, row_y)] for (arm, run) x vs y within a backend."""
    ts = sorted(t for (b, a, r, t) in I if b == backend and (a, r) == x and (b, y[0], y[1], t) in I)
    return [(t, I[(backend, *x, t)], I[(backend, *y, t)]) for t in ts if keep is None or keep(I[(backend, *x, t)])]


def resolve(I, keys, x, y, keep=None):
    P = [(b, t, rx, ry) for b in keys for t, rx, ry in pairs(I, b, x, y, keep)]
    a = sum(rx["success"] and ry["success"] for *_, rx, ry in P)
    bb = sum(rx["success"] and not ry["success"] for *_, rx, ry in P)
    cc = sum(ry["success"] and not rx["success"] for *_, rx, ry in P)
    dd = len(P) - a - bb - cc
    groups = [[rx["success"] - ry["success"] for b2, _t, rx, ry in P if b2 == b] for b in keys]
    lo, hi = strat_ci(groups)
    nl, nh = newcombe_paired(a, bb, cc, dd)
    mc = mcnemar_exact({f"{b}:{t}": rx["success"] for b, t, rx, ry in P}, {f"{b}:{t}": ry["success"] for b, t, rx, ry in P})
    return {"pairs": len(P), "x_resolved": a + bb, "y_resolved": a + cc, "x_only": bb, "y_only": cc,
            "diff": (bb - cc) / len(P), "bootstrap95": [lo, hi], "newcombe95": [nl, nh], "mcnemar_p": mc["p"]}


def paired_metric(I, keys, x, y, metric, keep=None):
    groups, diffs, excl = [], [], []
    for b in keys:
        g = []
        for t, rx, ry in pairs(I, b, x, y, keep):
            if rx[metric] is None or ry[metric] is None:
                excl.append([b, t, "x" if rx[metric] is None and ry[metric] is not None else
                             "y" if ry[metric] is None and rx[metric] is not None else "both"])
                continue
            g.append(rx[metric] - ry[metric])
        groups.append(g); diffs += g
    if not diffs:
        return {"n": 0, "excluded": excl}
    lo, hi = strat_ci(groups)
    return {"n": len(diffs), "mean_diff": statistics.mean(diffs), "ci95": [lo, hi],
            "wilcoxon_p": wilcoxon_signed_rank(diffs)["p"], "trimmed10": trimmed_mean(diffs), "excluded": excl}


def main(rows_path=HERE / "run_rows.jsonl", out_path=HERE / "analysis.json"):
    rows = [json.loads(l) for l in open(rows_path)]
    I = index(rows)
    R1, N1, N2 = ("recap", 1), ("none", 1), ("none", 2)
    sets = (("pooled", BACKENDS), ("modal", ("modal",)), ("kaggle", ("kaggle",)))
    out = {"primary_resolve_run1": {k: resolve(I, ks, R1, N1) for k, ks in sets}}
    metrics = {"repeat_rate": "registered", "repeat_rate_posthoc": "post-hoc",
               "repeats_after_compaction": "post-hoc (a) raw, registered steps",
               "repeats_after_compaction_posthoc": "post-hoc (a) raw, no post-submit submits",
               "repeats_per_call_after_compaction": "post-hoc (b) per call, registered steps",
               "repeats_per_call_after_compaction_posthoc": "post-hoc (b) per call, no post-submit submits",
               "compactions": "post-hoc (c) compactions per task"}
    out["repeats_run1"] = {m: {"label": lab, **{k: paired_metric(I, ks, R1, N1, m) for k, ks in sets}}
                           for m, lab in metrics.items()}
    # H-a / H-b, Kaggle replication (pre-registered, one-sided), and Modal discovery
    hyp = {}
    for b in BACKENDS:
        P = pairs(I, b, R1, N1)
        ha = [rx["compactions_before_first_submit"] - ry["compactions_before_first_submit"] for _t, rx, ry in P]
        la = sum(bool(rx["leaked_thought"]) for _t, rx, ry in P); lb = sum(bool(ry["leaked_thought"]) for _t, rx, ry in P)
        hyp[b] = {"role": "replication (pre-registered)" if b == "kaggle" else "discovery (post-hoc there)",
                  "H-a_more_fewer_equal": [sum(v > 0 for v in ha), sum(v < 0 for v in ha), sum(v == 0 for v in ha)],
                  "H-a_p_one_sided": wilcoxon_one_sided(ha)["p_one_sided"],
                  "H-b_leaks_recap_none": [la, lb, len(P)], "H-b_fisher_p_one_sided": fisher_one_sided(la, len(P), lb, len(P))}
    out["hypotheses_run1"] = hyp
    # per-protocol: recap runs with plumbing_failed excluded
    keep = lambda r: not r.get("plumbing_failed")
    out["per_protocol_run1"] = {
        "plumbing_failed": {b: sum(1 for (bb, a, r, t), x in I.items() if bb == b and (a, r) == R1 and x.get("plumbing_failed"))
                            for b in BACKENDS},
        "resolve": {k: resolve(I, ks, R1, N1, keep) for k, ks in sets},
        "repeat_rate": {k: paired_metric(I, ks, R1, N1, "repeat_rate", keep) for k, ks in sets},
        "repeat_rate_posthoc": {k: paired_metric(I, ks, R1, N1, "repeat_rate_posthoc", keep) for k, ks in sets}}
    # A/A calibration: none run 2 vs none run 1
    aa = {}
    for b in BACKENDS:
        P = pairs(I, b, N2, N1)
        ha = [rx["compactions_before_first_submit"] - ry["compactions_before_first_submit"] for _t, rx, ry in P]
        la = sum(bool(rx["leaked_thought"]) for _t, rx, ry in P); lb = sum(bool(ry["leaked_thought"]) for _t, rx, ry in P)
        aa[b] = {"resolve": resolve(I, (b,), N2, N1), "repeat_rate": paired_metric(I, (b,), N2, N1, "repeat_rate"),
                 "H-a_p_one_sided_run2_gt_run1": wilcoxon_one_sided(ha)["p_one_sided"],
                 "H-b_leaks_run1_run2": [lb, la], "H-b_p_one_sided_run2_gt_run1": fisher_one_sided(la, len(P), lb, len(P))}
    out["aa_calibration"] = aa
    # descriptive per arm/backend, run 1
    desc = {}
    for b in BACKENDS:
        for arm in ("none", "recap"):
            rs = [x for (bb, a, r, t), x in I.items() if bb == b and a == arm and r == 1]
            ov = [x for x in rs if x["context_overflow"]]
            desc[f"{b}/{arm}"] = {"n": len(rs), "resolved": sum(x["success"] for x in rs), "overflow": len(ov),
                                  "overflow_before_first_compaction": sum(1 for x in ov if x["overflow_phase"] == "before first compaction"),
                                  "cap_calls": sum(x["cap_calls"] for x in rs), "cap_time": sum(x["cap_time"] for x in rs),
                                  "cap_turns": sum(x["cap_turns"] for x in rs), "submit_loops": sum(x["submit_loop"] for x in rs),
                                  "compactions": sum(x["compactions"] for x in rs),
                                  "notes_written": sum(x.get("notes_written") or 0 for x in rs) if arm == "recap" else None,
                                  "runs_with_note": sum(1 for x in rs if x.get("notes_written")) if arm == "recap" else None}
    out["descriptive_run1"] = desc
    # every post-hoc test run so far (p values from this file's own computations)
    ph = []
    for m, lab in metrics.items():
        if m == "repeat_rate":
            continue
        for k, _ in sets:
            v = out["repeats_run1"][m][k]
            if v.get("n"):
                ph.append({"test": f"Wilcoxon signed-rank, recap - none, {lab}", "stratum": k, "p": v["wilcoxon_p"]})
    for k, _ in sets:
        v = out["per_protocol_run1"]["repeat_rate_posthoc"][k]
        if v.get("n"):
            ph.append({"test": "Wilcoxon, per-protocol, repeat rate post-hoc variant", "stratum": k, "p": v["wilcoxon_p"]})
    ph.append({"test": "H-a one-sided Wilcoxon (Modal discovery, post-hoc there)", "stratum": "modal",
               "p": hyp["modal"]["H-a_p_one_sided"]})
    ph.append({"test": "H-b one-sided Fisher (Modal discovery, post-hoc there)", "stratum": "modal",
               "p": hyp["modal"]["H-b_fisher_p_one_sided"]})
    out["posthoc_tests"] = {"count": len(ph), "tests": ph}
    json.dump(out, open(out_path, "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    o = main(*(Path(a) for a in sys.argv[1:3]))
    r = o["primary_resolve_run1"]["pooled"]
    print(f"pooled run 1: recap {r['x_resolved']} vs none {r['y_resolved']} of {r['pairs']}, diff {r['diff']:+.3f}, "
          f"Newcombe [{r['newcombe95'][0]:+.3f}, {r['newcombe95'][1]:+.3f}], McNemar p {r['mcnemar_p']:.3f}; "
          f"post-hoc tests: {o['posthoc_tests']['count']}")
