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
BACKENDS = ("modal", "kaggle")          # the registered 48 pairs (DESIGN.md section 10)
EXTENSION = "aws-l40s"                  # the post-48 replication backend (its own primary; see DESIGN.md)


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_paired(a, b, c, d):
    """Newcombe (1998) method 10, p1 - p2 for paired binary data (a: both, b: x only, c: y only,
    d: neither). Wilson score limits for each margin, combined with the continuity-corrected
    correlation: phi* = max(A - N/2, 0) / sqrt(efgh) when A = ad - bc > 0, else A / sqrt(efgh),
    with e, f, g, h the four margins (as in ratesci's MOVER_newc)."""
    n = a + b + c + d; p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n); l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    A = a * d - b * c
    phi = ((max(A - n / 2, 0) if A > 0 else A) / math.sqrt(den)) if den > 0 else 0.0
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


def sensitivity(I, keys):
    """Per-task mean of success over each arm's available runs, recap - none (paired, stratified)."""
    groups, n = [], 0
    for b in keys:
        ts = sorted({t for (bb, a, r, t) in I if bb == b and a == "recap"} & {t for (bb, a, r, t) in I if bb == b and a == "none"})
        g = []
        for t in ts:
            xs = [x["success"] for (bb, a, r, tt), x in I.items() if bb == b and a == "recap" and tt == t]
            ys = [x["success"] for (bb, a, r, tt), x in I.items() if bb == b and a == "none" and tt == t]
            g.append(sum(xs) / len(xs) - sum(ys) / len(ys))
        groups.append(g); n += len(g)
    diffs = [v for g in groups for v in g]
    lo, hi = strat_ci(groups)
    runs = {f"{a}/run{r}": sum(1 for (bb, aa, rr, t) in I if bb in keys and aa == a and rr == r)
            for (bb, a, r, t) in I if bb in keys}
    return {"tasks": n, "mean_diff": sum(diffs) / n, "bootstrap95": [lo, hi], "task_runs_used": runs}


def registered_blocks(I):
    """Every registered figure (and the descriptive run-1 blocks), from index I."""
    R1, N1 = ("recap", 1), ("none", 1)
    sets = (("pooled", BACKENDS), ("modal", ("modal",)), ("kaggle", ("kaggle",)))
    return {"primary_resolve_run1": {k: resolve(I, ks, R1, N1) for k, ks in sets},
            "repeat_rate_run1": {k: paired_metric(I, ks, R1, N1, "repeat_rate") for k, ks in sets},
            "sensitivity_resolve_all_available_runs": {k: sensitivity(I, ks) for k, ks in sets}}


def main(rows_path=HERE / "run_rows.jsonl", out_path=HERE / "analysis.json"):
    rows_all = [json.loads(l) for l in open(rows_path)]
    # exploratory_partial rows (the 7 finished Modal recap run-2 task-runs) enter NO registered
    # figure: only the labelled exploratory sensitivity variant and the descriptive discordant table.
    # Rows of the extension backend enter only the extension blocks below.
    ext_rows = [r for r in rows_all if r["backend"] == EXTENSION and not r.get("exploratory_partial")]
    rows_all = [r for r in rows_all if r["backend"] != EXTENSION]
    rows = [r for r in rows_all if not r.get("exploratory_partial")]
    I_all = index(rows_all)
    I = index(rows)
    assert not any(r.get("exploratory_partial") for r in I.values())
    R1, N1, N2 = ("recap", 1), ("none", 1), ("none", 2)
    sets = (("pooled", BACKENDS), ("modal", ("modal",)), ("kaggle", ("kaggle",)))
    out = {"primary_resolve_run1": {k: resolve(I, ks, R1, N1) for k, ks in sets},
           "sensitivity_resolve_all_available_runs": {k: sensitivity(I, ks) for k, ks in sets},
           "sensitivity_EXPLORATORY_with_partial_modal_recap_run2": {k: sensitivity(I_all, ks) for k, ks in sets},
           "exploratory_partial_rows": sorted(f"{r['backend']}/{r['arm']}/run{r['run']}/{r['task_id']}"
                                              for r in rows_all if r.get("exploratory_partial"))}
    metrics = {"repeat_rate": "registered: repeats per compaction",
               "repeat_rate_posthoc": "post-hoc: repeats per compaction, submit_patch calls after the first ok submit removed",
               "repeats_after_compaction": "post-hoc (a): RAW repeats per task at the registered steps (includes post-submit submit_patch loops)",
               "repeats_after_compaction_posthoc": "post-hoc (a): RAW repeats per task EXCLUDING submit_patch calls after the first ok submit",
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
    # post-hoc descriptive: each discordant run-1 pair, and whether the losing arm solved it in its other run
    disc = []
    for b in BACKENDS:
        for t, rx, ry in pairs(I, b, R1, N1):
            if rx["success"] == ry["success"]:
                continue
            winner, loser = ("recap", "none") if rx["success"] else ("none", "recap")
            other = I_all.get((b, loser, 2, t))        # descriptive: may use exploratory_partial rows
            disc.append({"backend": b, "task_id": t, "run1_winner": winner, "losing_arm": loser,
                         "losing_arm_solved_in_run2": None if other is None else bool(other["success"]),
                         "run2_row_is_exploratory_partial": None if other is None else bool(other.get("exploratory_partial")),
                         "recap_submit_loop_run1": rx["submit_loop"], "none_submit_loop_run1": ry["submit_loop"]})
    out["discordant_pairs_run1_posthoc_descriptive"] = disc
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
    # The Modal worker's own H-a counter (9/2/5, two-sided p 0.061) is NOT counted: it tested the same
    # hypothesis on the same traces and is superseded by the frozen detector below.
    ph.append({"test": "H-a one-sided Wilcoxon (Modal discovery, post-hoc there)", "stratum": "modal",
               "p": hyp["modal"]["H-a_p_one_sided"]})
    ph.append({"test": "H-b one-sided Fisher (Modal discovery, post-hoc there)", "stratum": "modal",
               "p": hyp["modal"]["H-b_fisher_p_one_sided"]})
    out["posthoc_tests"] = {"count": len(ph), "tests": ph}
    # number check: every registered figure is identical whether or not the exploratory_partial rows
    # exist in the index (run-1 and none-only figures), and the registered sensitivity mean is the one
    # computed without them.
    reg = registered_blocks(I)
    reg_all_but_sens = {k: v for k, v in registered_blocks(I_all).items() if k != "sensitivity_resolve_all_available_runs"}
    ok = (json.dumps({k: v for k, v in reg.items() if k != "sensitivity_resolve_all_available_runs"}, sort_keys=True, default=str)
          == json.dumps(reg_all_but_sens, sort_keys=True, default=str)
          and json.dumps(reg["primary_resolve_run1"], sort_keys=True, default=str) == json.dumps(out["primary_resolve_run1"], sort_keys=True, default=str)
          and json.dumps(reg["sensitivity_resolve_all_available_runs"], sort_keys=True, default=str)
          == json.dumps(out["sensitivity_resolve_all_available_runs"], sort_keys=True, default=str))
    out["check_no_registered_figure_uses_exploratory_partial"] = ok
    assert ok, "a registered figure depends on exploratory_partial rows"
    # extension (aws-l40s): its own two-sided primary on complete run-1 pairs, and the combined
    # stratified analysis (secondary, no confirmatory claim). Registered 48-pair figures above never see these rows.
    if ext_rows:
        IE = index(ext_rows)
        n_pairs = len(pairs(IE, EXTENSION, R1, N1))
        out["extension_primary_resolve_run1"] = resolve(IE, (EXTENSION,), R1, N1) if n_pairs else {"pairs": 0}
        IC = {**I, **IE}
        out["combined_secondary_resolve_run1"] = resolve(IC, BACKENDS + (EXTENSION,), R1, N1) if n_pairs else {"pairs": 0}
    json.dump(out, open(out_path, "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    o = main(*(Path(a) for a in sys.argv[1:3]))
    r = o["primary_resolve_run1"]["pooled"]
    print(f"pooled run 1: recap {r['x_resolved']} vs none {r['y_resolved']} of {r['pairs']}, diff {r['diff']:+.3f}, "
          f"Newcombe [{r['newcombe95'][0]:+.3f}, {r['newcombe95'][1]:+.3f}], McNemar p {r['mcnemar_p']:.3f}; "
          f"post-hoc tests: {o['posthoc_tests']['count']}")
