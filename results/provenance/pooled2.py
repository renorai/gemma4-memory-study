"""Pooled run-1: resolve (bootstrap + Newcombe paired score interval + exact McNemar) and the repeat
denominator check (post-hoc), per backend and pooled."""
import json, math, sys, random, statistics
from pathlib import Path
code = sys.argv[1]; sys.path.insert(0, code)
from harness.swegemma import read_trace
from stats import mcnemar_exact
from stats.repeats import compaction_turns, repeated_actions, wilcoxon_signed_rank, trimmed_mean
MP = Path("modal-probe")
KG = Path("kg")
A = "gemma-4-31b-it-qat-w4a16-ct"
SRC = {"modal": {"none": [MP / f"shard/ck2/out/{A}/none/run1/all/results"],
                 "recap": [MP / f"shard/ck7/out/{A}/recap/run1/all/results", MP / f"shard/rescue/out/{A}/recap/run1/all/results"]},
       "kaggle": {"none": [KG / f"base1/out/{A}/none/run1/all/results"], "recap": [KG / f"recap1/out/{A}/recap/run1/all/results"]}}

def drop_post_submit(steps):
    first, kept = None, []
    for s in steps:
        if s["tool"] == "submit_patch" and first is not None:
            continue
        kept.append(s)
        if s["tool"] == "submit_patch" and '"status": "ok"' in (s["output"] or ""):
            first = s["turn"]
    return kept

def measures(tr):
    c = compaction_turns(tr["root_prompt_tokens"])
    out = {"compactions": len(c)}
    for tag, steps in (("reg", tr["steps"]), ("ph", drop_post_submit(tr["steps"]))):
        rep = repeated_actions(steps, c)
        after = [s for s in steps if s.get("turn") is not None and c and s["turn"] >= c[0]]
        out[f"raw_{tag}"] = rep
        out[f"rate_{tag}"] = rep / len(c) if c else None
        out[f"per_call_{tag}"] = (rep / len(after)) if after else None
    return out

D = {}
for b, arms in SRC.items():
    D[b] = {}
    for arm, dirs in arms.items():
        D[b][arm] = {}
        for d in dirs:
            for l in open(d / "task_results.jsonl"):
                r = json.loads(l); t = r["instance_id"]
                D[b][arm][t] = {"success": int(r["resolved"]), **measures(read_trace(d / "traces" / f"trace_{t}.json"))}

def wilson(k, n, z=1.959964):
    if n == 0: return (0.0, 1.0)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def newcombe_paired(a, b, c, d):
    """Newcombe (1998) method 10: score interval for p1 - p2, paired (a: both 1, b: x only, c: y only, d: both 0)."""
    n = a + b + c + d; p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n); l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (p1 - p2 - dl, p1 - p2 + du)

def strat_ci(groups, n_boot=10000, seed=20261107):
    rng = random.Random(seed); st = []
    for _ in range(n_boot):
        s = n = 0
        for g in groups:
            for _ in range(len(g)):
                s += g[rng.randrange(len(g))]; n += 1
        st.append(s / n)
    st.sort(); return (st[250], st[9749])

out = {}
# 1. resolve
strata = {}
for b in ("modal", "kaggle"):
    ts = sorted(set(D[b]["none"]) & set(D[b]["recap"]))
    x = {t: D[b]["recap"][t]["success"] for t in ts}; y = {t: D[b]["none"][t]["success"] for t in ts}
    a = sum(x[t] and y[t] for t in ts); bb = sum(x[t] and not y[t] for t in ts); cc = sum(y[t] and not x[t] for t in ts); dd = len(ts) - a - bb - cc
    strata[b] = (ts, x, y, (a, bb, cc, dd))
def resolve_block(keys):
    ts = [(b, t) for b in keys for t in strata[b][0]]
    a, bb, cc, dd = [sum(strata[b][3][i] for b in keys) for i in range(4)]
    n = len(ts); diff = (bb - cc) / n
    groups = [[strata[b][1][t] - strata[b][2][t] for t in strata[b][0]] for b in keys]
    lo, hi = strat_ci(groups)
    nl, nh = newcombe_paired(a, bb, cc, dd)
    xx = {f"{b}:{t}": strata[b][1][t] for b in keys for t in strata[b][0]}; yy = {f"{b}:{t}": strata[b][2][t] for b in keys for t in strata[b][0]}
    return {"pairs": n, "none_resolved": a + cc, "recap_resolved": a + bb, "recap_only": bb, "none_only": cc, "both": a, "neither": dd,
            "diff": round(diff, 4), "bootstrap95" + ("_stratified" if len(keys) > 1 else ""): [round(lo, 4), round(hi, 4)],
            "newcombe95": [round(nl, 4), round(nh, 4)], "mcnemar_exact": mcnemar_exact(xx, yy)}
out["resolve"] = {"pooled": resolve_block(["modal", "kaggle"]), "modal": resolve_block(["modal"]), "kaggle": resolve_block(["kaggle"])}
# 2-3. repeats
def paired(metric, keys):
    diffs, groups, excl = [], [], []
    for b in keys:
        g = []
        for t in sorted(set(D[b]["none"]) & set(D[b]["recap"])):
            xr, yr = D[b]["recap"][t][metric], D[b]["none"][t][metric]
            if xr is None or yr is None:
                excl.append((b, t, "recap" if xr is None and yr is not None else "none" if yr is None and xr is not None else "both")); continue
            g.append(xr - yr)
        groups.append(g); diffs += g
    lo, hi = strat_ci(groups)
    xs = [D[b]["recap"][t][metric] for b in keys for t in D[b]["recap"] if t in D[b]["none"] and D[b]["recap"][t][metric] is not None]
    ys = [D[b]["none"][t][metric] for b in keys for t in D[b]["none"] if t in D[b]["recap"] and D[b]["none"][t][metric] is not None]
    return {"n": len(diffs), "mean_recap": round(statistics.mean(xs), 3), "mean_none": round(statistics.mean(ys), 3),
            "mean_paired_diff": round(statistics.mean(diffs), 3), "ci95": [round(lo, 3), round(hi, 3)],
            "wilcoxon_p": round(wilcoxon_signed_rank(diffs)["p"], 4), "trimmed10": round(trimmed_mean(diffs), 3),
            "excluded": excl}
for metric in ("rate_reg", "rate_ph", "raw_reg", "raw_ph", "per_call_reg", "per_call_ph", "compactions"):
    out[metric] = {k: paired(metric, keys) for k, keys in (("pooled", ["modal", "kaggle"]), ("modal", ["modal"]), ("kaggle", ["kaggle"]))}
json.dump(out, open(KG / "pooled_run1_v2.json", "w"), indent=1, default=str)
r = out["resolve"]
for k in ("pooled", "modal", "kaggle"):
    v = r[k]; print(f"RESOLVE {k}: {v['recap_resolved']} vs {v['none_resolved']} of {v['pairs']} (recap-only {v['recap_only']}, none-only {v['none_only']}) diff {v['diff']:+.3f}",
                    "bootstrap", v.get("bootstrap95_stratified", v.get("bootstrap95")), "Newcombe", v["newcombe95"], "McNemar p", v["mcnemar_exact"]["p"])
for metric in ("rate_reg", "rate_ph", "raw_reg", "raw_ph", "per_call_reg", "per_call_ph", "compactions"):
    for k in ("pooled", "modal", "kaggle"):
        v = out[metric][k]
        print(f"{metric:13s} {k:7s} n={v['n']:2d} recap {v['mean_recap']:7.3f} none {v['mean_none']:7.3f} diff {v['mean_paired_diff']:+8.3f} CI {v['ci95']} W p {v['wilcoxon_p']} trim {v['trimmed10']:+.3f}" + (f" excluded {v['excluded']}" if v['excluded'] and k != 'pooled' else ""))
