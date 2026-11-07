"""Paired statistics for the arm comparisons. Standard library only."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def paired_diffs(rows: Iterable[Mapping], arm_x: str, arm_y: str, *, model: str | None = None,
                 metric: str = "success") -> list[tuple[str, str, float]]:
    """(model, task_id, x - y) for every (model, task) that has BOTH arms.

    Raises if the two arms do not cover exactly the same (model, task) set, because
    a silently unpaired comparison is the easiest way to fool yourself.
    """
    by = defaultdict(dict)
    for r in rows:
        if model is not None and r["model"] != model:
            continue
        if r["arm"] in (arm_x, arm_y):
            key = (r["model"], r["task_id"])
            if r["arm"] in by[key]:
                raise ValueError(f"duplicate row for {key} arm {r['arm']}")
            by[key][r["arm"]] = float(r[metric])
    missing = [k for k, v in by.items() if len(v) != 2]
    if missing:
        raise ValueError(f"{len(missing)} unpaired (model, task) keys, e.g. {missing[:3]}")
    return [(m, t, v[arm_x] - v[arm_y]) for (m, t), v in sorted(by.items())]


def paired_bootstrap_ci(diffs: Sequence[tuple[str, str, float]] | Sequence[float], *,
                        n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
                        cluster_by_task: bool = True) -> dict:
    """Percentile bootstrap CI for the mean paired difference.

    With (model, task, diff) triples and cluster_by_task=True, tasks are resampled as
    clusters, keeping every model's diff for a drawn task together: this is the right
    unit when the same tasks are pooled across models.
    """
    if not diffs:
        raise ValueError("no paired differences")
    if isinstance(diffs[0], (int, float)):
        clusters = [[float(d)] for d in diffs]
    elif cluster_by_task:
        g = defaultdict(list)
        for _m, t, d in diffs:
            g[t].append(d)
        clusters = [g[t] for t in sorted(g)]
    else:
        clusters = [[d] for _m, _t, d in diffs]
    flat = [d for c in clusters for d in c]
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    n = len(clusters)
    stats = []
    for _ in range(n_boot):
        s = c = 0
        for _ in range(n):
            cl = clusters[rng.randrange(n)]
            s += sum(cl)
            c += len(cl)
        stats.append(s / c)
    stats.sort()
    lo = stats[int(math.floor((alpha / 2) * n_boot))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return {"mean": point, "lo": lo, "hi": hi, "n_pairs": len(flat), "n_clusters": n,
            "n_boot": n_boot, "alpha": alpha, "seed": seed}


def sign_test(diffs: Iterable[float]) -> dict:
    """Exact two-sided sign test on the non-zero paired differences (ties dropped)."""
    d = [x for x in diffs if x != 0]
    n, pos = len(d), sum(1 for x in d if x > 0)
    if n == 0:
        return {"n": 0, "pos": 0, "neg": 0, "p": 1.0}
    k = min(pos, n - pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"n": n, "pos": pos, "neg": n - pos, "p": min(1.0, 2 * tail)}


def per_task_means(diffs: Sequence[tuple[str, str, float]]) -> list[float]:
    """Mean difference per task across models. Pooled (model, task) pairs on the same task
    are not independent, so the sign test runs on these, matching the task-clustered bootstrap."""
    g = defaultdict(list)
    for _m, t, d in diffs:
        g[t].append(d)
    # Rounded so a cancelling mean of graded scores (e.g. 1/3 - 1/3) is an exact tie, not 1e-17.
    return [round(sum(v) / len(v), 12) for _t, v in sorted(g.items())]


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down. Returns adjusted p and reject flag per name."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m = len(order)
    out, running, stop = {}, 0.0, False
    for i, name in enumerate(order):
        adj = min(1.0, (m - i) * pvalues[name])
        running = max(running, adj)
        reject = (not stop) and pvalues[name] <= alpha / (m - i)
        if not reject:
            stop = True
        out[name] = {"p": pvalues[name], "p_holm": running, "reject": reject}
    return out


def cost_table(rows: Iterable[Mapping], *, gpus: int = 4,
               usd_per_mtok: Mapping[str, tuple[float, float]] | None = None) -> list[dict]:
    """Per (model, arm): success rate, tokens, wall time, GPU-hours.

    Our compute is free (Kaggle), so the measured dollar cost is 0. `usd_per_mtok`
    {model: (input, output)} optionally adds a HYPOTHETICAL API-equivalent column;
    it is labelled as such wherever it is printed.
    """
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["arm"])].append(r)
    out = []
    for (model, arm), rs in sorted(g.items()):
        n = len(rs)
        succ = sum(r["success"] for r in rs)
        tin = sum(r.get("tokens_in", 0) for r in rs)
        tout = sum(r.get("tokens_out", 0) for r in rs)
        wall = sum(r.get("wall_s", 0.0) for r in rs)
        row = {
            "model": model, "arm": arm, "n": n, "success_rate": succ / n,
            "tokens_in_mean": tin / n, "tokens_out_mean": tout / n,
            "tokens_per_success": (tin + tout) / succ if succ else None,
            "wall_h": wall / 3600, "gpu_h": wall * gpus / 3600, "usd_measured": 0.0,
        }
        if usd_per_mtok and model in usd_per_mtok:
            pi, po = usd_per_mtok[model]
            row["usd_hypothetical_api"] = (tin * pi + tout * po) / 1e6
        out.append(row)
    return out


def markdown_table(rows: Sequence[Mapping], cols: Sequence[str]) -> str:
    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows)


def mcnemar_exact(x: Mapping[str, int], y: Mapping[str, int]) -> dict:
    """Exact two-sided McNemar test on binary outcomes paired by task (same task set required).
    b = x wins (x=1, y=0), c = y wins. Equivalent to the exact sign test on the discordant pairs."""
    if set(x) != set(y):
        raise ValueError("McNemar needs the same tasks in both arms")
    b = sum(1 for t in x if x[t] == 1 and y[t] == 0)
    c = sum(1 for t in x if x[t] == 0 and y[t] == 1)
    st = sign_test([1] * b + [-1] * c)
    return {"n": len(x), "b": b, "c": c, "p": st["p"]}


def run_matrix(rows: Iterable[Mapping], model: str, arm: str) -> dict[int, dict[str, int]]:
    """{run: {task_id: success}} for one (model, arm)."""
    out: dict[int, dict[str, int]] = defaultdict(dict)
    for r in rows:
        if r["model"] == model and r["arm"] == arm:
            out[int(r["run"])][r["task_id"]] = int(r["success"])
    return dict(out)


def flip_rates(runs: Mapping[int, Mapping[str, int]]) -> dict:
    """Run-to-run noise among identical runs: mean pairwise flip rate over the common tasks,
    and the share of tasks whose outcome is not the same in every run."""
    ids = sorted(runs)
    if len(ids) < 2:
        return {"runs": len(ids), "n": 0, "pairwise": None, "any_flip": None}
    common = sorted(set.intersection(*(set(runs[i]) for i in ids)))
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    pw = [sum(runs[a][t] != runs[b][t] for t in common) / len(common) for a, b in pairs] if common else []
    anyf = sum(len({runs[i][t] for i in ids}) > 1 for t in common)
    return {"runs": len(ids), "n": len(common), "pairwise": sum(pw) / len(pw) if pw else None,
            "any_flip": anyf / len(common) if common else None}


def mean_over_runs(rows: Iterable[Mapping], metric: str = "success") -> list[dict]:
    """Collapse runs: one row per (model, arm, task) whose metric is the mean over its runs."""
    g = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            g[(r["model"], r["arm"], r["task_id"])].append(float(r[metric]))
    return [{"model": m, "arm": a, "task_id": t, metric: sum(v) / len(v), "n_runs": len(v)}
            for (m, a, t), v in sorted(g.items())]


def process_table(rows: Iterable[Mapping]) -> list[dict]:
    """Per (model, arm): runs, success, Hit@k, tool calls, model calls, tokens, recalls."""
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["arm"])].append(r)
    out = []
    for (model, arm), rs in sorted(g.items()):
        def mean(k):
            v = [x[k] for x in rs if x.get(k) is not None]
            return sum(v) / len(v) if v else None

        def median(k):
            v = sorted(x[k] for x in rs if x.get(k) is not None)
            return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else None
        succ = sum(x["success"] for x in rs)
        tok = sum((x.get("tokens_in") or 0) + (x.get("tokens_out") or 0) for x in rs)
        out.append({"model": model, "arm": arm, "runs": len({x["run"] for x in rs}), "n": len(rs),
                    "success_rate": succ / len(rs), "hit@1": mean("hit@1"), "hit@3": mean("hit@3"),
                    "hit@5": mean("hit@5"), "tool_calls_median": median("tool_calls"),
                    "model_calls_median": median("model_calls"), "tokens_in_mean": mean("tokens_in"),
                    "tokens_out_mean": mean("tokens_out"), "tokens_per_success": tok / succ if succ else None,
                    "wall_min_median": (median("wall_s") or 0) / 60, "recalls_mean": mean("recalls"),
                    "lessons_shown_mean": mean("lessons_shown")})
    return out
