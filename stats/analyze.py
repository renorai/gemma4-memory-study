"""The preregistered analysis (DESIGN.md section 2). Changing this file after the
first real run is a protocol deviation and must be logged in DESIGN.md section 10."""
from __future__ import annotations

from typing import Iterable, Mapping

from . import cost_table, holm, paired_bootstrap_ci, paired_diffs, sign_test

PRIMARY = ("graph", "flat")     # H1: graph-anchored beats flat experience memory
CO_PRIMARY = ("graph", "none")  # H2: graph-anchored beats no memory
SECONDARY = (("flat", "none"),)
ALPHA = 0.05
N_BOOT = 10_000
SEED = 20261107


def compare(rows: list[Mapping], x: str, y: str, model: str | None = None) -> dict:
    d = paired_diffs(rows, x, y, model=model)
    ci = paired_bootstrap_ci(d, n_boot=N_BOOT, alpha=ALPHA, seed=SEED)
    st = sign_test([v for _m, _t, v in d])
    return {"x": x, "y": y, "model": model or "pooled", **ci, "sign_p": st["p"],
            "wins": st["pos"], "losses": st["neg"]}


def analyze(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    models = sorted({r["model"] for r in rows})
    h1 = compare(rows, *PRIMARY)
    h2 = compare(rows, *CO_PRIMARY)
    adj = holm({"H1": h1["sign_p"], "H2": h2["sign_p"]}, ALPHA)
    # Decision rule, fixed before any run: a hypothesis is SUPPORTED only when the
    # bootstrap CI lies entirely above 0 AND the Holm-adjusted sign test rejects.
    for name, h in (("H1", h1), ("H2", h2)):
        h["p_holm"] = adj[name]["p_holm"]
        h["supported"] = bool(h["lo"] > 0 and adj[name]["reject"])
        h["verdict"] = ("SUPPORTED" if h["supported"]
                        else "CONTRADICTED" if h["hi"] < 0 and adj[name]["reject"]
                        else "NOT SUPPORTED (inconclusive)")
    per_model = [compare(rows, x, y, m) for m in models for (x, y) in (PRIMARY, CO_PRIMARY) + SECONDARY]
    return {"H1": h1, "H2": h2, "per_model": per_model, "cost": cost_table(rows)}
