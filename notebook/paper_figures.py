"""The paper's two figures, drawn from results/analysis.json and results/run_rows.jsonl only.

Figure 1 (forest): resolve-rate difference, recap - none, run 1, per backend and pooled, with the
Newcombe paired score 95% interval; beside it the A/A contrast (no memory run 2 - run 1) per backend,
and the minimal detectable effect (+/-18 points, DESIGN.md section 10) as dashed lines.
Figure 2 (adherence and delivery): per backend, recap arm run 1: runs with at least one agent note,
runs whose compactions all had something injected, and the share of compactions with injected state.

    python notebook/paper_figures.py      # writes paper/figures/{forest,adherence}.{png,pdf}

make_notebook.py inlines this file (up to the NOTEBOOK-STRIP line) into the notebook.
"""
import json
from pathlib import Path

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated, light surface)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MDE_PTS = 18                                 # DESIGN.md section 10, "Minimal detectable effect"


def delivery(rows):
    """Adherence and delivery per backend, recap arm, run 1 (row-derived; analysis.json has notes only)."""
    out = {}
    for b in ("modal", "kaggle"):
        rr = [r for r in rows if r["arm"] == "recap" and r["run"] == 1 and r["backend"] == b]
        comp = [r for r in rr if r["compactions"] > 0]
        inj = sum(r["injected_share"] * r["compactions"] for r in comp)
        n_comp = sum(r["compactions"] for r in comp)
        out[b] = {
            "runs": len(rr),
            "runs_with_note": sum(1 for r in rr if (r.get("notes_written") or 0) > 0),
            "runs_compacted": len(comp),
            "runs_all_injected": sum(1 for r in comp if r["injected_share"] >= 1.0),
            "runs_any_injected": sum(1 for r in comp if r["injected_share"] > 0),
            "compactions": n_comp,
            "compactions_injected": round(inj),
            "share_of_compactions": inj / n_comp,
            "mean_share_per_run": sum(r["injected_share"] for r in comp) / len(comp),
        }
    return out


def pts(v):
    """One decimal, half away from zero (6.25 -> +6.3), as the paper's text rounds."""
    from decimal import Decimal, ROUND_HALF_UP
    return f"{Decimal(repr(v)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):+}"


def _style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def forest(analysis, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    p, aa = analysis["primary_resolve_run1"], analysis["aa_calibration"]
    items = [("Modal H100, 16 tasks", p["modal"], BLUE), ("Kaggle 4xL4, 32 tasks", p["kaggle"], BLUE),
             ("Pooled, 48 tasks (primary)", p["pooled"], BLUE),
             ("A/A Modal: none run 2 - run 1", aa["modal"]["resolve"], ORANGE),
             ("A/A Kaggle: none run 2 - run 1", aa["kaggle"]["resolve"], ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    ys = [4.6, 3.6, 2.6, 1.0, 0.0]
    for y, (label, r, c) in zip(ys, items):
        lo, hi = (100 * v for v in r["newcombe95"])
        d = 100 * r["diff"]
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.plot([d], [y], "D" if "Pooled" in label else "o", color=c, ms=8 if "Pooled" in label else 6.5,
                mec=SURFACE, mew=1.5, zorder=3)
        ax.text(31.5, y, f"{pts(d)} [{pts(lo)}, {pts(hi)}]", va="center", ha="left", color=INK, fontsize=8)
    ax.axvline(0, color=INK2, lw=0.8)
    for m in (-MDE_PTS, MDE_PTS):
        ax.axvline(m, color=INK2, lw=0.8, ls=(0, (3, 3)))
    for m in (-MDE_PTS, MDE_PTS):
        ax.text(m, 5.3, f"MDE {m:+d}", ha="center", va="center", fontsize=7.5, color=INK2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(ys, [i[0] for i in items])
    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.6, 5.7)
    ax.set_xlabel("difference in resolve rate, percentage points (Newcombe 95% interval)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=2, label="recap - none, run 1"),
                       Line2D([], [], color=ORANGE, marker="o", lw=2, label="A/A: no memory against itself")],
              loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    plt.close(fig)


def adherence(analysis, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    dv = delivery(rows)
    desc = analysis["descriptive_run1"]
    metrics = [
        ("runs with >= 1 agent-written note",
         {b: (desc[f"{b}/recap"]["runs_with_note"], desc[f"{b}/recap"]["n"]) for b in dv}),
        ("compacted runs with something injected\nat every compaction",
         {b: (dv[b]["runs_all_injected"], dv[b]["runs_compacted"]) for b in dv}),
        ("compactions with something injected\n(explorer finding or notes page)",
         {b: (dv[b]["compactions_injected"], dv[b]["compactions"]) for b in dv}),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=200)
    h = 0.34
    for i, (label, vals) in enumerate(metrics):
        y = len(metrics) - 1 - i
        for j, (b, c, name) in enumerate((("modal", BLUE, "Modal H100"), ("kaggle", ORANGE, "Kaggle 4xL4"))):
            k, n = vals[b]
            yy = y + (h / 2 + 0.02) * (1 if j == 0 else -1)
            ax.barh(yy, 100 * k / n, height=h, color=c, label=name if i == 0 else None)
            ax.text(100 * k / n + 1.2, yy, f"{k}/{n} ({100 * k / n:.0f}%)", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(metrics)), [m[0] for m in reversed(metrics)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent (recap arm, run 1)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_figures(analysis, rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed: figures skipped")
        return delivery(rows)
    forest(analysis, out_dir / "forest")
    adherence(analysis, rows, out_dir / "adherence")
    return delivery(rows)


if __name__ == "__main__":  # NOTEBOOK-STRIP
    root = Path(__file__).resolve().parents[1]
    a = json.loads((root / "results" / "analysis.json").read_text())
    r = [json.loads(line) for line in (root / "results" / "run_rows.jsonl").read_text().splitlines() if line.strip()]
    print(json.dumps(make_figures(a, r, root / "paper" / "figures"), indent=1))
