"""Budget planner: projects wall time from MEASURED throughput before a long run starts.

The notebook measures, per model, the mean wall seconds of one episode on a small
calibration slice, then calls `plan()`. The run refuses to start when the projection
exceeds the limit. Nothing here is a measurement; it is arithmetic on measurements.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    name: str
    episodes: int
    sec_per_episode: float   # measured on the calibration slice, already divided by concurrency


def research_phases(model: str, n_tasks: int, sec_ep: float, sec_distill: float,
                    arms_with_memory: int = 2) -> list[Phase]:
    return [
        Phase(f"{model}:none(baseline)", n_tasks, sec_ep),
        Phase(f"{model}:distill", n_tasks, sec_distill),
        Phase(f"{model}:memory-arms", n_tasks * arms_with_memory, sec_ep),
    ]


def plan(phases: list[Phase], limit_h: float = 11.0) -> dict:
    """limit_h defaults to 11 h: the 12 h notebook budget minus 1 h for load and headroom."""
    rows = [{"phase": p.name, "episodes": p.episodes, "hours": p.episodes * p.sec_per_episode / 3600}
            for p in phases]
    total = sum(r["hours"] for r in rows)
    return {"rows": rows, "total_h": total, "limit_h": limit_h, "fits": total <= limit_h}
