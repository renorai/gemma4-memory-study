"""ADAPTER: the only place this package touches the Kaggle harness.

=====================================================================
ADAPTER — written BEFORE HARNESS_README.md was readable (2026-11-07).
Every name below is OUR name, not the harness's. When the dataset is
attached, implement `KaggleHarnessAdapter` (bottom of this file) by
mapping each method onto the real harness tools:

    get_code_neighbors   -> HarnessAdapter.neighbors
    search_similar_code  -> HarnessAdapter.similar_nodes
    get_code_subgraph    -> HarnessAdapter.subgraph
    (precomputed 256-d node embeddings) -> HarnessAdapter.embedding

Nothing else in memory/, runner/ or agent/ may import harness code
directly. If a real tool's semantics differ from the contract below
(for example, search_similar_code takes text rather than a node id),
change the adapter, record the difference in DESIGN.md section
"Adapter deviations", and leave the arms untouched.
=====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Task:
    """One harness task, normalised.

    task_id   stable id from the harness (used as `source_task` in lessons)
    repo      repository the task belongs to (the leave-one-repo-out key)
    prompt    the natural-language task statement given to the agent
    seed_nodes node ids the harness associates with the task, if it exposes
              any. ADAPTER: if the harness gives none, seeds come from
              `similar_nodes(query=prompt)` instead (see retrieval.py).
    """

    task_id: str
    repo: str
    prompt: str
    seed_nodes: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict, compare=False, hash=False)


class HarnessAdapter(Protocol):
    """Contract the memory arms rely on. Node ids are globally unique strings
    (the adapter must prefix them with the repo if the harness does not)."""

    def tasks(self) -> Sequence[Task]:
        """All tasks visible in this run (dev tasks, or hidden test tasks)."""

    def repo_of(self, node_id: str) -> str:
        """Repository a node belongs to."""

    def embedding(self, node_id: str) -> Sequence[float] | None:
        """The harness's precomputed node embedding (256-d in the real data),
        or None when the node has none. Must be a pure lookup: no model call."""

    def neighbors(self, node_id: str) -> Sequence[str]:
        """One-hop call-graph neighbours (callers and callees), deterministic order."""

    def similar_nodes(self, *, node_id: str | None = None, query: str | None = None,
                      k: int = 10) -> Sequence[tuple[str, float]]:
        """Nearest nodes by the harness's embedding search, (node_id, score) best first."""

    def subgraph(self, node_id: str, hops: int = 1) -> dict:
        """{"nodes": [...], "edges": [[src, dst], ...]} around a node."""


class KaggleHarnessAdapter:  # pragma: no cover - needs the dataset
    """ADAPTER STUB. Fill in once HARNESS_README.md is readable.

    Kept deliberately unimplemented so a run on the real harness cannot
    silently fall back to the fake one.
    """

    def __init__(self, harness_root: str):
        self.harness_root = harness_root
        raise NotImplementedError(
            "ADAPTER: map the harness tools onto HarnessAdapter; see the module docstring")
