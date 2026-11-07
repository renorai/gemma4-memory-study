"""The three memory arms.

    (a) NoMemory     returns nothing.
    (b) FlatMemory   BM25 over lesson text, query = task text. Ignores the graph.
    (c) GraphMemory  seeds = the task's code-graph nodes; lessons are scored by where
                     their anchor node sits relative to a seed: the seed itself,
                     its call-graph neighbourhood (decayed per hop), or a node whose
                     precomputed embedding is close to the seed's (cosine).

Fairness contract (DESIGN.md section 5): (b) and (c) read the SAME lesson pool,
return at most the SAME k lessons under the SAME character budget, and are
rendered by the SAME function. Only the selection differs.
"""
from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Sequence

from .adapter import HarnessAdapter, Task
from .schema import Lesson
from .store import LessonStore

ARMS = ("none", "flat", "graph")


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 5                      # lessons returned per call
    char_budget: int = 1600         # rendered-lesson characters per call
    per_node_cap: int = 2           # at most this many lessons from one anchor node
    # graph arm only
    n_seeds: int = 5                # seeds taken from similar_nodes() when the task has none
    max_hops: int = 2               # call-graph neighbourhood radius
    hop_decay: float = 0.6          # score of a lesson h hops away = hop_decay ** h
    emb_threshold: float = 0.5      # min cosine for embedding-transferred lessons
    emb_weight: float = 0.8         # embedding score = emb_weight * cosine (< 1 so an exact anchor wins)
    # flat arm only
    bm25_k1: float = 1.2
    bm25_b: float = 0.75


@dataclass(frozen=True)
class Retrieved:
    lesson: Lesson
    score: float
    via: str  # "self" | "hop:<h>" | "emb" | "bm25" — logged, never shown to the model


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase identifier-aware tokens; snake_case and CamelCase are also split."""
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        out.append(low)
        parts = [p for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", tok) if p]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


def _select(cands: list[Retrieved], cfg: RetrievalConfig) -> list[Retrieved]:
    """Shared top-k under per-node cap and char budget. Deterministic tie-break."""
    cands = sorted(cands, key=lambda r: (-r.score, r.lesson.lesson_id))
    chosen: list[Retrieved] = []
    per_node: Counter[str] = Counter()
    used = 0
    for r in cands:
        if len(chosen) >= cfg.k:
            break
        if per_node[r.lesson.node_id] >= cfg.per_node_cap:
            continue
        cost = len(render_one(r.lesson)) + 1
        if used + cost > cfg.char_budget:
            continue
        chosen.append(r)
        per_node[r.lesson.node_id] += 1
        used += cost
    return chosen


def render_one(lesson: Lesson) -> str:
    return f"- [{lesson.outcome}] {lesson.node_id}: {lesson.text}"


def render(retrieved: Sequence[Retrieved]) -> str:
    """Identical rendering for every arm. Empty string when nothing was retrieved."""
    if not retrieved:
        return ""
    lines = ["Lessons from your own earlier attempts on other tasks "
             "(they can be wrong; check them with the code tools):"]
    lines += [render_one(r.lesson) for r in retrieved]
    return "\n".join(lines)


class NoMemory:
    arm = "none"

    def __init__(self, *_args, **_kwargs):
        pass

    def for_task(self, task: Task) -> list[Retrieved]:
        return []

    def recall(self, *, query: str | None = None, node_id: str | None = None) -> list[Retrieved]:
        return []


class FlatMemory:
    """Text-similarity experience memory (ExpeL-style baseline), BM25 over lesson text."""

    arm = "flat"

    def __init__(self, store: LessonStore, cfg: RetrievalConfig = RetrievalConfig(), **_kwargs):
        self.cfg = cfg
        self.lessons = list(store)
        self.docs = [Counter(tokenize(l.text)) for l in self.lessons]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, query: str) -> list[Retrieved]:
        q = Counter(tokenize(query))
        k1, b = self.cfg.bm25_k1, self.cfg.bm25_b
        out = []
        for lesson, doc, dl in zip(self.lessons, self.docs, self.lens):
            s = 0.0
            for term in q:
                tf = doc.get(term, 0)
                if tf:
                    s += self.idf[term] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / (self.avgdl or 1)))
            if s > 0:
                out.append(Retrieved(lesson, s, "bm25"))
        return out

    def for_task(self, task: Task) -> list[Retrieved]:
        return _select(self._score(task.prompt), self.cfg)

    def recall(self, *, query: str | None = None, node_id: str | None = None) -> list[Retrieved]:
        # The flat arm sees a node id only as text, which is the point of the baseline.
        text = " ".join(x for x in (query, node_id) if x)
        return _select(self._score(text), self.cfg) if text else []


def _cosine(a: Sequence[float], b: Sequence[float], na: float, nb: float) -> float:
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


class GraphMemory:
    """Lessons anchored to code-graph nodes, retrieved via node embeddings + neighbourhood."""

    arm = "graph"

    def __init__(self, store: LessonStore, adapter: HarnessAdapter,
                 cfg: RetrievalConfig = RetrievalConfig(), **_kwargs):
        self.store, self.adapter, self.cfg = store, adapter, cfg
        # Pre-normalised embeddings of every node that carries a lesson.
        self.anchor_vecs: list[tuple[str, Sequence[float], float]] = []
        for node in sorted(store.nodes()):
            v = adapter.embedding(node)
            if v is not None:
                self.anchor_vecs.append((node, v, _norm(v)))

    def seeds(self, task: Task) -> list[str]:
        if task.seed_nodes:
            return list(task.seed_nodes)[: self.cfg.n_seeds]
        return [n for n, _ in self.adapter.similar_nodes(query=task.prompt, k=self.cfg.n_seeds)]

    def _from_seed(self, seed: str) -> dict[str, Retrieved]:
        best: dict[str, Retrieved] = {}

        def offer(lesson: Lesson, score: float, via: str):
            cur = best.get(lesson.lesson_id)
            if cur is None or score > cur.score:
                best[lesson.lesson_id] = Retrieved(lesson, score, via)

        # 1+2. the seed and its call-graph neighbourhood (BFS, deterministic order)
        seen = {seed: 0}
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            h = seen[node]
            for lesson in self.store.at_node(node):
                offer(lesson, self.cfg.hop_decay ** h, "self" if h == 0 else f"hop:{h}")
            if h < self.cfg.max_hops:
                for nb in self.adapter.neighbors(node):
                    if nb not in seen:
                        seen[nb] = h + 1
                        queue.append(nb)

        # 3. embedding transfer: anchors close to the seed in the harness's node-embedding
        #    space. This is what lets lessons from the 4 dev repos reach unseen repos.
        sv = self.adapter.embedding(seed)
        if sv is not None:
            sn = _norm(sv)
            for node, v, n in self.anchor_vecs:
                if node in seen:
                    continue
                cos = _cosine(sv, v, sn, n)
                if cos >= self.cfg.emb_threshold:
                    for lesson in self.store.at_node(node):
                        offer(lesson, self.cfg.emb_weight * cos, "emb")
        return best

    def _gather(self, seeds: Sequence[str]) -> list[Retrieved]:
        merged: dict[str, Retrieved] = {}
        for seed in seeds:
            for lid, r in self._from_seed(seed).items():
                if lid not in merged or r.score > merged[lid].score:
                    merged[lid] = r
        return _select(list(merged.values()), self.cfg)

    def for_task(self, task: Task) -> list[Retrieved]:
        return self._gather(self.seeds(task))

    def recall(self, *, query: str | None = None, node_id: str | None = None) -> list[Retrieved]:
        if node_id:
            return self._gather([node_id])
        if query:
            return self._gather([n for n, _ in self.adapter.similar_nodes(query=query, k=self.cfg.n_seeds)])
        return []


def make_arm(arm: str, store: LessonStore, adapter: HarnessAdapter,
             cfg: RetrievalConfig = RetrievalConfig()):
    if arm == "none":
        return NoMemory()
    if arm == "flat":
        return FlatMemory(store, cfg)
    if arm == "graph":
        return GraphMemory(store, adapter, cfg)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
