"""The memory arms (DESIGN.md section 3 and 5).

    none        nothing; the skill is not shipped at all.
    text        ReasoningBank-style: BM25 of the task text against each lesson's title + text.
    node        node-anchored: seeds = repo symbols the agent names (and the task text mentions);
                a lesson is eligible when it shares a STRONG transferable key with a seed.
    node_khop   node-anchored plus k-hop: as `node`, but also the seeds' call-graph neighbours
                within `hops`, each hop's score multiplied by `hop_decay`.
    shuffled    the control: exactly `node`, run on a pool whose key sets were permuted across
                lessons (LessonStore.shuffled_keys). Same text, same budget, same code path.

Fairness contract: every memory arm reads the same lesson pool (the shuffled arm the same pool
with its keys permuted), returns at most the same k under the same character budget through
the same `_select`, and renders with the same `render`. Only the key differs.

Python >= 3.8, standard library only: this file ships inside the submission's skill.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .keys import STRONG, WEIGHTS, has_strong, key_type, keys_for, mentioned_symbols

ARMS = ("none", "text", "node", "node_khop", "shuffled")
MODE = {"none": "none", "text": "text", "node": "node", "node_khop": "node_khop", "shuffled": "node"}


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 5                  # lessons returned per call
    char_budget: int = 1600     # rendered-lesson characters per call
    per_task_cap: int = 2       # at most this many lessons distilled from one source task
    n_seeds: int = 5            # symbols used as seeds per call
    hops: int = 2               # node_khop only: neighbourhood radius
    hop_decay: float = 0.6      # node_khop only: score multiplier per hop
    max_region: int = 60        # node_khop only: neighbourhood nodes considered per seed
    bm25_k1: float = 1.2        # text only
    bm25_b: float = 0.75        # text only


@dataclass(frozen=True)
class Retrieved:
    lesson: object
    score: float
    via: str  # "bm25" | "self" | "hop:<h>" -- logged by the offline replay, never shown


def render_one(lesson):
    return f"- [L:{lesson.lesson_id}] [{lesson.outcome}] {lesson.title}: {lesson.text} " \
           f"(seen at {lesson.anchor} in {lesson.repo})"


def render(retrieved):
    """Identical rendering for every arm. Empty string when nothing was retrieved."""
    if not retrieved:
        return ""
    lines = ["Lessons from your own earlier runs on OTHER repositories "
             "(hints, not facts; check each one in this repo before relying on it):"]
    lines += [render_one(r.lesson) for r in retrieved]
    return "\n".join(lines)


def _select(cands, cfg):
    """Shared top-k under the per-source-task cap and the character budget. Deterministic."""
    cands = sorted(cands, key=lambda r: (-r.score, r.lesson.lesson_id))
    chosen, per_task, used = [], Counter(), 0
    for r in cands:
        if len(chosen) >= cfg.k:
            break
        if per_task[r.lesson.source_task] >= cfg.per_task_cap:
            continue
        cost = len(render_one(r.lesson)) + 1
        if used + cost > cfg.char_budget:
            continue
        chosen.append(r)
        per_task[r.lesson.source_task] += 1
        used += cost
    return chosen


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text):
    """Lowercase identifier-aware tokens; snake_case and CamelCase are also split."""
    out = []
    for tok in _TOKEN.findall(text or ""):
        out.append(tok.lower())
        parts = [p for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", tok) if p]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


class NoMemory:
    arm = "none"

    def __init__(self, *_a, **_k):
        pass

    def recall(self, index=None, task_text="", symbols=()):
        return []


class TextMemory:
    """ReasoningBank-style text-keyed memory: BM25 over each item's title + content. ReasoningBank
    itself retrieves with an embedding model; the skill sandbox has none, so BM25 stands in, and
    the paper says so."""

    arm = "text"

    def __init__(self, store, cfg=RetrievalConfig()):
        self.cfg = cfg
        self.lessons = sorted(store, key=lambda l: l.lesson_id)
        self.docs = [Counter(tokenize(l.title + " " + l.text)) for l in self.lessons]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def recall(self, index=None, task_text="", symbols=()):
        q = Counter(tokenize(" ".join([task_text or ""] + list(symbols))))
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
        return _select(out, self.cfg)


class NodeMemory:
    """Node-anchored memory keyed by transferable keys; hops=0 is `node`, hops>0 is `node_khop`."""

    def __init__(self, store, cfg=RetrievalConfig(), hops=0, arm="node"):
        self.store, self.cfg, self.hops, self.arm = store, cfg, hops, arm
        self.idf = store.idf()

    def seeds(self, index, task_text, symbols):
        out = []
        for s in symbols:
            for q in index.lookup(s, limit=1):
                if q not in out:
                    out.append(q)
        for q in mentioned_symbols(index, task_text, limit=self.cfg.n_seeds):
            if q not in out:
                out.append(q)
        return out[: self.cfg.n_seeds]

    def score_keys(self, node_keys, factor, via, best):
        strong = [k for k in node_keys if key_type(k) in STRONG]
        cands = {}
        for k in strong:
            for lesson in self.store.with_key(k):
                cands[lesson.lesson_id] = lesson
        nk = set(node_keys)
        for lid, lesson in cands.items():
            shared = nk.intersection(lesson.keys)
            if not has_strong(shared):
                continue
            s = factor * sum(WEIGHTS[key_type(k)] * self.idf.get(k, 0.0) for k in shared)
            cur = best.get(lid)
            if cur is None or s > cur.score:
                best[lid] = Retrieved(lesson, s, via)

    def recall(self, index=None, task_text="", symbols=()):
        if index is None or not len(self.store):
            return []
        best = {}
        for seed in self.seeds(index, task_text, symbols):
            region = index.khop(seed, self.hops) if self.hops else {seed: 0}
            nodes = sorted(region, key=lambda n: (region[n], n))[: self.cfg.max_region]
            for node in nodes:
                h = region[node]
                self.score_keys(keys_for(index, node), self.cfg.hop_decay ** h,
                                "self" if h == 0 else f"hop:{h}", best)
        return _select(list(best.values()), self.cfg)


def make_arm(arm, store, cfg=RetrievalConfig()):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    mode = MODE[arm]
    if mode == "none":
        return NoMemory()
    if mode == "text":
        return TextMemory(store, cfg)
    if mode == "node":
        return NodeMemory(store, cfg, hops=0, arm=arm)
    return NodeMemory(store, cfg, hops=cfg.hops, arm=arm)
