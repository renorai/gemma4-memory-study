"""The memory arms (DESIGN.md section 3 and 5).

    none        nothing; the skill is not shipped at all.
    text        Renor's keyword lane alone over the whole pool (ReasoningBank-style text keying):
                keyword-OR with Porter stemming over each lesson's title + text.
    node        node-anchored: seeds = repo symbols the agent names (and the task text mentions);
                a lesson is eligible when it shares a STRONG transferable key with a seed.
    node_khop   node-anchored plus k-hop: as `node`, but also the seeds' call-graph neighbours
                within `hops`, each hop's score multiplied by `hop_decay`.
    shuffled    the control: exactly `node`, run on a pool whose key sets were permuted across
                lessons (LessonStore.shuffled_keys). Same text, same budget, same code path.

Renor's memory design, applied to code-structure-anchored lessons (memory/renor.py cites each
rule's TypeScript source): the candidate set is the lessons the arm's key admits (node: a shared
strong key with a seed or, for node_khop, a k-hop neighbour; text: any lesson matching a word);
`_select` then ranks them by Renor's keyword score (bm25 of the OR of the request's content words,
Porter-stemmed) against the issue text and named symbols, equal scores newest first, applies
supersession (a later correction is listed first and the lesson it corrects is marked and listed
right after it), and cuts the page to Renor's token budget. No embeddings anywhere.

Fairness contract: every memory arm reads the same lesson pool (the shuffled arm the same pool
with its keys permuted), returns at most the same k under the same token budget through the same
`_select`, and renders with the same `render`. Only the candidate set differs.

Python >= 3.8, standard library only: this file ships inside the submission's skill.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .keys import STRONG, WEIGHTS, has_strong, key_type, keys_for, mentioned_symbols
from .renor import DEFAULT_TOKEN_BUDGET, KeywordIndex, budget_cut, count_tokens, mark_corrected, rank

# recap, recap_shuffled: no lessons (mode none); the in-task working notes of memory/workmem.py,
# own notes or another task's notes (the control).
ARMS = ("none", "text", "node", "node_khop", "shuffled", "recap", "recap_shuffled")
MODE = {"none": "none", "text": "text", "node": "node", "node_khop": "node_khop", "shuffled": "node",
        "recap": "none", "recap_shuffled": "none"}


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 5                  # lessons returned per call; Renor's keyword window is k * 10
    token_budget: int = DEFAULT_TOKEN_BUDGET  # rendered-lesson tokens per call (get_context's 800)
    per_task_cap: int = 2       # at most this many lessons distilled from one source task
    n_seeds: int = 5            # symbols used as seeds per call
    hops: int = 2               # node_khop only: neighbourhood radius
    hop_decay: float = 0.6      # node_khop only: score multiplier per hop
    max_region: int = 60        # node_khop only: neighbourhood nodes considered per seed
    bm25_k1: float = 1.2        # Renor's keyword lane (FTS5 bm25 constants), every memory arm
    bm25_b: float = 0.75        # Renor's keyword lane (FTS5 bm25 constants), every memory arm
    recap_budget: int = 600     # recap arms only: tokens per "where was I" page (memory/workmem.py)
    recap_every: int = 6        # recap arms only: every n-th note also prints the recap (0: never)
    inject_budget: int = 1500   # recap arms only: tokens of the page injected every turn (oldest notes cut first)


@dataclass(frozen=True)
class Retrieved:
    lesson: object
    score: float
    via: str  # "bm25" | "self" | "hop:<h>" -- logged by the offline replay, never shown
    struct: float = 0.0         # node arms: the anchor-key score that admitted it (a tiebreak)
    superseded_by: str = ""     # lesson_id of the later lesson that corrects this one


def render_one(lesson, superseded_by=""):
    line = f"- [L:{lesson.lesson_id}] [{lesson.outcome}] {lesson.title}: {lesson.text} " \
           f"(seen at {lesson.anchor} in {lesson.repo})"
    if superseded_by:
        line += f" [superseded by L:{superseded_by}, listed above: prefer it]"
    return line


def render(retrieved):
    """Identical rendering for every arm. Empty string when nothing was retrieved."""
    if not retrieved:
        return ""
    lines = ["Lessons from your own earlier runs on OTHER repositories "
             "(hints, not facts; check each one in this repo before relying on it):"]
    lines += [render_one(r.lesson, r.superseded_by) for r in retrieved]
    return "\n".join(lines)


def _select(cands, cfg):
    """Shared by every memory arm: Renor's ordering, supersession and token budget. Deterministic.

    1. rank by keyword score; equal scores newest first (`at`); then the anchor-key score; then id.
    2. keep Renor's keyword window, k * 10.
    3. supersession over that page (renor.mark_corrected).
    4. the token-budgeted page: the longest run that fits cfg.token_budget, at most k lessons and
       at most per_task_cap from one source task."""
    ranked = rank(cands, score=lambda r: r.score, at=lambda r: r.lesson.at,
                  tiebreak=lambda r: (-round(r.struct, 6), r.lesson.lesson_id))[: cfg.k * 10]
    page, corrected_by = mark_corrected(
        ranked, statement=lambda r: r.lesson.statement, at=lambda r: r.lesson.at,
        source=lambda r: r.lesson.source_model, subject=lambda r: r.lesson.subject,
        replaces=lambda r: r.lesson.replaces, ident=lambda r: r.lesson.lesson_id)
    if corrected_by:
        by = {ranked[old].lesson.lesson_id: ranked[new].lesson.lesson_id for old, new in corrected_by.items()}
        page = [replace(r, superseded_by=by[r.lesson.lesson_id]) if r.lesson.lesson_id in by else r
                for r in page]

    def admit(r, kept):
        return len(kept) < cfg.k and \
            sum(1 for x in kept if x.lesson.source_task == r.lesson.source_task) < cfg.per_task_cap

    return budget_cut(page, lambda r: count_tokens(render_one(r.lesson, r.superseded_by)) + 1,
                      cfg.token_budget, admit)


def _keyword_index(store, cfg):
    return KeywordIndex({l.lesson_id: l.statement for l in store}, k1=cfg.bm25_k1, b=cfg.bm25_b)


def _query(task_text, symbols):
    return " ".join([task_text or ""] + list(symbols))


class NoMemory:
    arm = "none"

    def __init__(self, *_a, **_k):
        pass

    def recall(self, index=None, task_text="", symbols=()):
        return []


class TextMemory:
    """Text-keyed memory, ReasoningBank-style (each item's title + content is its key), searched
    with Renor's keyword lane: the OR of the request's content words, Porter-stemmed, bm25.
    ReasoningBank itself retrieves with an embedding model; the sandbox has none and Renor's lane
    needs none, and the paper says so."""

    arm = "text"

    def __init__(self, store, cfg=RetrievalConfig()):
        self.cfg = cfg
        self.lessons = {l.lesson_id: l for l in store}
        self.kw = _keyword_index(store, cfg)

    def recall(self, index=None, task_text="", symbols=()):
        scores = self.kw.scores(_query(task_text, symbols))
        return _select([Retrieved(self.lessons[i], s, "bm25") for i, s in scores.items() if s > 0], self.cfg)


class NodeMemory:
    """Node-anchored memory keyed by transferable keys; hops=0 is `node`, hops>0 is `node_khop`."""

    def __init__(self, store, cfg=RetrievalConfig(), hops=0, arm="node"):
        self.store, self.cfg, self.hops, self.arm = store, cfg, hops, arm
        self.idf = store.idf()
        self.kw = _keyword_index(store, cfg)

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
        if not best:
            return []
        kw = self.kw.scores(_query(task_text, symbols), ids=list(best))
        return _select([Retrieved(r.lesson, kw.get(lid, 0.0), r.via, struct=r.score)
                        for lid, r in best.items()], self.cfg)


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
