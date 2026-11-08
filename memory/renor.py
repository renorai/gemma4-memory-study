"""Renor's retrieval rules, ported to standard-library Python (Python >= 3.8, no dependencies).

Every rule here is a port of Renor's TypeScript engine and cites its source as file:line at
patrickpassosb/renor main 2fd556b647f2c85852af21177714d2ed20fafa98 (`R:` below). No embeddings,
no model call: the sandbox has neither, and Renor's keyword lane needs neither.

1. Keyword lane (`KeywordIndex`):
   - query = the request's content words, each a quoted FTS5 phrase, joined by OR
     (R:engine/src/retrieval/keyword.ts:89-93 `keywordOrMatch`), words as
     R:engine/src/memory/factTerms.ts:39-50 (`tokens`/`words`: lower case, `_` and `-` kept inside
     a word, at least 2 characters, de-duplicated), minus FTS_STOP_WORDS
     (R:engine/src/retrieval/keyword.ts:72-78). A hyphenated or snake_case word stays one quoted
     term, which FTS5 reads as a phrase of its parts ("billing-indexer" = billing then indexer).
   - index = FTS5 `porter unicode61` (R:engine/src/retrieval/keyword.ts:115): unicode61 splits on
     every non-alphanumeric character, porter stems each token (`porter_stem`, the reference
     algorithm SQLite's tokenizer implements; tokens under 3 characters pass through).
   - score = FTS5 `bm25()` (k1 1.2, b 0.75, idf floored at 1e-6, one combined column), ranked by
     R:engine/src/retrieval/keyword.ts:141-177 (`rankKeywordHits`: best first). Positive here:
     FTS5 negates it.
2. Ordering: equal scores (within 1e-4) newest first, R:engine/src/mvp/agentMemory.ts:605-618.
   The window considered is `k * 10` (R:engine/src/retrieval/keyword.ts:19, get_context's first
   keyword window is topK * 10).
3. Supersession (`infer_replaced`, `mark_corrected`): a later correction supersedes an earlier
   statement on the same subject; the superseded one is marked and listed right after its
   correction. Ports R:engine/src/mvp/inferReplaces.ts:16-79 (cue words, stop words, subject,
   shared-word thresholds, newest-of-best, tie leaves unlinked) and
   R:engine/src/mcp/correctedHits.ts:41-72 (read-time marking of a page and the reorder).
4. Token budget (`budget_cut`): the reply keeps the longest run of the ranked order that fits the
   token budget, passing over an item too big for the whole budget
   (R:engine/src/mvp/agentMemory.ts:972-974); get_context's default budget is 800 tokens
   (R:engine/src/mvp/tools/getContext.ts:26). Renor counts o200k tokens; the sandbox has no
   tokenizer, so `count_tokens` is a conservative estimate (see its docstring).
"""
from __future__ import annotations

import math
import re
from collections import Counter

# --- 1. keyword lane -----------------------------------------------------------------------------

# R:engine/src/retrieval/keyword.ts:72-78
FTS_STOP_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "where", "when",
    "why", "how", "do", "does", "did", "to", "for", "of", "in", "on", "at", "with", "and",
    "or", "my", "your", "our", "their", "his", "her", "it", "this", "that", "about",
    "am", "as", "be", "by", "he", "if", "me", "so", "up", "we",
])

_NOT_WORD = re.compile(r"[^\w-]+")      # R:factTerms.ts:42 [^\p{L}\p{N}\p{M}_-]+ -> space


def words(value):
    """R:engine/src/memory/factTerms.ts:39-50: lower-cased words in order, `_`/`-` kept, >= 2 chars, unique."""
    out, seen = [], set()
    for w in _NOT_WORD.sub(" ", (value or "").lower()).split():
        if len(w) >= 2 and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def or_terms(query):
    """The terms of R:engine/src/retrieval/keyword.ts:89-93 `keywordOrMatch`: content words, OR-joined."""
    return [w for w in words(query) if w not in FTS_STOP_WORDS]


# Porter (1980) stemmer, the reference ANSI C version (tartarus.org/martin/PorterStemmer) that
# SQLite's FTS5 `porter` tokenizer implements, including its departures `bli`->`ble`, `logi`->`log`.
class _Porter:
    def __init__(self, w):
        self.b, self.k, self.j = list(w), len(w) - 1, 0

    def cons(self, i):
        c = self.b[i]
        if c in "aeiou":
            return False
        if c == "y":
            return i == 0 or not self.cons(i - 1)
        return True

    def m(self):
        n, i, j = 0, 0, self.j
        while True:
            if i > j:
                return n
            if not self.cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > j:
                    return n
                if self.cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > j:
                    return n
                if not self.cons(i):
                    break
                i += 1
            i += 1

    def vowelinstem(self):
        return any(not self.cons(i) for i in range(self.j + 1))

    def doublec(self, j):
        return j >= 1 and self.b[j] == self.b[j - 1] and self.cons(j)

    def cvc(self, i):
        if i < 2 or not self.cons(i) or self.cons(i - 1) or not self.cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def ends(self, s):
        n = len(s)
        if n > self.k + 1 or "".join(self.b[self.k - n + 1:self.k + 1]) != s:
            return False
        self.j = self.k - n
        return True

    def setto(self, s):
        self.b[self.j + 1:self.k + 1] = list(s)
        self.k = self.j + len(s)

    def r(self, s):
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        if self.b[self.k] == "s":
            if self.ends("sses"):
                self.k -= 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k -= 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):
                self.setto("ate")
            elif self.ends("bl"):
                self.setto("ble")
            elif self.ends("iz"):
                self.setto("ize")
            elif self.doublec(self.k):
                self.k -= 1
                if self.b[self.k] in "lsz":
                    self.k += 1
            else:
                self.j = self.k
                if self.m() == 1 and self.cvc(self.k):
                    self.setto("e")

    def step1c(self):
        if self.ends("y") and self.vowelinstem():
            self.b[self.k] = "i"

    _STEP2 = {"a": (("ational", "ate"), ("tional", "tion")),
              "c": (("enci", "ence"), ("anci", "ance")),
              "e": (("izer", "ize"),),
              "l": (("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous")),
              "o": (("ization", "ize"), ("ation", "ate"), ("ator", "ate")),
              "s": (("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous")),
              "t": (("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")),
              "g": (("logi", "log"),)}
    _STEP3 = {"e": (("icate", "ic"), ("ative", ""), ("alize", "al")),
              "i": (("iciti", "ic"),),
              "l": (("ical", "ic"), ("ful", "")),
              "s": (("ness", ""),)}
    _STEP4 = {"a": ("al",), "c": ("ance", "ence"), "e": ("er",), "i": ("ic",), "l": ("able", "ible"),
              "n": ("ant", "ement", "ment", "ent"), "o": ("ion", "ou"), "s": ("ism",),
              "t": ("ate", "iti"), "u": ("ous",), "v": ("ive",), "z": ("ize",)}

    def _table(self, table, at):
        for suffix, repl in table.get(self.b[at], ()):
            if self.ends(suffix):
                self.r(repl)
                return

    def step4(self):
        for suffix in self._STEP4.get(self.b[self.k - 1], ()):
            if self.ends(suffix):
                if suffix == "ion" and not (self.j >= 0 and self.b[self.j] in "st"):
                    return
                break
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        self.j = self.k
        if self.b[self.k] == "e":
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self.doublec(self.k) and self.m() > 1:
            self.k -= 1

    def stem(self):
        if self.k <= 1:
            return "".join(self.b)
        self.step1ab()
        if self.k > 0:
            self.step1c()
            self._table(self._STEP2, self.k - 1)
            self._table(self._STEP3, self.k)
            self.step4()
            self.step5()
        return "".join(self.b[:self.k + 1])


_STEM_CACHE = {}


def porter_stem(word):
    """Porter stem of one lower-case token. Like FTS5's porter tokenizer, tokens shorter than 3 or
    longer than 64 characters, and tokens that are not plain ASCII letters, pass through."""
    got = _STEM_CACHE.get(word)
    if got is None:
        got = _Porter(word).stem() if 3 <= len(word) <= 64 and word.isascii() and word.isalpha() else word
        if len(_STEM_CACHE) < 50000:
            _STEM_CACHE[word] = got
    return got


_FTS_TOKEN = re.compile(r"[^\W_]+")     # unicode61: letters and digits; `_`, `-`, `.` separate


def fts_tokens(text):
    """What `porter unicode61` indexes: lower-cased alphanumeric runs, each Porter-stemmed."""
    return [porter_stem(t) for t in _FTS_TOKEN.findall((text or "").lower())]


class KeywordIndex:
    """FTS5 over one column per document, scored with FTS5's bm25(). `docs` maps id -> text."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.toks = {i: fts_tokens(t) for i, t in docs.items()}
        self.tf = {i: Counter(t) for i, t in self.toks.items()}
        self.n = len(self.toks)
        self.avgdl = (sum(len(t) for t in self.toks.values()) / self.n) if self.n else 0.0

    def _freq(self, doc_id, phrase):
        if len(phrase) == 1:
            return self.tf[doc_id].get(phrase[0], 0)
        if any(p not in self.tf[doc_id] for p in phrase):
            return 0
        toks, n = self.toks[doc_id], len(phrase)
        return sum(1 for i in range(len(toks) - n + 1) if toks[i:i + n] == phrase)

    def scores(self, query, ids=None):
        """bm25 of the OR query for each doc in `ids` (default all) that matches; non-matching docs absent.

        idf uses the whole index, as FTS5 does: nRow is every row, nHit the rows holding the phrase."""
        phrases = [p for p in (fts_tokens(t) for t in or_terms(query)) if p]
        if not phrases or not self.n:
            return {}
        ids = list(self.toks) if ids is None else [i for i in ids if i in self.toks]
        out = {}
        for phrase in phrases:
            freq = {i: self._freq(i, phrase) for i in self.toks}
            hits = sum(1 for f in freq.values() if f)
            if not hits:
                continue
            idf = math.log((self.n - hits + 0.5) / (hits + 0.5))
            if idf <= 0.0:
                idf = 1e-6
            for i in ids:
                f = freq.get(i, 0)
                if f:
                    dl = len(self.toks[i])
                    out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0)))
        return out


# --- 2. ordering ---------------------------------------------------------------------------------

def rank(items, score, at, tiebreak=lambda _x: ()):
    """R:engine/src/mvp/agentMemory.ts:605-618: best score first; scores within 1e-4 are equal and
    the newest (`at`, ISO-8601 text, compared as text) comes first. `tiebreak` orders what is left."""
    out = sorted(items, key=tiebreak)
    out.sort(key=lambda x: at(x) or "", reverse=True)
    out.sort(key=lambda x: -round(score(x), 4))
    return out


# --- 3. supersession -----------------------------------------------------------------------------

# R:engine/src/mvp/inferReplaces.ts:16-18
CUE = re.compile(r"\b(actually|instead|correction|corrected|not\s+[^,.;]{1,60}?\s*,?\s+but|no longer|"
                 r"rather than|replaced (?:by|with)|switched to|changed to|should be|is now|are now|"
                 r"was wrong|wrong:)\b", re.IGNORECASE)
CUE_WORDS = frozenset(["actually", "instead", "correction", "corrected", "not", "but", "no", "longer",
                       "rather", "than", "replaced", "by", "with", "switched", "to", "changed", "should",
                       "be", "is", "now", "are", "was", "wrong"])
STOP = frozenset(["the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "it", "this", "that",
                  "we", "i", "you", "use", "run", "with", "at", "as", "by"])
# R:engine/src/mvp/inferReplaces.ts:38-43
MIN_SHARED, MIN_OVERLAP = 2, 0.4
CROSS_MIN_SHARED, CROSS_MIN_OVERLAP = 3, 0.6

_CORR_TOKEN = re.compile(r"[^\W_][\w./:-]*")                  # R:inferReplaces.ts:26
_SUBJECT = re.compile(r"(?<![\w-])[^\W\d_]+(?:-[^\W\d_]+)+(?![\w-])")   # R:inferReplaces.ts:36


def reads_as_correction(text):
    """R:engine/src/mvp/inferReplaces.ts:20-22."""
    return bool(CUE.search(text or ""))


def _corr_tokens(s, drop_cues):
    """R:engine/src/mvp/inferReplaces.ts:24-31."""
    out = set()
    for w in _CORR_TOKEN.findall((s or "").lower()):
        if len(w) < 2 or w in STOP or (drop_cues and w in CUE_WORDS):
            continue
        out.add(w)
    return out


def subject_of(s):
    """R:engine/src/mvp/inferReplaces.ts:35-36: the first plain hyphenated name, or None."""
    m = _SUBJECT.search((s or "").lower())
    return m.group(0) if m else None


def infer_replaced(text, source, candidates, subject=None):
    """R:engine/src/mvp/inferReplaces.ts:49-79. `candidates` are dicts with id, excerpt, source, at
    and an optional explicit `subject`. Every candidate is in the same scope (a lesson pool is one
    scope), so the scope check of line 63 always passes. An explicit subject is a stated name: two
    statements with different explicit subjects never link (the rule of line 66 for stated names);
    without one, line 66's hyphenated-name rule applies. Returns the corrected candidate or None."""
    if not reads_as_correction(text):
        return None
    mine = _corr_tokens(text, True)
    lower = (text or "").lower()
    my_subject = subject_of(text)
    best, best_score, tie = None, -1.0, False
    for c in candidates:
        if subject and c.get("subject") and subject != c["subject"]:
            continue
        theirs0 = (c["excerpt"] or "").lower()
        their_subject = subject_of(c["excerpt"])
        if my_subject and their_subject and my_subject not in theirs0 and their_subject not in lower:
            continue
        theirs = _corr_tokens(c["excerpt"], True)
        if not theirs:
            continue
        shared = sum(1 for w in theirs if w in mine)
        score = shared / len(theirs)
        cross = (c.get("source") or "") != (source or "")
        if shared < (CROSS_MIN_SHARED if cross else MIN_SHARED) or \
                score < (CROSS_MIN_OVERLAP if cross else MIN_OVERLAP):
            continue
        if best is None or score > best_score or (score == best_score and c["at"] > best["at"]):
            best, best_score, tie = c, score, False
        elif score == best_score and c["at"] == best["at"]:
            tie = True
    return best if best is not None and not tie else None


def mark_corrected(page, statement, at, source, subject=None, replaces=None, ident=None):
    """R:engine/src/mcp/correctedHits.ts:41-72 over a ranked page of items.

    Returns (ordered page, {superseded index -> correcting index}) with indexes into `page`.
    An explicit `replaces` link (Renor's `remember` sets one at save time) is honoured first; then
    the read-time inference: a newer item that reads as a correction supersedes the older item it
    best matches (infer_replaced). A superseded item listed above its correction moves to just
    after it, so the current statement is read first; nothing else moves."""
    replaces = replaces or (lambda _x: "")
    subject = subject or (lambda _x: "")
    ident = ident or (lambda x: id(x))
    corrected_by = {}
    pos = {ident(x): i for i, x in enumerate(page)}
    for i, x in enumerate(page):
        j = pos.get(replaces(x) or None)
        if j is not None and j != i and j not in corrected_by:
            corrected_by[j] = i
    for i, x in enumerate(page):
        if not at(x) or not reads_as_correction(statement(x)):
            continue
        if replaces(x) in pos:
            continue  # its explicit link already says what it corrects
        cands = [{"id": j, "excerpt": statement(o), "source": source(o), "at": at(o), "subject": subject(o)}
                 for j, o in enumerate(page)
                 if j != i and at(o) and at(o) < at(x) and j not in corrected_by]
        found = infer_replaced(statement(x), source(x), cands, subject=subject(x))
        if found is not None:
            corrected_by[found["id"]] = i
    if not corrected_by:
        return list(page), {}
    order = list(range(len(page)))
    for old, by in corrected_by.items():
        frm, to = order.index(old), order.index(by)
        if frm > to:
            continue
        order.pop(frm)
        order.insert(order.index(by) + 1, old)
    return [page[i] for i in order], corrected_by


# --- 4. token budget -----------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET = 800   # R:engine/src/mvp/tools/getContext.ts:26
_PIECE = re.compile(r"[A-Za-z]+|\d{1,3}|[^\sA-Za-z\d]")


def count_tokens(text):
    """Estimate of o200k tokens with no tokenizer available: the larger of one token per word,
    number group or symbol, and one per 4 characters, plus one per 8 letters past the first 8 of a
    long word. It over-counts English prose against o200k, so a page under this estimate is under
    the same budget in real tokens."""
    pieces = _PIECE.findall(text or "")
    est = sum(1 + max(0, len(p) - 8) // 8 for p in pieces)
    return max(est, math.ceil(len(text or "") / 4))


def budget_cut(items, cost, budget, admit=lambda _x, _kept: True):
    """R:engine/src/mvp/agentMemory.ts:972-974: the longest run of the order that fits `budget`,
    except that an item too big for the whole budget is passed over. `admit(item, kept)` is an
    extra filter applied before the budget (the count and per-task caps); it sees what is kept."""
    out, used = [], 0
    for x in items:
        if not admit(x, out):
            continue
        c = cost(x)
        if c > budget:
            continue
        if used + c > budget:
            break
        out.append(x)
        used += c
    return out
