"""Renor's retrieval rules in Python (memory/renor.py), tested against Renor's own test cases.

Each ported case cites its TypeScript test by source path; that source snapshot is not part of this
study export. The FTS5 cross-check runs the same documents through
SQLite's real `porter unicode61` FTS5 table and bm25(), when this Python's sqlite3 has FTS5.
"""
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fake_harness as fh
from memory import Lesson, LessonStore, RepoIndex, RetrievalConfig, make_arm, render
from memory.renor import (KeywordIndex, budget_cut, count_tokens, infer_replaced, mark_corrected,
                          or_terms, porter_stem, rank, reads_as_correction, subject_of, words)
from runner.build_submission import build

M = fh.FAKE_MODEL


def L(text, title="lesson", keys=("name:send",), at="", subject="", replaces="", task="t0", anchor="a.send",
      repo="alpha", model=M):
    return Lesson(title=title, text=text, outcome="success", keys=tuple(keys), anchor=anchor, repo=repo,
                  source_task=task, source_model=model, at=at, subject=subject, replaces=replaces)


class KeywordTests(unittest.TestCase):
    def test_or_terms_port_keywordOrFallback(self):
        # engine/tests/context/keywordOrFallback.test.ts:13-17
        t = or_terms("how to run the test suite in billing-indexer with vitest threads")
        self.assertIn("vitest", t)
        self.assertNotIn("the", t)
        self.assertIn("billing-indexer", t)            # a hyphenated term stays one (phrase) term
        self.assertGreater(len(t), 3)
        # engine/tests/context/keywordOrFallback.test.ts:31-33
        self.assertEqual(or_terms("partner API client"), ["partner", "api", "client"])

    def test_words_port_factTerms(self):
        # engine/src/memory/factTerms.ts:39-50: >= 2 chars, `_`/`-` kept, unique, in order
        self.assertEqual(words("Parse_header parse_header a X-Sig, x-sig!"), ["parse_header", "x-sig"])

    def test_stemming_port_keyword_stem(self):
        # engine/tests/retrieval/keyword-stem.test.ts:45-51
        kw = KeywordIndex({"a.md": "We log with pino in every service", "b.md": "Run each test before pushing"})
        self.assertEqual(list(kw.scores("logging")), ["a.md"])
        self.assertEqual(list(kw.scores("tests")), ["b.md"])

    def test_porter_reference_vocabulary(self):
        # Porter (1980) examples; the algorithm SQLite's FTS5 porter tokenizer implements
        pairs = {"caresses": "caress", "ponies": "poni", "agreed": "agre", "motoring": "motor",
                 "hopping": "hop", "filing": "file", "happy": "happi", "relational": "relat",
                 "conditional": "condit", "digitizer": "digit", "vietnamization": "vietnam",
                 "hopefulness": "hope", "sensibiliti": "sensibl", "electrical": "electr",
                 "adjustment": "adjust", "adoption": "adopt", "generalization": "gener",
                 "controll": "control", "logging": "log", "tests": "test", "id": "id"}
        for w, s in pairs.items():
            self.assertEqual(porter_stem(w), s, w)

    def test_rare_word_ranks_first(self):
        kw = KeywordIndex({"a": "header header signature", "b": "signature only", "c": "unrelated"})
        got = kw.scores("signature header")
        self.assertEqual(set(got), {"a", "b"})
        self.assertGreater(got["a"], got["b"])

    def test_matches_sqlite_fts5_bm25(self):
        try:
            db = sqlite3.connect(":memory:")
            db.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')")
        except sqlite3.OperationalError:
            self.skipTest("this sqlite3 has no FTS5")
        rng = random.Random(7)
        vocab = ["send", "sending", "header", "headers", "signature", "hmac", "new", "client", "x",
                 "parse_header", "billing-indexer", "billing", "indexer", "connection", "dispatch",
                 "retries", "retry", "timeout", "timeouts", "tests", "testing", "value", "error"]
        docs = {i: " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 25))) for i in range(1, 41)}
        for i, d in docs.items():
            db.execute("INSERT INTO t(rowid, x) VALUES (?, ?)", (i, d))
        kw = KeywordIndex(docs)
        for _ in range(30):
            q = " ".join(rng.choice(vocab + ["the", "with", "is"]) for _ in range(rng.randint(1, 6)))
            match = " OR ".join('"%s"' % t.replace('"', '""') for t in or_terms(q))
            if not match:
                continue
            want = {r: -s for r, s in db.execute("SELECT rowid, bm25(t) FROM t WHERE t MATCH ?", (match,))}
            got = kw.scores(q)
            self.assertEqual(set(got), set(want), q)
            for r in want:
                self.assertAlmostEqual(got[r], want[r], places=9, msg=q)


class OrderAndBudgetTests(unittest.TestCase):
    def test_equal_scores_newest_first(self):
        # engine/src/mvp/agentMemory.ts:605-618: scores within 1e-4 tie, the newer comes first
        items = [("old", 1.00001, "2026-11-01"), ("new", 1.0, "2026-11-05"), ("best", 2.0, "2026-10-01")]
        got = rank(items, score=lambda x: x[1], at=lambda x: x[2])
        self.assertEqual([x[0] for x in got], ["best", "new", "old"])

    def test_budget_is_the_longest_run_that_fits(self):
        # engine/src/mvp/agentMemory.ts:972-974: stop at the first item that does not fit, but pass
        # over one that is bigger than the whole budget
        self.assertEqual(budget_cut([3, 900, 4, 5, 1], cost=lambda x: x, budget=10), [3, 4])
        self.assertEqual(budget_cut([3, 4], cost=lambda x: x, budget=10, admit=lambda x, kept: not kept), [3])

    def test_token_estimate_is_conservative(self):
        self.assertEqual(count_tokens(""), 0)
        s = "The function that calls hmac.new builds the X-Signature header; fix signature bugs there."
        self.assertGreaterEqual(count_tokens(s), len(s) // 4)


# engine/tests/mvp/infer-replaces.test.ts:8-9
OLD = "To deploy the Quillmoor service run npm run ship-prod from the repo root."
FIX = "Actually, to deploy the Quillmoor service run npm run deploy-prod from the repo root, not ship-prod."


def cand(i, excerpt, source="agent-a", at="2026-11-01T00:00:00Z"):
    return {"id": i, "excerpt": excerpt, "source": source, "at": at}


class InferReplacesTests(unittest.TestCase):
    """engine/tests/mvp/infer-replaces.test.ts:18-47 (the scope case does not apply: a pool is one scope)."""

    def test_issue_1457_pair(self):
        old = "Decision for acme-api: tests run on vitest, not jest."
        fix = "Decision for acme-api: tests run on node:test (vitest replaced by node:test)."
        self.assertEqual(infer_replaced(fix, "agent-a", [cand("a", old)])["id"], "a")

    def test_cues(self):
        self.assertTrue(reads_as_correction(FIX))
        self.assertFalse(reads_as_correction(OLD))

    def test_picks_the_overlapping_note(self):
        got = infer_replaced(FIX, "agent-a", [cand("a", OLD), cand("b", "Lunch is at noon on Fridays.")])
        self.assertEqual(got["id"], "a")

    def test_cross_client_needs_a_strong_match(self):
        self.assertEqual(infer_replaced(FIX, "agent-a", [cand("a", OLD, "agent-b")])["id"], "a")
        weak = "Actually, Quillmoor service deploys from root."
        self.assertEqual(infer_replaced(weak, "agent-a", [cand("a", OLD, "agent-a")])["id"], "a")
        self.assertIsNone(infer_replaced(weak, "agent-a", [cand("a", OLD, "agent-b")]))

    def test_different_subjects_never_link(self):
        a = "Decision for alpha-service: return Result values across module boundaries instead of throwing."
        b = "Decision for beta-service: return Result values across module boundaries instead of throwing."
        self.assertIsNone(infer_replaced(b, "agent-a", [cand("a", a, "agent-a")]))
        self.assertIsNone(infer_replaced(b, "agent-a", [cand("a", a, "agent-b")]))
        self.assertEqual(subject_of(a), "alpha-service")
        # stated subjects (a lesson's `subject`) differ: no link even when the words match
        c = cand("a", OLD)
        c["subject"] = "ship"
        self.assertIsNone(infer_replaced(FIX, "agent-a", [c], subject="deploy"))

    def test_no_correction_or_a_tie_links_nothing(self):
        self.assertIsNone(infer_replaced(OLD, "agent-a", [cand("a", OLD)]))
        self.assertIsNone(infer_replaced(FIX, "agent-a", [cand("a", OLD), cand("b", OLD)]))


def hit(i, time, text):
    return {"id": i, "text": text, "at": f"2026-11-08T{time}.000Z"}


C_OLD = hit(1, "10:00:00", "Decision for acme-api: tests run on vitest, not jest")
C_NEW = hit(2, "10:05:00", "Correction for acme-api: tests now run on bun test, not vitest")


def marked(page):
    out, by = mark_corrected(page, statement=lambda h: h["text"], at=lambda h: h["at"],
                             source=lambda h: "mcp", ident=lambda h: h["id"])
    return [h["id"] for h in out], {page[o]["id"]: page[n]["id"] for o, n in by.items()}


class CorrectedHitsTests(unittest.TestCase):
    """engine/tests/mcp/corrected-hits.test.ts:11-41 (scope case omitted: one scope)."""

    def test_marks_and_moves_after_its_correction(self):
        order, by = marked([C_OLD, hit(3, "09:00:00", "acme-api deploys on Fridays"), C_NEW])
        self.assertEqual(order, [3, 2, 1])
        self.assertEqual(by, {1: 2})

    def test_keeps_order_when_the_correction_comes_first(self):
        self.assertEqual(marked([C_NEW, C_OLD]), ([2, 1], {1: 2}))

    def test_unchanged_without_a_correction(self):
        self.assertEqual(marked([C_OLD, hit(4, "10:05:00", "acme-api: tests run on vitest with threads")]),
                         ([1, 4], {}))

    def test_an_older_note_never_corrects_a_newer_one(self):
        early = hit(5, "09:00:00", "Correction for acme-api: tests now run on bun test, not vitest")
        self.assertEqual(marked([early, C_OLD]), ([5, 1], {}))

    def test_different_subjects(self):
        other = hit(6, "10:05:00", "Correction for billing-api: tests now run on bun test, not vitest")
        self.assertEqual(marked([C_OLD, other])[1], {})

    def test_explicit_replaces_link_wins(self):
        page = [{"id": "a", "text": "x one", "at": "1", "r": ""}, {"id": "b", "text": "y two", "at": "2", "r": "a"}]
        out, by = mark_corrected(page, statement=lambda h: h["text"], at=lambda h: h["at"],
                                 source=lambda h: "", replaces=lambda h: h["r"], ident=lambda h: h["id"])
        self.assertEqual([h["id"] for h in out], ["b", "a"])
        self.assertEqual(by, {0: 1})


class StoreAndArmTests(unittest.TestCase):
    def test_remember_writes_subject_time_and_replaces(self):
        store = LessonStore()
        a = store.remember(L("To sign requests call hmac.new with sha1 in the signer.", anchor="p.auth.sign"),
                           at="2026-11-01T00:00:00Z")
        b = store.remember(L("Correction: to sign requests call hmac.new with sha256 in the signer, not sha1.",
                             anchor="q.auth.sign"), at="2026-11-02T00:00:00Z")
        self.assertEqual((a.subject, a.at, a.replaces), ("sign", "2026-11-01T00:00:00Z", ""))
        self.assertEqual(b.replaces, a.lesson_id)
        c = store.remember(L("Correction: to sign requests call hmac.new with sha256 in the signer, not sha1.",
                             anchor="q.client.send"), at="2026-11-03T00:00:00Z")
        self.assertEqual(c.replaces, "")                         # another subject: never supersedes
        self.assertEqual(Lesson.from_json(b.to_json()), b)
        legacy = json.loads(a.to_json())
        for k in ("subject", "at", "replaces"):
            legacy.pop(k)
        self.assertEqual(Lesson.from_json(json.dumps(legacy)).lesson_id, a.lesson_id)   # old files load

    def test_every_memory_arm_lists_the_correction_first_and_marks_the_old_one(self):
        with tempfile.TemporaryDirectory() as d:
            fh.write_repos(d)
            ix = RepoIndex.build(Path(d) / "beta")
            store = LessonStore()
            old = store.remember(L("The sender sets the signature header itself; fix it in send.",
                                   title="Signature header", task="t1"), at="2026-11-01T00:00:00Z")
            new = store.remember(L("Correction: the signature header is not set in send; it is set by "
                                   "the signer helper that send calls. Fix it in the signer instead.",
                                   title="Signature header", task="t2"), at="2026-11-02T00:00:00Z")
            self.assertEqual(new.replaces, old.lesson_id)
            for arm in ("text", "node", "node_khop"):
                got = make_arm(arm, store).recall(ix, "wrong signature header on send", ["Session.send"])
                self.assertEqual([r.lesson for r in got], [new, old], arm)
                self.assertEqual(got[1].superseded_by, new.lesson_id)
                self.assertIn(f"superseded by L:{new.lesson_id}", render(got))

    def test_keyword_score_orders_node_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            fh.write_repos(d)
            ix = RepoIndex.build(Path(d) / "beta")
            store = LessonStore([L("Retries need a timeout.", title="timeouts", task="t1"),
                                 L("The X-Signature header is computed by the signer.", title="signature", task="t2"),
                                 L("Unrelated to anything asked.", title="other", keys=("name:zzz",), task="t3")])
            got = make_arm("node", store).recall(ix, "wrong X-Signature header", ["Session.send"])
            self.assertEqual([r.lesson.title for r in got], ["signature", "timeouts"])  # gate, then keyword rank


class FakeHarnessSmoke(unittest.TestCase):
    """End to end through the built submission and the shipped recall.py, as the agent runs it."""

    def test_built_submission_recalls_renor_style(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fh.write_repos(d / "repos")
            store = LessonStore()
            old = store.remember(L("The sender sets the signature header itself; fix it in send.",
                                   title="Signature header", repo="alpha", task="t1"), at="2026-11-01T00:00:00Z")
            new = store.remember(L("Correction: the signature header is not set in send; the signer "
                                   "helper that calls hmac.new sets it. Fix it in the signer instead.",
                                   title="Signature header", repo="gamma", task="t2"), at="2026-11-02T00:00:00Z")
            alias = "gemma-4-12b-it-qat-w4a16-ct"
            build("node", store, alias, d / "sub", exclude_repos=("beta",), cfg=RetrievalConfig())
            self.assertIn("token_budget: 800", (d / "sub/skills/graph-memory/assets/config.yaml").read_text())
            task = d / "task.md"
            task.write_text("Requests sent with `Session.send` have a wrong X-Signature header.")
            script = d / "sub/skills/graph-memory/scripts/recall.py"
            out = subprocess.run([sys.executable, "-I", "-B", str(script), "--task-file", str(task),
                                  "--repo", str(d / "repos/beta"), "--symbol", "Session.send"],
                                 cwd=d / "sub/skills/graph-memory",
                                 capture_output=True, text=True, timeout=60).stdout
            first, second = out.index(f"[L:{new.lesson_id}]"), out.index(f"[L:{old.lesson_id}]")
            self.assertLess(first, second)
            self.assertIn(f"superseded by L:{new.lesson_id}", out)
            self.assertLessEqual(count_tokens(out), 800 + 60)          # page budget plus the header line


if __name__ == "__main__":
    unittest.main()
