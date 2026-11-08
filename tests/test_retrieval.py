import tempfile
import unittest
from pathlib import Path

import fake_harness as fh
from memory import (ARMS, Lesson, LessonError, LessonStore, NodeMemory, RepoIndex, RetrievalConfig,
                    TextMemory, keys_for, make_arm, mentioned_symbols, render)
from memory.renor import count_tokens
from memory.retrieval import render_one
from memory.keys import has_strong, split_tokens

M = fh.FAKE_MODEL


def L(keys, text="t", title="title", repo="alpha", task="t0", anchor="a.b", outcome="success", model=M):
    return Lesson(title=title, text=text, outcome=outcome, keys=tuple(keys), anchor=anchor, repo=repo,
                  source_task=task, source_model=model)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        fh.write_repos(cls.tmp.name)
        cls.ix = {r: RepoIndex.build(Path(cls.tmp.name) / r) for r in fh.REPOS}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class SchemaTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(LessonError):
            L(["name:x"], text="")
        with self.assertRaises(LessonError):
            L(["name:x"], text="x" * 401)
        with self.assertRaises(LessonError):
            L(["name:x"], title="x" * 81)
        with self.assertRaises(LessonError):
            L([])                                         # a lesson needs an anchor key
        with self.assertRaises(LessonError):
            L(["name:x"], outcome="maybe")
        with self.assertRaises(LessonError):
            L(["name:x"], model="claude-x")               # only Gemma 4 aliases author lessons
        with self.assertRaises(LessonError):
            L(["name:x"], model="gemma-4-31b")            # exact alias only
        L(["name:x"], model="gemma-4-31b-it-qat-w4a16-ct")

    def test_id_is_content_hash_and_roundtrips(self):
        a, b = L(["name:x"], text="same  text"), L(["name:y"], text="same text", task="other")
        self.assertEqual(a.lesson_id, b.lesson_id)       # keys and provenance are not in the hash
        self.assertEqual(Lesson.from_json(a.to_json()), a)


class IndexTests(Fixture):
    def test_symbols_calls_and_lookup_order(self):
        ix = self.ix["gamma"]
        self.assertIn("gamma.client.Connection.dispatch", ix.symbols)
        self.assertEqual(ix.callees("gamma.client.Connection.dispatch"),
                         ["gamma.client.Connection._transport", "gamma.client.make_request"])
        self.assertEqual(ix.khop("gamma.client.Connection.dispatch", 2)["gamma.auth.add_signature"], 2)
        self.assertEqual(ix.lookup("gamma.client.make_request"), ["gamma.client.make_request"])  # exact
        self.assertEqual(ix.lookup("Connection.dispatch"), ["gamma.client.Connection.dispatch"])  # suffix
        self.assertEqual(ix.lookup("connection.DISPATCH"), ["gamma.client.Connection.dispatch"])  # case
        self.assertEqual(ix.lookup("signat"), ["gamma.auth.add_signature"])                      # substring
        self.assertEqual(ix.lookup("zzz_nothing"), [])
        self.assertEqual(ix.symbols_at("gamma/auth.py", [9]), ["gamma.auth.add_signature"])

    def test_broken_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bad.py").write_text("def (:\n")
            (Path(d) / "ok.py").write_text("def f():\n    return 1\n")
            ix = RepoIndex.build(d)
            self.assertEqual(sorted(ix.symbols), ["ok.f"])


class KeyTests(Fixture):
    def test_keys_transfer_across_repos_under_different_names(self):
        ka = keys_for(self.ix["alpha"], "alpha.auth.sign")
        kg = keys_for(self.ix["gamma"], "gamma.auth.add_signature")
        self.assertIn("api:hmac.new", ka)
        self.assertIn("api:hmac.new", kg)                 # same API, different names
        self.assertNotIn("name:sign", kg)
        ks = keys_for(self.ix["beta"], "beta.client.Session.send")
        self.assertIn("name:send", ks)
        self.assertIn("name:send", keys_for(self.ix["alpha"], "alpha.client.Client.send"))
        self.assertTrue(has_strong(ka))
        self.assertEqual(keys_for(self.ix["alpha"], "nope"), [])

    def test_tokens_and_mentions(self):
        self.assertEqual(split_tokens("parse_HTTPHeaderLinks"), ["parse", "http", "header", "links"])
        got = mentioned_symbols(self.ix["beta"], "Requests sent with `Session.send` fail; see prepare()")
        self.assertEqual(got, ["beta.client.Session.send", "beta.client.prepare"])
        self.assertEqual(mentioned_symbols(self.ix["beta"], "the session is sent"), [])


class ArmTests(Fixture):
    def setUp(self):
        self.sign = L(["api:hmac.new", "kind:function", "name:sign"], text="signer lesson",
                      anchor="alpha.auth.sign", task="a1")
        self.send = L(["name:send", "kind:method", "tok:send"], text="sender lesson",
                      anchor="alpha.client.Client.send", task="a2")
        self.weak = L(["kind:method", "tok:client"], text="weak only", task="a3")
        self.store = LessonStore([self.sign, self.send, self.weak])

    def recall(self, arm, repo, symbols=(), text=""):
        return [r.lesson for r in make_arm(arm, self.store).recall(self.ix[repo], text, list(symbols))]

    def test_none(self):
        self.assertEqual(self.recall("none", "gamma", ["Connection.dispatch"]), [])

    def test_node_matches_by_strong_key_only(self):
        self.assertEqual(self.recall("node", "beta", ["Session.send"]), [self.send])
        self.assertEqual(self.recall("node", "gamma", ["Connection.dispatch"]), [])   # nothing recurs at hop 0
        self.assertEqual(self.recall("node", "gamma", ["add_signature"]), [self.sign])  # shared API

    def test_khop_reaches_two_hops_with_decay(self):
        got = make_arm("node_khop", self.store).recall(self.ix["gamma"], "", ["Connection.dispatch"])
        self.assertEqual([r.lesson for r in got], [self.sign])
        self.assertEqual(got[0].via, "hop:2")
        direct = make_arm("node_khop", self.store).recall(self.ix["gamma"], "", ["add_signature"])
        self.assertGreater(direct[0].struct, 0)
        self.assertAlmostEqual(got[0].struct, direct[0].struct * 0.6 ** 2)   # the anchor-key score decays

    def test_seeds_from_task_text(self):
        self.assertEqual(self.recall("node", "beta", text="`Session.send` is wrong"), [self.send])

    def test_text_arm_is_renor_keyword_over_title_and_text(self):
        self.assertEqual(self.recall("text", "gamma", text="signer"), [self.sign])
        self.assertEqual(self.recall("text", "gamma", text="unrelated words"), [])

    def test_shuffled_permutes_keys_only(self):
        sh = self.store.shuffled_keys(7)
        self.assertEqual(sorted(l.lesson_id for l in sh), sorted(l.lesson_id for l in self.store))
        self.assertEqual(sorted(l.text for l in sh), sorted(l.text for l in self.store))
        self.assertEqual(sorted(l.keys for l in sh), sorted(l.keys for l in self.store))
        orig = {l.lesson_id: l.keys for l in self.store}
        self.assertTrue(all(l.keys != orig[l.lesson_id] for l in sh))   # derangement
        self.assertEqual(sh.shuffled_keys(7).key_counts(), self.store.key_counts())

    def test_budget_cap_and_identical_rendering(self):
        many = LessonStore([L(["name:send"], text=f"lesson {i} " + "x" * 300, task=f"t{i // 3}",
                              anchor=f"a.b{i}") for i in range(12)])
        cfg = RetrievalConfig()
        for arm in ("text", "node", "node_khop"):
            got = make_arm(arm, many, cfg).recall(self.ix["beta"], "lesson send", ["Session.send"])
            self.assertLessEqual(len(got), cfg.k)
            self.assertTrue(got)
            used = sum(count_tokens(render_one(r.lesson, r.superseded_by)) + 1 for r in got)
            self.assertLessEqual(used, cfg.token_budget)
            per = {}
            for r in got:
                per[r.lesson.source_task] = per.get(r.lesson.source_task, 0) + 1
            self.assertLessEqual(max(per.values()), cfg.per_task_cap)
            for r in got:
                self.assertIn(f"[L:{r.lesson.lesson_id}]", render(got))
        self.assertEqual(render([]), "")

    def test_deterministic(self):
        a = make_arm("node_khop", self.store).recall(self.ix["gamma"], "x", ["Connection.dispatch"])
        b = make_arm("node_khop", self.store).recall(self.ix["gamma"], "x", ["Connection.dispatch"])
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            make_arm("graph", self.store)
        self.assertEqual(ARMS, ("none", "text", "node", "node_khop", "shuffled", "recap", "recap_shuffled"))
        self.assertIsInstance(make_arm("shuffled", self.store), NodeMemory)
        self.assertIsInstance(make_arm("text", self.store), TextMemory)


if __name__ == "__main__":
    unittest.main()
