import unittest

from fake_harness import FakeHarness, node
from memory import (FlatMemory, GraphMemory, Lesson, LessonError, LessonStore, NoMemory,
                    RetrievalConfig, make_arm, render)
from memory.adapter import Task


def L(n, text, task="t0", outcome="success", model="gemma-4-e4b"):
    return Lesson(node_id=n, repo=n.split(":")[0], text=text, outcome=outcome,
                  source_task=task, source_model=model)


class SchemaTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(LessonError):
            L("alpha:auth.sign", "")
        with self.assertRaises(LessonError):
            L("alpha:auth.sign", "x" * 401)
        with self.assertRaises(LessonError):
            L("alpha:auth.sign", "ok", outcome="maybe")
        with self.assertRaises(LessonError):
            L("alpha:auth.sign", "ok", model="claude-x")   # only Gemma authors lessons
        with self.assertRaises(LessonError):
            Lesson(node_id="alpha:auth.sign", repo="alpha", text="ok", outcome="success",
                   source_task="", source_model="gemma-4-e4b")

    def test_id_is_content_hash_and_roundtrips(self):
        a, b = L("alpha:auth.sign", "same  text"), L("alpha:auth.sign", "same text", task="other")
        self.assertEqual(a.lesson_id, b.lesson_id)   # whitespace normalised; provenance not in hash
        self.assertEqual(Lesson.from_json(a.to_json()), a)


class StoreTests(unittest.TestCase):
    def test_keyed_by_node_dedup_filter_and_persist(self):
        import tempfile, os
        s = LessonStore([L("alpha:auth.sign", "one"), L("alpha:auth.sign", "one"),
                         L("beta:auth.sign", "two", task="b1"), L("gamma:auth.sign", "x", model="gemma-4-31b")])
        self.assertEqual(len(s), 3)
        self.assertEqual([l.text for l in s.at_node("alpha:auth.sign")], ["one"])
        f = s.filtered(exclude_repos=["alpha"], exclude_tasks=["b1"], source_model="gemma-4-e4b")
        self.assertEqual(len(f), 0)
        f2 = s.filtered(exclude_repos=["alpha"])
        self.assertEqual(sorted(l.repo for l in f2), ["beta", "gamma"])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.jsonl")
            s.save(p)
            self.assertEqual([l.lesson_id for l in LessonStore.load(p)], [l.lesson_id for l in s])


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.h = FakeHarness()
        self.task = Task("beta-hard", "beta", "Which function adds the signature to requests made by client.send?",
                         (node("beta", "client.send"),))
        # g_only: no word overlap with the task, anchored 2 hops from the seed -> graph yes, flat no
        self.g_only = L(node("beta", "auth.sign"), "hmac secret is read lazily from env")
        # cross: other repo, same role as the seed -> graph via embedding
        self.cross = L(node("alpha", "client.send"), "dispatch retries twice before raising")
        # f_only: word overlap with the task, anchored on an unrelated far node in another repo
        self.f_only = L(node("gamma", "models.Response"), "the signature for requests made by client.send is added late")
        self.store = LessonStore([self.g_only, self.cross, self.f_only])

    def test_none(self):
        arm = NoMemory()
        self.assertEqual(arm.for_task(self.task), [])
        self.assertEqual(arm.recall(node_id="beta:client.send"), [])
        self.assertEqual(render([]), "")

    def test_flat_is_text_only(self):
        got = FlatMemory(self.store).for_task(self.task)
        ids = [r.lesson.lesson_id for r in got]
        self.assertEqual(ids[0], self.f_only.lesson_id)
        self.assertNotIn(self.g_only.lesson_id, ids)
        self.assertTrue(all(r.via == "bm25" for r in got))

    def test_graph_uses_hops_and_embeddings(self):
        got = {r.lesson.lesson_id: r for r in GraphMemory(self.store, self.h).for_task(self.task)}
        self.assertEqual(got[self.g_only.lesson_id].via, "hop:2")
        self.assertAlmostEqual(got[self.g_only.lesson_id].score, 0.36)
        self.assertEqual(got[self.cross.lesson_id].via, "emb")
        self.assertGreater(got[self.cross.lesson_id].score, 0.75)
        self.assertNotIn(self.f_only.lesson_id, got)   # far role: low cosine, not in neighbourhood

    def test_graph_self_anchor_wins(self):
        s = LessonStore([L(node("beta", "client.send"), "exact anchor"), self.cross])
        got = GraphMemory(s, self.h).for_task(self.task)
        self.assertEqual(got[0].via, "self")
        self.assertEqual(got[0].score, 1.0)

    def test_graph_seeds_from_similar_nodes_when_task_has_none(self):
        t = Task("q", "beta", "add an hmac signature header")
        seeds = GraphMemory(self.store, self.h).seeds(t)
        self.assertTrue(seeds[0].endswith(":auth.sign"))

    def test_same_budget_and_k_for_both_arms(self):
        many = LessonStore([L(node("beta", "auth.sign"), f"signature lesson {i} client send requests") for i in range(20)])
        cfg = RetrievalConfig(k=3, per_node_cap=10)
        for arm in (FlatMemory(many, cfg), GraphMemory(many, self.h, cfg)):
            self.assertEqual(len(arm.for_task(self.task)), 3)
        tight = RetrievalConfig(k=10, per_node_cap=10, char_budget=120)
        for arm in (FlatMemory(many, tight), GraphMemory(many, self.h, tight)):
            self.assertLessEqual(len(render(arm.for_task(self.task)).split("\n", 1)[1]), 120)

    def test_per_node_cap(self):
        many = LessonStore([L(node("beta", "auth.sign"), f"lesson {i}") for i in range(5)])
        self.assertEqual(len(GraphMemory(many, self.h, RetrievalConfig(per_node_cap=2)).for_task(self.task)), 2)

    def test_deterministic(self):
        a = [r.lesson.lesson_id for r in make_arm("graph", self.store, self.h).for_task(self.task)]
        b = [r.lesson.lesson_id for r in make_arm("graph", self.store, self.h).for_task(self.task)]
        self.assertEqual(a, b)

    def test_recall_tool_paths(self):
        g = GraphMemory(self.store, self.h)
        self.assertTrue(g.recall(node_id=node("beta", "client.send")))
        self.assertTrue(g.recall(query="hmac signature header"))
        self.assertEqual(g.recall(), [])
        self.assertTrue(FlatMemory(self.store).recall(query="signature client send"))

    def test_render_identical_across_arms(self):
        f = FlatMemory(self.store).recall(query="hmac secret lazily")
        g = GraphMemory(self.store, self.h).recall(node_id=node("beta", "auth.sign"))
        self.assertEqual(render(f[:1]), render(g[:1]))   # same lesson -> same text, whatever the arm

    def test_unknown_arm(self):
        with self.assertRaises(ValueError):
            make_arm("oracle", self.store, self.h)


if __name__ == "__main__":
    unittest.main()
