import json
import tempfile
import unittest
from pathlib import Path

from fake_harness import FakeAgent, FakeHarness, fake_generate
from memory import DistillStats, LessonStore, distill_attempt, touched_nodes
from runner.experiment import loro_memory, read_jsonl, run_loro
from stats.analyze import analyze

ROOT = Path(__file__).resolve().parents[1]


def attempt(**kw):
    a = {"run_id": "r", "task_id": "alpha-easy", "repo": "alpha", "model": "gemma-4-e4b",
         "outcome": "success", "prompt": "p", "error": None,
         "steps": [{"tool": "get_code_neighbors", "args": {"node_id": "alpha:client.send"},
                    "nodes": ["alpha:client.build_request", "alpha:utils.parse_headers"]}]}
    a.update(kw)
    return a


class DistillTests(unittest.TestCase):
    def setUp(self):
        self.h = FakeHarness()

    def test_touched_nodes_order(self):
        self.assertEqual(touched_nodes(attempt())[0], "alpha:client.send")

    def test_happy_path_uses_same_model_and_provenance(self):
        got = distill_attempt(attempt(), fake_generate, repo_of=self.h.repo_of)
        self.assertEqual(len(got), 1)
        l = got[0]
        self.assertEqual((l.node_id, l.repo, l.source_model, l.source_task, l.outcome),
                         ("alpha:client.send", "alpha", "gemma-4-e4b", "alpha-easy", "success"))

    def test_prompt_contains_anchors_and_outcome(self):
        seen = {}
        def gen(p):
            seen["p"] = p
            return "[]"
        distill_attempt(attempt(outcome="failure"), gen, repo_of=self.h.repo_of)
        self.assertIn("Outcome of the attempt (from the harness scorer): failure", seen["p"])
        self.assertIn("- alpha:client.build_request", seen["p"])
        self.assertNotIn("{{", seen["p"])

    def test_rejects_invented_anchor_gold_and_garbage(self):
        st = DistillStats()
        gen = lambda p: ('```json\n[{"node_id": "alpha:auth.sign", "lesson": "never touched"},'
                         '{"node_id": "alpha:client.send", "lesson": "answer is alpha:client.build_request"},'
                         '{"node_id": "alpha:client.send", "lesson": "kept"}]\n```')
        got = distill_attempt(attempt(), gen, repo_of=self.h.repo_of, stats=st,
                              forbidden_strings=["alpha:client.build_request"])
        self.assertEqual([l.text for l in got], ["kept"])
        self.assertEqual(st.dropped["anchor_not_touched"], 1)
        self.assertEqual(st.dropped["contains_gold"], 1)
        self.assertEqual(distill_attempt(attempt(), lambda p: "no json here", repo_of=self.h.repo_of, stats=st), [])
        self.assertEqual(st.dropped["unparseable"], 1)
        self.assertEqual(distill_attempt(attempt(steps=[]), fake_generate, repo_of=self.h.repo_of, stats=st), [])
        self.assertEqual(st.dropped["no_anchor"], 1)

    def test_non_gemma_model_cannot_author(self):
        st = DistillStats()
        got = distill_attempt(attempt(model="gpt-x"), fake_generate, repo_of=self.h.repo_of, stats=st)
        self.assertEqual(got, [])
        self.assertTrue(any(k.startswith("invalid:") for k in st.dropped))


class EndToEndFakeTests(unittest.TestCase):
    """Whole protocol on the fake harness: baseline -> distil -> LORO arms -> stats."""

    def run_once(self, out=None):
        h = FakeHarness()
        return run_loro(h, FakeAgent(h), fake_generate, "fake-policy", out_dir=out)

    def test_every_arm_every_task_and_no_leakage(self):
        rows, store, st = self.run_once()
        tasks = {t.task_id for t in FakeHarness().tasks()}
        for arm in ("none", "flat", "graph"):
            self.assertEqual({r["task_id"] for r in rows if r["arm"] == arm}, tasks)
        self.assertGreater(len(store), 0)
        for repo in ("alpha", "beta", "gamma"):
            fold = loro_memory(store, repo, "fake-policy")
            self.assertTrue(len(fold) > 0)
            self.assertFalse(any(l.repo == repo for l in fold))

    def test_mechanics_on_fixture(self):
        rows, _store, _st = self.run_once()
        by = {(r["arm"], r["task_id"]): r for r in rows}
        # no memory: easy solved, hard failed, in every repo
        for repo in ("alpha", "beta", "gamma"):
            self.assertEqual(by[("none", f"{repo}-easy")]["success"], 1)
            self.assertEqual(by[("none", f"{repo}-hard")]["success"], 0)
            self.assertEqual(by[("graph", f"{repo}-hard")]["success"], 1)   # cross-repo lesson via embedding
            self.assertGreater(by[("graph", f"{repo}-hard")]["n_lessons"], 0)

    def test_deterministic_and_written(self):
        with tempfile.TemporaryDirectory() as d:
            rows1, store1, _ = self.run_once(d)
            rows2, _, _ = self.run_once()
            self.assertEqual(rows1, rows2)
            self.assertEqual(read_jsonl(Path(d) / "rows-fake-policy.jsonl"), rows1)
            self.assertEqual([l.lesson_id for l in LessonStore.load(Path(d) / "lessons-fake-policy.jsonl")],
                             [l.lesson_id for l in store1])

    def test_analysis_runs_on_fake_rows(self):
        rows, _, _ = self.run_once()
        res = analyze(rows)
        self.assertIn(res["H1"]["verdict"], ("SUPPORTED", "CONTRADICTED", "NOT SUPPORTED (inconclusive)"))
        # 6 tasks is far too few to support anything: the rule must not fire on the toy
        self.assertFalse(res["H2"]["supported"])
        self.assertEqual(res["H2"]["n_pairs"], 6)


class SkillToolTests(unittest.TestCase):
    def test_configure_and_recall(self):
        import sys
        sys.path.insert(0, str(ROOT / "agent" / "skills" / "graph_memory"))
        import tool
        h = FakeHarness()
        rows, store, _ = run_loro(h, FakeAgent(h), fake_generate, "fake-policy")
        with tempfile.TemporaryDirectory() as d:
            store.save(Path(d) / "lessons.jsonl")
            cfg = json.loads((ROOT / "agent" / "memory_config.json").read_text())
            cfg["lessons_path"] = "lessons.jsonl"
            (Path(d) / "memory_config.json").write_text(json.dumps(cfg))
            self.assertEqual(tool.configure(h, Path(d) / "memory_config.json", Path(d) / "log.jsonl"), "graph")
            out = tool.recall_lessons(node_id="beta:client.send")
            self.assertTrue(out.startswith("Lessons from your own earlier attempts"))
            self.assertEqual(len((Path(d) / "log.jsonl").read_text().splitlines()), 1)
            cfg["arm"] = "none"
            (Path(d) / "memory_config.json").write_text(json.dumps(cfg))
            tool.configure(h, Path(d) / "memory_config.json")
            self.assertEqual(tool.recall_lessons(node_id="beta:client.send"), "no lessons for this code")


if __name__ == "__main__":
    unittest.main()
