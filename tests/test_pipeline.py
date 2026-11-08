"""The whole protocol on the fake harness, plus the ADAPTER readers and the distiller's rules."""
import json
import tempfile
import unittest
from pathlib import Path

import fake_harness as fh
from harness import (files_opened, lesson_ids_shown, outcome_of, patch_files, patch_old_lines,
                     read_run, read_tasks, read_trace, repo_short, symbols_touched)
from memory import LessonStore, RepoIndex
from pipeline.distill import DistillStats, anchor_candidates, distill_attempt, parse_completion
from pipeline.experiment import R_HEADLINE, read_jsonl, run_protocol
from runner.probes import key_recurrence_probe
from stats.analyze import VERDICTS, analyze

M = fh.FAKE_MODEL


class E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        fh.write_repos(cls.d / "repos")
        cls.tasks = fh.tasks()
        cls.rows, cls.store, cls.stats = run_protocol(
            fh.FakeEval(cls.d / "repos"), fh.fake_generate, cls.tasks,
            lambda t: cls.d / "repos" / t.repo, M, cls.d / "out")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def by(self, arm, run=1):
        return {r["task_id"]: r for r in self.rows if r["arm"] == arm and r["run"] == run}

    def test_every_arm_run_and_task(self):
        ids = {t.task_id for t in self.tasks}
        for arm, runs in (("none", R_HEADLINE), ("node", R_HEADLINE), ("text", 1), ("node_khop", 1), ("shuffled", 1)):
            for r in range(1, runs + 1):
                self.assertEqual(set(self.by(arm, r)), ids, (arm, r))

    def test_no_fold_ships_its_own_repo(self):
        for arm in ("node", "text", "node_khop", "shuffled"):
            for repo in fh.REPOS:
                p = self.d / "out" / M / arm / "run1" / f"fold-{repo}" / "submission" / "skills" / "graph-memory" / "assets" / "lessons.txt"
                fold = LessonStore.load(p)
                self.assertGreater(len(fold), 0)
                self.assertFalse(any(l.repo == repo for l in fold), (arm, repo))

    def test_none_ships_no_skill(self):
        sub = self.d / "out" / M / "none" / "run1" / "all" / "submission"
        self.assertFalse((sub / "skills").exists())
        self.assertNotIn("graph-memory", (sub / "prompts" / "coder.md").read_text())

    def test_mechanics_on_fixture(self):
        none, node, khop = self.by("none"), self.by("node"), self.by("node_khop")
        for repo in fh.REPOS:
            self.assertEqual(none[f"{repo}-digest"]["success"], 1)
            self.assertEqual(none[f"{repo}-header"]["success"], 0)
            self.assertEqual(khop[f"{repo}-header"]["success"], 1)   # API key two hops down
        self.assertEqual(node["alpha-header"]["success"], 1)          # name:send recurs in beta
        self.assertEqual(node["gamma-header"]["success"], 0)          # gamma says `dispatch`
        self.assertGreater(node["alpha-header"]["lessons_shown"], 0)

    def test_lessons_are_self_distilled_and_keyed_transferably(self):
        self.assertEqual(self.stats.attempts, 6)
        self.assertTrue(all(l.source_model == M for l in self.store))
        self.assertTrue(any("api:hmac.new" in l.keys for l in self.store))
        self.assertEqual(LessonStore.load(self.d / "out" / M / "lessons.jsonl").key_counts(), self.store.key_counts())

    def test_rows_are_written_and_deterministic(self):
        self.assertEqual(read_jsonl(self.d / "out" / M / "rows.jsonl"), self.rows)
        with tempfile.TemporaryDirectory() as d2:
            fh.write_repos(Path(d2) / "repos")
            rows2, _, _ = run_protocol(fh.FakeEval(Path(d2) / "repos"), fh.fake_generate, self.tasks,
                                       lambda t: Path(d2) / "repos" / t.repo, M, Path(d2) / "out")
        strip = lambda rs: [{k: v for k, v in r.items()} for r in rs]
        self.assertEqual(strip(rows2), strip(self.rows))

    def test_hit_at_k_and_process_columns(self):
        r = self.by("node_khop")["gamma-header"]
        self.assertEqual(r["hit@1"], 1)
        self.assertEqual(self.by("none")["gamma-header"]["hit@1"], 0)
        self.assertEqual(r["recalls"], 2)
        self.assertGreater(r["model_calls"], 0)

    def test_analysis_runs_and_does_not_fire_on_the_toy(self):
        res = analyze(self.rows, primary_models=(M,), headline=("node", "none"))
        h = res["primary"][M]
        self.assertIn(h["verdict"], VERDICTS)
        self.assertFalse(h["supported"])                 # 6 tasks can support nothing
        self.assertEqual(h["n_pairs"], 6)
        self.assertEqual(len(res["per_run_mcnemar"][M]), R_HEADLINE)
        self.assertEqual(res["flip_rate"][f"{M}/none"]["runs"], R_HEADLINE)
        self.assertTrue(all(c["label"].startswith("exploratory") for c in res["exploratory"]))
        self.assertIn(M, res["moderators"])

    def test_key_recurrence_probe(self):
        cache = {r: RepoIndex.build(self.d / "repos" / r) for r in fh.REPOS}
        p = key_recurrence_probe(self.tasks, lambda t: cache[t.repo])
        self.assertEqual(p["n"], 6)
        self.assertEqual(p["rate"], 1.0)                 # every fix site calls hmac.new


class AdapterTests(unittest.TestCase):
    def test_tasks_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            p.write_text(json.dumps({"instance_id": "encode__httpx-1", "repo": "encode/httpx",
                                     "problem_statement": "x", "patch": "+++ b/a.py\n"}) + "\n"
                         + json.dumps({"task_id": "t2", "repository": "fastapi", "problem_description": "y",
                                       "hints": "h"}) + "\n")
            a, b = read_tasks(p)
            self.assertEqual((a.task_id, a.repo, a.problem, a.gold_patch), ("encode__httpx-1", "httpx", "x", "+++ b/a.py\n"))
            self.assertEqual((b.repo, b.hints), ("fastapi", "h"))
            p.write_text(json.dumps({"repo": "x"}) + "\n")
            with self.assertRaises(ValueError):
                read_tasks(p)
        self.assertEqual(repo_short("Textualize__rich-12"), "rich")

    def test_outcome_fails_closed(self):
        self.assertEqual(outcome_of({"resolved": True}), 1)
        self.assertEqual(outcome_of({"success": 0}), 0)
        self.assertEqual(outcome_of({"passed": "false"}), 0)
        self.assertEqual(outcome_of({"score": 1.0}), 1)
        self.assertEqual(outcome_of({"status": "timeout"}), 0)
        with self.assertRaises(ValueError):
            outcome_of({"task_id": "t"})

    def test_atif_trace(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trace_t.json"
            p.write_text(json.dumps({"steps": [
                {"source": "user", "message": "fix", "timestamp": "2026-11-07T00:00:00Z"},
                {"source": "agent", "timestamp": "2026-11-07T00:01:00Z",
                 "tool_calls": [{"tool_call_id": "1", "function_name": "get_code_neighbors",
                                 "arguments": "{\"node\": \"Client.send\"}"}],
                 "observation": {"results": [{"source_call_id": "1", "content": "[...]"}]},
                 "metrics": {"prompt_tokens": 10, "completion_tokens": 2}},
                {"source": "agent", "timestamp": "2026-11-07T00:02:30Z",
                 "tool_calls": [{"id": "2", "function": {"name": "run_command", "arguments":
                                 {"command": "python3 x/recall.py --symbol pkg.f"}}}],
                 "observation": {"results": [{"source_call_id": "2", "content": "- [L:0123456789ab] ..."}]},
                 "metrics": {"prompt_tokens": 20, "completion_tokens": 3}}]}))
            t = read_trace(p)
        self.assertEqual((t["model_calls"], t["tokens_in"], t["tokens_out"], t["wall_s"]), (2, 30, 5, 150.0))
        self.assertEqual(t["steps"][0]["args"], {"node": "Client.send"})
        self.assertEqual(symbols_touched(t["steps"]), ["Client.send", "pkg.f"])
        self.assertEqual(lesson_ids_shown(t["steps"]), ["0123456789ab"])

    def test_patch_helpers(self):
        p = "--- a/x/y.py\n+++ b/x/y.py\n@@ -10,2 +10,3 @@\n-a\n+b\n--- a/z.py\n+++ b/z.py\n@@ -3 +3 @@\n"
        self.assertEqual(patch_files(p), ["x/y.py", "z.py"])
        self.assertEqual(patch_old_lines(p), {"x/y.py": [10, 11], "z.py": [3]})
        steps = [{"tool": "read_file", "args": {"path": "./x/y.py"}}, {"tool": "read_file", "args": {"path": "x/y.py"}},
                 {"tool": "write_file", "args": {"path": "/tmp/task.md"}}, {"tool": "edit_file", "args": {"path": "z.py"}}]
        self.assertEqual(files_opened(steps), ["x/y.py", "z.py"])


class DistillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        fh.write_repos(cls.tmp.name)
        cls.ix = RepoIndex.build(Path(cls.tmp.name) / "alpha")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def attempt(self, **kw):
        a = {"task_id": "alpha-header", "repo": "alpha", "model": M, "run": 1, "outcome": "failure",
             "prompt": "Requests sent with `Client.send` are wrong.",
             "patch": "--- a/alpha/client.py\n+++ b/alpha/client.py\n@@ -10,1 +10,1 @@\n",
             "steps": [{"tool": "get_code_neighbors", "args": {"node": "Client.send"}, "output": "[]"}]}
        a.update(kw)
        return a

    def test_anchors_are_only_touched_symbols(self):
        got = anchor_candidates(self.attempt(), self.ix)
        self.assertEqual(got[0], "alpha.client.build_request")   # the agent's own patch first
        self.assertIn("alpha.client.Client.send", got)
        self.assertNotIn("alpha.auth.sign", got)                 # never touched

    def test_rejects_invented_anchor_forbidden_text_and_garbage(self):
        st = DistillStats()
        bad = json.dumps([{"anchor": "alpha.auth.sign", "title": "t", "lesson": "x"},
                          {"anchor": "alpha.client.Client.send", "title": "t", "lesson": "SECRET_GOLD here"},
                          {"anchor": "alpha.client.Client.send", "title": "t", "lesson": "y" * 401}])
        out = distill_attempt(self.attempt(), lambda p: bad, self.ix, stats=st, forbidden_strings=["SECRET_GOLD"])
        self.assertEqual(out, [])
        self.assertEqual(st.dropped["anchor_not_touched"], 1)
        self.assertEqual(st.dropped["contains_forbidden"], 1)
        self.assertTrue(any(k.startswith("invalid:") for k in st.dropped))
        distill_attempt(self.attempt(), lambda p: "no json", self.ix, stats=st)
        self.assertEqual(st.dropped["unparseable"], 1)
        self.assertEqual(distill_attempt(self.attempt(model="gpt-x"), fh.fake_generate, self.ix), [])

    def test_keys_come_from_source_not_from_the_model(self):
        out = distill_attempt(self.attempt(), lambda p: json.dumps(
            [{"anchor": "alpha.client.Client.send", "title": "t", "lesson": "x", "keys": ["api:evil"]}]), self.ix)
        self.assertEqual(len(out), 1)
        self.assertIn("name:send", out[0].keys)
        self.assertNotIn("api:evil", out[0].keys)

    def test_prompt_has_no_gold_and_no_unfilled_slots(self):
        seen = {}
        distill_attempt(self.attempt(), lambda p: seen.setdefault("p", p) and "[]", self.ix)
        self.assertNotIn("{{", seen["p"])
        self.assertIn("- alpha.client.Client.send", seen["p"])
        self.assertEqual(parse_completion("<think>[1]</think> ```json\n[{\"a\": 1}]\n```"), [{"a": 1}])


if __name__ == "__main__":
    unittest.main()
