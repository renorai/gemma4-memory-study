"""The submission is declarative (HARNESS_README.md): the validator enforces the rules, and every
built arm passes it and runs its skill script in isolation."""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fake_harness as fh
import runner.validate as V
from memory import ARMS, SHIPPED, Lesson, LessonStore
import runner.build_submission as B
from runner.build_submission import build

ROOT = Path(__file__).resolve().parents[1]
ALIAS = "gemma-4-12b-it-qat-w4a16-ct"


def pool():
    def L(keys, text, repo, task):
        return Lesson(title="t", text=text, outcome="success", keys=tuple(keys), anchor=f"{repo}.x",
                      repo=repo, source_task=task, source_model=ALIAS)
    return LessonStore([L(["name:send"], "sender lesson from beta", "beta", "b1"),
                        L(["api:hmac.new"], "signer lesson from gamma", "gamma", "g1"),
                        L(["name:send"], "alpha's own lesson", "alpha", "a1")])


class BuiltArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.reports = {a: build(a, pool() if a != "none" else None, ALIAS, cls.d / a, exclude_repos=("alpha",))
                       for a in ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_arm_is_valid_and_single_model(self):
        for arm, r in self.reports.items():
            self.assertTrue(r["ok"], (arm, r["errors"]))
            self.assertEqual(r["model"], ALIAS)
            self.assertEqual(V.validate(self.d / arm)["errors"], [])

    def test_only_memory_arms_ship_the_skill_and_folds_exclude_their_repo(self):
        self.assertFalse((self.d / "none" / "skills").exists())
        for arm in ARMS[1:]:
            sk = self.d / arm / "skills" / "graph-memory"
            self.assertTrue((sk / "SKILL.md").is_file())
            lessons = LessonStore.load(sk / "assets" / "lessons.txt")
            self.assertEqual(len(lessons), 2)
            self.assertFalse(any(l.repo == "alpha" for l in lessons))
            self.assertIn(f"arm: {arm}", (sk / "assets" / "config.yaml").read_text())

    def test_memory_arms_declare_the_skill_and_none_does_not(self):
        self.assertEqual(V.declared_skills((self.d / "none" / "agent.yaml").read_text()), [])
        for arm in ARMS[1:]:
            self.assertEqual(V.declared_skills((self.d / arm / "agent.yaml").read_text()), ["skills/graph-memory"])
            # everything outside the skill section of agent.yaml is identical to the no-memory arm
            text = (self.d / arm / "agent.yaml").read_text()
            self.assertIn(B.NOTES_TOOL, text)          # memory arms only: the notes sub-agent
            text = text.replace(B.NOTES_TOOL, "")
            self.assertEqual(text[: text.index("\nskills:")] + "\n", (self.d / "none" / "agent.yaml").read_text())
            self.assertTrue((self.d / arm / "sub_agents" / "notes.yaml").is_file())
        self.assertFalse((self.d / "none" / "sub_agents" / "notes.yaml").exists())

    def test_prompts_differ_only_by_the_memory_section(self):
        none = (self.d / "none" / "prompts" / "coder.md").read_text()
        node = (self.d / "node" / "prompts" / "coder.md").read_text()
        self.assertNotIn("Memory from earlier runs", none)
        self.assertIn("Memory from earlier runs", node)
        for arm in ARMS[1:]:   # every memory arm gets the same instructions
            self.assertEqual((self.d / arm / "prompts" / "coder.md").read_text(), node)
        section = node.replace(none.split("## Method")[0], "").split("## Method")[0]
        self.assertTrue(section.startswith("## Memory from earlier runs"))

    def test_recall_is_the_explicit_second_action(self):
        """765dbb0: the first recall must come right after saving the issue, before explorer or any
        read; when the wording was softer the agent skipped it (DESIGN.md section 10)."""
        node = (self.d / "node" / "prompts" / "coder.md").read_text()
        i = node.index("Second action, right after saving the issue")
        self.assertIn("Do not skip it", node[i:i + 300])
        self.assertLess(i, node.index("`file_path` \"scripts/recall.py\""))
        self.assertLess(i, node.index("When `explorer` has named the likely fix site"))

    def test_shipped_code_is_verbatim_and_py38(self):
        for f in SHIPPED:
            shipped = (self.d / "node" / "skills" / "graph-memory" / "scripts" / "memory" / f).read_bytes()
            self.assertEqual(shipped, (ROOT / "memory" / f).read_bytes(), f)
        for p in (self.d / "node").rglob("*.py"):
            ast.parse(p.read_text(), feature_version=(3, 8))

    def test_recall_runs_isolated_inside_a_repo(self):
        repos = self.d / "repos"
        fh.write_repos(repos)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        for arm, symbol, expect in (("node", "Session.send", "sender lesson"),
                                    ("node_khop", "add_signature", "signer lesson"),
                                    ("text", None, "signer lesson")):
            script = self.d / arm / "skills" / "graph-memory" / "scripts" / "recall.py"
            repo = repos / ("beta" if arm == "node" else "gamma")
            (repo / "task.md").write_text("the signer is wrong")
            cmd = [sys.executable, "-I", str(script), "--task-file", "task.md"] + (["--symbol", symbol] if symbol else [])
            r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(expect, r.stdout, (arm, r.stdout))
            self.assertNotIn("alpha's own", r.stdout)
        r = subprocess.run([sys.executable, "-I", str(self.d / "node" / "skills" / "graph-memory" / "scripts" / "recall.py"),
                            "--symbol", "nothing_here"], cwd=repos / "beta", capture_output=True, text=True, env=env)
        # a status line first (recall.py: a broken setup must be visible in the trace), then the page
        self.assertEqual(r.stdout.strip().splitlines(),
                         ["working notes: off (this run has lessons, not notes: do not use the notes tool)",
                          "no lessons for this code"])

    def test_build_refuses_bad_inputs(self):
        with self.assertRaises(ValueError):
            build("node", None, ALIAS, self.d / "x")
        with self.assertRaises(ValueError):
            build("none", None, "gemma-4-31b-it", self.d / "x")      # not an exact alias
        with self.assertRaises(ValueError):
            build("graph", pool(), ALIAS, self.d / "x")

    def test_zip(self):
        r = build("node", pool(), ALIAS, self.d / "z", make_zip=True)
        import zipfile
        with zipfile.ZipFile(r["zip"]) as z:
            self.assertIn("agent.yaml", z.namelist())
            self.assertIn("skills/graph-memory/scripts/recall.py", z.namelist())


class ValidatorRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "s"
        build("node", pool(), ALIAS, self.root)
        self.assertTrue(V.validate(self.root)["ok"])

    def tearDown(self):
        self.tmp.cleanup()

    def errors(self):
        return " | ".join(V.validate(self.root)["errors"])

    def put(self, rel, text=""):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def test_file_types(self):
        for rel in ("notes.jsonl", "run.sh", "data.json", "Makefile", "w.bin"):
            self.put(rel)
            self.assertIn("not allowed", self.errors(), rel)
            (self.root / rel).unlink()
        self.put("adapter_config.json", "{}")
        self.put("adapter_model.safetensors")
        self.assertEqual(self.errors(), "")

    def test_python_only_inside_a_skill(self):
        self.put("main.py")
        self.assertIn("only inside skills", self.errors())
        (self.root / "main.py").unlink()
        self.put("skills/other/scripts/x.py")   # a skill without SKILL.md
        self.assertIn("only inside skills", self.errors())
        self.put("skills/other/SKILL.md", "---\nname: other\n---\n")
        self.assertIn("skills/other: not declared under skills:", self.errors())
        a = self.root / "agent.yaml"
        a.write_text(a.read_text() + "  - skills/other\n")
        self.assertEqual(self.errors(), "")

    def test_skills_must_be_declared_and_named_after_their_folder(self):
        a = self.root / "agent.yaml"
        base = a.read_text()
        a.write_text(base[: base.index("\nskills:")] + "\n")     # the first Kaggle smoke's failure
        self.assertIn("skills/graph-memory: not declared under skills:", self.errors())
        for bad, msg in (("../x", "traversal"), ("/abs", "traversal"), ("skills/missing", "not a folder")):
            a.write_text(base.replace("  - skills/graph-memory", f"  - {bad}"))
            self.assertIn(msg, self.errors(), bad)
        a.write_text(base)
        md = self.root / "skills" / "graph-memory" / "SKILL.md"
        md.write_text(md.read_text().replace("name: graph-memory", "name: graph-memo"))
        self.assertIn("does not match the folder name", self.errors())
        # ADK accepts snake_case only behind a feature flag (off by default): kebab-case only
        (self.root / "skills" / "graph-memory").rename(self.root / "skills" / "graph_memory")
        a.write_text(base.replace("skills/graph-memory", "skills/graph_memory"))
        md = self.root / "skills" / "graph_memory" / "SKILL.md"
        md.write_text(md.read_text().replace("name: graph-memo", "name: graph_memory"))
        self.assertIn("is not kebab-case", self.errors())

    def test_skill_files_must_be_where_run_skill_script_copies_them(self):
        # ADK materializes only scripts/, assets/ and references/ next to the script it runs
        self.put("skills/graph-memory/lessons.txt")
        self.assertIn("run_skill_script would not see it", self.errors())
        (self.root / "skills" / "graph-memory" / "lessons.txt").unlink()
        self.put("skills/graph-memory/assets/helper.py")
        self.assertIn(".py belongs in scripts/", self.errors())

    def test_single_model(self):
        y = self.root / "sub_agents" / "explorer.yaml"
        y.write_text(y.read_text().replace(ALIAS, "gemma-4-31b-it-qat-w4a16-ct"))
        self.assertIn("exactly one base model", self.errors())
        for f in ("agent.yaml", "sub_agents/explorer.yaml", "sub_agents/notes.yaml"):
            p = self.root / f
            if not p.exists():
                continue
            p.write_text(p.read_text().replace(ALIAS, "gemma-4-31b-it").replace("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-31b-it"))
        self.assertIn("not one of the four harness aliases", self.errors())

    def test_include_and_config_path_traversal(self):
        a = self.root / "agent.yaml"
        base = a.read_text()
        for bad, msg in (("../outside.md", "traversal"), ("/etc/passwd", "absolute"),
                         ("prompts/missing.md", "does not exist"), ("skills/graph-memory/scripts/recall.py", "extension")):
            a.write_text(base.replace("!include prompts/coder.md", f"!include {bad}"))
            self.assertIn(msg, self.errors(), bad)
        a.write_text(base)
        y = self.root / "sub_agents" / "explorer.yaml"
        y.write_text(y.read_text() + "  - agent_tool:\n      config_path: ../../x.yaml\n")
        self.assertIn("traversal", self.errors())

    def test_symlink_escape(self):
        target = Path(self.tmp.name) / "secret.md"
        target.write_text("x")
        (self.root / "prompts" / "link.md").symlink_to(target)
        self.assertIn("symlink", self.errors())

    def test_agent_class_placeholders_and_root(self):
        a = self.root / "agent.yaml"
        a.write_text(a.read_text().replace("agent_class: LlmAgent", "agent_class: PythonAgent"))
        self.assertIn("agent_class PythonAgent", self.errors())
        a.write_text(a.read_text().replace("agent_class: PythonAgent", "agent_class: LlmAgent"))
        p = self.root / "prompts" / "coder.md"
        p.write_text(p.read_text() + "\n{task_id}\n")
        self.assertIn("unknown state placeholder", self.errors())
        p.write_text(p.read_text().replace("{task_id}", "{{SKILL_DIR}}"))
        self.assertIn("unfilled build placeholder", self.errors())
        a.unlink()
        self.assertIn("agent.yaml missing", self.errors())

    def test_size_limit(self):
        with mock.patch.object(V, "MAX_BYTES", 1000):
            self.assertIn("not under 3 GiB", self.errors())

    def test_template_itself_is_not_a_submission(self):
        # the template keeps its build markers on purpose; only built directories are valid
        self.assertFalse(V.validate(ROOT / "submission")["ok"])

    def test_cli(self):
        self.assertEqual(V.main([str(self.root)]), 0)
        shutil.copyfile(self.root / "agent.yaml", self.root / "agent.json")
        self.assertEqual(V.main([str(self.root)]), 1)


class LessonlessArms(unittest.TestCase):
    """recap and recap_shuffled (DESIGN.md section 10): no lesson set required, skill shipped."""

    def test_recap_builds_without_lessons(self):
        from runner.build_submission import build
        model = "gemma-4-31b-it-qat-w4a16-ct"
        with tempfile.TemporaryDirectory() as d:
            for arm in ("recap", "recap_shuffled"):
                out = Path(d) / arm
                r = build(arm, None, model, out)
                self.assertEqual(r["n_lessons"], 0)
                sk = out / "skills" / "graph-memory"
                self.assertIn(f"arm: {arm}", (sk / "assets" / "config.yaml").read_text())
                self.assertEqual((sk / "assets" / "lessons.txt").read_text().strip(), "")
                tpl = Path(__file__).resolve().parents[1] / "submission" / "skills" / "graph-memory"
                for f in tpl.rglob("*"):          # every template file ships, except the two built ones
                    if f.is_file() and f.name not in ("config.yaml", "lessons.txt") and "__pycache__" not in f.parts:
                        self.assertTrue((sk / f.relative_to(tpl)).is_file(), f.relative_to(tpl))
            with self.assertRaises(ValueError):
                build("node", None, model, Path(d) / "node")

    def test_recap_runs_without_folds(self):
        from pipeline.experiment import eval_arm
        from harness.swegemma import Task
        calls = []

        def ev(sub, tasks, res):
            calls.append((Path(sub).parent.name, len(tasks)))
            Path(res).mkdir(parents=True, exist_ok=True)
            (Path(res) / "task_results.jsonl").write_text("")
        tasks = [Task(f"{r}_{i}", r, "issue") for r in ("a", "b") for i in range(2)]
        with tempfile.TemporaryDirectory() as d:
            eval_arm(ev, "recap", tasks, None, "gemma-4-31b-it-qat-w4a16-ct", 1, d)
        self.assertEqual(calls, [("all", 4)])


if __name__ == "__main__":
    unittest.main()
