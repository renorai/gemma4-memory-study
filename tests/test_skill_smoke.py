"""The skill smoke of the first real Kaggle run (8 Nov) failed three ways: write_file to
/tmp/task.md was rejected as path traversal, the skill was not declared under `skills:`, and
recall.py was not reachable from the task's working folder. This replays that smoke end to end
in fake_harness.Sandbox, which reproduces those harness rules, with the shipped prompt, the
built submission and the shipped recall.py: issue saved, recall.py called, notes in /tmp only."""
import hashlib
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import fake_harness as fh
import runner.validate as V
from memory import Lesson, LessonStore
from runner.build_submission import build

ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
# Backticks, $VARs, quotes, a backslash and a line that looks like a heredoc end: the save
# command must keep them verbatim.
ISSUE = ("Requests sent with `Client.send` have a wrong X-Signature header.\n"
         "Repro: SECRET=$HOME python -c 'print(\"x\")' \\\n"
         "EOF\n"
         "Expected the header to match hmac.new(SECRET, url).")


def tree(root):
    """{relative path: sha256} of every file under root, __pycache__ included."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class SkillSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        fh.write_repos(d / "repos")
        lessons = LessonStore([Lesson(
            title="Signature bugs live in the signer",
            text="The function that calls hmac.new builds the X-Signature header; fix it there.",
            outcome="success", keys=("name:send", "api:hmac.new"), anchor="beta.client.Session.send",
            repo="beta", source_task="beta-header", source_model=ALIAS)])
        self.sub = d / "sub"
        self.report = build("node", lessons, ALIAS, self.sub, exclude_repos=("alpha",))
        self.box = fh.Sandbox(self.sub, d / "repos" / "alpha", d / "box")
        self.ws_before = tree(self.box.workspace)
        self.sub_before = tree(self.sub)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_old_protocol_is_rejected_by_the_fake_harness(self):
        # the first real run's failure, reproduced: write_file cannot reach /tmp
        self.assertIn("path traversal", self.box.write_file("/tmp/task.md", ISSUE))
        self.assertFalse((self.box.tmp / "task.md").exists())

    def test_skill_smoke_passes_end_to_end(self):
        self.assertTrue(self.report["ok"], self.report["errors"])
        self.assertEqual(self.box.declared_skills(), ["skills/graph-memory"])

        calls = list(fh.follow_memory_protocol(self.box, ISSUE))
        calls.append(self.box.recall_call(ISSUE, ["alpha.client.Client.send"]))
        tools = [c[0] for c in calls]
        self.assertEqual(tools, ["run_command", "run_skill_script", "run_skill_script"])

        # 1. the issue is saved, verbatim, in /tmp
        self.assertEqual((self.box.tmp / "task.md").read_text(), ISSUE + "\n")

        # 2. recall.py is reached through run_skill_script and prints a lesson for the fix site
        for _, args, out in calls[1:]:
            self.assertEqual(args["file_path"], "scripts/recall.py")
            self.assertIn("/workspace", args["args"])
            self.assertFalse(out.startswith(("Error", "memory unavailable")), out)
            self.assertNotIn("Traceback", out)
        self.assertIn("[L:", calls[2][2])
        self.assertIn("hmac.new", calls[2][2])
        steps = [{"tool": t, "args": a, "output": o} for t, a, o in calls]
        self.assertEqual(sum(fh_is_recall(s) for s in steps), 2)

        # 3. notes go to /tmp only: /workspace (what submit_patch diffs) and the submission are
        #    byte-identical, with no __pycache__, and /tmp holds just the issue
        self.assertEqual(tree(self.box.workspace), self.ws_before)
        self.assertEqual(tree(self.sub), self.sub_before)
        self.assertEqual(sorted(tree(self.box.tmp)), ["task.md"])

    def test_undeclared_skill_is_not_found_and_fails_validation(self):
        a = self.sub / "agent.yaml"
        text = a.read_text()
        a.write_text(text[: text.index("\nskills:")] + "\n")
        out = self.box.run_skill_script("graph-memory", "scripts/recall.py", ["--repo", "/workspace"])
        self.assertIn("not found", out)
        self.assertIn("not declared under skills:", " ".join(V.validate(self.sub)["errors"]))

    def test_dict_args_and_one_list_like_symbol_also_work(self):
        list(fh.follow_memory_protocol(self.box, ISSUE))
        out = self.box.run_skill_script("graph-memory", "scripts/recall.py", {
            "task-file": "/tmp/task.md", "repo": "/workspace", "symbol": "['alpha.client.Client.send']"})
        self.assertIn("[L:", out)
        self.assertEqual(tree(self.box.workspace), self.ws_before)

    def test_run_command_fallback_from_the_working_folder(self):
        list(fh.follow_memory_protocol(self.box, ISSUE))
        script = "/submission/skills/graph-memory/scripts/recall.py"   # mapped by the Sandbox
        out = self.box.run_command(f"python3 {script} --task-file /tmp/task.md --repo /workspace "
                                   "--symbol alpha.client.Client.send")
        self.assertIn("[L:", out)
        out = self.box.run_command(f"python3 {script} --task-file /tmp/task.md --symbol alpha.client.Client.send")
        self.assertIn("[L:", out)                     # --repo defaults to the working folder
        self.assertEqual(tree(self.box.workspace), self.ws_before)
        self.assertEqual(tree(self.sub), self.sub_before)

    def test_repo_defaults_to_workspace_inside_adks_copy(self):
        script = self.sub / "skills" / "graph-memory" / "scripts" / "recall.py"
        spec = importlib.util.spec_from_file_location("shipped_recall", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        skill = script.parent.parent
        self.assertEqual(mod.default_repo(cwd=str(skill), skill_dir=str(skill)), "/workspace")
        self.assertEqual(mod.default_repo(cwd=str(self.box.workspace), skill_dir=str(skill)),
                         str(self.box.workspace.resolve()))
        self.assertEqual(mod.split_symbols(["a.b", "['c.d', 'e']", "f,g"]), ["a.b", "c.d", "e", "f", "g"])


def _has_adk():
    try:
        return importlib.util.find_spec("google.adk") is not None
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(_has_adk(), "google-adk not installed (it is, in the competition wheelhouse)")
class RealAdk(SkillSmoke):
    """The same built skill through the real google-adk: load_skill_from_dir, then the exact
    wrapper code run_skill_script executes (materialize, chdir into a temp dir, runpy)."""

    def run_real(self, args):
        import subprocess
        import sys
        from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
        from google.adk.skills import load_skill_from_dir
        from google.adk.tools.skill_toolset import _SkillScriptCodeExecutor
        skill = load_skill_from_dir(self.sub / "skills" / "graph-memory")
        self.assertEqual(sorted(skill.resources.list_assets()), ["config.yaml", "lessons.txt"])
        code = _SkillScriptCodeExecutor(UnsafeLocalCodeExecutor(), 60)._build_wrapper_code(
            skill, "scripts/recall.py", [self.box._map(a) for a in args])
        r = subprocess.run([sys.executable, "-c", code], cwd=self.box.workspace, capture_output=True,
                           text=True, timeout=120)
        return r.returncode, (r.stdout + r.stderr).strip()

    def test_real_adk_loads_and_runs_the_skill(self):
        list(fh.follow_memory_protocol(self.box, ISSUE))
        args = memory_args(self.box.prompt) + ["--symbol", "alpha.client.Client.send"]
        rc, out = self.run_real(args)
        self.assertEqual(rc, 0, out)
        self.assertIn("[L:", out)
        self.assertEqual(tree(self.box.workspace), self.ws_before)
        self.assertEqual(tree(self.sub), self.sub_before)
        self.assertEqual(sorted(tree(self.box.tmp)), ["task.md"])


def memory_args(prompt):
    return list(fh.memory_protocol(prompt)["args"])


def fh_is_recall(step):
    from harness.swegemma import is_recall
    return is_recall(step)


if __name__ == "__main__":
    unittest.main()
