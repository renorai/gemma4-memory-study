"""The recap arm (memory/workmem.py): in-task working notes and the "where was I" page.

End to end in fake_harness.Sandbox with the built submission and the shipped prompt: notes are
written, a recap after a simulated compaction gives back the goal, hypothesis, edits and failed
attempts, nothing is written outside /tmp, the token budget holds, a later note supersedes an
earlier one, and the shuffled control shows another task's notes at the same length."""
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import fake_harness as fh
from memory import LessonStore, renor, workmem as W
from runner.build_submission import build

ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
ISSUE = ("Requests sent with `Client.send` have a wrong X-Signature header.\n"
         "Expected the header to match hmac.new(SECRET, url).")
SCRIPT = "scripts/memory/workmem.py"


def tree(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def note_protocol(prompt):
    """The note and recap arguments, read from the shipped prompt (so the tests use the
    instructions the model is given)."""
    sec = prompt[prompt.index("## Working notes"):]
    note = re.search(r'`file_path` "([^"]+)", `args` (\["note"[^\]]*\])', sec)
    recap = re.search(r'`args` (\["recap"[^\]]*\])', sec)
    if not note or not recap:
        raise AssertionError("working-notes section lost its note or recap arguments")
    return note.group(1), json.loads(note.group(2)), json.loads(recap.group(1))


class Unit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "notes")
        self.task = os.path.join(self.tmp.name, "task.md")
        Path(self.task).write_text(ISSUE)
        self.cfg = {"arm": "recap"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_wm(self, *argv, cfg=None):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            W.main(list(argv) + ["--dir", self.dir, "--task-file", self.task], config=cfg or self.cfg)
        return buf.getvalue().strip()

    def test_note_is_written_with_keys_from_flags_and_text(self):
        out = self.run_wm("note", "--kind", "edit", "--file", "/workspace/pkg/client.py",
                          "made Session.send pass the url to sign()")
        self.assertTrue(out.startswith("noted #1 (edit"), out)
        notes = W.load(self.dir)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["files"], ["pkg/client.py"])
        self.assertIn("Session.send", notes[0]["about"])
        self.assertEqual(notes[0]["subject"], "pkg/client.py")

    def test_note_text_and_keys_are_capped(self):
        self.run_wm("note", "--kind", "tried", *sum((["--about", f"m.f{i}"] for i in range(9)), []), "x " * 500)
        n = W.load(self.dir)[0]
        self.assertLessEqual(len(n["text"]), W.MAX_TEXT)
        self.assertLessEqual(len(n["about"]), W.MAX_KEYS)

    def test_supersession(self):
        self.run_wm("note", "--kind", "hypothesis", "--about", "Client.send", "the header is built before the body")
        self.run_wm("note", "--kind", "hypothesis", "--about", "sign", "sign() hashes the path, not the full url")
        self.run_wm("note", "--kind", "edit", "--file", "pkg/client.py", "first edit of client")
        self.run_wm("note", "--kind", "edit", "--file", "pkg/client.py", "second edit of client")
        self.run_wm("note", "--kind", "tried", "--about", "sign", "hash the full url in sign")
        self.run_wm("note", "--kind", "failed", "--about", "sign", "hashing the full url broke test_sign_query")
        self.run_wm("note", "--kind", "failed", "--about", "Client.send", "reordering headers in send changed nothing")
        notes = W.load(self.dir)
        gone = W.superseded(notes)
        self.assertEqual(gone, {1: 2, 3: 4, 5: 6})
        page = self.run_wm("recap", "--focus", "sign")
        self.assertIn("Hypothesis: sign() hashes the path", page)
        self.assertNotIn("built before the body", page)       # superseded hypothesis
        self.assertNotIn("first edit of client", page)        # superseded edit, same file
        self.assertIn("second edit of client", page)
        self.assertIn("broke test_sign_query", page)          # failed attempts accumulate ...
        self.assertIn("reordering headers", page)             # ... both are kept
        self.assertNotIn("hash the full url in sign\n", page + "\n")  # tried, then its failure

    def test_renor_correction_cue_supersedes_an_accumulating_note(self):
        self.run_wm("note", "--kind", "failed", "--about", "sign", "pytest tests/test_sign.py fails on the query string case")
        self.run_wm("note", "--kind", "failed", "--about", "sign",
                    "correction: pytest tests/test_sign.py fails on the fragment case, not the query string")
        self.assertEqual(W.superseded(W.load(self.dir)), {1: 2})

    def test_recap_ranks_by_focus_and_is_newest_first_on_ties(self):
        self.run_wm("note", "--kind", "failed", "--about", "parse_url", "parse_url rejects ipv6 hosts")
        self.run_wm("note", "--kind", "failed", "--about", "sign", "sign drops the port")
        self.run_wm("note", "--kind", "failed", "--about", "encode", "encode doubles the slash")
        page = self.run_wm("recap", "--focus", "parse_url")
        failed = page[page.index("Failed"):].splitlines()[1:]
        self.assertIn("ipv6", failed[0])
        self.assertIn("encode doubles", failed[1])      # no match: newest first
        self.assertIn("Goal: Requests sent with `Client.send`", page)

    def test_token_budget_holds(self):
        for i in range(60):
            kind = W.KINDS[i % len(W.KINDS)]
            self.run_wm("note", "--kind", kind, "--about", f"mod{i}.func{i}",
                        f"step {i}: " + "inspected the request signing path and its callers " * 4)
        for budget in (600, 300, 150):
            page = self.run_wm("recap", "--focus", "mod7.func7", cfg={"arm": "recap", "recap_budget": str(budget)})
            self.assertLessEqual(renor.count_tokens(page), budget, budget)
            self.assertIn("more; recap --focus", page)
        page = self.run_wm("recap", "--budget", "5000")      # the agent cannot raise the cap
        self.assertLessEqual(renor.count_tokens(page), W.DEFAULT_BUDGET)

    def test_every_nth_note_prints_the_recap(self):
        outs = [self.run_wm("note", "--kind", "tried", f"ran check {i}") for i in range(1, 13)]
        with_recap = [i + 1 for i, o in enumerate(outs) if "where was I" in o]
        self.assertEqual(with_recap, [6, 12])
        # The ack names the notes file, which memory/plumbing.py reads back (_NOTES_PATH) to check
        # where notes landed. Its token cost depends on the folder path, so the note's own cost is
        # measured with the path replaced; the path cost is stated in DESIGN.md section 10.
        path = os.path.join(self.dir, "notes.jsonl")
        self.assertIn(f"[{path}]", outs[0])
        self.assertLess(renor.count_tokens(outs[0].replace(path, "P")), 20)    # a plain note costs a few tokens

    def test_writes_only_under_tmp(self):
        for bad in ("/workspace/notes", "/home", "/tmp", "relative/dir"):
            with self.assertRaises(ValueError):
                W.safe_dir(bad)
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            W.main(["note", "hello", "--dir", "/workspace/x"], config=self.cfg)
        self.assertIn("working notes unavailable (ValueError", buf.getvalue())
        self.assertFalse(os.path.exists("/workspace/x"))

    def test_lesson_arms_have_notes_off(self):
        self.assertEqual(self.run_wm("note", "hello", cfg={"arm": "node"}), "working notes are off in this run")
        self.assertFalse(os.path.exists(self.dir))


def _steps(task_text, notes):
    head = "cat > /tmp/task.md <<'RENOR_ISSUE_EOF'\n" + task_text + "\nRENOR_ISSUE_EOF"
    steps = [{"tool": "run_command", "args": {"command": head}}]
    for i, (k, about, text) in enumerate(notes):
        if i % 2:
            steps.append({"tool": "run_skill_script", "args": {
                "skill_name": "graph-memory", "file_path": SCRIPT, "args": ["note", "--kind", k, "--about", about, text]}})
        else:
            steps.append({"tool": "run_command", "args": {
                "command": f"python3 skills/graph-memory/{SCRIPT} note --kind {k} --about {about} '{text}'"}})
    steps.append({"tool": "run_skill_script", "args": {"file_path": SCRIPT, "args": ["recap"]}})
    return steps


class Control(unittest.TestCase):
    OTHER = [("hypothesis", "Parser.feed", "feed() keeps a stale buffer between chunks"),
             ("edit", "lib/parser.py", "reset the buffer in Parser.feed"),
             ("failed", "Parser.close", "flushing in close() double-emits the last token"),
             ("next", "Parser.feed", "run tests/test_parser.py")]
    THIRD = [("failed", "Pool.acquire", "raising the pool size hides the leak"),
             ("hypothesis", "Pool.release", "release() skips connections closed by the server")]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pool = W.donors_from_traces({"other-1": _steps("Parser emits tokens twice", self.OTHER),
                                          "other-2": _steps("Pool leaks connections", self.THIRD),
                                          "empty": _steps("no notes here", [])})

    def tearDown(self):
        self.tmp.cleanup()

    def test_harvest_reads_both_tool_forms_and_the_issue(self):
        self.assertEqual([d["task"] for d in self.pool], ["other-1", "other-2"])
        self.assertEqual([n["text"] for n in self.pool[0]["notes"]], [t for _, _, t in self.OTHER])
        self.assertEqual(self.pool[0]["task_hash"], W.task_hash("Parser emits tokens twice\n"))
        self.assertTrue(W.control_ready(self.pool))
        self.assertFalse(W.control_ready(()))

    def test_donor_module_round_trip(self):
        path = Path(self.tmp.name) / "recap_donors.py"
        W.write_donor_module(self.pool, path)
        spec = importlib.util.spec_from_file_location("donors", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(list(mod.DONORS), self.pool)

    def test_control_shows_another_tasks_notes_at_the_same_length(self):
        own = [W.make_note(i + 1, k, t, [a]) for i, (k, a, t) in enumerate([
            ("hypothesis", "sign", "sign() hashes the path, not the full url"),
            ("edit", "pkg/client.py", "pass the full url to sign() in Session.send"),
            ("failed", "Client.send", "reordering headers in send changed nothing"),
            ("next", "sign", "run tests/test_sign.py")])]
        real, _ = W.page(own, "sign", 600, ISSUE)
        ctrl = W.control_page(own, "sign", 600, ISSUE, repo=None, donors=self.pool)
        self.assertIn("Goal: Requests sent with `Client.send`", ctrl)      # own goal kept
        for _, _, t in [("", "", "sign() hashes"), ("", "", "reordering headers")]:
            self.assertNotIn(t, ctrl)                                       # own notes gone
        self.assertTrue(any(t in ctrl for _, _, t in self.OTHER + self.THIRD))
        self.assertLessEqual(renor.count_tokens(ctrl), renor.count_tokens(real))
        self.assertGreaterEqual(renor.count_tokens(ctrl), 0.6 * renor.count_tokens(real))

    def test_a_task_never_gets_its_own_notes(self):
        own_hash = W.task_hash("Parser emits tokens twice\n")
        notes = W.donor_notes(own_hash, donors=self.pool)
        self.assertTrue(notes)
        self.assertFalse(any(n["text"] in {t for _, _, t in self.OTHER} for n in notes))

    def test_without_donors_the_control_says_so(self):
        out = W.control_page([W.make_note(1, "tried", "x")], "", 600, ISSUE, donors=())
        self.assertIn("unavailable (no donor notes", out)


class FakeHarness(unittest.TestCase):
    """The built recap arm in the fake sandbox, driven by the arguments in the shipped prompt."""

    ARM = "recap"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        fh.write_repos(d / "repos")
        self.sub = d / "sub"
        self.report = build(self.ARM, LessonStore([]), ALIAS, self.sub)
        self.box = fh.Sandbox(self.sub, d / "repos" / "alpha", d / "box")
        self.ws_before, self.sub_before = tree(self.box.workspace), tree(self.sub)
        self.script, self.note_args, self.recap_args = note_protocol(self.box.prompt)
        self.real_tmp_before = os.path.exists("/tmp/renor-notes")

    def tearDown(self):
        self.tmp.cleanup()

    def wm(self, args):
        # the sandbox maps /tmp to its own folder; the shipped default is /tmp/renor-notes
        return self.box.run_skill_script("graph-memory", self.script,
                                         args + ["--dir", "/tmp/renor-notes", "--task-file", "/tmp/task.md"])

    def note(self, kind, about, text):
        a = list(self.note_args)
        a[a.index("--kind") + 1], a[a.index("--about") + 1], a[-1] = kind, about, text
        return self.wm(a)

    def test_build_is_valid_and_prompt_names_the_shipped_script(self):
        self.assertTrue(self.report["ok"], self.report["errors"])
        self.assertEqual(self.script, SCRIPT)
        self.assertTrue((self.sub / "skills" / "graph-memory" / SCRIPT).is_file())
        self.assertIn(f"arm: {self.ARM}", (self.sub / "skills/graph-memory/assets/config.yaml").read_text())
        self.assertIn("recap_budget: 600", (self.sub / "skills/graph-memory/assets/config.yaml").read_text())

    def test_notes_then_recap_after_a_simulated_compaction(self):
        calls = list(fh.follow_memory_protocol(self.box, ISSUE))
        self.assertTrue(calls[1][2].startswith("working notes: on"), calls[1][2])
        # the agent works and takes notes ...
        self.assertTrue(self.note("hypothesis", "Client.send", "the header is built before the url is final").startswith("noted #1"))
        self.note("tried", "sign", "python -m pytest -x -q tests/test_client.py: 1 failure in test_signature")
        self.note("hypothesis", "sign", "sign() hashes the path, not the full url")
        self.note("edit", "alpha/client.py", "Client.send now passes req.url to sign()")
        self.note("failed", "Client.send", "moving the header after _transport changed nothing")
        self.assertIn("where was I", self.note("next", "sign", "rerun tests/test_client.py, then submit"))  # 6th
        # ... compaction: the agent's history is a summary now; it holds only what the recap returns
        history = []
        rec = self.wm(list(self.recap_args))
        history.append(rec)
        self.assertIn("Goal: Requests sent with `Client.send` have a wrong X-Signature header.", rec)
        self.assertIn("Hypothesis: sign() hashes the path", rec)
        self.assertNotIn("built before the url", rec)
        self.assertIn("Next: rerun tests/test_client.py", rec)
        self.assertIn("alpha/client.py", rec[rec.index("Edited:"):])
        self.assertIn("moving the header after _transport changed nothing", rec[rec.index("Failed"):])
        self.assertLessEqual(renor.count_tokens(rec), 600)
        # a scripted agent that skips any attempt the recap lists as failed does not repeat it
        planned = ["moving the header after _transport", "pass the full url to sign"]
        self.assertEqual([p for p in planned if p not in rec], ["pass the full url to sign"])

    def test_nothing_is_written_outside_tmp(self):
        list(fh.follow_memory_protocol(self.box, ISSUE))
        for i in range(8):
            self.note("tried", "Client.send", f"check {i}")
        self.wm(list(self.recap_args))
        self.assertEqual(tree(self.box.workspace), self.ws_before)   # what submit_patch diffs
        self.assertEqual(tree(self.sub), self.sub_before)            # no __pycache__ in the skill
        self.assertEqual(sorted(tree(self.box.tmp)), ["renor-notes/notes.jsonl", "task.md"])
        self.assertEqual(os.path.exists("/tmp/renor-notes"), self.real_tmp_before)

    def test_run_command_fallback(self):
        list(fh.follow_memory_protocol(self.box, ISSUE))
        script = f"/submission/skills/graph-memory/{SCRIPT}"
        out = self.box.run_command(f"python3 {script} note --kind failed --about sign "
                                   "'patching sign alone left test_signature red' --dir /tmp/renor-notes")
        self.assertTrue(out.startswith("noted #1"), out)
        out = self.box.run_command(f"python3 {script} recap --focus sign --dir /tmp/renor-notes "
                                   "--task-file /tmp/task.md")
        self.assertIn("patching sign alone", out)
        self.assertEqual(tree(self.box.workspace), self.ws_before)
        self.assertEqual(tree(self.sub), self.sub_before)


class FakeHarnessControl(FakeHarness):
    """The control, after the operator step: donors harvested from (fake) recap-arm traces are
    written into the built skill's memory/recap_donors.py."""
    ARM = "recap_shuffled"

    def setUp(self):
        super().setUp()
        pool = W.donors_from_traces({"other-1": _steps("Parser emits tokens twice", Control.OTHER),
                                     "other-2": _steps("Pool leaks connections", Control.THIRD)})
        W.write_donor_module(pool, self.sub / "skills/graph-memory/scripts/memory/recap_donors.py")
        self.sub_before = tree(self.sub)

    def test_notes_then_recap_after_a_simulated_compaction(self):
        list(fh.follow_memory_protocol(self.box, ISSUE))
        self.note("hypothesis", "sign", "sign() hashes the path, not the full url")
        self.note("failed", "Client.send", "moving the header after _transport changed nothing")
        rec = self.wm(list(self.recap_args))
        self.assertIn("Goal: Requests sent with `Client.send`", rec)
        self.assertNotIn("sign() hashes the path", rec)     # never the agent's own notes
        self.assertNotIn("_transport", rec)
        self.assertTrue(any(t in rec for _, _, t in Control.OTHER + Control.THIRD), rec)
        self.assertEqual(len(W.load(str(self.box.tmp / "renor-notes"))), 2)   # own notes still recorded

    def test_run_command_fallback(self):
        list(fh.follow_memory_protocol(self.box, ISSUE))
        script = f"/submission/skills/graph-memory/{SCRIPT}"
        out = self.box.run_command(f"python3 {script} note --kind failed --about sign "
                                   "'patching sign alone left test_signature red' --dir /tmp/renor-notes")
        self.assertTrue(out.startswith("noted #1"), out)
        out = self.box.run_command(f"python3 {script} recap --focus sign --dir /tmp/renor-notes "
                                   "--task-file /tmp/task.md")
        self.assertNotIn("patching sign alone", out)
        self.assertIn("where was I", out)
        self.assertEqual(tree(self.box.workspace), self.ws_before)


def _has_adk():
    try:
        return importlib.util.find_spec("google.adk") is not None
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(_has_adk(), "google-adk not installed (it is, in the competition wheelhouse)")
class RealAdk(FakeHarness):
    """The built recap skill through the real google-adk: load_skill_from_dir, then the exact
    wrapper code run_skill_script executes (materialize scripts/ incl. scripts/memory/, chdir into a
    temp dir, runpy)."""

    def wm(self, args):
        import subprocess
        import sys
        from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
        from google.adk.skills import load_skill_from_dir
        from google.adk.tools.skill_toolset import _SkillScriptCodeExecutor
        skill = load_skill_from_dir(self.sub / "skills" / "graph-memory")
        self.assertIn("memory/workmem.py", skill.resources.list_scripts())
        args = args + ["--dir", "/tmp/renor-notes", "--task-file", "/tmp/task.md"]
        code = _SkillScriptCodeExecutor(UnsafeLocalCodeExecutor(), 60)._build_wrapper_code(
            skill, self.script, [self.box._map(a) for a in args])
        r = subprocess.run([sys.executable, "-c", code], cwd=self.box.workspace, capture_output=True,
                           text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        return (r.stdout + r.stderr).strip()


if __name__ == "__main__":
    unittest.main()
