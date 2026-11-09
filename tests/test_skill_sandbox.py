"""The first Modal recap check (9 Nov, commit ff9bf75d) failed: under swegemma --sandbox subprocess,
run_skill_script runs the skill's wrapper through the sandbox, but only the COMMAND is rewritten;
the arguments "/workspace" and "/tmp/task.md" stay literal and name the host's folders. recall.py
printed `memory unavailable (no repository at /workspace; pass --repo)`, never `working notes: on`,
the prompt then told the agent to skip working notes, and no note was ever written. Notes would also
have gone to the host's /tmp/renor-notes, shared by every concurrent task.

These tests reproduce that sandbox (fake_harness.Sandbox mode "subprocess": literal arguments,
TMPDIR/HOME/PYTHONPATH of the task, ADK's self-extracting wrapper) and check the fix: the scripts
find the task's own folders, notes are per task, the first recall proves the plumbing is alive,
and memory/plumbing.py flags a run whose plumbing failed."""
import json
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path

import fake_harness as fh
from memory import Lesson, LessonStore, renor
from memory.plumbing import plumbing
from runner.build_submission import build

ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
ISSUE = "Requests sent with `Client.send` have a wrong X-Signature header."
PROBE = ("import json, os, sys\n"
         "print(json.dumps({'argv': sys.argv[1:], 'tmpdir': os.environ.get('TMPDIR'),\n"
         "                  'host_workspace': os.path.isdir('/workspace')}))\n")


def tree(root):
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def note_args(sub):
    """The notes sub-agent's call (built sub_agents/notes.yaml): script and argument template."""
    text = (Path(sub) / "sub_agents" / "notes.yaml").read_text()
    script = re.search(r'`file_path`\s+"([^"]+)"', text).group(1)
    return script, json.loads(re.search(r'`args` (\["note"[^\]]*\])', text).group(1)), ["recap"]


class Base(unittest.TestCase):
    ARM = "recap"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        fh.write_repos(self.d / "repos")
        lessons = LessonStore([Lesson(
            title="Signature bugs live in the signer",
            text="The function that calls hmac.new builds the X-Signature header; fix it there.",
            outcome="success", keys=("name:send", "api:hmac.new"), anchor="beta.client.Session.send",
            repo="beta", source_task="beta-header", source_model=ALIAS)])
        self.sub = self.d / "sub"
        build(self.ARM, None if self.ARM.startswith("recap") else lessons, ALIAS, self.sub,
              exclude_repos=() if self.ARM.startswith("recap") else ("alpha",))
        self.host_notes = os.path.exists("/tmp/renor-notes")

    def tearDown(self):
        self.tmp.cleanup()

    def box(self, name, mode="subprocess"):
        return fh.Sandbox(self.sub, self.d / "repos" / "alpha", self.d / name, mode=mode)

    def note(self, box, kind, about, text):
        script, args, _ = note_args(self.sub)
        a = list(args)
        a[-1] = f"{kind} {about}: {text}"
        return box.run_skill_script("graph-memory", script, a)

    def recap(self, box):
        script, _, args = note_args(self.sub)
        return box.run_skill_script("graph-memory", script, list(args))


class SubprocessSandboxFacts(Base):
    def test_skill_arguments_are_not_rewritten(self):
        """The 9 Nov failure's cause: the script sees the literal arguments and the host's folders."""
        (self.sub / "skills" / "graph-memory" / "scripts" / "probe.py").write_text(PROBE)
        box = self.box("box")
        got = json.loads(box.run_skill_script("graph-memory", "scripts/probe.py",
                                              ["--task-file", "/tmp/task.md", "--repo", "/workspace"]))
        self.assertEqual(got["argv"], ["--task-file", "/tmp/task.md", "--repo", "/workspace"])
        self.assertEqual(got["tmpdir"], str(box.tmp))
        if not os.path.isdir("/workspace"):          # the host of every Kaggle/Modal run
            self.assertFalse(got["host_workspace"])


class RecapArmInSubprocessSandbox(Base):
    def test_first_recall_proves_the_plumbing_is_alive(self):
        box = self.box("box")
        calls = list(fh.follow_memory_protocol(box, ISSUE))
        first = calls[1][2].splitlines()[0]
        self.assertTrue(first.startswith("working notes: on (notes "), calls[1][2])
        self.assertIn(str(box.tmp) + "/renor-notes/", first)
        self.assertNotIn("memory unavailable", calls[1][2])

    def test_notes_and_recap_stay_in_the_task(self):
        box = self.box("box")
        ws_before = tree(box.workspace)
        list(fh.follow_memory_protocol(box, ISSUE))
        ack = self.note(box, "failed", "Client.send", "moving the header after _transport changed nothing")
        self.assertTrue(ack.startswith("noted #1"), ack)
        self.assertIn(f"[{box.tmp}/renor-notes/", ack)
        rec = self.recap(box)
        self.assertIn("Goal: Requests sent with `Client.send`", rec)    # read from the TASK's /tmp/task.md
        self.assertIn("moving the header after _transport changed nothing", rec)
        self.assertEqual(tree(box.workspace), ws_before)                # nothing for submit_patch to diff
        files = tree(box.tmp)
        self.assertIn("task.md", files)
        self.assertEqual([f for f in files if f.endswith("notes.jsonl")],
                         [f"renor-notes/{f.split('/')[1]}/notes.jsonl" for f in files if f.endswith("notes.jsonl")])
        self.assertEqual(os.path.exists("/tmp/renor-notes"), self.host_notes)   # the host's /tmp untouched

    def test_two_concurrent_tasks_read_only_their_own_notes(self):
        boxes = [self.box("box_a"), self.box("box_b")]
        for b in boxes:
            list(fh.follow_memory_protocol(b, ISSUE))
        outs = {}

        def work(i, b):
            for k in range(4):
                self.note(b, "tried", "Client.send", f"task {'AB'[i]} check {k}")
            outs[i] = self.recap(b)
        threads = [threading.Thread(target=work, args=(i, b)) for i, b in enumerate(boxes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertIn("task A check 3", outs[0])
        self.assertNotIn("task B", outs[0])
        self.assertIn("task B check 3", outs[1])
        self.assertNotIn("task A", outs[1])
        notes = [next(p for p in b.tmp.rglob("notes.jsonl")) for b in boxes]
        self.assertNotEqual(notes[0], notes[1])


class LessonArmInSubprocessSandbox(Base):
    ARM = "node"

    def test_recall_finds_the_task_repo(self):
        box = self.box("box")
        calls = list(fh.follow_memory_protocol(box, ISSUE))
        calls.append(box.recall_call(ISSUE, ["alpha.client.Client.send"]))
        lines = calls[2][2].splitlines()
        self.assertTrue(lines[0].startswith("working notes: off"), calls[2][2])
        self.assertFalse(any(l.startswith("memory error") for l in lines), calls[2][2])
        self.assertIn("hmac.new", calls[2][2])

    def test_container_mode_still_works(self):
        box = self.box("box", mode="container")
        calls = list(fh.follow_memory_protocol(box, ISSUE))
        calls.append(box.recall_call(ISSUE, ["alpha.client.Client.send"]))
        self.assertTrue(calls[2][2].startswith("working notes: off"), calls[2][2])
        self.assertIn("hmac.new", calls[2][2])


class Prompt(unittest.TestCase):
    def test_no_disk_hunting_and_no_quiet_skip(self):
        text = (Path(__file__).resolve().parents[1] / "runner" / "prompt_parts" / "memory.md").read_text()
        self.assertNotIn("find /", text)
        self.assertNotIn("Otherwise skip this section", text)
        self.assertNotIn("python3 {{SKILL_DIR}}", text)
        self.assertIn("memory error:", text)


def _trace(path, calls, prompt_tokens):
    """An ATIF trace in the shape the real harness writes: one tool call per step, its output in
    observation.content. `calls` are (author, tool, args, output); `prompt_tokens` holds one value
    per coder step (sub-agent steps are not coder turns)."""
    steps = [{"step_id": 1, "source": "user", "message": "issue"}]
    pts = iter(prompt_tokens)
    for i, (author, tool, args, out) in enumerate(calls, 2):
        pt = next(pts) if author == "coder" else 500
        if tool is None:            # a sub-agent's final answer: text, no tool call
            steps.append({"step_id": i, "source": "agent", "extra": {"author": author}, "message": out,
                          "metrics": {"prompt_tokens": pt, "completion_tokens": 10}})
            continue
        if tool in ("notes", "explorer"):   # the coder's agent_tool call carries no observation
            steps.append({"step_id": i, "source": "agent", "extra": {"author": author},
                          "tool_calls": [{"tool_call_id": f"c{i}", "function_name": tool, "arguments": args}],
                          "metrics": {"prompt_tokens": pt, "completion_tokens": 10}})
            continue
        if tool == "run_skill_script":
            content = json.dumps({"stdout": out, "status": "success"})
        elif tool == "notes":
            content = json.dumps({"result": out})
        else:
            content = out
        steps.append({"step_id": i, "source": "agent", "extra": {"author": author},
                      "tool_calls": [{"tool_call_id": f"c{i}", "function_name": tool, "arguments": args}],
                      "observation": {"content": content},
                      "metrics": {"prompt_tokens": pt, "completion_tokens": 10}})
    Path(path).write_text(json.dumps({"steps": steps}))
    return path


class PlumbingAssertions(Base):
    """Automatic recall: the coder calls the `notes` tool; the sub-agent runs workmem `note --page`
    (acks in the sub-agent's steps) and returns the page, which output_key injects from then on."""

    def calls(self, box, notes_before_compaction=True):
        out = [("coder",) + c for c in fh.follow_memory_protocol(box, ISSUE)]
        script, args, _ = note_args(self.sub)
        for k in range(2 if notes_before_compaction else 0):
            a = list(args)
            a[-1] = f"tried Client.send: check {k}"
            page = box.run_skill_script("graph-memory", script, a)
            out.append(("coder", "notes", {"request": a[-1]}, None))
            out.append(("notes", "run_skill_script",
                        {"skill_name": "graph-memory", "file_path": script, "args": a}, page))
            out.append(("notes", None, None, page.split("\n", 1)[1]))
        while sum(1 for c in out if c[0] == "coder") < 7:
            out.append(("coder", "run_command", {"command": "ls"}, "x"))
        return out

    def test_injected_page_before_each_compaction(self):
        box = self.box("box")
        pts = [9000, 10000, 11000, 12000, 13000, 4000, 4500]          # compaction at coder turn 5
        ok = plumbing(_trace(self.d / "ok.json", self.calls(box), pts), "recap")
        self.assertEqual(ok["skill_status"], "on")
        self.assertTrue(ok["writes_in_task_dir"])
        self.assertEqual(ok["notes_written"], 2)
        self.assertEqual(ok["injected_share"], 1.0)
        self.assertTrue(ok["recall_runs_after_compaction"])
        self.assertEqual(len(ok["recap_tokens_per_call"]), 2)
        self.assertTrue(all(0 < t <= 1500 for t in ok["recap_tokens_per_call"]))
        self.assertFalse(ok["plumbing_failed"])

    def test_no_note_before_the_compaction_is_flagged(self):
        box = self.box("box")
        pts = [9000, 10000, 11000, 12000, 13000, 4000, 4500]
        miss = plumbing(_trace(self.d / "miss.json", self.calls(box, notes_before_compaction=False), pts),
                        "recap")
        self.assertEqual(miss["notes_written"], 0)
        self.assertIs(miss["recall_runs_after_compaction"], False)
        self.assertTrue(miss["plumbing_failed"])

    def test_explorer_finding_alone_counts_as_injected_state(self):
        """The fallback: no note written, but the explorer's answer (output_key explorer_finding)
        was in state before the compaction. Plumbing works; note writes are an adherence outcome."""
        box = self.box("box")
        calls = self.calls(box, notes_before_compaction=False)
        calls[3:3] = [("coder", "explorer", {"request": "where is the signer?"}, None),
                      ("explorer", None, None, "alpha/auth.py:3 sign() builds the X-Signature header")]
        calls = calls[:-1]                      # keep 7 coder turns
        pts = [9000, 10000, 11000, 12000, 13000, 4000, 4500]
        row = plumbing(_trace(self.d / "ex.json", calls, pts), "recap")
        self.assertEqual(row["notes_written"], 0)
        self.assertEqual(row["explorer_injections"], 1)
        self.assertTrue(row["recall_runs_after_compaction"])
        self.assertFalse(row["plumbing_failed"])

    def test_the_9_nov_failure_is_flagged(self):
        calls = [("coder", "run_skill_script", {"skill_name": "graph-memory", "file_path": "scripts/recall.py",
                                                "args": ["--task-file", "/tmp/task.md", "--repo", "/workspace"]},
                  "memory unavailable (no repository at /workspace; pass --repo)")]
        row = plumbing(_trace(self.d / "old.json", calls, [5000]), "recap")
        self.assertIsNone(row["skill_status"])
        self.assertEqual(row["notes_written"], 0)
        self.assertTrue(row["plumbing_failed"])
        self.assertIsNone(plumbing(_trace(self.d / "none.json", calls, [5000]), "none")["plumbing_failed"])


class InjectedPage(Base):
    def test_page_is_capped_oldest_first(self):
        from memory import workmem as W
        notes = [W.make_note(i, "tried", f"check number {i} " + "word " * 25) for i in range(1, 120)]
        page = W.inject_page(notes, ISSUE, 1500)
        self.assertLessEqual(renor.count_tokens(page), 1500)
        self.assertIn("check number 119", page)                    # newest kept
        self.assertNotIn("check number 1 ", page)                  # oldest dropped first
        self.assertIn("older notes dropped", page)

    def test_note_line_with_leading_kind_returns_the_page(self):
        box = self.box("box")
        list(fh.follow_memory_protocol(box, ISSUE))
        out = self.note(box, "failed", "Client.send", "moving the header changed nothing")
        self.assertTrue(out.startswith("noted #1 (failed: Client.send)"), out)
        self.assertIn("where was I", out)                          # --page: the page follows the ack
        self.assertIn("Failed", out)


class ShuffledControlInjection(Base):
    ARM = "recap_shuffled"

    def test_the_control_injects_another_tasks_notes(self):
        from memory import workmem as W
        pool = [{"task": "other", "task_hash": W.task_hash("Parser emits tokens twice\n"),
                 "notes": [{"kind": "failed", "text": "the lexer cache double-counts tokens",
                            "about": ["Lexer.next"], "files": []}]},
                {"task": "third", "task_hash": W.task_hash("Pool leaks connections\n"),
                 "notes": [{"kind": "hypothesis", "text": "release() skips closed sockets",
                            "about": ["Pool.release"], "files": []}]}]
        W.write_donor_module(pool, self.sub / "skills/graph-memory/scripts/memory/recap_donors.py")
        box = self.box("box")
        list(fh.follow_memory_protocol(box, ISSUE))
        out = self.note(box, "failed", "Client.send", "moving the header changed nothing")
        self.assertTrue(out.startswith("noted #1"), out)
        page = out.split("\n", 1)[1]
        self.assertNotIn("moving the header", page)               # never the agent's own notes
        self.assertTrue("lexer cache" in page or "release() skips" in page, page)


if __name__ == "__main__":
    unittest.main()
